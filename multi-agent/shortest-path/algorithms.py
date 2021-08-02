import yaml
import networkx as nx
import cvxpy as cp
import numpy as np


def parse_yaml(filepath):
    # Parse YAML
    with open(filepath, 'r') as stream:
        try:
            raw_dict = yaml.safe_load(stream)
            print(filepath, "loaded successfully")
        except yaml.YAMLError as e:
            print(e)
    # Return
    return raw_dict


def create_graph(raw_dict):
    # Parse dictionary content
    dimensions = raw_dict['map']['dimensions']
    obstacles = raw_dict['map']['obstacles']
    # Create Graph
    graph = nx.grid_2d_graph(dimensions[0], dimensions[1])
    for obs in obstacles:
        graph.nodes[tuple(obs)]['obstacle'] = True
    # Return
    return graph


# Calculates shortest path for agent0 with fixed obstacles and dynamic obstacles from
# other agents' generated CBS paths/solution.
def mapf(graph, raw_solution):
    # Extract Solution Data except agent0
    makespan = raw_solution['statistics']['makespan']
    schedule = raw_solution['schedule']
    agent0 = schedule.pop('agent0')
    # occupied_nodes[t] = list of nodes occupied by agents (except agent0) at time=t.
    occupied_nodes = [[] for _ in range(makespan + 1)]
    for path in schedule.values():
        for pos in path:
            occupied_nodes[pos['t']].append((pos['x'], pos['y']))

    # edges[t] = list of edges going from node at time t to t+1
    # weights[t] = list of weights of corresponding edges
    edges = []
    edge2idx = []
    weights = []
    for t in range(makespan):
        edges_t = []
        edge2idx_t = {}
        weights_t = []
        for n in graph.nodes:
            next_nodes = [nei for nei in graph[n]]
            next_nodes.append(n)
            for next_node in next_nodes:
                edge = (n, next_node)
                edge2idx_t[edge] = len(edges_t)
                edges_t.append(edge)
                if graph.nodes[next_node].get('obstacle') or next_node in occupied_nodes[t + 1]:
                    weights_t.append(1000)
                elif next_node == n:
                    weights_t.append(0.99)
                else:
                    weights_t.append(1)
        edges.append(edges_t)
        edge2idx.append(edge2idx_t)
        weights.append(weights_t)

    # Matrix A and b
    A = np.zeros((makespan, len(graph.nodes), 2 * len(graph.edges) + len(graph.nodes)))
    b = np.zeros((makespan, len(graph.nodes)))
    start = (agent0[0]['x'], agent0[0]['y'])
    goal = (agent0[-1]['x'], agent0[-1]['y'])
    for t in range(makespan):
        for i, node in enumerate(graph.nodes):
            for nei in graph[node]:
                if (node, nei) in edge2idx[t]:
                    j = edge2idx[t][node, nei]
                    A[t, i, j] = 1
                if (nei, node) in edge2idx[t]:
                    j = edge2idx[t][nei, node]
                    A[t, i, j] = -1
            if node == start:
                b[t, i] = 1
            elif node == goal:
                b[t, i] = -1

    # Solve with cvxpy
    x = cp.Variable((makespan, 2 * len(graph.edges) + len(graph.nodes)))
    cost = cp.sum(cp.multiply(x, np.array(weights)))
    prob = cp.Problem(cp.Minimize(cost), [A @ x == b])
    value = prob.solve(solver=cp.GUROBI)
    if value == float('inf'):
        print("shortest path LP failed")
        return []


def main_mapf(problem_file, solution_file):
    raw_problem = parse_yaml(problem_file)
    raw_solution = parse_yaml(solution_file)
    graph = create_graph(raw_problem)
    mapf(graph, raw_solution)
