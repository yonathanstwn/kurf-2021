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


def check_obstacles(graph, path, occupied_nodes):
    obstacles = [n for n in graph.nodes if graph.nodes[n].get('obstacle')]
    for t, n in enumerate(path):
        if n in obstacles or n in occupied_nodes[t]:
            return False
    return True


# Calculates shortest path for agent0 with fixed obstacles and dynamic obstacles from
# other agents' generated CBS paths/solution.
def mapf(graph, raw_solution):
    # Extract Solution Data with agent0 stored separately.
    makespan = raw_solution['statistics']['makespan']
    schedule = raw_solution['schedule']
    agent0 = schedule.pop('agent0')

    # occupied_nodes[t] = list of nodes occupied by agents (except agent0) at time=t.
    occupied_nodes = [[] for _ in range(makespan + 1)]
    for path in schedule.values():
        for pos in path:
            occupied_nodes[pos['t']].append((pos['x'], pos['y']))

    edges = []
    weights = []
    edge_t_2idx = {}  # key: tuple[edge, t] where t=0 means edge going from time=0 to time=1
    for t in range(makespan):
        for n in graph.nodes:
            next_nodes = [nei for nei in graph[n]]
            next_nodes.append(n)
            for next_node in next_nodes:
                if next_node not in occupied_nodes[t + 1]:
                    edge = (n, next_node)
                    idx = len(edges)
                    edge_t_2idx[(edge, t)] = idx
                    edges.append(edge)
                    if graph.nodes[next_node].get('obstacle'):
                        weights.append(1000)
                    elif next_node == n:
                        weights.append(0.99)
                    else:
                        weights.append(1)

    # Matrix A and b
    A = np.zeros((len(graph.nodes) * (makespan + 1), len(edges)))
    b = np.zeros(len(graph.nodes) * (makespan + 1))
    start = (agent0[0]['x'], agent0[0]['y'])
    goal = (agent0[-1]['x'], agent0[-1]['y'])
    if start == goal:
        print("Automatic success - same start and goal")
        return [start], True
    goal_time = agent0[-1]['t']
    for t in range(makespan + 1):
        for idx, n in enumerate(graph.nodes):
            neighbours = [nei for nei in graph[n]]
            neighbours.append(n)
            for nei in neighbours:
                j = edge_t_2idx.get(((n, nei), t))
                if j is not None:
                    A[len(graph.nodes) * t + idx, j] = 1
                j = edge_t_2idx.get(((nei, n), t - 1))
                if j is not None:
                    A[len(graph.nodes) * t + idx, j] = -1
            if n == start and t == 0:
                b[idx] = 1
            elif n == goal and t == goal_time:
                b[len(graph.nodes) * t + idx] = -1

    # Solve with cvxpy
    x = cp.Variable(len(edges), boolean=True)
    cost = cp.sum(cp.multiply(x, weights))
    prob = cp.Problem(cp.Minimize(cost), [A @ x == b])
    value = prob.solve(solver=cp.ECOS_BB)
    if value == float('inf'):
        print("shortest path LP failed")
        return []

    # Sanity Check
    path = [start]
    success = False
    for i, v in enumerate(x.value):
        if round(v) == 1 and edges[i][0] == path[-1]:
            path.append(edges[i][1])
    cbs_path = [(pos['x'], pos['y']) for pos in agent0]
    if len(path) == len(cbs_path) and check_obstacles(graph, path, occupied_nodes):
        success = True
        print("Multi-Agent SP Success!")
    else:
        print("Multi-Agent SP Fail!")

    # Return
    return path, success


def main_mapf(problem_file, solution_file):
    raw_problem = parse_yaml(problem_file)
    raw_solution = parse_yaml(solution_file)
    graph = create_graph(raw_problem)
    return mapf(graph, raw_solution)
