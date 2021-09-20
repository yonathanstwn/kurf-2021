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
    for t in range(1, len(path)):
        if path[t] in obstacles or path[t] in occupied_nodes[t - 1].values() or occupied_nodes[t - 1].get(path[t]) == \
                path[t - 1]:
            return False
    return True


# Calculates shortest path for specified agent with fixed obstacles and dynamic obstacles from
# other agents' generated CBS paths/solution.
def mapf(graph, raw_solution, agent_name, start, goal):
    # Extract Solution Data with specified agent stored separately.
    makespan = raw_solution['statistics']['makespan']
    schedule = raw_solution['schedule']
    agent = schedule.pop(agent_name)

    # occupied_nodes[t] = list of edges corresponding to movement of agents (except specified agent)
    # from time=t to time=t+1
    occupied_nodes = [{} for _ in range(makespan)]
    for path in schedule.values():
        if len(path) == 1:
            for i in range(makespan):
                temp = (path[0]['x'], path[0]['y'])
                occupied_nodes[i][temp] = temp
        for t in range(len(path) - 1):
            temp = (path[t + 1]['x'], path[t + 1]['y'])
            occupied_nodes[t][(path[t]['x'], path[t]['y'])] = temp
            if t + 1 == len(path) - 1:
                for i in range(t + 1, makespan):
                    occupied_nodes[i][temp] = temp

    edges = []
    weights = []
    edge_t_2idx = {}  # key: tuple[edge, t] where t=0 means edge going from time=0 to time=1
    for t in range(makespan):
        for n in graph.nodes:
            next_nodes = [nei for nei in graph[n]]
            next_nodes.append(n)
            for next_node in next_nodes:
                # Check to avoid collisions with other agents
                if next_node not in occupied_nodes[t].values() and occupied_nodes[t].get(next_node) != n:
                    edge = (n, next_node)
                    idx = len(edges)
                    edge_t_2idx[(edge, t)] = idx
                    edges.append(edge)
                    if graph.nodes[next_node].get('obstacle'):
                        weights.append(1000)
                    elif next_node == n and n == goal:
                        weights.append(0.90)
                    elif next_node == n:
                        weights.append(0.99)
                    else:
                        weights.append(1)

    # Matrix A and b
    A = np.zeros((len(graph.nodes) * (makespan + 1), len(edges)))
    b = np.zeros(len(graph.nodes) * (makespan + 1))
    if start == goal:
        print("Automatic success - same start and goal")
        return [start], True
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
            elif n == goal and t == makespan:
                b[len(graph.nodes) * t + idx] = -1

    # Solve with cvxpy
    x = cp.Variable(len(edges), boolean=True)
    cost = cp.sum(cp.multiply(x, weights))
    prob = cp.Problem(cp.Minimize(cost), [A @ x == b])
    value = prob.solve(solver=cp.ECOS_BB)
    if value == float('inf'):
        print("Shortest Path LP FAIL!")
        return [], False

    # Extract path
    path = [start]
    success = False
    for i, v in enumerate(x.value):
        if round(v) == 1 and edges[i][0] == path[-1]:
            path.append(edges[i][1])
    # Remove trailing duplicate nodes, i.e. when agent is already in goal node.
    idx = len(path) - 1
    while path[idx] == path[idx - 1]:
        path.pop()
        idx -= 1

    # Sanity Check
    cbs_path = [(pos['x'], pos['y']) for pos in agent]
    if len(path) == len(cbs_path) and check_obstacles(graph, path, occupied_nodes):
        success = True
        print("Multi-Agent SP Success!")
    else:
        print("Multi-Agent SP Fail!")

    # Return
    return path, success


def main_mapf(problem_file, solution_file, agent_name):
    raw_problem = parse_yaml(problem_file)
    raw_solution = parse_yaml(solution_file)
    graph = create_graph(raw_problem)
    agent = raw_solution['schedule'][agent_name]
    start = (agent[0]['x'], agent[0]['y'])
    goal = (agent[-1]['x'], agent[-1]['y'])
    return mapf(graph, raw_solution, agent_name, start, goal)
