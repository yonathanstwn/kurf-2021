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


def inv_mapf(graph, raw_solution, desired_path, agent_name):
    # Extract Solution Data with specified agent stored separately.
    ori_makespan = raw_solution['statistics']['makespan']
    schedule = raw_solution['schedule']
    schedule.pop(agent_name)

    # Determine max t
    max_t = max(len(desired_path) - 1, ori_makespan)

    # occupied_nodes[t] = list of nodes occupied by agents (except agent0) at time=t.
    occupied_nodes = [[] for _ in range(ori_makespan + 1)]
    for path in schedule.values():
        for pos in path:
            occupied_nodes[pos['t']].append((pos['x'], pos['y']))

    # Auxiliary variables
    edges = []
    edge_t_2idx = {}  # key: tuple[edge, t] where t=0 means edge going from time=0 to time=1
    for t in range(max_t):
        for i, n in enumerate(graph.nodes):
            next_nodes = [nei for nei in graph[n]]
            next_nodes.append(n)
            for next_node in next_nodes:
                if t + 1 > ori_makespan or next_node not in occupied_nodes[t + 1]:
                    edge = (n, next_node)
                    idx = len(edges)
                    edge_t_2idx[(edge, t)] = idx
                    edges.append(edge)
    edge2lidx = {}
    for i, n in enumerate(graph.nodes):
        for nei in graph[n]:
            edge = (n, nei)
            edge2lidx[edge] = i

    # Matrix A
    A = np.zeros((len(graph.nodes) * (max_t + 1), len(edges)))
    for t in range(max_t + 1):
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

    # Desired x
    xzero = np.zeros(len(edges))
    for p in range(len(desired_path) - 1):
        j = edge_t_2idx[((desired_path[p], desired_path[p + 1]), p)]
        xzero[j] = 1

    # l_original
    l_original = []
    for i, n in enumerate(graph.nodes):
        if graph.nodes[n].get('obstacle'):
            l_original.append(1)
        else:
            l_original.append(0)

    # - inverse optimization problem -
    # Variables
    l_ = cp.Variable(len(l_original), boolean=True)
    pi_ = cp.Variable((len(graph.nodes) * (max_t + 1)))
    lambda_ = cp.Variable(len(edges) * max_t)
    # Cost
    cost = cp.norm1(l_ - l_original)
    # Constraints
    constraints = []
    for j, edge in enumerate(edges):
        edge_w = 0.99
        if edge[0] != edge[1]:
            i = edge2lidx[edge]
            edge_w = l_[i] * 10000 + 1
        if xzero[j] == 1:
            # sum_i a_ij * pi_i = edge_w,              for all j in desired path
            constraints.append(cp.sum(cp.multiply(A[:, j], pi_)) == edge_w)
        else:
            # sum_i a_ij * pi_i + lambda_j = edge_w,   for all j not in desired path
            constraints.append(cp.sum(cp.multiply(A[:, j], pi_)) + lambda_[j] == edge_w)
    # lambda >= 0, for all j not in desired path.
    for j in range(len(edges)):
        if xzero[j] == 0:
            constraints.append(lambda_[j] >= 0)
    # solve with cvxpy
    prob = cp.Problem(cp.Minimize(cost), constraints)
    value = prob.solve(solver=cp.ECOS_BB)
    if value == float('inf'):
        print("inverse shortest path FAILED")
        return []

    for i, n in enumerate(graph.nodes):
        print(n, l_original[i], l_.value[i])

    # New obstacles set
    cells = [n for n in graph.nodes]
    new_obstacles = []
    for i, v in enumerate(l_.value):
        if round(v) == 1:
            new_obstacles.append(cells[i])

    # Return
    return new_obstacles


# Create default desired path from shortest path with zero obstacles
# Example:
# Start: (0, 0), Goal: (2, 3)
# Desired Path: [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3)]
def create_desired_path(raw_problem, agent_name):
    dp = []
    start, goal = [], []
    for agent in raw_problem['agents']:
        if agent['name'] == agent_name:
            start, goal = agent['start'], agent['goal']
    x_diff = goal[0] - start[0]
    if x_diff != 0:
        x_step = int(x_diff / abs(x_diff))
        for i in range(start[0], goal[0], x_step):
            dp.append((i, start[1]))
    y_diff = goal[1] - start[1]
    if y_diff != 0:
        y_step = int(y_diff / abs(y_diff))
        for i in range(start[1], goal[1] + y_step, y_step):
            dp.append((goal[0], i))
    else:
        dp.append((goal[0], goal[1]))
    return dp


def create_new_schedule(old_schedule, new_path, agent_name):
    new_schedule = old_schedule
    new_schedule['schedule'][agent_name] = []
    for t, p in enumerate(new_path):
        pos = {'x': p[0], 'y': p[1], 't': t}
        new_schedule['schedule'][agent_name].append(pos)
    return new_schedule


def create_new_dct(old_dct, new_path, agent_name, new_obstacles):
    new_dct = old_dct
    for agent in new_dct['agents']:
        if agent['name'] == agent_name:
            agent['start'] = new_path[0]
            agent['goal'] = new_path[-1]
    new_dct['map']['obstacles'] = new_obstacles
    return new_dct


def main_inv_mapf(problem_file, solution_file, agent_name, desired_path=None):
    raw_problem = parse_yaml(problem_file)
    if desired_path is None:
        desired_path = create_desired_path(raw_problem, agent_name)
    raw_solution = parse_yaml(solution_file)
    graph = create_graph(raw_problem)
    new_obstacles = inv_mapf(graph, raw_solution, desired_path, agent_name)
    new_schedule = create_new_schedule(raw_solution, desired_path, agent_name)
    new_dct = create_new_dct(raw_problem, desired_path, agent_name, new_obstacles)
    return new_dct, new_schedule
