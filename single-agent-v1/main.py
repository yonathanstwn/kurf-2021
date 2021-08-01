import yaml
import networkx as nx
import cvxpy as cp
import numpy as np
from additional import visualize


def create_schedule(path):
    schedule = {'schedule': {'agent0': []}}
    for i in range(len(path)):
        schedule['schedule']['agent0'].append({'x': path[i][0], 'y': path[i][1], 't': i})
    return schedule


def create_new_dct(new_graph, desired_path, dimensions):
    new_dct = {'agents': [{'goal': list(desired_path[-1]), 'name': 'agent0',
                           'start': list(desired_path[0])}],
               'map': {'dimensions': dimensions, 'obstacles': []}}

    obstacles = []
    for node in new_graph.nodes:
        temp = 0
        for adj_node in new_graph[node]:
            if new_graph[adj_node][node]['weight'] > 1:
                temp += 1
        if temp == len(new_graph[node]):
            obstacles.append(node)
    new_dct['map']['obstacles'] = obstacles

    return new_dct


def get_cost(graph, path):
    cost = 0
    for i in range(len(path) - 1):
        cost += graph[path[i]][path[i + 1]]["weight"]
    return cost


def parse_yaml(filepath):
    # Parse YAML to dictionary
    with open(filepath, 'r') as stream:
        try:
            dct = yaml.safe_load(stream)
            print(filepath, "loaded successfully")
        except yaml.YAMLError as e:
            print(e)

    # Parse dictionary content
    agents = dct['agents']
    dimensions = dct['map']['dimensions']
    obstacles = dct['map']['obstacles']

    # Creating graph
    G = nx.grid_2d_graph(dimensions[0], dimensions[1])
    nx.set_edge_attributes(G, 1, "weight")
    for obs in obstacles:
        obs = tuple(obs)
        for adj in G[obs]:
            G[obs][adj]['weight'] = 100000

    # Start and goal as list[tuple[int,int]] extendable to MAPF
    starts = []
    goals = []
    for agent in agents:
        starts.append(tuple(agent['start']))
        goals.append((tuple(agent['goal'])))

    # Return
    return dct, G, starts, goals, dimensions


def find_shortest_path(graph, starts, goals):
    edge2index = {}
    edges = []
    weights = []
    for (i, j) in graph.edges:
        edge2index[i, j] = len(edges)
        edges.append([i, j])
        weights.append(graph[i][j]["weight"])
        edge2index[j, i] = len(edges)
        edges.append([j, i])
        weights.append(graph[j][i]["weight"])

    A = []
    b = []
    for n in graph.nodes:
        # sum_j x_ij - sum_j x_ji
        line = [0] * len(edges)
        for i, j in edge2index.keys():
            if i == n:
                line[edge2index[i, j]] += 1
        for j, i in edge2index.keys():
            if i == n:
                line[edge2index[j, i]] -= 1
        A.append(line)
        # = 1/-1/0
        if n == starts[0]:
            b.append(1)
        elif n == goals[0]:
            b.append(-1)
        else:
            b.append(0)
    A = np.array(A)
    b = np.array(b)

    # solve with cvxpy
    x = cp.Variable(len(edges), boolean=True)
    cost = cp.sum(cp.multiply(x, weights))
    prob = cp.Problem(cp.Minimize(cost), [A @ x == b])
    value = prob.solve(solver=cp.GUROBI)
    if value == float('inf'):
        print("shortest path LP failed")
        return []

    # recover path
    path = [starts[0]]
    while path[-1] != goals[0]:
        for i in range(len(edges)):
            if x.value[i] > 0.1 and edges[i][0] == path[-1]:
                path.append(edges[i][1])
                break
    return path


def inv_shortest_path(graph, desired_path):
    edge2index = {}
    edges = []
    weights = []
    for (i, j) in graph.edges:
        edge2index[i, j] = len(edges)
        edges.append([i, j])
        weights.append(graph[i][j]["weight"])
        edge2index[j, i] = len(edges)
        edges.append([j, i])
        weights.append(graph[j][i]["weight"])

    # node2index = {}
    # nodes = []
    # s = 0
    # t = 0
    # for n in graph.nodes:
    #     node2index[n] = len(nodes)
    #     nodes.append(n)
    #     if n == desired_path[0]:
    #         s = node2index[n]
    #     if n == desired_path[-1]:
    #         t = node2index[n]

    # # optimal x
    # path = nx.shortest_path(graph, source=desired_path[0], target=desired_path[-1], weight="weight")
    # xstar = np.zeros(len(edges))
    # for p in range(len(path) - 1):
    #     j = edge2index[path[p], path[p + 1]]
    #     xstar[j] = 1

    # desired x
    xzero = np.zeros(len(edges))
    for p in range(len(desired_path) - 1):
        j = edge2index[desired_path[p], desired_path[p + 1]]
        xzero[j] = 1

    A = []
    # b = []
    for n in graph.nodes:
        # sum_j x_ij - sum_j x_ji
        line = [0] * len(edges)
        for i, j in edge2index.keys():
            if i == n:
                line[edge2index[i, j]] += 1
        for j, i in edge2index.keys():
            if i == n:
                line[edge2index[j, i]] -= 1
        A.append(line)
    A = np.array(A)
    #     # = 1/-1/0
    #     if n == starts[0]:
    #         b.append(1)
    #     elif n == goals[0]:
    #         b.append(-1)
    #     else:
    #         b.append(0)
    # b = np.array(b)

    # inverse optimization problem
    w_ = cp.Variable(len(weights))
    pi_ = cp.Variable(len(graph.nodes))
    lambda_ = cp.Variable(len(edges))

    # cost
    cost = cp.norm1(w_ - weights)

    # constraints
    constraints = []
    for j in range(len(edges)):
        w_j = w_[j]
        if xzero[j] == 1:
            constraints.append(cp.sum(cp.multiply(A[:, j], pi_)) == w_j)
        else:
            constraints.append(cp.sum(cp.multiply(A[:, j], pi_)) + lambda_[j] == w_j)
    for j in range(len(edges)):
        if xzero[j] == 0:
            constraints.append(lambda_[j] >= 0)

    # solve with cvxpy
    prob = cp.Problem(cp.Minimize(cost), constraints)
    value = prob.solve()
    if value == float('inf'):
        print("  inverse shortest path MILP failed")
        return []

    # new graph
    new_G = graph.copy()
    new_weights = [round(i) for i in w_.value]
    changed = 0
    for i in range(len(weights)):
        if new_weights[i] < weights[i]:
            new_G.edges[edges[i]]['weight'] = 1
            changed += 1
        if new_weights[i] > weights[i]:
            new_G.edges[edges[i]]['weight'] = 100000
            changed += 1

    # Sanity check
    success = False
    new_path = nx.shortest_path(new_G, source=desired_path[0], target=desired_path[-1], weight="weight")
    if get_cost(new_G, new_path) == get_cost(new_G, desired_path):
        print("inverse shortest path: success")
        success = True
    else:
        print("inverse shortest path: fail")

    # Return
    return new_G, success


if __name__ == '__main__':
    # filepath = input("YAML File Path: ")
    filepath = '//examples/map_8by8_obst12_agents1_ex13.yaml'
    dct, G, starts, goals, dimensions = parse_yaml(filepath)

    # desired_path = [(5, 4), (5, 3), (5, 2), (5, 1), (6, 1), (7, 1)]
    # new_G, success = inv_shortest_path(G, desired_path)
    # new_schedule = create_schedule(desired_path)
    # new_dct = create_new_dct(new_G, desired_path, dimensions)
    #
    # animation = visualize.Animation(new_dct, new_schedule)
    # animation.show()

    # path = find_shortest_path(G, starts, goals)
    # schedule = create_schedule(path)
    # animation = visualize.Animation(dct, schedule)
    # animation.show()
