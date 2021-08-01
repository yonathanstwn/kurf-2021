import yaml
import networkx as nx
import cvxpy as cp
import numpy as np


# Parse YAML to create graph
def parse_yaml(filepath):
    # Parse YAML
    with open(filepath, 'r') as stream:
        try:
            raw_dict = yaml.safe_load(stream)
            print(filepath, "loaded successfully")
        except yaml.YAMLError as e:
            print(e)
    # Extract data
    dimensions = raw_dict['map']['dimensions']
    obstacles = raw_dict['map']['obstacles']
    # Create Graph
    graph = nx.Graph()
    for r in range(dimensions[0]):
        for c in range(dimensions[1]):
            # horizontal edges
            if c < dimensions[0] - 1:
                graph.add_edge((r, c), (r, c + 0.5))
                graph.add_edge((r, c + 0.5), (r, c + 1))
            # vertical edges
            if r < dimensions[1] - 1:
                graph.add_edge((r, c), (r + 0.5, c))
                graph.add_edge((r + 0.5, c), (r + 1, c))
    # Flag Portals and setup area types
    for n in graph.nodes:
        if isinstance(n[0], int) and isinstance(n[1], int):
            graph.nodes[n]['portal'] = False
            graph.nodes[n]['area_type'] = 0
        else:
            graph.nodes[n]['portal'] = True
    # Setup obstacles
    for obs in obstacles:
        graph.nodes[tuple(obs)]['area_type'] = 1
    # Return
    return graph, raw_dict


# Insert portal nodes to raw path, e.g. [(0,0), (0,1)] to [(0,0), (0,0.5), (0,1)]
def insert_portals(path):
    new_path = []
    for i in range(len(path) - 1):
        portal = ((path[i][0] + path[i + 1][0]) / 2, (path[i][1] + path[i + 1][1]) / 2)
        new_path.extend([path[i], portal])
    return new_path


# Delete portal nodes from output path, e.g. [(0,0), (0,0.5), (0,1)] to [(0,0), (0,1)]
def delete_portals(path):
    return [path[i] for i in range(len(path)) if i % 2 == 0]


# Calculate cost of given path in given graph
def get_cost(graph, path):
    cost = 0
    for i in range(len(path) - 1):
        cost += graph[path[i]][path[i + 1]]["weight"]
    return cost


def create_aux_vars(graph, desired_path):
    # Nodes, edges, non-portal nodes (varnodes), insert portals to desired path
    dp = insert_portals(desired_path)
    nodes = list(graph.nodes)
    edge2index = {}
    edges = []
    edge2varnodeindex = {}
    varnodes = []
    for (i, j) in graph.edges:
        edge2index[i, j] = len(edges)
        edges.append([i, j])
        edge2index[j, i] = len(edges)
        edges.append([j, i])
        if not graph.nodes[i]["portal"]:
            vn = i
        else:
            vn = j
        if vn in varnodes:
            idx = varnodes.index(vn)
        else:
            idx = len(varnodes)
            varnodes.append(vn)
        edge2varnodeindex[i, j] = idx
        edge2varnodeindex[j, i] = idx

    # A matrix
    A = np.zeros([len(nodes), len(edges)])
    for i in range(len(nodes)):
        for nei in graph.adj[nodes[i]]:
            j = edge2index[nodes[i], nei]
            A[i, j] = 1
            j = edge2index[nei, nodes[i]]
            A[i, j] = -1

    # Desired x
    xzero = np.zeros(len(edges))
    for p in range(len(dp) - 1):
        j = edge2index[dp[p], dp[p + 1]]
        xzero[j] = 1

    # Return
    return dp, nodes, edges, edge2index, varnodes, edge2varnodeindex, A, xzero


# Sanity Check
def check(dp, new_graph):
    success = False
    new_path = nx.shortest_path(new_graph, source=dp[0], target=dp[-1], weight="weight")
    if get_cost(new_graph, new_path) == get_cost(new_graph, dp):
        success = True
        print("inverse shortest path: success")
    else:
        print("inverse shortest path: fail")
    return success


# ISP Discrete
def isp_discrete(graph, desired_path, area_costs=None, allowed_area_types=None, dist_per_edge=0.5):
    if allowed_area_types is None:
        allowed_area_types = [0, 1]
    if area_costs is None:
        area_costs = [1, 1000]

    dp, nodes, edges, edge2index, varnodes, edge2varnodeindex, A, xzero = create_aux_vars(graph, desired_path)

    # l_original
    l_original = np.zeros(len(varnodes) * len(allowed_area_types))
    for idx in range(len(varnodes)):
        node = varnodes[idx]
        for k in range(len(allowed_area_types)):
            if allowed_area_types[k] == graph.nodes[node]["area_type"]:
                l_original[len(allowed_area_types) * idx + k] = 1
            else:
                l_original[len(allowed_area_types) * idx + k] = 0

    # - inverse optimization problem -
    # Variables
    l_ = cp.Variable(len(l_original), boolean=True)
    pi_ = cp.Variable(len(nodes))
    lambda_ = cp.Variable(len(edges))
    # Cost
    cost = cp.norm1(l_ - l_original)
    # Constraints
    constraints = []
    for j in range(len(edges)):
        edge = edges[j]
        i = edge2varnodeindex[edge[0], edge[1]]
        # edge's new cost d_j = sum_(k in areas) dist_j * ac_k * l_ik
        d_j = 0
        for k in range(len(allowed_area_types)):
            ac_k = area_costs[allowed_area_types[k]]
            d_j += dist_per_edge * ac_k * l_[len(allowed_area_types) * i + k]
        if xzero[j] == 1:
            # sum_i a_ij * pi_i = d_j,              for all j in desired path
            constraints.append(cp.sum(cp.multiply(A[:, j], pi_)) == d_j)
        else:
            # sum_i a_ij * pi_i + lambda_j = d_j,   for all j not in desired path
            constraints.append(cp.sum(cp.multiply(A[:, j], pi_)) + lambda_[j] == d_j)
        # sum_k l_ik = 1, for all i
    for i in range(len(varnodes)):
        idx = len(allowed_area_types) * i
        constraints.append(cp.sum(l_[idx:idx + len(allowed_area_types)]) == 1)
        # lambda >= 0, for all j not in desired path.
    for j in range(len(edges)):
        if xzero[j] == 0:
            constraints.append(lambda_[j] >= 0)
    # solve with cvxpy
    prob = cp.Problem(cp.Minimize(cost), constraints)
    value = prob.solve(solver=cp.GUROBI)
    if value == float('inf'):
        print("inverse shortest path FAILED")
        return []

    # new graph - weights added to edges
    new_graph = graph.copy()
    l_new = [round(i) for i in l_.value]
    for i in range(len(l_original)):
        if l_original[i] != l_new[i]:
            vn_idx = i // len(allowed_area_types)
            vn = varnodes[vn_idx]
            new_graph.nodes[vn]['area_type'] = l_new[i]
    for vn in varnodes:
        area_type = new_graph.nodes[vn]['area_type']
        for adj in new_graph[vn]:
            new_graph[adj][vn]['weight'] = area_costs[area_type] * dist_per_edge

    # Sanity Check
    success = check(dp, new_graph)
    # Return
    return new_graph, success


# ISP Continuous
def isp_continuous(graph, desired_path, area_costs=None, allowed_area_types=None):
    if allowed_area_types is None:
        allowed_area_types = [0, 1]
    if area_costs is None:
        area_costs = [0.5, 500]

    dp, nodes, edges, edge2index, varnodes, edge2varnodeindex, A, xzero = create_aux_vars(graph, desired_path)

    # w_original
    w_original = [area_costs[graph.nodes[vn]['area_type']] for vn in varnodes]

    # - inverse optimization problem -
    # Variables
    w_ = cp.Variable(len(w_original))
    pi_ = cp.Variable(len(nodes))
    lambda_ = cp.Variable(len(edges))
    # Cost
    cost = cp.norm1(w_ - w_original)
    # Constraints
    constraints = []
    for j in range(len(edges)):
        edge = edges[j]
        i = edge2varnodeindex[edge[0], edge[1]]
        w_j = w_[i]
        if xzero[j] == 1:
            # sum_i a_ij * pi_i = w_j,              for all j in desired path
            constraints.append(cp.sum(cp.multiply(A[:, j], pi_)) == w_j)
        else:
            # sum_i a_ij * pi_i + lambda_j = d_j,   for all j not in desired path
            constraints.append(cp.sum(cp.multiply(A[:, j], pi_)) + lambda_[j] == w_j)
    # lambda >= 0, for all j not in desired path.
    for j in range(len(edges)):
        if xzero[j] == 0:
            constraints.append(lambda_[j] >= 0)
    # solve with cvxpy
    prob = cp.Problem(cp.Minimize(cost), constraints)
    value = prob.solve(solver=cp.GUROBI)
    if value == float('inf'):
        print("inverse shortest path FAILED")
        return []

    # new graph - weights added to edges
    new_graph = graph.copy()
    w_new = [round(i) for i in w_.value]
    for i in range(len(w_original)):
        if w_new[i] <= (area_costs[0] + area_costs[1]) / 2:
            new_graph.nodes[varnodes[i]]['area_type'] = 0
        else:
            new_graph.nodes[varnodes[i]]['area_type'] = 1
    for vn in varnodes:
        area_type = new_graph.nodes[vn]['area_type']
        for adj in new_graph[vn]:
            new_graph[adj][vn]['weight'] = area_costs[area_type]

    # Sanity Check
    success = check(dp, new_graph)
    # Return
    return new_graph, success
