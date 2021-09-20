import yaml
import networkx as nx
import cvxpy as cp
import numpy as np
import argparse
import subprocess
from path import *
from additional import visualize
import copy
# from additional import shortest_path as sp


def create_problem_yaml(problem_dct, filename):
    with open(filename, 'w') as stream:
        yaml.dump(problem_dct, stream, default_flow_style=None)
        print(filename, "created successfully")


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


# def create_desired_path(graph, raw_solution, agent_name, waypoints):
#     agent = raw_solution['schedule'][agent_name]
#     abs_start = (agent[0]['x'], agent[0]['y'])
#     abs_goal = (agent[-1]['x'], agent[-1]['y'])
#     desired_path = []
#     start = abs_start
#     goal = waypoints[0]
#     for i in range(len(waypoints) - 1):
#         if start == goal:
#             start = waypoints[i]
#             goal = waypoints[i + 1]
#             continue
#         partial_path, success = sp.mapf(graph, raw_solution, agent_name, start, goal)
#         if success:
#             desired_path.extend(partial_path[:-1])
#             start = waypoints[i]
#             goal = waypoints[i + 1]
#         else:
#             print("Creating Desired Path For", agent_name, "From Waypoints FAIL!")
#             return [], success
#     partial_path, success = sp.mapf(graph, raw_solution, agent_name, goal, abs_goal)
#     if success:
#         print("Creating Desired Path For", agent_name, "From Waypoints SUCCESS!")
#         desired_path.extend(partial_path)
#     else:
#         print("Creating Desired Path For", agent_name, "From Waypoints FAIL!")
#         return [], success
#     return desired_path, success


def inv_mapf(graph, raw_solution, desired_paths, agent_names, verbose=False):
    # Extract Solution Data with specified agent stored separately.
    ori_makespan = copy.deepcopy(raw_solution['statistics']['makespan'])
    schedule = copy.deepcopy(raw_solution['schedule'])
    for agent_name in agent_names:
        schedule.pop(agent_name)

    # Check validity of desired path
    for path in schedule.values():
        for t, pos in enumerate(path):
            n = (pos['x'], pos['y'])
            for desired_path in desired_paths:
                if n == tuple(desired_path[min(t, len(desired_path) - 1)]):
                    print("INVALID DESIRED PATH - Desired path of agent collides with other agents")
                    return []
    for desired_path in desired_paths:
        for t in range(ori_makespan, len(desired_path)):
            for path in schedule.values():
                last_node = (path[-1]['x'], path[-1]['y'])
                if tuple(desired_path[t]) == last_node:
                    print("INVALID DESIRED PATH - Desired path of agent collides with other agents")
                    return []
    # for dp1 in desired_paths:
    #     for dp2 in desired_paths:
    #         for i in range(len(dp1)):
    #
    #             if

    # Determine max t
    max_t = 0
    for desired_path in desired_paths:
        max_t = max(len(desired_path) - 1, ori_makespan, max_t)

    # l_original
    l_original = []
    for i, n in enumerate(graph.nodes):
        if graph.nodes[n].get('obstacle'):
            l_original.append(1)
        else:
            l_original.append(0)

    # occupied_nodes[t] = list of edges corresponding to movement of agents (except agent0)
    # from time=t to time=t+1
    # nodes_passed = set of all the nodes that are ever passed by any agent at any time
    occupied_nodes = [{} for _ in range(ori_makespan + 1)]
    nodes_passed = set()
    for path in schedule.values():
        for t, pos in enumerate(path):
            n = (pos['x'], pos['y'])
            nodes_passed.add(n)
        if len(path) == 1:
            for i in range(ori_makespan + 1):
                temp = (path[0]['x'], path[0]['y'])
                occupied_nodes[i][temp] = temp
        for t in range(len(path) - 1):
            temp = (path[t + 1]['x'], path[t + 1]['y'])
            occupied_nodes[t][(path[t]['x'], path[t]['y'])] = temp
            if t + 1 == len(path) - 1:
                for i in range(t + 1, ori_makespan + 1):
                    occupied_nodes[i][temp] = temp

    # Auxiliary variables
    edges = []
    edge_t_2idx = {}  # key: tuple[edge, t] where t=0 means edge going from time=0 to time=1
    for t in range(max_t):
        for i, n in enumerate(graph.nodes):
            next_nodes = [nei for nei in graph[n]]
            next_nodes.append(n)
            for next_node in next_nodes:
                # Check to avoid collisions with other agents
                if (next_node not in occupied_nodes[min(t, ori_makespan)].values() and
                        occupied_nodes[min(t, ori_makespan)].get(next_node) != n and
                        n not in occupied_nodes[min(t, ori_makespan)].keys()):
                    edge = (n, next_node)
                    idx = len(edges)
                    edge_t_2idx[(edge, t)] = idx
                    edges.append(edge)
    cells = []
    edge2lidx = {}
    node2lidx = {}
    for i, n in enumerate(graph.nodes):
        cells.append(n)
        node2lidx[n] = i
        edge2lidx[(n, n)] = i
        for nei in graph[n]:
            edge = (nei, n)
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

    # List of Desired x
    xzeros = np.zeros((len(desired_paths), len(edges)))
    for i, desired_path in enumerate(desired_paths):
        for p in range(len(desired_path) - 1):
            j = edge_t_2idx[((tuple(desired_path[p]), tuple(desired_path[p + 1])), p)]
            xzeros[i, j] = 1

    # Debug Info
    if verbose:
        print("-------------------- DEBUG --------------------")
        print("-------------------- MAX T --------------------")
        print(max_t)
        print("-------------------- OCCUPIED NODES --------------------")
        for t, dct in enumerate(occupied_nodes):
            print(t, dct)
        print("-------------------- EDGES --------------------")
        for i, e in enumerate(edges):
            print(i, e)
        print("-------------------- EDGE_T TO INDEX --------------------")
        for e in edge_t_2idx.items():
            print(e)
        print("-------------------- EDGE TO L INDEX --------------------")
        for e in edge2lidx.items():
            print(e)
        print("-------------------- NODE TO L INDEX --------------------")
        for e in node2lidx.items():
            print(e)
        print("-------------------- MATRIX A --------------------")
        for r, edges_set in enumerate(A):
            node = cells[r % len(cells)]
            t = r // len(cells)
            print("----- ROW:", r, "NODE:", node, "T:", t, "-----")
            for i, v in enumerate(edges_set):
                if round(v) == 1 or round(v) == -1:
                    print(edges[i], v)
        print("-------------------- L ORIGINAL --------------------")
        for i, v in enumerate(l_original):
            if round(v) == 1:
                print(i, cells[i], v)
        print("-------------------- X ZERO --------------------")
        for x in xzeros:
            print("-----")
            for i, v in enumerate(x):
                if round(v) == 1:
                    print(i, edges[i], v)
        print("-------------------- DEBUG --------------------")

    # - inverse optimization problem -
    # Variables
    l_ = cp.Variable(len(l_original), boolean=True)
    pi_vector = []
    lambda_vector = []
    for i in range(len(desired_paths)):
        pi_vector.append(cp.Variable(len(graph.nodes) * (max_t + 1)))
        lambda_vector.append(cp.Variable(len(edges)))
    # Cost
    cost = cp.norm1(l_ - l_original)
    # Constraints
    constraints = []
    for j, edge in enumerate(edges):
        i = edge2lidx[edge]
        edge_w = l_[i] * 1000 + 1
        for i in range(len(desired_paths)):
            if xzeros[i, j] == 1:
                # sum_i a_ij * pi_i = edge_w,              for all j in desired path
                constraints.append(cp.sum(cp.multiply(A[:, j], pi_vector[i])) == edge_w)
            else:
                # sum_i a_ij * pi_i + lambda_j = edge_w,   for all j not in desired path
                constraints.append(cp.sum(cp.multiply(A[:, j], pi_vector[i])) + lambda_vector[i][j] == edge_w)
                # lambda >= 0, for all j not in desired path.
                # NOTE THIS IS DIFFERENT FROM ORIGINAL CONSTRAINTS (ORIGINAL: >= 0)
                # Otherwise new obstacles are not created
                constraints.append(lambda_vector[i][j] >= 1)
    # l_[node] == 0 for all nodes in other agents' paths
    for n in nodes_passed:
        constraints.append(l_[node2lidx[n]] == 0)
    # solve with cvxpy
    prob = cp.Problem(cp.Minimize(cost), constraints)
    value = prob.solve(solver=cp.GUROBI)
    if value == float('inf'):
        print("inverse shortest path FAILED")
        return []

    # New obstacles set
    new_obstacles = []
    for i, v in enumerate(l_.value):
        if round(v) == 1:
            new_obstacles.append(list(cells[i]))

    # Return
    return new_obstacles


def create_new_schedule(old_schedule, new_paths, agent_names):
    new_schedule = copy.deepcopy(old_schedule)
    for agent_name in agent_names:
        new_schedule['schedule'][agent_name] = []
    for i, new_path in enumerate(new_paths):
        agent_name = agent_names[i]
        for t, p in enumerate(new_path):
            pos = {'x': p[0], 'y': p[1], 't': t}
            new_schedule['schedule'][agent_name].append(pos)
    return new_schedule


def create_new_problem(old_problem, new_paths, agent_names, new_obstacles):
    new_problem = copy.deepcopy(old_problem)
    for agent in new_problem['agents']:
        for i, agent_name in enumerate(agent_names):
            if agent['name'] == agent_name:
                agent['start'] = list(new_paths[i][0])
                agent['goal'] = list(new_paths[i][-1])
    new_problem['map']['obstacles'] = new_obstacles
    return new_problem


def sanity_check(new_cbs_solution, agent_names, desired_paths):
    for i, agent_name in enumerate(agent_names):
        if len(new_cbs_solution['schedule'][agent_name]) != len(desired_paths[i]):
            print("Multi-Agent ISP Fail!")
            return False
    print("Multi-Agent ISP Success!")
    return True


def generate_cbs_solution(filepath):
    os.chdir(CBS_DIR_PATH)
    subprocess.run('./cbs -i ' + filepath + ' -o output.yaml', shell=True, capture_output=True)
    os.chdir(ROOT_PATH)


def generate_animation(old_problem, new_problem, new_schedule):
    animation = visualize.Animation(old_problem, new_problem, new_schedule)
    animation.show()


def main_inv_mapf(problem_file, verbose=False, animate=False):
    # Parsing and generating CBS solution of original problem file
    problem_fullpath = EXAMPLES_PATH + "/" + problem_file
    generate_cbs_solution(problem_fullpath)
    raw_problem = parse_yaml(problem_fullpath)
    raw_solution = parse_yaml(SOLUTION_YAML)
    graph = create_graph(raw_problem)

    # Handling desired path of the agent and get agent name
    desired_paths = []
    agent_names = []
    for agent in raw_problem['agents']:
        if agent.get('waypoints') is not None:
            desired_paths.append(agent['waypoints'])
            agent_names.append(agent['name'])

    # Multi-Agent ISP
    print("Running Multi-Agent ISP Algorithm...")
    new_obstacles = inv_mapf(graph, raw_solution, desired_paths, agent_names, verbose)

    # Create new schedule and problem dict and created a new problem yaml file
    new_schedule = create_new_schedule(raw_solution, desired_paths, agent_names)
    new_problem = create_new_problem(raw_problem, desired_paths, agent_names, new_obstacles)
    new_filename = "additional/build/new_problem.yaml"
    create_problem_yaml(new_problem, new_filename)

    # Sanity Check
    generate_cbs_solution(ROOT_PATH + "/" + new_filename)
    new_cbs_solution = parse_yaml(SOLUTION_YAML)
    success = sanity_check(new_cbs_solution, agent_names, desired_paths)

    # Animation
    if animate:
        generate_animation(raw_problem, new_problem, new_schedule)

    # Return
    return new_problem, success


if __name__ == '__main__':
    # Terminal Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_file", help="input problem filepath")
    parser.add_argument("-v", "--verbose", action="store_true", help="outputs debug information")
    parser.add_argument("-a", "--animate", action="store_true", help="shows animation")
    args = parser.parse_args()
    # Main SP Function
    main_inv_mapf(args.problem_file, args.verbose, args.animate)
