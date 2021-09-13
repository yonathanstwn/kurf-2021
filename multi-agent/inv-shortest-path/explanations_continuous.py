import yaml
import networkx as nx
import cvxpy as cp
import numpy as np
import argparse
import subprocess
from path import *
from additional import visualize


def create_yaml(data, filename):
    with open(filename, 'w') as stream:
        yaml.dump(data, stream)
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


def inv_mapf(graph, raw_solution, desired_path, agent_name):
    # Extract Solution Data with specified agent stored separately.
    ori_makespan = raw_solution['statistics']['makespan']
    schedule = raw_solution['schedule']
    schedule.pop(agent_name)

    # Check validity of desired path
    for path in schedule.values():
        for t, pos in enumerate(path):
            n = (pos['x'], pos['y'])
            if n == desired_path[min(t, len(desired_path) - 1)]:
                print("INVALID DESIRED PATH - Desired path of agent collides with other agents")
                return []
    for t in range(ori_makespan, len(desired_path)):
        for path in schedule.values():
            last_node = (path[-1]['x'], path[-1]['y'])
            if desired_path[t] == last_node:
                print("INVALID DESIRED PATH - Desired path of agent collides with other agents")
                return []

    # Determine max t
    max_t = max(len(desired_path) - 1, ori_makespan)

    # w_original
    w_original = []
    for i, n in enumerate(graph.nodes):
        if graph.nodes[n].get('obstacle'):
            w_original.append(100)
        else:
            w_original.append(1)

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
                        occupied_nodes[min(t, ori_makespan)].get(next_node) != n):
                    edge = (n, next_node)
                    idx = len(edges)
                    edge_t_2idx[(edge, t)] = idx
                    edges.append(edge)
    edge2lidx = {}
    node2lidx = {}
    for i, n in enumerate(graph.nodes):
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

    # Desired x
    xzero = np.zeros(len(edges))
    for p in range(len(desired_path) - 1):
        j = edge_t_2idx[((tuple(desired_path[p]), tuple(desired_path[p + 1])), p)]
        xzero[j] = 1

    # - inverse optimization problem -
    # Variables
    w_ = cp.Variable(len(w_original))
    pi_ = cp.Variable(len(graph.nodes) * (max_t + 1))
    lambda_ = cp.Variable(len(edges))
    # Cost
    cost = cp.norm1(w_ - w_original)
    # Constraints
    constraints = []
    for j, edge in enumerate(edges):
        i = edge2lidx[edge]
        if xzero[j] == 1:
            # sum_i a_ij * pi_i = edge_w,              for all j in desired path
            constraints.append(cp.sum(cp.multiply(A[:, j], pi_)) == w_[i])
        else:
            # sum_i a_ij * pi_i + lambda_j = edge_w,   for all j not in desired path
            constraints.append(cp.sum(cp.multiply(A[:, j], pi_)) + lambda_[j] == w_[i])
    # lambda >= 0, for all j not in desired path.
    for j in range(len(edges)):
        if xzero[j] == 0:
            constraints.append(lambda_[j] >= 0)
    # l_[node] == 0 for all nodes in other agents' paths
    for n in nodes_passed:
        constraints.append(w_[node2lidx[n]] == 1)
    # solve with cvxpy
    prob = cp.Problem(cp.Minimize(cost), constraints)
    value = prob.solve()
    if value == float('inf'):
        print("inverse shortest path FAILED")
        return []

    # New obstacles set
    cells = [n for n in graph.nodes]
    new_obstacles = []
    for i, v in enumerate(w_.value):
        print(w_original[i], v, cells[i])
        if round(v) > 5000:
            new_obstacles.append(list(cells[i]))

    # Return
    return new_obstacles


# Create default desired path which is shortest path with zero obstacles
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


def create_new_problem(old_problem, new_path, agent_name, new_obstacles):
    new_problem = old_problem
    for agent in new_problem['agents']:
        if agent['name'] == agent_name:
            agent['start'] = list(new_path[0])
            agent['goal'] = list(new_path[-1])
    new_problem['map']['obstacles'] = new_obstacles
    return new_problem


def sanity_check(new_cbs_solution, agent_name, desired_path):
    if len(new_cbs_solution['schedule'][agent_name]) == len(desired_path):
        print("Multi-Agent ISP Success!")
        return True
    else:
        print("Multi-Agent ISP Fail!")
        return False


def generate_cbs_solution(filepath):
    os.chdir(CBS_DIR_PATH)
    subprocess.run('./cbs -i ' + filepath + ' -o output.yaml', shell=True, capture_output=True)
    os.chdir(ROOT_PATH)


def generate_animation(new_problem, new_schedule):
    animation = visualize.Animation(new_problem, new_schedule)
    animation.show()


def main_inv_mapf(problem_file, agent_name):
    # Parsing and generating CBS solution of original problem file
    problem_fullpath = EXAMPLES_PATH + "/" + problem_file
    generate_cbs_solution(problem_fullpath)
    raw_problem = parse_yaml(problem_fullpath)

    # Handling desired path of the agent
    desired_path = []
    for agent in raw_problem['agents']:
        if agent['name'] == agent_name and agent.get('waypoints') is not None:
            desired_path = agent['waypoints']
    if len(desired_path) == 0:
        desired_path = create_desired_path(raw_problem, agent_name)

    # Multi-Agent ISP
    raw_solution = parse_yaml(SOLUTION_YAML)
    graph = create_graph(raw_problem)
    new_obstacles = inv_mapf(graph, raw_solution, desired_path, agent_name)

    # Create new schedule and problem dict and created a new problem yaml file
    new_schedule = create_new_schedule(raw_solution, desired_path, agent_name)
    new_problem = create_new_problem(raw_problem, desired_path, agent_name, new_obstacles)
    new_filename = "additional/build/new_problem.yaml"
    create_yaml(new_problem, new_filename)

    # Sanity Check
    generate_cbs_solution(new_filename)
    new_cbs_solution = parse_yaml(SOLUTION_YAML)
    success = sanity_check(new_cbs_solution, agent_name, desired_path)

    # Return
    return new_problem, new_schedule, success, new_cbs_solution


if __name__ == '__main__':
    # Terminal Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_file", help="input problem filepath")
    parser.add_argument("agent_name", help="the agent that will have a new desired path")
    args = parser.parse_args()
    # Main SP Function
    new_problem, new_schedule, success, new_cbs_solution = main_inv_mapf(args.problem_file, args.agent_name)
    generate_animation(new_problem, new_schedule)
