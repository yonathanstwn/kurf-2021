import argparse
from algorithms import *
from visualize import *


# Create default desired path from shortest path with zero obstacles
# Example:
# Start: (0, 0), Goal: (2, 3)
# Desired Path: [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3)]
def create_desired_path(old_dct):
    dp = []
    start, goal = old_dct['agents'][0]['start'], old_dct['agents'][0]['goal']
    print(start, goal)

    x_diff = goal[0] - start[0]
    if x_diff != 0:
        x_step = int(x_diff/abs(x_diff))
        for i in range(start[0], goal[0], x_step):
            dp.append((i, start[1]))

    y_diff = goal[1] - start[1]
    if y_diff != 0:
        y_step = int(y_diff/abs(y_diff))
        for i in range(start[1], goal[1] + y_step, y_step):
            dp.append((goal[0], i))
    else:
        dp.append((goal[0], goal[1]))

    print(dp)
    return dp


# Main ISP function
# Versions:
# 0 : Discrete ISP
# 1 : Continuous ISP
def main_isp(filepath, version, desired_path=None, vis=False):
    graph, old_dct = parse_yaml(filepath)
    if desired_path is None:
        desired_path = create_desired_path(old_dct)

    new_graph, success = None, False

    if version == 0:
        new_graph, success = isp_discrete(graph, desired_path)
    if version == 1:
        new_graph, success = isp_continuous(graph, desired_path)
    if vis:
        animate(new_graph, desired_path, old_dct)

    return success


# Main
if __name__ == '__main__':
    # Terminal Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="raw input yaml file")
    parser.add_argument("version", help="0 (Discrete) or 1 (Continuous)", type=int)
    args = parser.parse_args()
    # Main ISP Function
    main_isp(args.filepath, 0, vis=True)
