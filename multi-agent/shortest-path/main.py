import argparse
import subprocess
from path import *
from algorithms import *


def main_sp(example_number, agent_name):
    problem_file = EXAMPLES_PATH + "/agents5/map_8by8_obst12_agents5_ex" + example_number + ".yaml"
    generate_cbs_solution(problem_file)
    path, success = main_mapf(problem_file, SOLUTION_YAML, agent_name)
    return success


def generate_cbs_solution(filepath):
    os.chdir(CBS_DIR_PATH)
    subprocess.run('./cbs -i ' + filepath + ' -o output.yaml', shell=True, capture_output=True)


def generate_animation(filepath):
    os.chdir(CBS_DIR_PATH)
    subprocess.run('python3 ' + VISUALIZE_PATH + ' ' + filepath + ' output.yaml', shell=True, capture_output=True)


if __name__ == '__main__':
    # Terminal Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("ex_no", help="input file example number")
    parser.add_argument("agent_name", help="input agent name")
    args = parser.parse_args()
    # Main SP Function
    main_sp(args.ex_no, args.agent_name)

