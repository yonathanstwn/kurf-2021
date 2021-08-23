import argparse
import subprocess
from path import *
from algorithms import *
from additional import visualize


def generate_cbs_solution(filepath):
    os.chdir(CBS_DIR_PATH)
    subprocess.run('./cbs -i ' + filepath + ' -o output.yaml', shell=True, capture_output=True)


def generate_animation(new_dct, new_schedule):
    animation = visualize.Animation(new_dct, new_schedule)
    animation.show()


def main_isp(example_number, agent_name):
    problem_file = EXAMPLES_PATH + "/agents5/map_8by8_obst12_agents5_ex" + example_number + ".yaml"
    generate_cbs_solution(problem_file)
    return main_inv_mapf(problem_file, SOLUTION_YAML, agent_name)


if __name__ == '__main__':
    # Terminal Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("ex_no", help="input file example number")
    parser.add_argument("agent_name", help="the agent that will have a new desired path")
    args = parser.parse_args()
    # Main SP Function
    new_dct, new_schedule = main_isp(args.ex_no, args.agent_name)
    generate_animation(new_dct, new_schedule)
