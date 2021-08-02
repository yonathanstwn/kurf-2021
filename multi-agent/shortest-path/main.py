import subprocess
from path import *
from algorithms import *


def generate_cbs_solution(filepath):
    os.chdir(CBS_DIR_PATH)
    subprocess.run('./cbs -i ' + filepath + ' -o output.yaml', shell=True, capture_output=True)


if __name__ == '__main__':
    problem_file = EXAMPLES_PATH + "/agents5/map_8by8_obst12_agents5_ex0.yaml"
    generate_cbs_solution(problem_file)
    main_mapf(problem_file, SOLUTION_YAML)