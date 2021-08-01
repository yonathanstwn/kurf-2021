from main import *
import subprocess
import os

switch = True

if switch:
    # Inverse Shortest Path Test
    # All success: 100/100
    success_count = 0
    failed = []
    for i in range(100):
        filepath = "/Users/yonathan/PycharmProjects/kurf-2021-old/examples/map_8by8_obst12_agents1_ex" + str(i) + ".yaml"
        dct, G, starts, goals, dimensions = parse_yaml(filepath)
        desired_path = [(5, 4), (5, 3), (5, 2), (5, 1), (6, 1), (7, 1)]
        new_G, success = inv_shortest_path(G, desired_path)
        new_schedule = create_schedule(desired_path)
        new_dct = create_new_dct(new_G, desired_path, dimensions)
        if success:
            success_count += 1
        else:
            failed.append("ex" + str(i))
    print("Success: " + str(success_count) + "/100")
    print("Failed:", failed)

else:
    # Shortest Path Test
    # Test fail when start == goal
    os.chdir('build')
    success_count = 0
    failed = []
    for i in range(100):
        # My code
        filepath = "/Users/yonathan/PycharmProjects/kurf-2021-old/examples/map_8by8_obst12_agents1_ex" + str(i) + ".yaml"
        dct, G, starts, goals, dimensions = parse_yaml(filepath)
        path = find_shortest_path(G, starts, goals)
        # Whoenig's code
        out = subprocess.run('./cbs -i ' + filepath + ' -o output.yaml', shell=True, capture_output=True).stdout.decode()
        actual_cost = out.split()[2]
        if int(actual_cost) == len(path) - 1:
            success_count += 1
        else:
            failed.append("ex" + str(i))
    print("Success: " + str(success_count) + "/100")
    print("Failed:", failed)



