from main import *

# Main
if __name__ == '__main__':
    success_dis_count = 0
    failed_dis = []
    success_con_count = 0
    failed_con = []
    for i in range(100):
        filepath = "/Users/yonathan/PycharmProjects/kurf-mapf-2021/examples/map_8by8_obst12_agents1_ex" + str(i) + ".yaml"
        success_dis = main_isp(filepath, 0)
        success_con = main_isp(filepath, 1)
        success_dis_count += 1 if success_dis else failed_dis.append("ex" + str(i))
        success_con_count += 1 if success_con else failed_con.append("ex" + str(i))
    print("Success Discrete: " + str(success_dis_count) + "/100")
    print("Failed:", failed_dis)
    print("Success Discrete: " + str(success_con_count) + "/100")
    print("Failed:", failed_con)

