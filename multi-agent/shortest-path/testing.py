from main import *

if __name__ == '__main__':
    fail = []
    for i in range(100):
        if not main_sp(str(i), "agent2"):
            print("Multi-Agent SP Ex" + str(i) + " FAIL!")
            fail.append(i)
    print("success: " + str(100 - len(fail)) + "/100")
    print("FAIL:", fail)



