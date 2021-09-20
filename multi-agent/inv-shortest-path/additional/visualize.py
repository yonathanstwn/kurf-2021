#!/usr/bin/env python3
import yaml
from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import argparse

Colors = ['orange']


class Animation:
    def __init__(self, old_map, new_map, schedule):
        self.old_map = old_map
        self.schedule = schedule
        self.new_map = new_map

        aspect = new_map["map"]["dimensions"][0] / new_map["map"]["dimensions"][1]

        self.fig = plt.figure(frameon=False, figsize=(4 * aspect, 4))
        self.ax = self.fig.add_subplot(111, aspect='equal')
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)
        # self.ax.set_frame_on(False)

        self.patches = []
        self.artists = []
        self.agents = dict()
        self.agent_names = dict()
        # create boundary patch
        xmin = -0.5
        ymin = -0.5
        xmax = new_map["map"]["dimensions"][0] - 0.5
        ymax = new_map["map"]["dimensions"][1] - 0.5

        # self.ax.relim()
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        # self.ax.set_xticks([])
        # self.ax.set_yticks([])
        # plt.axis('off')
        # self.ax.axis('tight')
        # self.ax.axis('off')

        # MODIFIED BY Yonathan Setiawan
        # to show new and deleted obstacles.
        new_map_obstacles = set([tuple(o) for o in new_map['map']['obstacles']])
        old_map_obstacles = set([tuple(o) for o in old_map['map']['obstacles']])
        common_obstacles = new_map_obstacles & old_map_obstacles
        deleted_obstacles = old_map_obstacles - new_map_obstacles
        new_obstacles = new_map_obstacles - old_map_obstacles
        self.patches.append(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor='none', edgecolor='red'))
        for o in common_obstacles:
            x, y = o[0], o[1]
            self.patches.append(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='red', edgecolor='red'))
        for o in deleted_obstacles:
            x, y = o[0], o[1]
            self.patches.append(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='lightsteelblue', edgecolor='lightsteelblue'))
        for o in new_obstacles:
            x, y = o[0], o[1]
            self.patches.append(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='darkviolet', edgecolor='darkviolet'))

        # create agents:
        self.T = 0
        # draw goals first
        for d, i in zip(new_map["agents"], range(0, len(new_map["agents"]))):
            if "goal" in d:
                goals = [d["goal"]]
            if "potentialGoals" in d:
                goals = [goal for goal in d["potentialGoals"]]
            for goal in goals:
                self.patches.append(
                    Rectangle((goal[0] - 0.25, goal[1] - 0.25), 0.5, 0.5, facecolor=Colors[i % len(Colors)],
                              edgecolor='black', alpha=0.5))

        for d, i in zip(new_map["agents"], range(0, len(new_map["agents"]))):
            name = d["name"]
            self.agents[name] = Circle((d["start"][0], d["start"][1]), 0.3, facecolor=Colors[i % len(Colors)],
                                       edgecolor='black')
            self.agents[name].original_face_color = Colors[i % len(Colors)]
            self.patches.append(self.agents[name])
            self.T = max(self.T, schedule["schedule"][name][-1]["t"])
            self.agent_names[name] = self.ax.text(d["start"][0], d["start"][1], name.replace('agent', ''))
            self.agent_names[name].set_horizontalalignment('center')
            self.agent_names[name].set_verticalalignment('center')
            self.artists.append(self.agent_names[name])

        # self.ax.set_axis_off()
        # self.fig.axes[0].set_visible(False)
        # self.fig.axes.get_yaxis().set_visible(False)

        # self.fig.tight_layout()

        self.anim = animation.FuncAnimation(self.fig, self.animate_func,
                                            init_func=self.init_func,
                                            frames=int(self.T + 1) * 10,
                                            interval=100,
                                            blit=True)

    def save(self, file_name, speed):
        self.anim.save(
            file_name,
            "ffmpeg",
            fps=10 * speed,
            dpi=200),
        # savefig_kwargs={"pad_inches": 0, "bbox_inches": "tight"})

    def show(self):
        plt.show()

    def init_func(self):
        for p in self.patches:
            self.ax.add_patch(p)
        for a in self.artists:
            self.ax.add_artist(a)
        return self.patches + self.artists

    def animate_func(self, i):
        for agent_name in self.schedule["schedule"]:
            agent = self.schedule["schedule"][agent_name]
            pos = self.getState(i / 10, agent)
            p = (pos[0], pos[1])
            self.agents[agent_name].center = p
            self.agent_names[agent_name].set_position(p)

        # reset all colors
        for _, agent in self.agents.items():
            agent.set_facecolor(agent.original_face_color)

        # check drive-drive collisions
        agents_array = [agent for _, agent in self.agents.items()]
        for i in range(0, len(agents_array)):
            for j in range(i + 1, len(agents_array)):
                d1 = agents_array[i]
                d2 = agents_array[j]
                pos1 = np.array(d1.center)
                pos2 = np.array(d2.center)
                if np.linalg.norm(pos1 - pos2) < 0.7:
                    d1.set_facecolor('red')
                    d2.set_facecolor('red')
                    print("COLLISION! (agent-agent) ({}, {})".format(i, j))

        return self.patches + self.artists

    def getState(self, t, d):
        idx = 0
        while idx < len(d) and d[idx]["t"] < t:
            idx += 1
        if idx == 0:
            return np.array([float(d[0]["x"]), float(d[0]["y"])])
        elif idx < len(d):
            posLast = np.array([float(d[idx - 1]["x"]), float(d[idx - 1]["y"])])
            posNext = np.array([float(d[idx]["x"]), float(d[idx]["y"])])
        else:
            return np.array([float(d[-1]["x"]), float(d[-1]["y"])])
        dt = d[idx]["t"] - d[idx - 1]["t"]
        t = (t - d[idx - 1]["t"]) / dt
        pos = (posNext - posLast) * t + posLast
        return pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("map", help="input file containing map")
    parser.add_argument("schedule", help="schedule for agents")
    parser.add_argument('--video', dest='video', default=None,
                        help="output video file (or leave empty to show on screen)")
    parser.add_argument("--speed", type=int, default=1, help="speedup-factor")
    args = parser.parse_args()

    with open(args.map) as map_file:
        map = yaml.load(map_file)

    with open(args.schedule) as states_file:
        schedule = yaml.load(states_file)

    animation = Animation(map, schedule)

    if args.video:
        animation.save(args.video, args.speed)
    else:
        animation.show()
