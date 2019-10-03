import numpy as np
import env.context as ctx
from wf_gen_funcs import tree_data_wf, read_workflow
import actor
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_schedule(wfl):
    schedule = wfl.schedule
    worst = wfl.worst_time
    n = wfl.n
    colors = ["r", "g", "b", "yellow", "orange", "purple", "brown", "violet"]
    fig, ax = plt.subplots(1)
    m = len(schedule.keys())
    keys = list(schedule.keys())
    for k in range(m):
        items = schedule[keys[k]]
        for it in items:
            print("Task {}, st {} end {}".format(it.task, it.st_time, it.end_time))
            coords = (it.st_time, k)
            rect = patches.Rectangle(coords, it.end_time - it.st_time, 1, fill=True, facecolor="r", label=it.task, alpha=0.5, edgecolor="black")
            ax.add_patch(rect)
            ax.text(coords[0] + (it.end_time-it.st_time)/3, coords[1]+0.5, str(it.task))

    plt.legend()
    plt.ylim(0,m)
    plt.xlim(0, worst)
    plt.title(wfl.wf_name)
    plt.show()
