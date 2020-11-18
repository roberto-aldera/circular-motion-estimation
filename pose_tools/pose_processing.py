# Comparing RO to INS data
import numpy as np
import matplotlib.pyplot as plt
from pose_utils import *
import settings


def process_ro():
    RO_se3s, RO_timestamps = get_poses_from_file(settings.RO_PATH)
    RO_speeds = get_speeds(RO_se3s, RO_timestamps)
    fig = plt.figure(figsize=(15, 5))
    plt.plot(RO_speeds, color=colour_ro, label="RO")
    # plt.xlim(1100,1500)
    plt.ylim(-5, 20)
    plt.title("Velocity")
    plt.legend()
    plt.grid()
    plt.savefig(settings.FIGURE_PATH + "ro_speeds.png")

    x_RO, y_RO = get_poses(RO_se3s)
    plt.figure(figsize=(10, 10))
    plt.plot(x_RO, y_RO, '-', color=colour_ro, markersize=1)
    plt.axis('equal')
    plt.title('Cumulative poses comparison for integrated odometry readings')
    plt.xlabel('X position from start [m]')
    plt.ylabel('Y position from start [m]')
    line2, = plt.plot([], [], color=colour_ro, label='RO', linewidth=10.0)
    # plt.xlim(150, 200)
    # plt.ylim(0,20)
    plt.legend(handles=[line2], loc='best')
    plt.grid()
    plt.savefig(settings.FIGURE_PATH + "all_poses.png", bbox_inches='tight', pad_inches=0.5)


def process_ins():
    INS_se3s, INS_timestamps = get_poses_from_file(settings.INS_PATH)
    INS_speeds = get_speeds(INS_se3s, INS_timestamps)
    fig = plt.figure(figsize=(15, 5))
    plt.plot(INS_speeds, color=colour_ro, label="RO")
    # plt.xlim(1100,1500)
    plt.ylim(-5, 20)
    plt.title("Velocity")
    plt.legend()
    plt.grid()
    plt.savefig(settings.FIGURE_PATH + "INS_speeds.png")

    x_INS, y_INS = get_poses(INS_se3s)
    plt.figure(figsize=(10, 10))
    plt.plot(x_INS, y_INS, '-', color=colour_ins, markersize=1)
    plt.axis('equal')
    plt.title('Cumulative poses comparison for integrated odometry readings')
    plt.xlabel('X position from start [m]')
    plt.ylabel('Y position from start [m]')
    line2, = plt.plot([], [], color=colour_ins, label='INS', linewidth=10.0)
    # plt.xlim(150, 200)
    # plt.ylim(0,20)
    plt.legend(handles=[line2], loc='best')
    plt.grid()
    plt.savefig(settings.FIGURE_PATH + "INS_poses.png", bbox_inches='tight', pad_inches=0.5)

    save_poses_to_csv(INS_se3s, INS_timestamps, "ins", settings.FIGURE_PATH)


if __name__ == "__main__":
    print("Running main...")
    colour_ro = "tab:green"
    colour_ins = "black"
    # process_ro()
    process_ins()
