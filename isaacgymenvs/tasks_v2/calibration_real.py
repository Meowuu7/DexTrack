"""Example script for using joint impedance control."""
import argparse
import pickle
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deoxys import config_root
from deoxys.experimental.motion_utils import joint_interpolation_traj
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger

logger = get_deoxys_example_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument(
        "--controller-cfg", type=str, default="joint-impedance-controller.yml"
    )
    args = parser.parse_args()
    return args

def generate_sin_wave(last_q):
    """ 生成正弦波目标位置序列 """
    amplitude = 0.1 # 正弦波振幅
    dt = 0.05      # 控制时间步
    period = 2.0     # 正弦波周期
    steps = 40 * 8      # 步数
    sin_wave_targets = [last_q + amplitude * np.sin(2 * np.pi * t * dt / period) for t in range(steps)]
    return np.array(sin_wave_targets)

def main():
    args = parse_args()

    robot_interface = FrankaInterface(
        config_root + f"/{args.interface_cfg}", use_visualizer=False
    )
    controller_cfg = YamlConfig(config_root + f"/{args.controller_cfg}").as_easydict()

    controller_type = "JOINT_IMPEDANCE"


    while robot_interface.state_buffer_size == 0:
        logger.warn("Robot state not received")
        time.sleep(0.5)

    last_q = np.array(robot_interface.last_q)
    sin_wave_targets = generate_sin_wave(last_q)
    i = 0
    all_trajectory = []
    target_trajectory = []
    first_iteration = True
    num_timesteps = 40 * 8
    num_dofs = 7

    for target_pos in sin_wave_targets:
        start_time = time.time()  # 记录当前时间
        target_trajectory.extend(target_pos.tolist())
        if not first_iteration:  # 如果不是第一次迭代，执行读取和记录位置
            last_pos = np.array(robot_interface.last_q)
            # print("actual_pos = ",last_pos)
            all_trajectory.extend(last_pos.tolist())
        else:
            first_iteration = False  # 第一次迭代后，设置标志为False
        # print("i",i )
        # print("taget_pos = ",target_pos)
        action = target_pos.tolist() + [-1.0]
        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )
        i+=1
        elapsed_time = time.time() - start_time
        # print("elapsed_time = ",elapsed_time)
        # 确保每次循环的间隔是 0.05 秒
        if elapsed_time < 0.05:
            time.sleep(0.05 - elapsed_time)  # 如果执行时间少于 0.05秒，补充剩余的时间
        # end_time = time.time()
        # print("loop time = ", end_time - start_time)
    
    last_pos = np.array(robot_interface.last_q)
    all_trajectory.extend(last_pos.tolist())

    robot_interface.close()

    target_trajectory = np.array(target_trajectory).reshape(num_timesteps, num_dofs)
    actual_trajectory = np.array(all_trajectory).reshape(num_timesteps, num_dofs)

    save_path = '/home/janebek/Janebek/deoxys_control/actual_franka_sinwave_trajectory.npy'
    np.save(save_path, actual_trajectory)

    time_steps = np.linspace(0, num_timesteps - 1, num_timesteps) * 0.05  # 假设总时长为1秒
    
    fig, axs = plt.subplots(3, 3, figsize=(20, 15))
    axs = axs.flatten()  # 将二维数组扁平化，方便通过索引访问

    for i in range(num_dofs):  
        axs[i].plot(time_steps, target_trajectory[:, i], 'r--', label='real_target')
        axs[i].plot(time_steps, actual_trajectory[:, i], 'b-', label='real_tracking')
        axs[i].set_title(f'DOF {i+1}')
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('Position')
        axs[i].legend()
        axs[i].grid(True)

    # 隐藏多余的子图
    for i in range(num_dofs, 9):
        axs[i].axis('off')  # 隐藏多余的子图区域

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
