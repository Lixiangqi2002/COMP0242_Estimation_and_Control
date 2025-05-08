import os

import numpy as np
import matplotlib.pyplot as plt
from docutils.nodes import reference
from simulation_and_control import SinusoidalReference

from Lab3.linear_ref import LinearReference
from Lab3.polynomial_ref import PolynomialReference


def load_data(file_path):
    """Load data from a given text file."""
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)  # Skip the header
    return data

def plot_comparison(reff, constraints1_pos, constraints1_vel, constraints2_pos, constraints2_vel, constraints3_pos, constraints3_vel, constraints4_pos, constraints4_vel, constraints5_pos, constraints5_vel, constraints6_pos, constraints6_vel, constraints7_pos, constraints7_vel,constraints8_pos, constraints8_vel, num_joints):
    """Plot the comparison of positions and velocities."""

    episode_duration = 0.2  # duration in seconds
    current_time = 0
    time_step = 0.001
    steps = int(episode_duration / time_step)
    q_d_all, qd_d_all = [], []
    lower_pos = [-2.8973, -1.7628, -2.8973, 0.0, -2.8973, -0.0175, -2.8973]
    upper_pos = [2.8973, 1.7628, 2.8973, 3.002, 2.8973, 3.7525, 2.8973]
    lower_vel = [-2.175, -2.175, -2.175, -2.175, -2.61, -2.61, -2.61]
    upper_vel = [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]
    if reff == "sin":
        amplitudes = [np.pi / 4, np.pi / 6, np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4]
        frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints
        amplitude = np.array(amplitudes)
        frequency = np.array(frequencies)
        q_init = np.array([0.0, 1.0323, 0.0, 0.8247, 0.0, 1.57, 0.0])
        ref = SinusoidalReference(amplitude, frequency, q_init)  # Initialize the reference
    elif reff == "linear":
        # slope = [0.3, 0.1, -0.1, 0.15, -0.2, 0.1, 0.05]
        slope = [0.5, 0.2, -0.1, 0.3, -0.4, 0.1, 0.05]
        intercept = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        slope = np.array(slope)
        intercept = np.array(intercept)
        q_init = np.array([0.0, 1.0323, 0.0, 0.8247, 0.0, 1.57, 0.0])
        ref = LinearReference(slope, intercept, q_init)
    elif reff == "polynomial":
        coefficients = [
            [0.1, 0.5, -2.0],  # Joint 1: 0.1*t^2 + 0.5*t - 2.0 (stays within [-2.8973, 2.8973])
            [0.0, 0.0, 0.0],  # Joint 2: -0.05*t^2 + 0.3*t - 1.5 (stays within [-1.7628, 1.7628])
            [0.2, 0.3, -1.5],  # Joint 3: 0.2*t^2 (stays within [-2.8973, 2.8973])
            [0.1, -0.1, 1.0],
            [0.15, 0.2, -2.5],
            [-0.1, 0.4, 0.1],
            [0.05, 0.25, 0.0]
        ]
        coefficients = np.array(coefficients)
        q_init = np.array([0.0, 1.0323, 0.0, 0.8247, 0.0, 1.57, 0.0])
        ref = PolynomialReference(coefficients, q_init)

    for i in range(steps-1):
        print(i)
        q_d, qd_d = ref.get_values(current_time)
        q_d_all.append(q_d)
        qd_d_all.append(qd_d)
        current_time += time_step

    # plotting
    for i in range(num_joints):
        plt.figure(figsize=(12, 10))
        colors = ['r', 'g', 'c', 'k', 'm', 'y', 'brown', 'b']  # 不同颜色组合
        linestyles = ['-', '--', '-.', ':']  # 不同线型
        # Position plot for joint i
        plt.subplot(2, 1, 1)
        upper_pos_limit = upper_pos[i] if isinstance(upper_pos[i], list) else [upper_pos[i]] * len(q_d_all)
        lower_pos_limit = lower_pos[i] if isinstance(lower_pos[i], list) else [lower_pos[i]] * len(q_d_all)
        # plt.plot(upper_pos_limit, color='grey', linewidth=2)
        # plt.plot(lower_pos_limit, color='grey', linewidth=2)
        plt.fill_between(range(len(q_d_all)), upper_pos_limit, lower_pos_limit, color='green', alpha=0.2, label="Position Constraints")

        plt.plot([q[i] for q in q_d_all], label=f'Desired Position - Joint {i + 1}', linestyle='--',
                 color='green', linewidth=2)
        plt.plot(constraints1_pos[:, i], label=f'Measured Position for No Constraints ', linestyle=linestyles[0],
                 color=colors[0], linewidth=1)
        plt.plot(constraints2_pos[:, i], label=f'Measured Position for Ts = 0.01 ',
                 linestyle=linestyles[1],color=colors[1],linewidth=1)
        plt.plot(constraints3_pos[:, i], label=f'Measured Position for Ts = 0.025 ',
                 linestyle=linestyles[2],color=colors[2],  linewidth=1)
        plt.plot(constraints4_pos[:, i], label=f'Measured Position for Ts = 0.05 ',
                 linestyle=linestyles[3],color=colors[3], linewidth=1)
        plt.plot(constraints5_pos[:, i], label=f'Measured Position for Ts = 0.075 ',
                 color=colors[4],  linewidth=1)
        plt.plot(constraints6_pos[:, i], label=f'Measured Position for Ts = 0.10 ',
                 color=colors[5],linestyle=linestyles[1], linewidth=1)
        plt.plot(constraints7_pos[:, i], label=f'Measured Position for Ts = 0.15 ',
                 color=colors[6],linestyle=linestyles[2], linewidth=1)
        plt.plot(constraints8_pos[:, i], label=f'Measured Position for Ts = 0.20 ',
                 color=colors[7],linestyle=linestyles[3], linewidth=1)
        plt.title(f'Position Comparison for Joint {i + 1} With Different Constraints Ts')
        plt.xlabel('Time steps')
        plt.ylabel('Position')
        plt.legend()
        plt.grid()

        # Velocity plot for joint i
        plt.subplot(2, 1, 2)
        upper_vel_limit = upper_vel[i] if isinstance(upper_vel[i], list) else [upper_vel[i]] * len(q_d_all)
        lower_vel_limit = lower_vel[i] if isinstance(lower_vel[i], list) else [lower_vel[i]] * len(q_d_all)
        # plt.plot(upper_vel_limit, color='grey', linewidth=2)
        # plt.plot(lower_vel_limit, color='grey', linewidth=2)
        plt.fill_between(range(len(q_d_all)), upper_vel_limit, lower_vel_limit, color='green', alpha=0.2, label="Velocity Constraints")

        plt.plot([qd[i] for qd in qd_d_all], label=f'Desired Velocity - Joint {i + 1}',
                 linestyle='--', color='g', linewidth=2)
        plt.plot(constraints1_vel[:, i], label=f'Measured Velocity for No Constraints ', linestyle=linestyles[0],
                 color=colors[0],
                 linewidth=1)
        plt.plot(constraints2_vel[:, i], label=f'Measured Velocity for Ts = 0.01 ',
                 linestyle=linestyles[1],color=colors[1], linewidth=1)
        plt.plot(constraints3_vel[:, i], label=f'Measured Velocity for Ts = 0.025 ',
                 linestyle=linestyles[2],color=colors[2], linewidth=1)
        plt.plot(constraints4_vel[:, i], label=f'Measured Velocity for Ts = 0.05 ',
                 linestyle=linestyles[3],color=colors[3], linewidth=1)
        plt.plot(constraints5_vel[:, i], label=f'Measured Velocity for Ts = 0.075 ',
                 color=colors[4], linewidth=1)
        plt.plot(constraints6_vel[:, i], label=f'Measured Velocity for Ts = 0.10 ',
                 color=colors[5], linestyle=linestyles[1], linewidth=1)
        plt.plot(constraints7_vel[:, i], label=f'Measured Velocity for Ts = 0.15 ',
                 color=colors[6], linestyle=linestyles[2], linewidth=1)
        plt.plot(constraints8_vel[:, i], label=f'Measured Velocity for Ts = 0.20 ',
                 color=colors[7], linestyle=linestyles[3], linewidth=1)
        plt.title(f'Position Comparison for Joint {i + 1} With Different Constraints Ts')
        plt.xlabel('Time steps')
        plt.ylabel('Velocity')
        plt.legend()
        plt.grid()

        # Save the figure
        plt.tight_layout()
        image_dir = "report_figures/constraints"
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)  # Create the directory if it doesn't exist

        image_path = os.path.join(image_dir, f"joint_{i + 1}_{reff}.png")
        plt.savefig(image_path, format='png', dpi=300)
        plt.show()

def main():
    # File paths for small damping coeff
    """
        cur : [0, 1, 2]   ------ represent the reference type for comparison
        linear : Q_1000000000_10000.txt
        sin : Q_10000000_1000.txt
        polynomial : Q_1000000000_10000.txt
    """
    cur = 0
    reference = ["linear", "sin", "polynomial"]
    config_name = ["Q_1000000000_10000.txt", "Q_10000000_1000.txt", "Q_1000000000_10000.txt"]
    config = config_name[cur]
    ref_type = reference[cur]

    txt_path = "report_figures/constraints/"+ ref_type +"/"
    constraints1_POS = txt_path  + "constraint_False_pos_" + config
    constraints1_VEL = txt_path  + "constraint_False_vel_" + config
    constraints2_POS = txt_path  + "constraint_True_0.01_pos_" + config
    constraints2_VEL = txt_path  + "constraint_True_0.01_vel_" + config
    constraints3_POS = txt_path  + "constraint_True_0.025_pos_" + config
    constraints3_VEL = txt_path  + "constraint_True_0.025_vel_" + config
    constraints4_POS = txt_path  + "constraint_True_0.05_pos_" + config
    constraints4_VEL = txt_path  + "constraint_True_0.05_vel_" + config
    constraints5_POS = txt_path  + "constraint_True_0.075_pos_" + config
    constraints5_VEL = txt_path  + "constraint_True_0.075_vel_" + config
    constraints6_POS = txt_path + "constraint_True_0.1_pos_" + config
    constraints6_VEL = txt_path + "constraint_True_0.1_vel_" + config
    constraints7_POS = txt_path + "constraint_True_0.15_pos_" + config
    constraints7_VEL = txt_path + "constraint_True_0.15_vel_" + config
    constraints8_POS = txt_path + "constraint_True_0.2_pos_" + config
    constraints8_VEL = txt_path + "constraint_True_0.2_vel_" + config

    # Load data from files
    constraints1_pos = load_data(constraints1_POS)
    constraints1_vel = load_data(constraints1_VEL)
    constraints2_pos = load_data(constraints2_POS)
    constraints2_vel = load_data(constraints2_VEL)
    constraints3_pos = load_data(constraints3_POS)
    constraints3_vel = load_data(constraints3_VEL)
    constraints4_pos = load_data(constraints4_POS)
    constraints4_vel = load_data(constraints4_VEL)
    constraints5_pos = load_data(constraints5_POS)
    constraints5_vel = load_data(constraints5_VEL)
    constraints6_pos = load_data(constraints6_POS)
    constraints6_vel = load_data(constraints6_VEL)
    constraints7_pos = load_data(constraints7_POS)
    constraints7_vel = load_data(constraints7_VEL)
    constraints8_pos = load_data(constraints8_POS)
    constraints8_vel = load_data(constraints8_VEL)

    # Determine number of joints based on the shape of the data
    num_joints = constraints1_pos.shape[1]

    # Plot comparison
    plot_comparison(ref_type, constraints1_pos, constraints1_vel, constraints2_pos, constraints2_vel, constraints3_pos, constraints3_vel, constraints4_pos, constraints4_vel, constraints5_pos, constraints5_vel,
                    constraints6_pos, constraints6_vel, constraints7_pos, constraints7_vel,constraints8_pos, constraints8_vel,  num_joints)

if __name__ == '__main__':
    main()
