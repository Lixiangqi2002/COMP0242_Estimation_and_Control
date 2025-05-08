import os

import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load data from a given text file."""
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)  # Skip the header
    return data

def plot_comparison(true_pos, true_vel, false_pos, false_vel, num_joints):
    """Plot the comparison of positions and velocities."""
    for i in range(num_joints):
        plt.figure(figsize=(12, 10))

        # Position plot for joint i
        plt.subplot(2, 1, 1)
        plt.plot(true_pos[:, i], label=f'Measured Position (with damping) - Joint {i + 1}', color='r',
                 linestyle='-', linewidth=2)
        plt.plot(false_pos[:, i], label=f'Measured Position (no damping) - Joint {i + 1}',
                 color='b', linestyle='--', linewidth=1)
        plt.title(f'Position Comparison for Joint {i + 1}')
        plt.xlabel('Time steps')
        plt.ylabel('Position')
        plt.legend()
        plt.grid()

        # Velocity plot for joint i
        plt.subplot(2, 1, 2)
        plt.plot(true_vel[:, i], label=f'Measured Velocity (with damping) - Joint {i + 1}', color='r',
                 linestyle='-', linewidth=2)
        plt.plot(false_vel[:, i], label=f'Measured Velocity (no damping) - Joint {i + 1}',
                 color='b', linestyle='--', linewidth=1)

        plt.title(f'Velocity Comparison for Joint {i + 1}')
        plt.xlabel('Time steps')
        plt.ylabel('Velocity')
        plt.legend()
        plt.grid()

        # Save the figure
        plt.tight_layout()
        # image_dir = "report_figure/damping"
        # if not os.path.exists(image_dir):
        #     os.makedirs(image_dir)  # Create the directory if it doesn't exist
        #
        # image_path = os.path.join(image_dir, f"joint_{i + 1}_damping_comparison_small.png")
        #
        # plt.savefig(image_path, format='png', dpi=300)
        plt.show()

def main():
    compare_damping_large_coeff = True
    if compare_damping_large_coeff:
        # # File paths for large damping coeff
        damping_true_pos_path = "measurements/output_data_pos_damping_True.txt"
        damping_true_vel_path = "measurements/output_data_vel_damping_True.txt"
        damping_false_pos_path = "measurements/output_data_pos_damping_False.txt"
        damping_false_vel_path = "measurements/output_data_vel_damping_False.txt"
    else:
        # File paths for small damping coeff
        damping_true_pos_path = "measurements/output_data_pos_damping_True_small.txt"
        damping_true_vel_path = "measurements/output_data_vel_damping_True_small.txt"
        damping_false_pos_path = "measurements/output_data_pos_damping_False_small.txt"
        damping_false_vel_path = "measurements/output_data_vel_damping_False_small.txt"

    # Load data from files
    true_pos = load_data(damping_true_pos_path)
    true_vel = load_data(damping_true_vel_path)
    false_pos = load_data(damping_false_pos_path)
    false_vel = load_data(damping_false_vel_path)

    # Determine number of joints based on the shape of the data
    num_joints = true_pos.shape[1]

    # Plot comparison
    plot_comparison(true_pos, true_vel, false_pos, false_vel, num_joints)

if __name__ == '__main__':
    main()
