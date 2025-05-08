import os

import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load data from a given text file."""
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)  # Skip the header
    return data

def plot_comparison(ratio, QR1_pos, QR1_vel, QR2_pos, QR2_vel, QR3_pos, QR3_vel, QR4_pos, QR4_vel, QR5_pos, QR5_vel, num_joints):
    """Plot the comparison of positions and velocities."""
    for i in range(num_joints):
        plt.figure(figsize=(12, 10))

        # Position plot for joint i
        plt.subplot(2, 1, 1)
        plt.plot(QR1_pos[:, i], label=f'Measured Position for Q/R = {ratio}, R = 10 ', color='r',
                  linewidth=1)
        plt.plot(QR2_pos[:, i], label=f'Measured Position for Q/R = {ratio}, R = 1 ',
                 color='b',  linewidth=1)
        plt.plot(QR3_pos[:, i], label=f'Measured Position for Q/R = {ratio}, R = 0.1 ',
                 color='g',  linewidth=1)
        plt.plot(QR4_pos[:, i], label=f'Measured Position for Q/R = {ratio}, R = 0.01 ',
                 color='y', linewidth=1)
        plt.plot(QR5_pos[:, i], label=f'Measured Position for Q/R = {ratio}, R = 0.001 ',
                 color='pink',  linewidth=1)
        plt.title(f'Position Comparison for Joint {i + 1}')
        plt.xlabel('Time steps')
        plt.ylabel('Position')
        plt.legend()
        plt.grid()

        # Velocity plot for joint i
        plt.subplot(2, 1, 2)
        plt.plot(QR1_vel[:, i], label=f'Measured Velocity for Q/R = {ratio}, R = 10 ', color='r',
                 linewidth=1)
        plt.plot(QR2_vel[:, i], label=f'Measured Velocity for Q/R = {ratio}, R = 1 ',
                 color='b', linewidth=1)
        plt.plot(QR3_vel[:, i], label=f'Measured Velocity for Q/R = {ratio}, R = 0.1',
                 color='g', linewidth=1)
        plt.plot(QR4_vel[:, i], label=f'Measured Velocity for Q/R = {ratio}, R = 0.01 ',
                 color='y', linewidth=1)
        plt.plot(QR5_vel[:, i], label=f'Measured Velocity for Q/R = {ratio}, R = 0.001 ',
                 color='pink', linewidth=1)
        plt.title(f'Velocity Comparison for Joint {i + 1}')
        plt.xlabel('Time steps')
        plt.ylabel('Velocity')
        plt.legend()
        plt.grid()

        # Save the figure
        plt.tight_layout()
        # image_dir = "report_figure/Q&R"
        # if not os.path.exists(image_dir):
        #     os.makedirs(image_dir)  # Create the directory if it doesn't exist
        #
        # image_path = os.path.join(image_dir, f"joint_{i + 1}_Q_R_{ratio}.png")
        #
        # plt.savefig(image_path, format='png', dpi=300)
        plt.show()

def main():

    # File paths for small damping coeff
    ratio = 1000000 # 1000000, 10000000, 1000000000

    QR1_POS = f"measurements/output_data_pos_QR_{ratio*10}_10.txt"
    QR1_VEL = f"measurements/output_data_vel_QR_{ratio*10}_10.txt"
    QR2_POS = f"measurements/output_data_pos_QR_{ratio}_1.txt"
    QR2_VEL = f"measurements/output_data_vel_QR_{ratio}_1.txt"
    QR3_POS = f"measurements/output_data_pos_QR_{int(ratio/10)}_0.1.txt"
    QR3_VEL = f"measurements/output_data_vel_QR_{int(ratio/10)}_0.1.txt"
    QR4_POS = f"measurements/output_data_pos_QR_{int(ratio/100)}_0.01.txt"
    QR4_VEL = f"measurements/output_data_vel_QR_{int(ratio/100)}_0.01.txt"
    QR5_POS = f"measurements/output_data_pos_QR_{int(ratio/1000)}_0.001.txt"
    QR5_VEL = f"measurements/output_data_vel_QR_{int(ratio/1000)}_0.001.txt"

    # Load data from files
    QR1_pos = load_data(QR1_POS)
    QR1_vel = load_data(QR1_VEL)
    QR2_pos = load_data(QR2_POS)
    QR2_vel = load_data(QR2_VEL)
    QR3_pos = load_data(QR3_POS)
    QR3_vel = load_data(QR3_VEL)
    QR4_pos = load_data(QR4_POS)
    QR4_vel = load_data(QR4_VEL)
    QR5_pos = load_data(QR5_POS)
    QR5_vel = load_data(QR5_VEL)

    # Determine number of joints based on the shape of the data
    num_joints = QR1_pos.shape[1]

    # Plot comparison
    plot_comparison(ratio, QR1_pos, QR1_vel, QR2_pos, QR2_vel, QR3_pos, QR3_vel, QR4_pos, QR4_vel, QR5_pos, QR5_vel, num_joints)

if __name__ == '__main__':
    main()
