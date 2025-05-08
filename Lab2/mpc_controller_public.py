import numpy as np
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, dyn_cancel, SinusoidalReference, \
    CartesianDiffKin
from regulator_model import RegulatorModel


def initialize_simulation(conf_file_name):
    """Initialize simulation and dynamic model."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)

    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]

    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    return sim, dyn_model, num_joints


def print_joint_info(sim, dyn_model, controlled_frame_name):
    """Print initial joint angles and limits."""
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos, init_R = dyn_model.ComputeFK(init_joint_angles, controlled_frame_name)

    print(f"Initial joint angles: {init_joint_angles}")

    lower_limits, upper_limits = sim.GetBotJointsLimit()
    print(f"Lower limits: {lower_limits}")
    print(f"Upper limits: {upper_limits}")

    joint_vel_limits = sim.GetBotJointsVelLimit()
    print(f"Joint velocity limits: {joint_vel_limits}")


def getSystemMatrices(sim, num_joints, damping_coefficients=None):
    """
    Get the system matrices A and B according to the dimensions of the state and control input.

    Parameters:
    sim: Simulation object
    num_joints: Number of robot joints
    damping_coefficients: List or numpy array of damping coefficients for each joint (optional)

    Returns:
    A: State transition matrix
    B: Control input matrix
    """
    num_states = 2 * num_joints
    num_controls = num_joints

    time_step = sim.GetTimeStep()

    if damping_coefficients is None:
        damping_coefficients = np.zeros(num_joints)
    else:
        damping_coefficients = np.array(damping_coefficients)
        print(damping_coefficients)

    D = np.diag(damping_coefficients)

    # Identity matrix
    I = np.eye(num_joints)

    # A matrix (state transition matrix)
    A = np.block([
        [I, time_step * I],
        [np.zeros((num_joints, num_joints)), (I - time_step * D)]
    ])
    # print(A.shape)

    # B matrix (control input matrix)
    B = np.block([
        [np.zeros((num_joints, num_joints))],
        [time_step * I]
    ])
    # print(B.shape)
    return A, B


def getCostMatrices(Q_value, R_value, num_joints):
    """
    Get the cost matrices Q and R for the MPC controller.

    Returns:
    Q: State cost matrix
    R: Control input cost matrix
    """
    num_states = 2 * num_joints
    num_controls = num_joints

    # Q = 1 * np.eye(num_states)  # State cost matrix
    Q = Q_value * np.eye(num_states)
    Q[num_joints:, num_joints:] = 0.0
    print(Q.shape)
    R = R_value * np.eye(num_controls)  # Control input cost matrix # 0.1
    print(R.shape)

    return Q, R


def main():
    # Configuration
    """
        When consider damping, the config file should also be changed!

        "motor_damping": [0], means no damping
        "motor_damping": [1], means damping exists
    """
    damping = [0.5, 0.6, 0.2, 0.1, 0.3, 0.35, 0.8]

    consider_damping = True
    if consider_damping==True:
        conf_file_name =  "pandaconfig_damping.json"
    else:
        conf_file_name = "pandaconfig.json"
    controlled_frame_name = "panda_link8"

    # Initialize simulation and dynamic model
    sim, dyn_model, num_joints = initialize_simulation(conf_file_name)
    cmd = MotorCommands()

    # Print joint information
    print_joint_info(sim, dyn_model, controlled_frame_name)

    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all = [], [], [], []
    # Define reference state (target position and velocity)

    q_ref = np.array([0, 0, 0, 0, 0, 0, 0])  # Example reference position
    qd_ref = np.array([0, 0, 0, 0, 0, 0, 0])  # Example reference velocity
    x_ref = np.hstack((q_ref, qd_ref))  # Full reference state vector
    print(x_ref)


    # Define the matrices and consider the damping
    Q_value = 10000
    R_value = 0.001


    if consider_damping:
        A, B = getSystemMatrices(sim, num_joints, damping_coefficients=damping)
    else:
        A, B = getSystemMatrices(sim, num_joints)
    Q, R = getCostMatrices(Q_value, R_value, num_joints)

    # Measuring all the state
    num_states = 2 * num_joints
    C = np.eye(num_states)

    # Horizon length
    N_mpc = 10

    # Initialize the regulator model
    regulator = RegulatorModel(A, B, C, Q, R, N_mpc, num_states, num_joints, num_states, x_ref)
    # Compute the matrices needed for MPC optimization
    S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
    H, F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)
    # Main control loop
    episode_duration = 10
    current_time = 0
    time_step = sim.GetTimeStep()
    steps = int(episode_duration / time_step)
    sim.ResetPose()
    # sim.SetSpecificPose([0, 0, 0, 0, 0, 0, 0])
    # testing loop
    for i in range(steps):
        # measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)

        x0_mpc = np.vstack((q_mes, qd_mes))
        x0_mpc = x0_mpc.flatten()
        # Compute the optimal control sequence
        H_inv = np.linalg.inv(H)
        # use x_ref to update
        u_mpc = -H_inv @ F @ (x0_mpc - x_ref)

        # Return the optimal control sequence
        u_mpc = u_mpc[0:num_joints]

        # Control command old env
        # cmd.tau_cmd = dyn_cancel(dyn_model, q_mes, qd_mes, u_mpc)
        # sim.Step(cmd, "torque")  # Simulation step with torque command
        # Control command new env
        angular_wheels_velocity_cmd = u_mpc
        cmd.SetControlCmd(angular_wheels_velocity_cmd, ["torque"] * 7)  # Simulation step with torque command
        sim.Step(cmd, "torque")
        # print(cmd.tau_cmd)
        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # simulation_time = sim.GetTimeSinceReset()

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)

        # time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
        # print(f"Current time: {current_time}")


    # # Write q_mes_all and qd_mes_all to a text file
    # output_file_path = "measurements/output_data_pos_damping_"+str(consider_damping)+"_small.txt"
    # with open(output_file_path, 'w') as f:
    #     for q in q_mes_all:
    #         f.write(','.join(map(str, q)) + '\n')  # Write each position as a comma-separated string
    # output_file_path = "output_data_vel_damping_"+str(consider_damping)+"_small.txt"
    # with open(output_file_path, 'w') as f:
    #     for qd in qd_mes_all:
    #         f.write(','.join(map(str, qd)) + '\n')  # Write each velocity as a comma-separated string
    # print(f"Data has been written to {output_file_path}.")

    # # Write q_mes_all and qd_mes_all to a text file for Q/R
    # output_file_path = "measurements/output_data_pos_QR_" + str(Q_value)+"_"+ str(R_value) + ".txt"
    # with open(output_file_path, 'w') as f:
    #     for q in q_mes_all:
    #         f.write(','.join(map(str, q)) + '\n')  # Write each position as a comma-separated string
    # output_file_path = "output_data_vel_QR_" + str(Q_value)+"_"+ str(R_value) + ".txt"
    # with open(output_file_path, 'w') as f:
    #     for qd in qd_mes_all:
    #         f.write(','.join(map(str, qd)) + '\n')  # Write each velocity as a comma-separated string
    # print(f"Data has been written to {output_file_path}.")

    # Plotting
    for i in range(num_joints):
        plt.figure(figsize=(10, 8))

        # Position plot for joint i
        plt.subplot(2, 1, 1)
        plt.plot([q[i] for q in q_mes_all], label=f'Measured Position - Joint {i + 1}')
        plt.plot([q_ref[i] for _ in q_mes_all], 'r--', label=f'Reference Position - Joint {i + 1}')  # 添加参考线
        plt.title(f'Position Tracking for Joint {i + 1}')
        plt.xlabel('Time steps')
        plt.ylabel('Position')
        plt.legend()

        # Velocity plot for joint i
        plt.subplot(2, 1, 2)
        plt.plot([qd[i] for qd in qd_mes_all], label=f'Measured Velocity - Joint {i + 1}')
        plt.plot([qd_ref[i] for _ in qd_mes_all], 'r--', label=f'Reference Velocity - Joint {i + 1}')  # 添加参考线
        plt.title(f'Velocity Tracking for Joint {i + 1}')
        plt.xlabel('Time steps')
        plt.ylabel('Velocity')
        plt.legend()

        # Ensure the directory exists before saving
        # image_dir = "report_figure/Q&R"
        # if not os.path.exists(image_dir):
        #     os.makedirs(image_dir)  # Create the directory if it doesn't exist
        #
        # image_path = os.path.join(image_dir, f"joint_{i + 1}_Q_"+str(Q_value)+"_R_"+str(R_value)+".png")
        # plt.savefig(image_path, format='png', dpi=300)  # Save the image
        # print(f"Image saved!")

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()