import numpy as np
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, dyn_cancel, SinusoidalReference
from polynomial_ref import PolynomialReference
from linear_ref import LinearReference
from tracker_model import TrackerModel
from tracker_model_constraints import ConstraintsTrackerModel

init_joint_angles = []


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
    return lower_limits, upper_limits, joint_vel_limits


def getSystemMatricesContinuos(num_joints, damping_coefficients=None):
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

    # Initialize A matrix
    A = np.zeros((num_states, num_states))

    # Upper right quadrant of A (position affected by velocity)
    A[:num_joints, num_joints:] = np.eye(num_joints)

    # Lower right quadrant of A (velocity affected by damping)
    # if damping_coefficients is not None:
    #    damping_matrix = np.diag(damping_coefficients)
    #    A[num_joints:, num_joints:] = np.eye(num_joints) - time_step * damping_matrix

    # Initialize B matrix
    B = np.zeros((num_states, num_controls))

    # Lower half of B (control input affects velocity)
    B[num_joints:, :] = np.eye(num_controls)

    return A, B


# Example usage:
# sim = YourSimulationObject()
# num_joints = 6  # Example: 6-DOF robot
# damping_coefficients = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05]  # Example damping coefficients
# A, B = getSystemMatrices(sim, num_joints, damping_coefficients)


def getCostMatrices(num_joints, Q_p, Q_v):
    """
    Get the cost matrices Q and R for the MPC controller.

    Returns:
    Q: State cost matrix
    R: Control input cost matrix
    """
    num_states = 2 * num_joints
    num_controls = num_joints

    # Q = 1 * np.eye(num_states)  # State cost matrix
    p_w = Q_p  # 10000
    v_w = Q_v  # 10
    Q_diag = np.array([p_w, p_w, p_w, p_w, p_w, p_w, p_w, v_w, v_w, v_w, v_w, v_w, v_w, v_w])
    Q = np.diag(Q_diag)

    print(Q)

    R = 0.1 * np.eye(num_controls)  # Control input cost matrix

    return Q, R


def main():
    # Configuration
    reference = ["sin", "linear", "polynomial"]
    control_type = ["all", "pos"]
    constraints = True
    reff = reference[0]
    control = control_type[0]

    conf_file_name = "pandaconfig.json"
    controlled_frame_name = "panda_link8"
    # Initialize simulation and dynamic model
    sim, dyn_model, num_joints = initialize_simulation(conf_file_name)
    cmd = MotorCommands()
    # Print joint information
    lower_limits, upper_limits, joint_vel_limits = print_joint_info(sim, dyn_model, controlled_frame_name)
    lower_vel_limits = [-2.175, -2.175, -2.175, -2.175, -2.61, -2.61, -2.61]
    upper_vel_limits = joint_vel_limits
    print(lower_vel_limits)
    Q_p, Q_v = 0, 0

    if reff == "sin":
        amplitudes = [np.pi / 4, np.pi / 6, np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4]
        frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints
        amplitude = np.array(amplitudes)
        frequency = np.array(frequencies)
        ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference
        if control == "all":
            Q_p = 10000000
            Q_v = 1000
        elif control == "pos":
            Q_p = 10000000
            Q_v = 0
    elif reff == "linear":
        slope = [0.5, 0.2, -0.1, 0.3, -0.4, 0.1, 0.05]
        intercept = [0.1, 0.0, 0.2, 0.1, 0.05, -0.1, 0.3]
        slope = np.array(slope)
        intercept = np.array(intercept)
        ref = LinearReference(slope, intercept, sim.GetInitMotorAngles())
        if control == "all":
            Q_p = 1000000000
            Q_v = 10000
        elif control == "pos":
            Q_p = 10000000
            Q_v = 0
    elif reff == "polynomial":

        coefficients = [
            [0.1, 0.5, -2.0],  # Joint 1: 0.1*t^2 + 0.5*t - 2.0 (stays within [-2.8973, 2.8973])
            [0.0, 0.0, 0.0],  # Joint 2: 0
            [0.2, 0.3, -1.5],  # Joint 3: 0.2*t^2 + 0.3*t - 1.5 (stays within [-2.8973, 2.8973])
            [0.1, -0.1, 1.0],
            [0.15, 0.2, -2.5],
            [-0.1, 0.4, 0.1],
            [0.05, 0.25, 0.0]
        ]
        coefficients = np.array(coefficients)
        ref = PolynomialReference(coefficients, sim.GetInitMotorAngles())
        if control == "all":
            Q_p = 1000000000
            Q_v = 10000
        elif control == "pos":
            Q_p = 100000000
            Q_v = 0

    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all = [], [], [], []
    regressor_all = np.array([])

    # Define the matrices
    A, B = getSystemMatricesContinuos(num_joints)
    Q, R = getCostMatrices(num_joints, Q_p, Q_v)

    # Measuring all the state
    num_states = 2 * num_joints
    C = np.eye(num_states)

    # Horizon length
    N_mpc = 5

    # Initialize the regulator model
    if constraints:
        c = 0.2
        slack_penalty_weight = np.ones(14) * c # slack for 7 joints' position and velocity
        tracker = ConstraintsTrackerModel(A, B, C, Q, R, N_mpc, num_states, num_joints, num_states, sim.GetTimeStep(),
                                          [lower_limits, upper_limits], [lower_vel_limits, upper_vel_limits],
                                          slack_penalty_weight)
    else:
        tracker = TrackerModel(A, B, C, Q, R, N_mpc, num_states, num_joints, num_states, sim.GetTimeStep())
    # Compute the matrices needed for MPC optimization
    S_bar, S_bar_C, T_bar, T_bar_C, Q_hat, Q_bar, R_bar = tracker.propagation_model_tracker_fixed_std()
    # print(S_bar.shape)
    H, Ftra = tracker.tracker_std(S_bar, T_bar, Q_hat, Q_bar, R_bar)
    # Main control loop
    episode_duration = 0.2 # duration in seconds
    current_time = 0
    time_step = sim.GetTimeStep()
    steps = int(episode_duration / time_step)
    sim.ResetPose()

    # testing loop
    u_mpc = np.zeros(num_joints)
    # for i in range(steps-2500, steps):
    for i in range(steps):
        print(i)
        # measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)
        # Compute sinusoidal reference trajectory
        # Ensure q_init is within the range of the amplitude

        x0_mpc = np.vstack((q_mes, qd_mes))
        x0_mpc = x0_mpc.flatten()
        x_ref = []
        # generate the predictive trajectory for N steps
        for j in range(N_mpc):
            q_d, qd_d = ref.get_values(current_time + j * time_step)

            # here i need to stack the q_d and qd_d
            x_ref.append(np.vstack((q_d.reshape(-1, 1), qd_d.reshape(-1, 1))))

        x_ref = np.vstack(x_ref).flatten()

        # Compute the optimal control sequence
        if constraints:
            u_star = tracker.computesolution(x_ref, x0_mpc, u_mpc, H, Ftra, T_bar, S_bar)
        else:
            u_star = tracker.computesolution(x_ref, x0_mpc, u_mpc, H, Ftra, True)
        # Return the optimal control sequence
        u_mpc += u_star[:num_joints]


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

        q_d, qd_d = ref.get_values(current_time)

        q_d_all.append(q_d)
        qd_d_all.append(qd_d)

        # time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
        print(f"Time: {current_time}")
        print(time_step)

    # save the data into .txt file
    # if constraints:
    #     txt_name1 = reff+f"/constraint_{constraints}_{c}_pos_Q_{Q_p}_{Q_v}.txt"
    #     txt_name2 = reff+f"/constraint_{constraints}_{c}_vel_Q_{Q_p}_{Q_v}.txt"
    # else:
    #     txt_name1 = reff+f"/constraint_{constraints}_pos_Q_{Q_p}_{Q_v}.txt"
    #     txt_name2 = reff+f"/constraint_{constraints}_vel_Q_{Q_p}_{Q_v}.txt"
    # output_file_path = f"report_figures/constraints/" + txt_name1
    # with open(output_file_path, 'w') as f:
    #     for q in q_mes_all:
    #         f.write(','.join(map(str, q)) + '\n')  # Write each position as a comma-separated string
    # output_file_path = f"report_figures/constraints/" + txt_name2
    # with open(output_file_path, 'w') as f:
    #     for qd in qd_mes_all:
    #         f.write(','.join(map(str, qd)) + '\n')  # Write each velocity as a comma-separated string
    # print(f"Data has been written to {output_file_path}.")
    # Plotting
    for i in range(num_joints):
        plt.figure(figsize=(10, 8))

        # Position plot for joint i
        plt.subplot(2, 1, 1)
        time_steps = range(len(q_mes_all))  # Assuming time steps correspond to the length of q_mes_all

        # Measured and desired position plot
        plt.plot(time_steps, [q[i] for q in q_mes_all], label=f'Measured Position - Joint {i + 1}', color='b')
        plt.plot(time_steps, [q[i] for q in q_d_all], label=f'Desired Position - Joint {i + 1}', linestyle='--',
                 color='g')

        # Ensure that the limits have the correct length or replicate them
        upper_pos_limit = upper_limits[i] if isinstance(upper_limits[i], list) else [upper_limits[i]] * len(
            time_steps)
        lower_pos_limit = lower_limits[i] if isinstance(lower_limits[i], list) else [lower_limits[i]] * len(
            time_steps)

        plt.plot(time_steps, upper_pos_limit, label="Upper Position Constraint", linestyle='-', color='r',
                 alpha=0.7)
        plt.plot(time_steps, lower_pos_limit, label="Lower Position Constraint", linestyle='-', color='r',
                 alpha=0.7)

        plt.title(f'Position Tracking for Joint {i + 1}')
        plt.xlabel('Time steps')
        plt.ylabel('Position')
        plt.legend()

        # Velocity plot for joint i
        plt.subplot(2, 1, 2)
        # Measured velocity
        plt.plot(time_steps, [qd[i] for qd in qd_mes_all], label=f'Measured Velocity - Joint {i + 1}', color='b')

        # Desired velocity plot if the control mode is "all"
        if control == "all":
            plt.plot(time_steps, [qd[i] for qd in qd_d_all], label=f'Desired Velocity - Joint {i + 1}',
                     linestyle='--', color='g')

        # Ensure that the limits have the correct length or replicate them
        upper_vel_limit = upper_vel_limits[i] if isinstance(upper_vel_limits[i], list) else [upper_vel_limits[
                                                                                                 i]] * len(
            time_steps)
        lower_vel_limit = lower_vel_limits[i] if isinstance(lower_vel_limits[i], list) else [lower_vel_limits[
                                                                                                 i]] * len(
            time_steps)

        plt.plot(time_steps, upper_vel_limit, label="Upper Velocity Constraint", linestyle='-', color='r',
                 alpha=0.7)
        plt.plot(time_steps, lower_vel_limit, label="Lower Velocity Constraint", linestyle='-', color='r',
                 alpha=0.7)

        plt.title(f'Velocity Tracking for Joint {i + 1}')
        plt.xlabel('Time steps')
        plt.ylabel('Velocity')
        plt.legend()
        # image_dir = "report_figures/constraints"
        # if not os.path.exists(image_dir):
        #     os.makedirs(image_dir)  # Create the directory if it doesn't exist
        # if constraints:
        #     image_path = os.path.join(image_dir,
        #                               f"joint_{i + 1}_constraint_{constraints}_{c}_ref_{reff}_control_{control}_Q_{Q_p}_{Q_v}_locaL.png")
        # else:
        #     image_path = os.path.join(image_dir,
        #                               f"joint_{i + 1}_constraint_{constraints}_ref_{reff}_control_{control}_Q_{Q_p}_{Q_v}_locaL.png")
        #
        # plt.savefig(image_path, format='png', dpi=300)  # Save the image
        # print(f"Image saved!")
        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()

        plt.close()




if __name__ == '__main__':
    main()