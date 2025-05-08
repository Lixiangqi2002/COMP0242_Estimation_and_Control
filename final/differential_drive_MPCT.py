import math
from os import terminal_size

import numpy as np
import time
import os
import matplotlib.pyplot as plt
from numpy.ma.extras import average
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, \
    CartesianDiffKin, differential_drive_controller_adjusting_bearing
from simulation_and_control import differential_drive_regulation_controller, regulation_polar_coordinates, \
    regulation_polar_coordinate_quat, wrap_angle, velocity_to_wheel_angular_velocity
import pinocchio as pin

from final.robot_localization_system import Map, RobotEstimator, FilterConfiguration, FilterConfigurationMPC
from regulator_model import RegulatorModel


# global variables
W_range = 0.5 ** 2  # Measurement noise variance (range measurements)
W_bearing = (np.pi * 0.5 / 180.0) ** 2  # bearing measurements

# WE CHANGED THE FUNCTION IN THE LAST COMMIT (1 novemeber 2024, 16:45)
def landmark_range_observations(base_position, landmarks, W_range=W_range):
    y = []
    C = []
    W = W_range
    for lm in landmarks:
        # True range measurement (with noise)
        dx = lm[0] - base_position[0]
        dy = lm[1] - base_position[1]
        range_meas = np.sqrt(dx ** 2 + dy ** 2) + np.random.normal(0, np.sqrt(W))

        y.append(range_meas)

    y = np.array(y)
    return y


def landmark_range_bearing_observations(_x_true_pos, _x_true_theta, landmarks):
    y, y_r, y_b = [], [], []

    for lm in landmarks:
        # Calculate the true range and bearing
        dx = lm[0] - _x_true_pos[0]
        dy = lm[1] - _x_true_pos[1]
        range_true = np.sqrt(dx ** 2 + dy ** 2)
        bearing_true = np.arctan2(dy, dx) - _x_true_theta

        # Add noise to the measurements
        range_meas = range_true + np.random.normal(0, np.sqrt(W_range))
        bearing_meas = bearing_true + np.random.normal(0, np.sqrt(W_bearing))

        # Normalize bearing to [-π, π]
        bearing_meas = wrap_angle(bearing_meas)

        # Append both range and bearing to y
        y.extend([range_meas, bearing_meas])
        # y_r.append(range_meas)
        # y_b.append(bearing_meas)

    # Convert y to an array with all range and bearing values
    y = np.array(y)
    # y_r = np.array(y_r)
    # y_b = np.array(y_b)
    return y


def quaternion2bearing(q_w, q_x, q_y, q_z):
    quat = pin.Quaternion(q_w, q_x, q_y, q_z)
    quat.normalize()  # Ensure the quaternion is normalized

    # Convert quaternion to rotation matrix
    rot_quat = quat.toRotationMatrix()

    # Convert rotation matrix to Euler angles (roll, pitch, yaw)
    base_euler = pin.rpy.matrixToRpy(rot_quat)  # Returns [roll, pitch, yaw]

    # Extract the yaw angle
    bearing_ = base_euler[2]

    return bearing_


def init_simulator(conf_file_name):
    """Initialize simulation and dynamic model."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)

    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]

    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    return sim, dyn_model, num_joints


def main():
    """
    Parameters
    route_num : [0,1,2,3,4]  ----  For task 4, testing on different routes
            0 -> [2,3,45] -> [0,0,0]
            1 -> [1,2,45]  -> [0,-1,0]
            2 -> [2,2,90] -> [0,0,0]
            3- > [1,3,180] -> [0, -1, 0] Difficult route, need to try multiple times
            4 -> [-0.5,-0.5,90] -> [1,1,-17]
    start_point_task2 : [-1, 0, 1, 2, 3]  ---- For task 2, testing linearization approach on different starting state
                        -1 -> not testing this function, but on task 4
                        0 -> [-4, 3, 45]
                        1 -> [1, 1, 45]
                        2 -> [1.5, 1, 45]
                        3 -> [2, 3, 45]
    terminal_cost : True/ False  ---- For terminal cost turn on
    dynamic_AB : True/ False  ---- For different linearization approach

    Time for 1 timestep
    time duration for terminal cost= True: 0.000648
    time duration for terminal cost= False: 0.004392
    """
    """ FLYING PARAMETERS """
    route_num = 0
    start_point_task2 = -1
    if start_point_task2 == -1:
        conf_file_name = f"robotnik_route_num_{route_num}.json"  # Configuration file for the robot
    else:
        conf_file_name = f"robotnik_task2_start_point_{start_point_task2}.json"
    sim, dyn_model, num_joints = init_simulator(conf_file_name)
    terminal_cost = True
    dynamic_AB = True
    if dynamic_AB:
        if terminal_cost:
            N_mpc = 2
            y_coeff = 0.105
            Qcoeff = [y_coeff, 3. * y_coeff, y_coeff * 3.1]
            Rcoeff = [0.0045, 0.001]
            if route_num == 0:
                # init_link_base_orientation [[0, 0, 0.3827, 0.9239]] [2,3,45] -> [0,0,0]
                ######### car flying but using N=2 same QR as no terminal cost ###########################
                # N_mpc = 2
                # y_coeff = 0.45
                # Qcoeff = [y_coeff, 1.65 * y_coeff, y_coeff * 3.9]
                # Rcoeff = [0.005, 0.001]

                ######### car no flying but using N=2 different QR as no terminal cost ###########################
                toler = 0.05
                goal_state = np.array([0, 0, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 1:
                # init_link_base_orientation [[0, 0, 0.3827, 0.9239]] [1,2,45]  -> [0,-1,0]
                toler = 0.1
                goal_state = np.array([0, -1, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 2:
                # init_link_base_orientation [[0, 0, 1.414, 1.414]] [2,2,90] -> [0,0,0]
                N_mpc = 2
                toler = 0.1
                goal_state = np.array([0, 0, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 3:
                # init_link_base_orientation [[0, 0, 1, 0]] [1,3,180] -> [0, -1, 0]
                toler = 0.1
                goal_state = np.array([0, -1, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 4:
                # init_link_base_orientation[[0, 0, 1.414, 1.414]] [-0.5,-0.5,90] -> [1,1,-17]
                toler = 0.1
                goal_state = np.array([1, 1, -0.3])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
        else:
            N_mpc = 40
            y_coeff = 0.45
            Qcoeff = [y_coeff, 1.65 * y_coeff, y_coeff * 3.9]
            Rcoeff = [0.005, 0.001]
            if route_num == 0:
                # init_link_base_orientation [[0, 0, 0.3827, 0.9239]] [2,3,45] -> [0,0,0]
                toler = 0.05
                goal_state = np.array([0, 0, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 1:
                # init_link_base_orientation [[0, 0, 0.3827, 0.9239]] [1,2,45]  -> [0,-1,0]
                toler = 0.1
                goal_state = np.array([0, -1, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 2:
                # init_link_base_orientation [[0, 0, 1.414, 1.414]] [2,2,90] -> [0,0,0]
                toler = 0.1
                goal_state = np.array([0, 0, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 3:
                # init_link_base_orientation [[0, 0, 1, 0]] [1,3,180] -> [0, -1, 0]
                toler = 0.1
                goal_state = np.array([0, -1, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 4:
                # init_link_base_orientation[[0, 0, 1.414, 1.414]] [-0.5,-0.5,90] -> [1,1,-17]
                toler = 0.1
                goal_state = np.array([1, 1, -0.3])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
    else:  # ONLY FOR ROUTE 0 terminal_cost = False
        if terminal_cost == False:
            N_mpc = 40
            y_coeff = 0.45  # 0.6
            Qcoeff = [y_coeff, 1.65 * y_coeff, y_coeff * 3.85]
            Rcoeff = [0.005, 0.001]
            toler = 0.05
            goal_state = np.array([0, 0, 0])  # Goal position (x, y) and orientation θ
            goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity

    """ NO FLYING PARAMETERS """
    """
    if dynamic_AB:
        if terminal_cost:
            N_mpc = 5
            if route_num == 0:
                # init_link_base_orientation [2,3,0] [[0, 0, 0.3827, 0.9239]]
                y_coeff = 0.12
                Qcoeff = [0.5 * y_coeff, 3 * y_coeff, y_coeff * 2.15]
                Rcoeff = [0.0045, 0.001]
                toler = 0.05
                goal_state = np.array([0, 0, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 1:
                # init_link_base_orientation [1,2,0] [[0, 0, 0.3827, 0.9239]]
                Qcoeff = [0.1, 0.07, 0.231]
                Rcoeff = [0.0045, 0.001]
                toler = 0.1
                goal_state = np.array([0, 0, -0.77])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 2:
                # init_link_base_orientation [2,2,0] [[0, 0, 1.414, 1.414]]
                y_coeff = 0.06
                Qcoeff = [y_coeff, 3.05 * y_coeff, y_coeff * 3.2]
                Rcoeff = [0.0045, 0.001]
                toler = 0.1
                goal_state = np.array([0, 0, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 3:
                # init_link_base_orientation [[0, 0, 1,0]]  [1,3,180]
                y_coeff = 0.06
                Qcoeff = [1.3 * y_coeff, 2.5 * y_coeff, y_coeff * 3.85]
                Rcoeff = [0.0045, 0.001]
                toler = 0.1
                goal_state = np.array([0, -1, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 4:
                # init_link_base_orientation  [-0.5,-0.5,0]  [[0, 0, 1.414, 1.414]]
                y_coeff = 0.06
                Qcoeff = [1.5 * y_coeff, 1.9 * y_coeff, y_coeff * 5]
                Rcoeff = [0.0045, 0.001]
                toler = 0.1
                goal_state = np.array([1, 1, -0.3])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
        else:
            N_mpc = 40
            if route_num == 0:
                # init_link_base_orientation [[0, 0, 0.3827, 0.9239]]
                y_coeff = 0.45
                Qcoeff = [y_coeff, 1.65 * y_coeff, y_coeff * 3.9]
                Rcoeff = [0.005, 0.001]
                toler = 0.05
                goal_state = np.array([0, 0, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 1:
                # init_link_base_orientation [[0, 0, 0.3827, 0.9239]]
                y_coeff = 0.45
                Qcoeff = [y_coeff, 1.75 * y_coeff, y_coeff * 3.85]
                Rcoeff = [0.005, 0.001]
                toler = 0.1
                goal_state = np.array([0, 0, -0.77])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 2:
                # init_link_base_orientation [[0, 0, 1.414, 1.414]]
                y_coeff = 0.45
                Qcoeff = [y_coeff, 2. * y_coeff, y_coeff * 2.4]
                Rcoeff = [0.005, 0.001]
                toler = 0.1
                goal_state = np.array([0, 0, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 3:
                # init_link_base_orientation [[0, 0, 1,0]]  [1,3,180]
                y_coeff = 0.45
                Qcoeff = [y_coeff, 2.1 * y_coeff, y_coeff * 2.3]
                Rcoeff = [0.005, 0.001]
                toler = 0.1
                goal_state = np.array([0, -1, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 4:
                # init_link_base_orientation [[0, 0, 1.414, 1.414]]
                y_coeff = 0.45
                Qcoeff = [y_coeff, 2.2 * y_coeff, y_coeff * 3.4]
                Rcoeff = [0.005, 0.001]
                toler = 0.1
                goal_state = np.array([1, 1, -0.3])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
    else: # ONLY FOR ROUTE 0
        if terminal_cost == False:
            N_mpc = 40
            y_coeff = 0.45  # 0.6
            Qcoeff = [y_coeff, 1.65 * y_coeff, y_coeff * 3.85]
            Rcoeff = [0.005, 0.001]
            toler = 0.05
            goal_state = np.array([0, 0, 0])  # Goal position (x, y) and orientation θ
            goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity
        else:
            # init_link_base_orientation [2,3,0] [[0, 0, 0.3827, 0.9239]]
            N_mpc = 5
            y_coeff = 0.12
            Qcoeff = [0.5 * y_coeff, 3 * y_coeff, y_coeff * 2.15]
            Rcoeff = [0.0045, 0.001]
            toler = 0.05
            goal_state = np.array([0, 0, 0])  # Goal position (x, y) and orientation θ
            goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
    """
    x_his = []
    y_his = []
    theta_his = []
    x_his_real = []
    y_his_real = []
    theta_his_real = []
    desire_state_all = []
    time_durations = []

    current_time = 0

    # Initialize data storage
    base_pos_all, base_bearing_all = [], []  #

    # initializing MPC
    # Define the matrices
    num_states = 3
    num_controls = 2

    # Measuring all the state
    C = np.eye(num_states)


    # adjusting floor friction
    floor_friction = 100
    sim.SetFloorFriction(floor_friction)
    # getting time step
    time_step = sim.GetTimeStep()
    # Initialize the regulator model
    regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states)
    regulator.setCostMatrices(Qcoeff, Rcoeff)
    if not dynamic_AB:
        regulator.updateSystemMatrices(sim, goal_state, goal_action)
        if terminal_cost:
            regulator.dlqr()
    ##### robot parameters ########
    wheel_radius = 0.11
    wheel_base_width = 0.46

    ##### MPC control action #######
    v_linear = 0
    v_angular = 0
    cmd = MotorCommands()  # Initialize command structure for motors
    angular_wheels_velocity_cmd = np.array([0.0, 0.0, 0.0, 0.0])
    init_interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
    cmd.SetControlCmd(angular_wheels_velocity_cmd, init_interface_all_wheels)

    while True:
        start_time = time.time()

        # Get the measurements from the simulator ###########################################
        # measurements of the robot without noise (just for comparison purpose) #############
        base_pos_no_noise = sim.bot[0].base_position
        base_ori_no_noise = sim.bot[0].base_orientation
        base_bearing_no_noise_ = quaternion2bearing(base_ori_no_noise[3], base_ori_no_noise[0], base_ori_no_noise[1],
                                                    base_ori_no_noise[2])
        base_bearing_no_noise_ = wrap_angle(base_bearing_no_noise_)
        base_lin_vel_no_noise = sim.bot[0].base_lin_vel
        base_ang_vel_no_noise = sim.bot[0].base_ang_vel

        # # True state propagation (with process noise)
        # ##### advance simulation ##################################################################
        sim.Step(cmd, "torque")
        # Measurements of the current state (real measurements with noise) ##################################################################
        base_pos = sim.GetBasePosition()
        base_ori = sim.GetBaseOrientation()
        base_bearing_ = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        print(np.array([base_pos[0], base_pos[1], base_bearing_]))
        base_bearing_ = wrap_angle(base_bearing_)

        # Figure out what the controller should do next
        # MPC section/ low level controller section ##################################################################
        # Compute the matrices needed for MPC optimization
        # TODO here you want to update the matrices A and B at each time step if you want to linearize around the current points
        if dynamic_AB:
            # Update the linearized point based on the current state and control
            state_x_for_linear = np.array(
                [base_pos[0] - goal_state[0], base_pos[1] - goal_state[1], base_bearing_ - goal_state[2]])
            cur_u_for_linear = np.array([v_linear - goal_action[0], v_angular - goal_action[1]])

            # Update system matrices for the current state
            regulator.updateSystemMatrices(sim, state_x_for_linear, cur_u_for_linear)
            if terminal_cost:
                flag = regulator.dlqr()
                if flag == False:
                    break

        S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
        H, F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)

        x0_mpc = np.vstack((base_pos[0], base_pos[1], base_bearing_))
        x0_mpc = x0_mpc.flatten()
        # Compute the optimal control sequence
        H_inv = np.linalg.inv(H)
        u_mpc = -H_inv @ F @ (x0_mpc - goal_state)
        # Return the optimal control sequence
        u_mpc = u_mpc[0:num_controls]
        v_linear, v_angular = u_mpc[0], u_mpc[1]

        # Prepare control command to send to the low level controller
        left_wheel_velocity, right_wheel_velocity = velocity_to_wheel_angular_velocity(u_mpc[0], u_mpc[1],
                                                                                       wheel_base_width, wheel_radius)
        angular_wheels_velocity_cmd = np.array(
            [right_wheel_velocity, left_wheel_velocity, left_wheel_velocity, right_wheel_velocity])
        interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
        cmd.SetControlCmd(angular_wheels_velocity_cmd, interface_all_wheels)

        # Store data for plotting if necessary
        base_pos_all.append(base_pos)
        base_bearing_all.append(base_bearing_)
        x_his_real.append(base_pos_no_noise[0])
        y_his_real.append(base_pos_no_noise[1])
        theta_his_real.append(base_bearing_no_noise_)

        x_his.append(base_pos[0])
        y_his.append(base_pos[1])
        theta_his.append(base_bearing_)
        desire_state_all.append(goal_state)

        # Update current time
        current_time += time_step
        end_time = time.time()
        timestep_duration = end_time - start_time
        time_durations.append(timestep_duration)
        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
    average_duration = sum(time_durations) / len(time_durations)
    print(f"Average timestep duration: {average_duration:.6f} seconds")
    # Plotting
    #########################################################################################
    # 1. Plot of final x, y, trajectory and theta
    base_pos_all = np.array(base_pos_all)
    # x_est_history = np.array(x_est_history)
    # Sigma_est_history = np.array(Sigma_est_history)
    desire_state_all = np.array(desire_state_all)
    x_his = np.array(x_his)
    y_his = np.array(y_his)
    theta_his = np.array(theta_his)
    # Get the first, final position and orientation (theta)
    first_pos = base_pos_all[0]
    first_x, first_y = first_pos[0], first_pos[1]
    first_theta = base_bearing_all[2]
    final_pos = base_pos_all[-1]
    final_x, final_y = final_pos[0], final_pos[1]
    final_theta = base_bearing_all[-1]  # Get the last bearing value
    plt.figure(figsize=(10, 6))
    plt.plot(base_pos_all[:, 0], base_pos_all[:, 1], label="Trajectory", color='blue')

    plt.scatter(goal_state[0], goal_state[1], color="red", marker="x", label="target point")
    arrow_length = 0.5  # Length of the arrow to indicate direction
    plt.arrow(goal_state[0], goal_state[1], arrow_length * np.cos(goal_state[2]), arrow_length * np.sin(goal_state[2]),
              label=f'theta = 0', head_width=0.3,
              head_length=0.3, fc='red', ec='red')
    plt.scatter(base_pos_all[0, 0], base_pos_all[0, 1], color="green", marker="o", label="Start Point")
    plt.arrow(first_x, first_y, arrow_length * np.cos(first_theta), arrow_length * np.sin(first_theta),
              label=f'theta = {math.degrees(first_theta)}', head_width=0.3, head_length=0.3, fc='green', ec='green')
    plt.scatter(final_x, final_y, color="orange", marker="o", label="End Point")
    arrow_dx = arrow_length * np.cos(final_theta)
    arrow_dy = arrow_length * np.sin(final_theta)
    final_theta = math.degrees(final_theta)
    plt.arrow(final_x, final_y, arrow_dx, arrow_dy, label=f'theta = {final_theta}', head_width=0.3, head_length=0.3,
              fc='orange', ec='orange')
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title("Robot Trajectory with Start and End Points")
    plt.legend()
    plt.grid()
    plt.axis('equal')  # Ensure equal scaling of the axes
    # plt.savefig(f"task2/dynamic_AB_{dynamic_AB}_N_{N_mpc}_Q_{Qcoeff}_R_{Rcoeff}.png")
    # plt.savefig(f'task4/MPCT_FINAL/route_{route_num}_trajectory_terminal_{terminal_cost}.png')
    plt.show()

    #########################################################################################
    # 2. plot control reponse
    plt.figure()
    states = ["x_response", "y_response", "theta_response"]
    his = [x_his, y_his, theta_his]
    his_real = [x_his_real, y_his_real, theta_his_real]
    for s in range(len(states)):
        desire_state = desire_state_all[:, s]
        cur_his = his[s]
        cur_his_real = his_real[s]
        if cur_his[0] > desire_state[0]:
            plt.gca().invert_yaxis()  # Invert y-axis
            invert = True
        else:
            invert = False
        # x_his remains within x range of desired state and stays within that bound
        tolerance = toler * (np.max(cur_his) - np.min(desire_state))
        try:
            # first crossing exists ---> rising time and overshoot
            if invert:
                rising_time_index = next(i for i in range(1, len(cur_his)) if
                                         cur_his[i - 1] > desire_state[i - 1] and cur_his[i] <= desire_state[i])
                overshoot_value = np.min(cur_his)
                overshoot_index = np.argmin(cur_his)
            else:
                rising_time_index = next(i for i in range(1, len(cur_his)) if
                                         cur_his[i - 1] < desire_state[i - 1] and cur_his[i] >= desire_state[i])
                overshoot_value = np.max(cur_his)
                overshoot_index = np.argmax(cur_his)
            rising_time = rising_time_index / 1000
            # Find overshoot (highest point in x_his)
            no_crossing_peak = 0
            try:
                # second crossing exists ----> effect on the settling time
                if invert:
                    second_crossing_index = next(i for i in range(rising_time_index + 1, len(cur_his)) if
                                                 # abs(cur_his[i]-cur_his[i-10])>=0.1 and
                                                 cur_his[i - 1] < desire_state[i - 1] and cur_his[i] >= desire_state[i])
                else:
                    second_crossing_index = next(i for i in range(rising_time_index + 1, len(cur_his)) if
                                                 # abs(cur_his[i]-cur_his[i-10])>=0.1 and
                                                 cur_his[i - 1] >= desire_state[i - 1] and cur_his[i] < desire_state[i])
                subsequent_index = second_crossing_index + max(np.argmax(cur_his[second_crossing_index:]),
                                                               np.argmin(cur_his[second_crossing_index:]))
            except StopIteration or NameError:
                # no second crossing exists
                second_crossing_index = 0
                subsequent_index = 0
        except StopIteration or ValueError:
            # no first crossing exists
            no_crossing_peak_start = next((i for i, x in enumerate(cur_his) if abs(x - desire_state[i]) <= tolerance),
                                          len(cur_his) - 1)
            try:
                no_crossing_peak = np.argmin(cur_his[no_crossing_peak_start:-1])
            except ValueError:
                no_crossing_peak = np.argmin(cur_his[:-1])
            rising_time_index = -0.5
            overshoot_value = -0.5
            overshoot_index = 0
            second_crossing_index = 0
            subsequent_index = 0

        # Determine settling time
        distance = cur_his_real[-1] - desire_state[-1]
        settling_index = len(cur_his_real) - 1
        for i in reversed(range(len(cur_his_real))):
            cur_dis = cur_his_real[i] - desire_state[i]
            if abs(cur_dis - distance) > 0.01 and i > rising_time_index:
                settling_index = i + 1
                break
        print(settling_index)
        # Plotting
        plt.plot(cur_his, label='Response', linewidth=0.7)
        plt.plot(desire_state, label='Desired State', linestyle='dashed', color='red')

        plt.hlines(desire_state[-1], 0, len(cur_his) - 1, color='gray', linestyle=':')
        if settling_index != len(cur_his) - 1:
            settling_time = settling_index / 1000  # Assuming each point is a time step for simplicity
            # Calculate steady state error (distance from the avg_after_settling to desired state)
            avg_after_settling = average(cur_his[settling_index:-1])
            steady_state_error = abs(avg_after_settling - desire_state[-1])
            plt.axvline(settling_index, color='green', linestyle='--', label=f'Settling Time: {settling_time}')
            plt.text(len(cur_his) - 1, desire_state[-1], f"Steady State Error: {steady_state_error:.2f}", ha='right',
                     color='blue')
        if rising_time_index != -0.5:
            plt.axvline(rising_time_index, color='purple', linestyle='--', label=f'Rising Time: {rising_time}')
        if overshoot_value != -0.5:
            plt.plot(overshoot_index, overshoot_value, 'ro')  # Mark the overshoot point
            plt.text(overshoot_index, overshoot_value, f"Overshoot: {overshoot_value:.2f}", ha='center', color='red')
        plt.title("Control Response for " + states[s])
        plt.legend()
        plt.xlabel("Time Steps")
        plt.ylabel("Position")
        # plt.savefig(f"task2/dynamic_AB_{dynamic_AB}_N_{N_mpc}_Q_{Qcoeff}_R_{Rcoeff}_{states[s]}.png")
        # plt.savefig(f'task4/MPCT_FINAL/route_{route_num}_{states[s]}_terminal_{terminal_cost}.png')
        plt.show()


if __name__ == '__main__':
    main()