import math

import numpy as np
import time
import os
import matplotlib.pyplot as plt
from numpy.ma.extras import average
from simulation_and_control import pb, MotorCommands, PinWrapper
from simulation_and_control import wrap_angle,velocity_to_wheel_angular_velocity
import pinocchio as pin

from final.robot_localization_system import Map, RobotEstimator, FilterConfiguration, FilterConfigurationMPC
from regulator_model import RegulatorModel

# global variables
W_range = 0.5 ** 2  # Measurement noise variance (range measurements)
W_bearing = (np.pi * 0.5/ 180.0) ** 2 # bearing measurements


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
    y,y_r, y_b = [], [], []

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

    # Convert y to an array with all range and bearing values
    y = np.array(y)
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
    robot_noise : [-1, 0, 0.01, 0.05]  ---- For task 3, testing EKF on different robot noise
                        -1 -> not testing this function, but on task 4
                        0 -> no robot noise, V=0.5
                        0.01 -> base_pos_cov = base_ori_cov = 0.01, V=0.5
                        0.05 -> base_pos_cov = base_ori_cov = 0.05, V=1.6
    terminal_cost : True/ False  ---- For terminal cost turn on
    dynamic_AB : True/ False  ---- For different linearization approach

    Time for 1 timestep
    time duration for terminal cost= True: 0.0013
    time duration for terminal cost= False: 0.0046
    """
    # Define the goal state and control for fixed linear
    route_num = 0
    robot_noise = -1
    terminal_cost = True
    dynamic_AB = True
    # EKF parameters
    if robot_noise==0.05:
        V = 1.6
    else:
        V = 0.5
    # Configuration for the simulation
    if robot_noise==-1:
        conf_file_name = f"robotnik_route_num_{route_num}.json"  # Configuration file for the robot
    else:
        conf_file_name = f"robotnik_task3_robot_noise_{robot_noise}.json"
    print(conf_file_name)
    sim, dyn_model, num_joints = init_simulator(conf_file_name)
    """ FLYING PARAMETERS """
    if dynamic_AB:
        if terminal_cost:
            N_mpc = 2
            y_coeff = 0.105
            Qcoeff = [y_coeff, 3. * y_coeff, y_coeff * 3.1]
            Rcoeff = [0.0045, 0.001]
            if route_num == 0:
                # init_link_base_orientation [[0, 0, 0.3827, 0.9239]] [2,3,45] -> [0,0,0]
                ######### car flying but using same QR as no terminal cost ###########################
                # N_mpc = 2
                # x0 = [2, 3, 0.7854]
                # y_coeff = 0.45
                # Qcoeff = [y_coeff, 1.65 * y_coeff, y_coeff * 3.9]
                # Rcoeff = [0.005, 0.001]
                # toler = 0.05
                # goal_state = np.array([0, 0, 0])  # Goal position (x, y) and orientation θ
                # goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)

                ######### car no flying but using different QR as no terminal cost ###########################
                x0 = [2, 3, 0.7854]
                toler = 0.05
                goal_state = np.array([0, 0, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 1:
                # init_link_base_orientation [[0, 0, 0.3827, 0.9239]] [1,2,45]  -> [0,-1,0]
                x0 = [1, 2, 0.7854]
                toler = 0.1
                goal_state = np.array([0, -1, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 2:
                # init_link_base_orientation [[0, 0, 1.414, 1.414]] [2,2,90] -> [0,0,0]
                x0 = [2, 2, 1.57]
                toler = 0.1
                goal_state = np.array([0, 0, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 3:
                # init_link_base_orientation [[0, 0, 1, 0]] [1,3,180] -> [0, -1, 0]
                toler = 0.1
                x0 = [1, 3, 3.14]
                goal_state = np.array([0, -1, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 4:
                # init_link_base_orientation[[0, 0, 1.414, 1.414]] [-0.5,-0.5,90] -> [1,1,-17]
                N_mpc = 2
                x0 = [-0.5, -0.5, 1.57]
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
                x0 = [2, 3, 0.7854]
                toler = 0.05
                goal_state = np.array([0, 0, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 1:
                # init_link_base_orientation [[0, 0, 0.3827, 0.9239]] [1,2,45]  -> [0,-1,0]
                x0 = [1, 2, 0.7854]
                toler = 0.1
                goal_state = np.array([0, -1, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 2:
                # init_link_base_orientation [[0, 0, 1.414, 1.414]] [2,2,90] -> [0,0,0]
                x0 = [2, 2, 1.57]
                toler = 0.1
                goal_state = np.array([0, 0, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 3:
                # init_link_base_orientation [[0, 0, 1, 0]] [1,3,180] -> [0, -1, 0]
                x0 = [1, 3, 3.14]
                toler = 0.1
                goal_state = np.array([0, -1, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 4:
                # init_link_base_orientation[[0, 0, 1.414, 1.414]] [-0.5,-0.5,90] -> [1,1,-17]
                x0 = [-0.5, -0.5, 1.57]
                toler = 0.1
                goal_state = np.array([1, 1, -0.3])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
    else:
        if route_num == 0:
            terminal_cost = False
            N_mpc = 40
            y_coeff = 0.45  # 0.6
            Qcoeff = [y_coeff, 1.65 * y_coeff, y_coeff * 3.85]
            Rcoeff = [0.005, 0.001]
            toler = 0.05
            goal_state = np.array([0, 0, 0])  # Goal position (x, y) and orientation θ
            goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)

    """ NO FLYING PARAMETERS """
    """
    if dynamic_AB:
        if terminal_cost:
            N_mpc = 5
            if route_num == 0:
                # init_link_base_orientation [[0, 0, 0.3827, 0.9239]]
                x0 = [2,3,0.7854]
                y_coeff = 0.12
                Qcoeff = [0.5 * y_coeff, 3 * y_coeff, y_coeff * 2.15]
                Rcoeff = [0.0045, 0.001]
                toler = 0.05
                goal_state = np.array([0, 0, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 1:
                # init_link_base_orientation [[0, 0, 0.3827, 0.9239]]
                x0 = [1,2,0.7854]
                Qcoeff = [0.1, 0.07, 0.231]
                Rcoeff = [0.0045, 0.001]
                toler = 0.1
                goal_state = np.array([0, 0, -0.78])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 2:
                # init_link_base_orientation [[0, 0, 1.414, 1.414]]
                y_coeff = 0.06
                x0 = [2, 2, 1.57]
                Qcoeff = [y_coeff, 3.05 * y_coeff, y_coeff * 3.2]
                Rcoeff = [0.0045, 0.001]
                toler = 0.1
                goal_state = np.array([0, 0, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 3:
                # init_link_base_orientation [[0, 0, 1.414, 1.414]]
                y_coeff = 0.06
                x0 = [1, 3, 0]
                Qcoeff = [1.3 * y_coeff, 2.5 * y_coeff, y_coeff * 3.85]
                Rcoeff = [0.0045, 0.001]
                toler = 0.1
                goal_state = np.array([0, -1,0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 4:
                # init_link_base_orientation[[0, 0, 1.414, 1.414]]
                y_coeff = 0.06
                x0 =  [-0.5,-0.5, 1.57]
                Qcoeff = [1.5 * y_coeff, 1.9 * y_coeff, y_coeff * 5]
                Rcoeff = [0.0045, 0.001]
                toler = 0.1
                goal_state = np.array([1, 1, 0.3])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
        else:
            N_mpc = 40
            if route_num == 0:
                # init_link_base_orientation [[0, 0, 0.3827, 0.9239]]
                y_coeff = 0.45
                x0 = [2,3,0.7854]
                Qcoeff = [y_coeff, 1.65 * y_coeff, y_coeff * 3.9]
                Rcoeff = [0.005, 0.001]
                toler = 0.05
                goal_state = np.array([0, 0, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 1:
                # init_link_base_orientation [[0, 0, 0.3827, 0.9239]]
                y_coeff = 0.45
                x0 = [1,2,0.7854]
                Qcoeff = [y_coeff, 1.8 * y_coeff, y_coeff * 3.85]
                Rcoeff = [0.005, 0.001]
                toler = 0.1
                goal_state = np.array([0, 0, -0.77])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 2:
                # init_link_base_orientation [[0, 0, 1.414, 1.414]]
                y_coeff = 0.45
                x0 = [2, 2, 1.57]
                Qcoeff = [y_coeff, 2. * y_coeff, y_coeff * 2.4]
                Rcoeff = [0.005, 0.001]
                toler = 0.1
                goal_state = np.array([0, 0, 0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 3:
                # init_link_base_orientation [[0, 0, 1.414, 1.414]]
                x0 = [1, 3, 0]
                y_coeff = 0.45
                Qcoeff = [y_coeff, 2.1 * y_coeff, y_coeff * 2.3]
                Rcoeff = [0.005, 0.001]
                toler = 0.1
                goal_state = np.array([0, -1,0])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
            elif route_num == 4:
                # init_link_base_orientation [[0, 0, 1.414, 1.414]]
                x0 = [-0.5,-0.5, 1.57]
                y_coeff = 0.45
                Qcoeff = [y_coeff, 2.2 * y_coeff, y_coeff * 3.4]
                Rcoeff = [0.005, 0.001]
                toler = 0.1
                goal_state = np.array([1, 1,0.3 ])  # Goal position (x, y) and orientation θ
                goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
    else:
        if route_num == 0:
            terminal_cost = False
            N_mpc = 40
            y_coeff = 0.45 # 0.6
            Qcoeff = [y_coeff, 1.65*y_coeff, y_coeff*3.85]
            Rcoeff = [0.005,0.001]
            toler = 0.05
            goal_state = np.array([0, 0, 0])  # Goal position (x, y) and orientation θ
            goal_action = np.array([0.0001, 0.])  # No movement (linear velocity, angular velocity)
    """

    """
    Construct map and estimator for EKF
    """
    map = Map()
    landmarks = map.landmarks
    landmarks_number = map.landmarks.shape[0]
    print(landmarks_number)
    x_est_history = []
    Sigma_est_history = []
    x_his=[]
    y_his=[]
    theta_his=[]
    x_his_real = []
    y_his_real = []
    time_durations = []
    theta_his_real = []
    desire_state_all = []
    x_true_history = []
    # adjusting floor friction
    floor_friction = 100
    sim.SetFloorFriction(floor_friction)
    # getting time step
    time_step = sim.GetTimeStep()
    current_time = 0
    # Initialize data storage
    base_pos_all, base_bearing_all = [], []#

    # initializing MPC
    # Define the matrices
    num_states = 3
    num_controls = 2
    # Measuring all the state
    C = np.eye(num_states)


    # Initialize the estimator
    filter_config = FilterConfigurationMPC(x0,V)
    estimator = RobotEstimator(filter_config, map)
    estimator.start()
    x_est, Sigma_est = estimator.estimate()
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
    time_init = 0

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

        # True state propagation (with process noise)
        ##### advance simulation ##################################################################
        sim.Step(cmd, "torque")
        time_step = 0.001
        # Kalman filter prediction
        estimator.set_control_input(np.array([v_linear, v_angular]))
        estimator.predict_to(time_init)
        time_init += time_step
        # Measurements of the current state (real measurements with noise) ##################################################################
        base_pos = sim.GetBasePosition()
        base_ori = sim.GetBaseOrientation()
        base_bearing_ = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        print(np.array([base_pos[0], base_pos[1], base_bearing_]))
        base_bearing_ = wrap_angle(base_bearing_)
        # y = landmark_range_observations(base_pos, landmarks)
        y = landmark_range_bearing_observations(base_pos, base_bearing_, landmarks)
        # Update the filter with the latest observations
        innovation, innovation_covariance = estimator.update_from_landmark_range_bearing_observations(y)
        # Get the current state estimate
        x_est, Sigma_est = estimator.estimate()
        print(x_est)
        nis = estimator.calculate_nis(innovation, innovation_covariance)
        estimator.nis_history.append(nis)

        # Figure out what the controller should do next
        # MPC section/ low level controller section ##################################################################
        # Compute the matrices needed for MPC optimization
        # TODO here you want to update the matrices A and B at each time step if you want to linearize around the current points
        if dynamic_AB:
            # Update the linearized point based on the current state and control
            state_x_for_linear = np.array([x_est[0]-goal_state[0], x_est[1]-goal_state[1], x_est[2]-goal_state[2]])
            cur_u_for_linear = np.array([v_linear-goal_action[0], v_angular-goal_action[1]])

            # Update system matrices for the current state
            regulator.updateSystemMatrices(sim, state_x_for_linear, cur_u_for_linear)
            if terminal_cost:
                flag = regulator.dlqr()
                if flag == False:
                    break

        S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
        H,F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)
        x0_mpc = np.vstack((x_est[0], x_est[1], x_est[2]))
        x0_mpc = x0_mpc.flatten()
        # Compute the optimal control sequence
        H_inv = np.linalg.inv(H)
        u_mpc = -H_inv @ F @ (x0_mpc - goal_state)
        # Return the optimal control sequence
        u_mpc = u_mpc[0:num_controls]
        v_linear, v_angular = u_mpc[0], u_mpc[1]

        # Prepare control command to send to the low level controller
        left_wheel_velocity,right_wheel_velocity = velocity_to_wheel_angular_velocity(u_mpc[0],u_mpc[1], wheel_base_width, wheel_radius)
        angular_wheels_velocity_cmd = np.array([right_wheel_velocity, left_wheel_velocity, left_wheel_velocity, right_wheel_velocity])
        interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
        cmd.SetControlCmd(angular_wheels_velocity_cmd, interface_all_wheels)

        # Store data for plotting if necessary
        base_pos_all.append(base_pos)
        base_bearing_all.append(base_bearing_)
        x_est_history.append(x_est)
        x_true_history.append(np.array([base_pos_no_noise[0], base_pos_no_noise[1], base_bearing_no_noise_]))
        Sigma_est_history.append(np.diagonal(Sigma_est))
        x_his.append(base_pos[0])
        y_his.append(base_pos[1])
        theta_his.append(base_bearing_)
        desire_state_all.append(goal_state)
        x_his_real.append(base_pos_no_noise[0])
        y_his_real.append(base_pos_no_noise[1])
        theta_his_real.append(base_bearing_no_noise_)
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
    x_est_history = np.array(x_est_history)
    Sigma_est_history = np.array(Sigma_est_history)
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
    plt.plot(x_est_history[:, 0], x_est_history[:, 1], label='Estimated Path', color="purple", linestyle="--")
    plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='*', color='red', label='Landmarks')
    plt.scatter(goal_state[0], goal_state[1], color="red", marker="x", label="target point")
    arrow_length = 0.5  # Length of the arrow to indicate direction
    plt.arrow(goal_state[0], goal_state[1], arrow_length * np.cos(goal_state[2]), arrow_length * np.sin(goal_state[2]), label=f'theta = 0', head_width=0.3,
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
    t3_fig_name = f"robot_noise_{robot_noise}_V_{V}"
    # plt.savefig(f"task3/{t3_fig_name}.png")
    # plt.savefig(f'task4/MPCK_FINAL/route_{route_num}_trajectory_terminal_{terminal_cost}_MPCK.png')
    plt.show()

    #########################################################################################
    # 2. Plot the 2 standard deviation and error history for each state.
    estimation_error_x = x_est_history[:,0] - base_pos_all[:,0]
    estimation_error_y = x_est_history[:,1] - base_pos_all[:,1]
    estimation_error_theta = x_est_history[:,2] - base_bearing_all
    estimation_error_theta = wrap_angle(estimation_error_theta)

    plt.figure()
    two_sigma = 2 * np.sqrt(Sigma_est_history[:, 0])
    plt.plot(estimation_error_x, label=' estimation error for x', linewidth=0.7)
    # plt.plot(base_pos_all[:,0], label=' real trajectory for x', linewidth=0.7)
    plt.axhline(y=0.1, label=' 10cm ', color='green', linestyle='--', linewidth=3)
    plt.axhline(y=-0.1, label=' -10cm ', color='green', linestyle='--', linewidth=3)
    plt.plot(two_sigma, label=' 2σ ', linestyle='dashed', color='red')
    plt.plot(-two_sigma, label=' -2σ ', linestyle='dashed', color='red')
    plt.title("X")
    plt.legend()
    # plt.savefig(f"task3/{t3_fig_name}_x.png")
    # plt.savefig(f"task4/MPCK_FINAL/route_{route_num}_x_terminal_{terminal_cost}_MPCK.png")
    plt.show()

    plt.figure()
    two_sigma = 2 * np.sqrt(Sigma_est_history[:, 1])
    plt.plot(estimation_error_y, label=' estimation error for y', linewidth=0.7)
    # plt.plot(base_pos_all[:,1], label=' real trajectory for y', linewidth=0.7)
    plt.axhline(y=0.1, label=' 10cm ', color='green', linestyle='--', linewidth=3)
    plt.axhline(y=-0.1, label=' -10cm ', color='green', linestyle='--', linewidth=3)
    plt.plot(two_sigma, label=' 2σ ', linestyle='dashed', color='red')
    plt.plot(-two_sigma, label=' -2σ ', linestyle='dashed', color='red')
    plt.title("Y")
    plt.legend()
    # plt.savefig(f"task3/{t3_fig_name}_y.png")
    # plt.savefig(f"task4/MPCK_FINAL/route_{route_num}_y_terminal_{terminal_cost}_MPCK.png")
    plt.show()

    plt.figure()
    two_sigma = 2 * np.sqrt(Sigma_est_history[:, 2])
    plt.plot(estimation_error_theta, label=' estimation error for theta', linewidth=0.7)
    plt.axhline(y=(np.pi * 1/ 180.0), label=' 1 degree ', color='green', linestyle='--', linewidth=3)
    plt.axhline(y=-(np.pi * 1/ 180.0), label=' -1 degree ', color='green', linestyle='--', linewidth=3)
    plt.plot(two_sigma, label=' 2σ ', linestyle='dashed', color='red')
    plt.plot(-two_sigma, label=' -2σ ', linestyle='dashed', color='red')
    plt.title("theta")
    plt.legend()
    # plt.savefig(f"task3/{t3_fig_name}_theta.png")
    # plt.savefig(f"task4/MPCK_FINAL/route_{route_num}_theta_terminal_{terminal_cost}_MPCK.png")
    plt.show()

    #########################################################################################
    # 3. plot NIS
    plt.figure()
    nis_history = np.array(estimator.nis_history)
    plt.plot(nis_history, label='NIS')
    # The expected value of the NIS will be approximately equal to the dimension of the observation vector.
    # each landmark has two contributed measurements: range & bearing
    plt.axhline(y=2*landmarks_number, color='red', linestyle='--', label=f'Expected NIS ({2*landmarks_number})')
    plt.title("Normalized Innovation Squared (NIS) over Time")
    plt.xlabel("Time Step")
    plt.ylabel("NIS")
    plt.legend()
    # plt.savefig(f"task3/{t3_fig_name}_nis.png")
    # plt.savefig(f"task4/MPCK_FINAL/route_{route_num}_nis_terminal_{terminal_cost}_MPCK.png")
    plt.show()


    #########################################################################################
    # 4. plot control reponse
    plt.figure()
    states = ["x_response", "y_response", "theta_response"]
    his = [x_his, y_his, theta_his]
    his_real = [x_his_real, y_his_real, theta_his_real]
    for s in range(len(states)):
        desire_state = desire_state_all[:, s]
        cur_his = his[s]
        cur_his_real = his_real[s]

        if cur_his[0]>desire_state[0]:
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
                subsequent_index = second_crossing_index + max(np.argmax(cur_his[second_crossing_index:]), np.argmin(cur_his[second_crossing_index:]))
            except StopIteration or NameError:
                # no second crossing exists
                second_crossing_index = 0
                subsequent_index = 0
        except StopIteration or ValueError:
            # no first crossing exists
            no_crossing_peak_start = next((i for i, x in enumerate(cur_his) if abs(x - desire_state[i]) <= tolerance) , len(cur_his) - 1)
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
            if abs(cur_dis - distance) > 0.01:
                settling_index = i + 1
                break
        print(settling_index)
        # Plotting
        plt.plot(cur_his, label='Response', linewidth=0.7)
        plt.plot(desire_state, label='Desired State', linestyle='dashed', color='red')

        plt.hlines(desire_state[-1], 0, len(cur_his) - 1, color='gray', linestyle=':')
        if settling_index!=len(cur_his)-1:
            settling_time = settling_index / 1000  # Assuming each point is a time step for simplicity
            # Calculate steady state error (distance from the avg_after_settling to desired state)
            avg_after_settling = average(cur_his[settling_index:-1])
            steady_state_error = abs(avg_after_settling - desire_state[-1])
            plt.axvline(settling_index, color='green', linestyle='--', label=f'Settling Time: {settling_time}')
            plt.text(len(cur_his) - 1, desire_state[-1], f"Steady State Error: {steady_state_error:.2f}", ha='right', color='blue')
        if rising_time_index!=-0.5:
            plt.axvline(rising_time_index, color='purple', linestyle='--', label=f'Rising Time: {rising_time}')
        if overshoot_value != -0.5:
            plt.plot(overshoot_index, overshoot_value, 'ro')  # Mark the overshoot point
            plt.text(overshoot_index, overshoot_value, f"Overshoot: {overshoot_value:.2f}", ha='center', color='red')
        plt.title("Control Response for "+states[s])
        plt.legend()
        plt.xlabel("Time Steps")
        plt.ylabel("Position")
        # plt.savefig(f"task3/{t3_fig_name}_{states[s]}.png")
        # plt.savefig(f'task4/MPCK_FINAL/route_{route_num}_{states[s]}_terminal_{terminal_cost}_MPCK.png')
        plt.show()
    
    
    

if __name__ == '__main__':
    main()