import os 
import numpy as np
from cv2 import threshold
from numpy.fft import fft, fftfreq
import time
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl

# Configuration for the simulation
conf_file_name = "pandaconfig.json"  # Configuration file for the robot
cur_dir = os.path.dirname(os.path.abspath(__file__))
sim = pb.SimInterface(conf_file_name, conf_file_path_ext = cur_dir)  # Initialize simulation interface

# Get active joint names from the simulation
ext_names = sim.getNameActiveJoints()
ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

source_names = ["pybullet"]  # Define the source for dynamic modeling

# Create a dynamic model of the robot
dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False,0,cur_dir)
num_joints = dyn_model.getNumberofActuatedJoints()

init_joint_angles = sim.GetInitMotorAngles()

print(f"Initial joint angles: {init_joint_angles}")



# single joint tuning
# episode_duration is specified in seconds
def simulate_with_given_pid_values(sim_, kp, joints_id, regulation_displacement=0.1, episode_duration=10, plot=False):
    # here we reset the simulator each time we start a new test
    sim_.ResetPose()

    # updating the kp value for the joint we want to tune
    kp_vec = np.array([10]*dyn_model.getNumberofActuatedJoints())
    kp_vec[joints_id] = kp
    kd = np.array([0]*dyn_model.getNumberofActuatedJoints()) # all zero kd
    # IMPORTANT: to ensure that no side effect happens, we need to copy the initial joint angles
    q_des = init_joint_angles.copy()
    qd_des = np.array([0]*dyn_model.getNumberofActuatedJoints())

    q_des[joints_id] = q_des[joints_id] + regulation_displacement
   
    time_step = sim_.GetTimeStep()
    current_time = 0
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors

    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all,  = [], [], [], []
    is_steady_oscillation = False

    steps = int(episode_duration/time_step)
    # testing loop
    for i in range(steps):
        # measure current state
        q_mes = sim_.GetMotorAngles(0)
        qd_mes = sim_.GetMotorVelocities(0)
        qdd_est = sim_.ComputeMotorAccelerationTMinusOne(0)
        # Compute sinusoidal reference trajectory
        # Ensure q_init is within the range of the amplitude
        
        # Control command
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)  # Zero torque command
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)  # Simulation step with torque command
        sim.Step(cmd, "torque")

        # Exit logic with 'q' key
        keys = sim_.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim_.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        
        #simulation_time = sim.GetTimeSinceReset()

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(q_des)
        qd_d_all.append(qd_des)
        #cur_regressor = dyn_model.ComputeDyanmicRegressor(q_mes,qd_mes, qdd_est)
        #regressor_all = np.vstack((regressor_all, cur_regressor))

        #time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
        #print("current time in seconds",current_time)


    # TODO make the plot for the current joint
    q_mes_all_array = np.array(q_mes_all)
    q_d_all_array = np.array(q_d_all)

    # Find peaks of current oscillation curves
    peaks, _ = find_peaks(q_mes_all_array[:, joints_id])
    peak_values = q_mes_all_array[peaks, joints_id]
    # print(peak_values)

    # determine whether the oscillation is steady: determine the variance of peak values
    is_steady_oscillation, var_peaks = is_sustained_oscillation(peak_values, threshold=0.05)

    # Plotting the measured joint angles
    plt.figure(figsize=(10, 5))
    time_vector = np.linspace(0, episode_duration, q_mes_all_array.shape[0])
    plt.plot(time_vector, q_mes_all_array[:,joints_id], label=f'Joint '+str(joints_id + 1))
    plt.plot(time_vector, q_d_all_array[:,joints_id], label=f'Reference for Joint '+str(joints_id + 1))

    # Plot the peaks and mark peaks on the plot using *
    peak_times = time_vector[peaks]
    plt.plot(peak_times, peak_values, "*", label="Peaks")

    plt.title(f'Measured Joint Angles with Kp = {kp} for Joint {joints_id + 1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Angle (rad)')
    plt.grid()
    plt.legend()

    # save the steady oscillation for current kp value
    if is_steady_oscillation:
        # Ensure the directory exists before saving
        image_dir = "report_figure/joint_"+str(joints_id+1)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)  # Create the directory if it doesn't exist

        image_path = os.path.join(image_dir, f"joint_{joints_id + 1}_kp_{kp}_var_{var_peaks}.png")
        plt.savefig(image_path, format='png', dpi=300)  # Save the image
        print(f"Image saved!")

    # display the figure on screen
    plt.show()
    plt.close()

    return q_mes_all, is_steady_oscillation, var_peaks



def is_sustained_oscillation(peak_value_array, threshold=0.1):
    """ Calculate the variance of all the peak values"""
    peak_variance = np.var(peak_value_array)
    print(f"Variance of peak values: {peak_variance}")
    return peak_variance < threshold, peak_variance


def perform_frequency_analysis(data, dt, joints_id, plot=True, save=False):
    # Remove the DC component in the data
    data = data - np.mean(data)

    n = len(data)
    yf = fft(data)
    xf = fftfreq(n, dt)[:n//2]
    power = 2.0/n * np.abs(yf[:n//2])
    amplitude = np.sqrt(power)
    print(xf.shape)
    # Find the dominant frequency
    dominant_frequency_index = np.argmax(amplitude)
    dominant_frequency = xf[dominant_frequency_index]
    print("Dominant Frequency: ", dominant_frequency)

    cutoff = int(dominant_frequency_index/2)
    amplitude_plot = amplitude[cutoff:-1]
    xf_plot = xf[cutoff:-1]


    # Optional: Plot the spectrum
    plt.figure()
    plt.plot(xf_plot, amplitude_plot, label='Amplitude Spectrum')
    plt.title("FFT of the signal")
    plt.xlabel("Frequency in Hz")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.axvline(x=dominant_frequency, color='r', linestyle='--')
    plt.legend()

    if save:
        image_dir = "report_figure/joint_" + str(joints_id + 1)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)  # Create the directory if it doesn't exist

        image_path = os.path.join(image_dir, f"joint_{joints_id + 1}_kp_{kp}_frequency.png")
        plt.savefig(image_path, format='png', dpi=300)  # Save the image
        print(f"Image saved!")


    plt.show()
    plt.close()

    return xf, power, dominant_frequency


# TODO Implement the table in thi function
def calculate_k_parameters(ku, tu, control_type='PD'):
    """ Implement the table in thi function to calculate the Kp and Kd"""
    kp = 0.0
    ki = 0.0
    kd = 0.0
    if control_type=='P':
        kp = 0.5*ku
        kd = 0
        ki = 0
    elif control_type=='PI':
        kp = 0.45*ku
        ti = 0.83*tu
        ki = 0.54 * (ku/tu)
        kd = 0
    elif control_type=='PD':
        kp = 0.8*ku
        td = 0.125*tu
        kd =0.1*ku*tu
        ki = 0
    elif control_type=="classic PID":
        kp = 0.6*ku
        ki = 1.2 * (ku/tu)
        kd = 0.075 * ku * tu

    return kp, ki, kd


def PD_controller_function(sim_, regulation_displacement, joints_id, kp, kd, episode_duration):
    """
    Simulates the response of the system using a PID controller

    :param sim_: simulation envrionment
    :param regulation_displacement: desired motion
    :param joints_id: current joint id
    :param kp: parameter for P item
    :param kd: parameter for D item
    :param episode_duration: simulation period
    :return: measured motion of current joints
    """

    # here we reset the simulator each time we start a new test
    sim_.ResetPose()

    # updating the kp value for the joint we want to tune
    kp_vec = np.array([1000] * dyn_model.getNumberofActuatedJoints())
    kp_vec[joints_id] = kp

    kd_vec = np.array([0] * dyn_model.getNumberofActuatedJoints())
    kd_vec[joints_id] = kd

    # IMPORTANT: to ensure that no side effect happens, we need to copy the initial joint angles
    q_des = init_joint_angles.copy()
    qd_des = np.array([0] * dyn_model.getNumberofActuatedJoints())

    q_des[joints_id] = q_des[joints_id] + regulation_displacement

    time_step = sim_.GetTimeStep()
    current_time = 0
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors

    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all, = [], [], [], []

    steps = int(episode_duration / time_step)
    # testing loop
    for i in range(steps):
        # measure current state
        q_mes = sim_.GetMotorAngles(0)
        qd_mes = sim_.GetMotorVelocities(0)
        qdd_est = sim_.ComputeMotorAccelerationTMinusOne(0)
        # Compute sinusoidal reference trajectory
        # Ensure q_init is within the range of the amplitude

        # Control command
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)  # Zero torque command
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)  # Simulation step with torque command
        sim.Step(cmd, "torque")

        # Exit logic with 'q' key
        keys = sim_.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim_.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # simulation_time = sim.GetTimeSinceReset()

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(q_des)
        qd_d_all.append(qd_des)
        # cur_regressor = dyn_model.ComputeDyanmicRegressor(q_mes,qd_mes, qdd_est)
        # regressor_all = np.vstack((regressor_all, cur_regressor))

        # time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
        # print("current time in seconds",current_time)
    return q_mes_all



def plot_pid_results(joints_id, ku, q_mes_all, regulation_displacement, plot):
    """
    Plots the results of the PID controller simulation.

    :param joints_id:  current joint
    :param ku:  current joint
    :param q_mes_all:  the measured motion of current joint
    :param regulation_displacement:  the desired angle = initial joint angles + regulation displacement
    :param plot:  choose to plot on screen
    """
    plt.figure(figsize=(10, 5))
    q_mes_all = np.array(q_mes_all)
    # Generate time vector
    dt = sim.GetTimeStep()
    time_vector = np.linspace(0, len(q_mes_all) * dt, len(q_mes_all))

    # Plot measured joint angles
    plt.plot(time_vector, q_mes_all[:, joints_id], label='Measured Joint Angle', color='blue')

    # Plot desired joint angle
    desired_angle = regulation_displacement + init_joint_angles[joints_id]
    plt.axhline(y=desired_angle, color='red', linestyle='--', label='Desired Joint Angle')

    plt.title(f'PD Controller Response for Joint {joints_id+1} with Ku {ku}')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Angle (rad)')
    plt.grid()
    plt.legend()

    # image_dir = "report_figure/real_control"
    # if not os.path.exists(image_dir):
    #     os.makedirs(image_dir)  # Create the directory if it doesn't exist
    # image_path = os.path.join(image_dir, f"joint_{joints_id+1}_ku_{ku}.png")
    # plt.savefig(image_path, format='png', dpi=300)  # Save the image
    # print(f"Image saved!")

    plt.show()
    plt.close()


if __name__ == '__main__':
    regulation_displacement = 0.1  # Displacement from the initial joint position
    """
        id = #Joint - 1
        Joint 1 (16.6): 13.28, 2.49
        Joint 2 (16): 12.8, 2.67
        Joint 3 (8.45): 6.76, 1.81
        Joint 4 (2): 1.6, 0.8695
        Joint 5 (16.6): 13.28, 2.49
        Joint 6 (15.75): 12.6, 2.48
        Joint 7 (16.5): 13.2, 2.60
    """
    id = 0 # 0～6
    # JOINT 2 use 10s simulation
    # Other joints use 30s
    test_duration = 30  # in seconds
    if id ==0:
        kp = 16.6
    elif id ==1:
        kp = 16
        test_duration = 10  # in seconds
    elif id == 2:
        kp = 8.45
    elif id == 3:
        kp = 2
    elif id ==4:
        kp = 16.6
    elif id == 5:
        kp = 15.75
    elif id == 6:
        kp = 16.5

    # TODO using simulate_with_given_pid_values() and perform_frequency_analysis() write you code to test different Kp values
    # for each joint, bring the system to oscillation and compute the the PD parameters using the Ziegler-Nichols method
    joint_kp_values = 0
    min_var_peak = 1
    cur_joint_value = []
    ku_cur = 0

    print("################################################################################")
    print("################################################################################")
    print("################################################################################")
    print("Current Joint id:", id)
    print("Current kp:" , kp)

    # 1. measurements of the motion; 2. whether it is a steady oscillation; 3. the variance for all peak values of current oscillation
    q_mes_all, is_steady_oscillation, var_peak = simulate_with_given_pid_values(sim, kp, id, regulation_displacement, episode_duration=test_duration, plot=True)
    q_mes_all_array = np.array(q_mes_all)
    dt = sim.GetTimeStep()

    q_joint_mes_all = []

    for mes in q_mes_all:
        q_joint_mes_all.append(mes[0])
    xf, power, dominant_frequency = perform_frequency_analysis(q_joint_mes_all, dt, id, plot=True, save=True)

    # if steady oscillation, calculate the kp & kd for simulation control
    if is_steady_oscillation:
        ku_cur = kp
        print(f"Is steady oscillation: {is_steady_oscillation}")
        print("ku: ", ku_cur)

        # Calculate and validate the kp, kd
        parameters = []
        dt = sim.GetTimeStep()

        # frequency analysis
        xf, power, dominant_frequency = perform_frequency_analysis(np.array(q_mes_all).T[id], dt, id, plot=False, save=False)
        Tu = 1 / dominant_frequency

        # Calculate the kp and kd
        kp, ki, kd = calculate_k_parameters(ku_cur, Tu)
        print("(kp, ki, kd) for joint " + str(id + 1) + ": (" + str(kp) + ", " + str(ki) + ", " + str(kd) + ")")

        q_mes = PD_controller_function(sim, regulation_displacement, id, kp, kd, episode_duration=test_duration)
        plot_pid_results(id, ku_cur, q_mes, regulation_displacement, plot=True)


    # #########################################################################################################################
    # VALIDATE PROCESS!!!
    # # Use PD controller to plot the figure for current joints
    """
    id = #Joint - 1
    Joint 1 (16.6): 13.28, 2.49
    Joint 2 (16): 12.8, 2.67
    Joint 3 (8.45): 6.76, 1.81
    Joint 4 (2): 1.6, 0.8695
    Joint 5 (16.6): 13.28, 2.49
    Joint 6 (15.75): 12.6, 2.48
    Joint 7 (16.5): 13.2, 2.60
    """
    if id == 0:
        ku_cur = 16.6
        kp = 13.28
        kd = 2.49
    elif id ==1:
        ku_cur = 16
        kp = 12.8
        kd = 2.67
    elif id ==2:
        ku_cur = 8.45
        kp = 6.76
        kd = 1.81
    elif id ==3:
        ku_cur = 2
        kp = 1.6
        kd = 0.8695
    elif id ==4:
        ku_cur = 16.6
        kp = 13.28
        kd = 2.49
    elif id ==5:
        ku_cur = 15.75
        kp = 12.6
        kd = 2.48
    elif id ==6:
        ku_cur = 16.5
        kp = 13.2
        kd = 2.6

    # call the PD controller and plot the result figures
    q_mes = PD_controller_function(sim, regulation_displacement, id, kp, kd, episode_duration=test_duration)
    plot_pid_results(id, ku_cur, q_mes, regulation_displacement, plot=True)


