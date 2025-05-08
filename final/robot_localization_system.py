#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tornado.process import task_id


# from final.standalone_localization_tester import wrap_angle
def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

class FilterConfiguration(object):
    def __init__(self):
        # Process and measurement noise covariance matrices
        self.V = np.diag([0.1, 0.1, 0.05]) ** 2  # Process noise covariance
        # Measurement noise variance (range measurements)
        self.W_range = 0.5 ** 2
        self.W_bearing = (np.pi * 0.5/ 180.0) ** 2

        # Initial conditions for the filter
        self.x0 = np.array([2.0, 3.0, np.pi / 4])
        self.Sigma0 = np.diag([1.0, 1.0, 0.5]) ** 2


class FilterConfigurationMPC(object):
    def __init__(self,x0,V=0):
        # Process noise covariance matrix V
        self.V = np.diag([V, V, V]) ** 2

        # Measurement noise variance (range and bearing)
        self.W_range = 0.25
        self.W_bearing = (np.pi * 0.5 / 180.0) ** 2

        # Initial state estimate x0 and covariance Sigma0
        self.x0 = np.array(x0)
        self.Sigma0 = np.diag([0.5,0.5,0.5]) ** 2



class Map(object):
    def __init__(self, task="MPC"):
        if task == "1-1":
            # Original 3 points for 1-1
            self.landmarks = np.array([
                    [5, 10],
                    [15, 5],
                    [10, 15]])
        elif task == "1-2-circle":
            # TODO: Task 1-2
            # Circle：
            radius = 40
            num_landmarks = 100
            center = [-3,2]
            angles = np.linspace(0, 2 * np.pi, num_landmarks, endpoint=False)
            x_values = radius * np.cos(angles) + center[0]# x
            y_values = radius * np.sin(angles) + center[1] # y 坐标
            self.landmarks = np.vstack([x_values, y_values]).T
        elif task == "1-2-grid":
             # GRID：
            x_min, x_max = -50, 30
            y_min, y_max = -45, 35
            num_points_x = 10 # 5
            num_points_y = 10 # 5
            x_values = np.linspace(x_min, x_max, num_points_x)
            y_values = np.linspace(y_min, y_max, num_points_y)

            x_grid, y_grid = np.meshgrid(x_values, y_values)
            self.landmarks = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
        elif task == "1-4-circle":
            # TODO: Task 1-4
            # Circle：
            radius = 40
            num_landmarks = 25
            center = [-3,2]
            angles = np.linspace(0, 2 * np.pi, num_landmarks, endpoint=False)
            x_values = radius * np.cos(angles) + center[0]# x
            y_values = radius * np.sin(angles) + center[1] # y 坐标
            self.landmarks = np.vstack([x_values, y_values]).T
        elif task == "1-4-grid":
             # GRID：
            x_min, x_max = -50, 30
            y_min, y_max = -45, 35
            num_points_x = 5
            num_points_y = 5
            x_values = np.linspace(x_min, x_max, num_points_x)
            y_values = np.linspace(y_min, y_max, num_points_y)

            x_grid, y_grid = np.meshgrid(x_values, y_values)
            self.landmarks = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
        else:
            # # TODO: Task 3 add some landmark for EKF
            x_min, x_max = -3, 5
            y_min, y_max = -3, 5
            num_points_x = 5 # 5
            num_points_y = 6 # 5
            x_values = np.linspace(x_min, x_max, num_points_x)
            y_values = np.linspace(y_min, y_max, num_points_y)

            x_grid, y_grid = np.meshgrid(x_values, y_values)
            self.landmarks = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

class RobotEstimator(object):

    def __init__(self, filter_config, map):
        # Variables which will be used
        self._config = filter_config
        self._map = map
        self.nis_history = []

    # This method MUST be called to start the filter
    def start(self):
        self._t = 0
        self._set_estimate_to_initial_conditions()

    def set_control_input(self, u):
        self._u = u

    # Predict to the time. The time is fed in to
    # allow for variable prediction intervals.
    def predict_to(self, time):
        # What is the time interval length?
        dt = time - self._t

        # Store the current time
        self._t = time

        # Now predict over a duration dT
        self._predict_over_dt(dt)

    # Return the estimate and its covariance
    def estimate(self):
        return self._x_est, self._Sigma_est

    # This method gets called if there are no observations
    def copy_prediction_to_estimate(self):
        self._x_est = self._x_pred
        self._Sigma_est = self._Sigma_pred

    # This method sets the filter to the initial state
    def _set_estimate_to_initial_conditions(self):
        # Initial estimated state and covariance
        self._x_est = self._config.x0
        self._Sigma_est = self._config.Sigma0

    # Predict to the time
    def _predict_over_dt(self, dt):
        v_c = self._u[0]
        omega_c = self._u[1]
        V = self._config.V
        # print(dt)
        # Predict the new state
        self._x_pred = self._x_est + np.array([
            v_c * np.cos(self._x_est[2]) * dt,
            v_c * np.sin(self._x_est[2]) * dt,
            omega_c * dt
        ])
        self._x_pred[-1] = np.arctan2(np.sin(self._x_pred[-1]),
                                      np.cos(self._x_pred[-1]))

        # Predict the covariance
        A = np.array([
            [1, 0, -v_c * np.sin(self._x_est[2]) * dt],
            [0, 1,  v_c * np.cos(self._x_est[2]) * dt],
            [0, 0, 1]
        ])

        self._kf_predict_covariance(A, self._config.V * dt)

    # Predict the EKF covariance; note the mean is
    # totally model specific, so there's nothing we can
    # clearly separate out.
    def _kf_predict_covariance(self, A, V):
        self._Sigma_pred = A @ self._Sigma_est @ A.T + V

    # Implement the Kalman filter update step.
    def _do_kf_update(self, nu, C, W):

        # Kalman Gain
        SigmaXZ = self._Sigma_pred @ C.T
        SigmaZZ = C @ SigmaXZ + W
        K = SigmaXZ @ np.linalg.inv(SigmaZZ)
        # print(K.shape)
        # State update
        self._x_est = self._x_pred + K @ nu

        # Covariance update
        self._Sigma_est = (np.eye(len(self._x_est)) - K @ C) @ self._Sigma_pred

    def update_from_landmark_range_observations(self, y_range):
        # Predicted the landmark measurements and build up the observation Jacobian
        y_pred = []
        C = []
        x_pred = self._x_pred
        for lm in self._map.landmarks:
            # x, y, theta
            dx_pred = lm[0] - x_pred[0]
            dy_pred = lm[1] - x_pred[1]
            range_pred = np.sqrt(dx_pred**2 + dy_pred**2)
            y_pred.append(range_pred)

            # Jacobian of the measurement model
            C_range = np.array([
                -(dx_pred) / range_pred,
                -(dy_pred) / range_pred,
                0
            ])
            C.append(C_range)
        # Convert lists to arrays
        C = np.array(C)
        y_pred = np.array(y_pred)

        # Innovation. Look new information! (geddit?)
        nu = y_range - y_pred

        # Since we are observing a bunch of landmarks
        # build the covariance matrix. Note you could
        # swap this to just calling the ekf update call
        # multiple times, once for each observation,
        # as well
        W_landmarks = self._config.W_range * np.eye(len(self._map.landmarks))
        self._do_kf_update(nu, C, W_landmarks)

        # Angle wrap afterwards
        self._x_est[-1] = np.arctan2(np.sin(self._x_est[-1]), np.cos(self._x_est[-1]))

    # TODO： calculate NIS
    def calculate_nis(self, innovation, innovation_covariance):
        nis = innovation.T @ np.linalg.inv(innovation_covariance) @ innovation
        return nis


    # TODO: Make EKF process both range and bearing observations
    def update_from_landmark_range_bearing_observations(self, y):
        # Initialize arrays for predictions and Jacobian matrix
        y_pred = []
        C = []
        x_pred = self._x_pred
        # print(len(self._map.landmarks))
        for lm in self._map.landmarks:
            # Calculate predicted range and bearing
            dx_pred = lm[0] - x_pred[0]
            dy_pred = lm[1] - x_pred[1]
            range_pred = np.sqrt(dx_pred ** 2 + dy_pred ** 2)
            bearing_pred = np.arctan2(dy_pred, dx_pred) - x_pred[2]

            # Normalize predicted bearing to [-π, π]
            bearing_pred = wrap_angle(bearing_pred)

            # Append predicted range and bearing to y_pred
            y_pred.extend([range_pred, bearing_pred])

            # Jacobian for range measurement
            C_range = np.array([
                -(dx_pred) / range_pred,
                -(dy_pred) / range_pred,
                0
            ])

            # Jacobian for bearing measurement
            C_bearing = np.array([
                dy_pred / (range_pred ** 2),
                -dx_pred / (range_pred ** 2),
                -1
            ])

            # Stack range and bearing Jacobians
            C.append(np.vstack((C_range, C_bearing)))

        # Convert predictions and Jacobian to arrays
        C = np.vstack(C)
        y_pred = np.array(y_pred)

        # Calculate innovation (y - y_pred)
        nu = y - y_pred

        # Normalize bearing innovation to [-π, π] for every second element
        nu[1::2] = wrap_angle(nu[1::2])

        # Covariance for measurement noise (both range and bearing)
        W_range = self._config.W_range
        W_bearing = self._config.W_bearing
        W_landmarks = np.diag([W_range, W_bearing] * (len(y) // 2))
        # Calculate innovation covariance S
        S = C @ self._Sigma_pred @ C.T + W_landmarks
        # Perform Kalman Filter update
        self._do_kf_update(nu, C, W_landmarks)

        # Normalize the estimated angle to [-π, π]
        self._x_est[-1] = wrap_angle(self._x_est[-1])

        return nu, S

