import numpy as np
from scipy.optimize import minimize


class ConstraintsTrackerModel:
    def __init__(self, A_continuos, B_continuos, C_continuos, Q, R, N, q, m, n, delta, pos_bounds, vel_bounds, slack_penalty_weight):
        A_cont = A_continuos
        B_cont = B_continuos
        C_cont = C_continuos
        self.Q = Q
        self.R = R
        self.N = N  # prediction horizon
        self.q = q  # number of outputs
        self.m = m  # Number of control inputs
        self.orig_n = n  # Number of states
        self.delta = delta
        self.slack_penalty_weight = slack_penalty_weight  # 惩罚松弛变量的权重

        # Add position and velocity bounds
        self.pos_bounds = pos_bounds  # [lower_bound, upper_bound] for position
        self.vel_bounds = vel_bounds  # [lower_bound, upper_bound] for velocity

        # Compute matrix A and B as before
        A_upper_left = np.eye(self.orig_n) + self.delta * A_cont
        A_upper_right = self.delta * B_cont
        A_lower_left = np.zeros((self.m, self.orig_n))
        A_lower_right = np.eye(self.m)

        A_upper = np.hstack((A_upper_left, A_upper_right))
        A_lower = np.hstack((A_lower_left, A_lower_right))
        A = np.vstack((A_upper, A_lower))

        self.A = np.block([
            [np.eye(self.orig_n) + self.delta * A_cont, self.delta * B_cont],
            [np.zeros((self.m, self.orig_n)), np.eye(self.m)]
        ])
        self.B = np.vstack((self.delta * B_cont, np.eye(self.m)))
        self.C = np.hstack((C_cont, np.zeros((self.q, self.m))))
        self.n = A.shape[0]  # Extended state dimension


    def tracker_std(self, S_bar, T_bar, Q_hat, Q_bar, R_bar):
        # Compute H
        H = R_bar + S_bar.T @ Q_bar @ S_bar

        # Compute F_tra
        first = -Q_hat @ S_bar
        second = T_bar.T @ Q_bar @ S_bar
        F_tra = np.vstack([first, second])  # Stack first and second vertically

        return H, F_tra

    def propagation_model_tracker_fixed_std(obj):
        # Determine sizes and initialize matrices
        S_bar = np.zeros((obj.n * obj.N, obj.m * obj.N))
        S_bar_C = np.zeros((obj.q * obj.N, obj.m * obj.N))
        T_bar = np.zeros((obj.n * obj.N, obj.n))
        T_bar_C = np.zeros((obj.q * obj.N, obj.n))
        Q_hat = np.zeros((obj.q * obj.N, obj.n * obj.N))
        Q_bar = np.zeros((obj.n * obj.N, obj.n * obj.N))
        R_bar = np.zeros((obj.m * obj.N, obj.m * obj.N))

        # Loop to calculate matrices
        for k in range(1, obj.N + 1):
            for j in range(1, k + 1):
                idx_row_S = slice(obj.n * (k - 1), obj.n * k)
                idx_col_S = slice(obj.m * (k - j), obj.m * (k - j + 1))
                S_bar[idx_row_S, idx_col_S] = np.linalg.matrix_power(obj.A, j - 1) @ obj.B

                idx_row_SC = slice(obj.q * (k - 1), obj.q * k)
                S_bar_C[idx_row_SC, idx_col_S] = obj.C @ np.linalg.matrix_power(obj.A, j - 1) @ obj.B

            idx_row_T = slice(obj.n * (k - 1), obj.n * k)
            T_bar[idx_row_T, :] = np.linalg.matrix_power(obj.A, k)

            idx_row_TC = slice(obj.q * (k - 1), obj.q * k)
            T_bar_C[idx_row_TC, :] = obj.C @ np.linalg.matrix_power(obj.A, k)

            idx_row_QH = slice(obj.q * (k - 1), obj.q * k)
            idx_col_QH = slice(obj.n * (k - 1), obj.n * k)
            Q_hat[idx_row_QH, idx_col_QH] = obj.Q @ obj.C

            idx_row_col_QB = slice(obj.n * (k - 1), obj.n * k)
            Q_bar[idx_row_col_QB, idx_row_col_QB] = obj.C.T @ obj.Q @ obj.C

            idx_row_col_R = slice(obj.m * (k - 1), obj.m * k)
            R_bar[idx_row_col_R, idx_row_col_R] = obj.R

        return S_bar, S_bar_C, T_bar, T_bar_C, Q_hat, Q_bar, R_bar

    def computesolution(self, x_ref, x_cur, u_cur, H, F_tra, T_bar, S_bar):
        F = np.dot(np.hstack([x_ref, x_cur, u_cur]), F_tra)

        lower_bounds = []
        upper_bounds = []

        for _ in range(self.N):
            lower_bounds.extend(self.pos_bounds[0])  # Position lower bounds
            lower_bounds.extend(self.vel_bounds[0])  # Velocity lower bounds
            upper_bounds.extend(self.pos_bounds[1])  # Position upper bounds
            upper_bounds.extend(self.vel_bounds[1])  # Velocity upper bounds

        # Convert bounds to numpy arrays for use in the solver
        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)

        slack_size = len(lower_bounds)
        slack_penalty_weight = np.tile(self.slack_penalty_weight, self.N)
        T_s = np.eye(slack_size) * slack_penalty_weight  # punishment matrix (diagonal)

        # objective function: 0.5 * u^T * H * u + F^T * u + s^T * T_s * s
        def objective(variables, H, F, T_s):
            u = variables[:self.m * self.N]  #  u
            s = variables[self.m * self.N:]  #  s
            return 0.5 * np.dot(u.T, np.dot(H, u)) + np.dot(F, u) + np.dot(s.T, np.dot(T_s, s))


        u0 = np.zeros(self.m * self.N)  # u
        s0 = np.zeros(slack_size)  # s=0
        initial_guess = np.hstack([u0, s0])

        result = minimize(
            fun=objective,
            x0=initial_guess,
            args=(H, F, T_s),
            method='SLSQP',
            constraints=[
                {
                    'type': 'ineq',  # upper_bound + s >= S_bar @ u
                    'fun': lambda variables: (upper_bounds + variables[self.m * self.N:])
                                             - np.dot(S_bar[:self.orig_n * self.N, :], variables[:self.m * self.N])
                },
                {
                    'type': 'ineq',  # S_bar @ u + s >= lower_bound
                    'fun': lambda variables: np.dot(S_bar[:self.orig_n  * self.N, :], variables[:self.m * self.N]) + variables[
                                                                                                     self.m * self.N:] - lower_bounds
                },
                {
                    'type': 'ineq',  # s >= 0
                    'fun': lambda variables: variables[self.m * self.N:]
                }
            ]

        )

        u_star = result.x[:self.m * self.N]  # 优化出的控制输入
        return u_star

