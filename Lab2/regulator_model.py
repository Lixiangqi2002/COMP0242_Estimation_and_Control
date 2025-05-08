import numpy as np
    
class RegulatorModel:
    def __init__(self, A, B, C, Q, R, N, q, m, n, x_ref):
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.N = N
        self.q = q
        self.m = m
        self.n = n
        self.x_ref = x_ref

    # def compute_H_and_F(self, S_bar, T_bar, Q_bar, R_bar):
    #     # Compute H
    #     H = np.dot(S_bar.T, np.dot(Q_bar, S_bar)) + R_bar
    #     print(self.x_ref)
    #     if np.all(self.x_ref == 0):
    #         # Compute F
    #         F = np.dot(S_bar.T, np.dot(Q_bar, T_bar))
    #     else:
    #         # Compute F for reference tracking
    #         F = np.dot(S_bar.T, np.dot(Q_bar, (T_bar - np.dot(np.tile(np.eye(self.q), (self.N, 1)), self.x_ref))))
    #
    #     return H, F

    def compute_H_and_F(self, S_bar, T_bar, Q_bar, R_bar):
        # Compute H
        H = np.dot(S_bar.T, np.dot(Q_bar, S_bar)) + R_bar

        # If reference is provided, compute F with reference tracking
        if np.any(self.x_ref != 0):  # Check if the reference is non-zero
            # Tile the reference vector to match the horizon length
            x_ref_tiled = np.tile(self.x_ref, self.N).reshape(-1, 1)  # Shape: (N*q, 1)
            # Compute F for reference tracking (account for deviation from x_ref)
            F = np.dot(S_bar.T, np.dot(Q_bar, (T_bar - x_ref_tiled)))
        else:
            # Compute F without reference tracking (regulator problem)
            F = np.dot(S_bar.T, np.dot(Q_bar, T_bar))

        return H, F

    def propagation_model_regulator_fixed_std(self):
        S_bar = np.zeros((self.N*self.q, self.N*self.m))
        T_bar = np.zeros((self.N*self.q, self.n))
        Q_bar = np.zeros((self.N*self.q, self.N*self.q))
        R_bar = np.zeros((self.N*self.m, self.N*self.m))

        for k in range(1, self.N + 1):
            for j in range(1, k + 1):
                S_bar[(k-1)*self.q:k*self.q, (k-j)*self.m:(k-j+1)*self.m] = np.dot(np.dot(self.C, np.linalg.matrix_power(self.A, j-1)), self.B)

            T_bar[(k-1)*self.q:k*self.q, :self.n] = np.dot(self.C, np.linalg.matrix_power(self.A, k))

            Q_bar[(k-1)*self.q:k*self.q, (k-1)*self.q:k*self.q] = self.Q
            R_bar[(k-1)*self.m:k*self.m, (k-1)*self.m:k*self.m] = self.R

        return S_bar, T_bar, Q_bar, R_bar