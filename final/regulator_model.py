from logging import exception

import numpy as np
from exceptiongroup import catch
from numpy.linalg import LinAlgError
from scipy.linalg import solve_discrete_lyapunov, solve_discrete_are, pinv, eig


class RegulatorModel:
    def __init__(self, N, q, m, n):
        self.A = None
        self.B = None
        self.C = None
        self.Q = None
        self.R = None
        self.N = N
        self.q = q #  output dimension
        self.m = m #  input dimension
        self.n = n #  state dimension
        self.P = None # Terminal Cost
        self.K = None

    def dlqr(self, n=None):
        # If n is None, set it to a zero matrix with the appropriate dimensions
        if n is None:
            n = np.zeros((self.A.shape[0], self.B.shape[1]))

        # Solve the discrete Riccati equation to get S
        try:
            self.P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        except LinAlgError:
            return False
        # Compute the optimal gain matrix K
        br_inv = np.linalg.inv(self.R + self.B.T @ self.P  @ self.B)
        k = br_inv @ (self.B.T @ self.P  @ self.A + n.T)

        # Compute the closed-loop poles
        clp = eig(self.A - self.B @ k)[0]
        return True

    def compute_H_and_F(self, S_bar, T_bar, Q_bar, R_bar):
        # Compute H
        H = np.dot(S_bar.T, np.dot(Q_bar, S_bar)) + R_bar

        # Compute F
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
        # Add terminal cost P to the last block of Q_bar
        if self.P is not None:
            Q_bar[-self.q:, -self.q:] = self.P
        # print(S_bar.shape, (self.q, self.m)) # N*N
        # print(T_bar.shape, (self.q, self.n)) # N*1
        # print(Q_bar.shape, (self.q, self.q)) # N*N
        # # print(Q_bar)
        # print(R_bar.shape, (self.m, self.m)) # N*N
        return S_bar, T_bar, Q_bar, R_bar


    def updateSystemMatrices(self,sim,cur_x,cur_u):
        """
        Get the system matrices A and B according to the dimensions of the state and control input.
        
        Parameters:
        num_states, number of system states
        num_controls, number oc conttrol inputs
        cur_x, current state around which to linearize
        cur_u, current control input around which to linearize
       
        
        Returns:
        A: State transition matrix
        B: Control input matrix
        """
        # Check if state_x_for_linearization and cur_u_for_linearization are provided
        if cur_x is None or cur_u is None:
            raise ValueError(
                "state_x_for_linearization and cur_u_for_linearization are not specified.\n"
                "Please provide the current state and control input for linearization.\n"
                "Hint: Use the goal state (e.g., zeros) and zero control input at the beginning.\n"
                "Also, ensure that you implement the linearization logic in the updateSystemMatrices function."
            )

        A =[]
        B = []
        num_states = self.n
        num_controls = self.m
        num_outputs = self.q
        time_step = sim.GetTimeStep()
        theta_0 = cur_x[2]
        v_0 = cur_u[0]
        # get A and B matrices by linearinzing the cotinuous system dynamics
        # The linearized continuous-time system is:

        # \[
        # \dot{\mathbf{x}} = A_c (\mathbf{x} - \mathbf{x}_0) + B_c (\mathbf{u} - \mathbf{u}_0).
        # \]

        # \textbf{Compute \( A_c = \left. \dfrac{\partial \mathbf{f}}{\partial \mathbf{x}} \right|_{(\mathbf{x}_0, \mathbf{u}_0)} \):}

        # \[
        # A_c = \begin{bmatrix}
        # \frac{\partial \dot{x}}{\partial x} & \frac{\partial \dot{x}}{\partial y} & \frac{\partial \dot{x}}{\partial \theta} \\
        # \frac{\partial \dot{y}}{\partial x} & \frac{\partial \dot{y}}{\partial y} & \frac{\partial \dot{y}}{\partial \theta} \\
        # \frac{\partial \dot{\theta}}{\partial x} & \frac{\partial \dot{\theta}}{\partial y} & \frac{\partial \dot{\theta}}{\partial \theta}
        # \end{bmatrix}.
        # \]

        # Compute the partial derivatives:

        # \begin{align*}
        # \frac{\partial \dot{x}}{\partial x} &= 0, & \frac{\partial \dot{x}}{\partial y} &= 0, & \frac{\partial \dot{x}}{\partial \theta} &= -v_0 \sin(\theta_0), \\
        # \frac{\partial \dot{y}}{\partial x} &= 0, & \frac{\partial \dot{y}}{\partial y} &= 0, & \frac{\partial \dot{y}}{\partial \theta} &= v_0 \cos(\theta_0), \\
        # \frac{\partial \dot{\theta}}{\partial x} &= 0, & \frac{\partial \dot{\theta}}{\partial y} &= 0, & \frac{\partial \dot{\theta}}{\partial \theta} &= 0.
        # \end{align*}

        # Thus,

        # \[
        # A_c = \begin{bmatrix}
        # 0 & 0 & -v_0 \sin(\theta_0) \\
        # 0 & 0 & v_0 \cos(\theta_0) \\
        # 0 & 0 & 0
        # \end{bmatrix}.
        # \]
        A_c = np.array([
            [0, 0, -v_0 * np.sin(theta_0)],
            [0, 0, v_0 * np.cos(theta_0)],
            [0, 0, 0]
        ])
        # \textbf{Compute \( B_c = \left. \dfrac{\partial \mathbf{f}}{\partial \mathbf{u}} \right|_{(\mathbf{x}_0, \mathbf{u}_0)} \):}

        # \[
        # B_c = \begin{bmatrix}
        # \frac{\partial \dot{x}}{\partial v} & \frac{\partial \dot{x}}{\partial \omega} \\
        # \frac{\partial \dot{y}}{\partial v} & \frac{\partial \dot{y}}{\partial \omega} \\
        # \frac{\partial \dot{\theta}}{\partial v} & \frac{\partial \dot{\theta}}{\partial \omega}
        # \end{bmatrix}.
        # \]

        # Compute the partial derivatives:

        # \begin{align*}
        # \frac{\partial \dot{x}}{\partial v} &= \cos(\theta_0), & \frac{\partial \dot{x}}{\partial \omega} &= 0, \\
        # \frac{\partial \dot{y}}{\partial v} &= \sin(\theta_0), & \frac{\partial \dot{y}}{\partial \omega} &= 0, \\
        # \frac{\partial \dot{\theta}}{\partial v} &= 0, & \frac{\partial \dot{\theta}}{\partial \omega} &= 1.
        # \end{align*}

        # Thus,

        # \[
        # B_c = \begin{bmatrix}
        # \cos(\theta_0) & 0 \\
        # \sin(\theta_0) & 0 \\
        # 0 & 1
        # \end{bmatrix}.
        # \]
        B_c = np.array([
            [np.cos(theta_0), 0],
            [np.sin(theta_0), 0],
            [0, 1]
        ])


        # then linearize A and B matrices
        #\[
        # A = I + \Delta t \cdot A_c,
        # \]
        # \[
        # B = \Delta t \cdot B_c,
        # \]

        # where \( I \) is the identity matrix.

        # TODO: Compute \( A \):
        # \[
        # A = \begin{bmatrix}
        # 1 & 0 & -v_0 \Delta t \sin(\theta_0) \\
        # 0 & 1 & v_0 \Delta t \cos(\theta_0) \\
        # 0 & 0 & 1
        # \end{bmatrix}.
        # \]
        A = np.eye(num_states) + time_step * A_c
        A = np.array([
            [1, 0, -v_0 * time_step * np.sin(theta_0)],
            [0, 1, v_0 * time_step * np.cos(theta_0)],
            [0, 0, 1]
        ])
        # print(A)

        # TODO: Compute \( B \):
        # \[
        # B = \begin{bmatrix}
        # \Delta t \cos(\theta_0) & 0 \\
        # \Delta t \sin(\theta_0) & 0 \\
        # 0 & \Delta t
        # \end{bmatrix}.
        # \]
        B = time_step * B_c
        B = np.array([
            [time_step * np.cos(theta_0), 0],
            [time_step * np.sin(theta_0), 0],
            [0, time_step]
        ])
        #updating the state and control input matrices
        self.A = A
        # self.A += 1e-4 * np.eye(self.A.shape[0])
        self.B = B
        self.C = np.eye(num_outputs)

        def is_controllable(A, B):
            n = A.shape[0]
            eigenvalues, _ = np.linalg.eig(A)
            for l in eigenvalues:
                hautus_matrix = np.hstack((l*np.eye(n)-A, B))
                rank = np.linalg.matrix_rank(hautus_matrix)
                if rank<n:
                    return False
            return True

        if is_controllable(self.A, self.B):
            print("Controllable")
        else:
            print("Not Controllable")


    # TODO you can change this function to allow for more passing a vector of gains
    def setCostMatrices(self,Qcoeff,Rcoeff):
        """
        Get the cost matrices Q and R for the MPC controller.

        Returns:
        Q: State cost matrix
        R: Control input cost matrix
        """
        num_states = self.n
        num_controls = self.m

        # Q = 1 * np.eye(num_states)  # State cost matrix
        Q = Qcoeff * np.eye(num_states)
        # print(Q)

        R = Rcoeff * np.eye(num_controls)  # Control input cost matrix

        self.Q = Q
        self.R = R