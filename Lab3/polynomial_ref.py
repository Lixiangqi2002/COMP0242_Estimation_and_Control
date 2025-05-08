import numpy as np

class PolynomialReference:
    def __init__(self, coefficients, q_init):
        """
        coefficients: List of polynomial coefficients for each joint (7 joints).
        q_init: Initial position for each joint.
        """
        self.coefficients = np.array(coefficients)
        self.q_init = np.array(q_init).reshape(-1, 1)  # Ensure q_init is a column vector

        # Ensure the number of coefficient sets matches the number of joints
        if not (self.coefficients.shape[0] == self.q_init.size):
            expected_num_elements = self.q_init.size
            raise ValueError(f"Number of coefficient sets must match the number of joints. "
                             f"Expected: {expected_num_elements}, Received: {self.coefficients.shape[0]}.")

    def get_values(self, time):
        """
        Calculate the position and velocity at a given time for polynomial reference.

        Parameters:
        time (float): The time at which to evaluate the position and velocity.

        Returns:
        tuple: The position (q_d) and velocity (qd_d) at the given time (both as 7x1 column vectors).
        """
        q_d = np.zeros((self.q_init.size, 1))  # Initialize as a column vector
        qd_d = np.zeros((self.q_init.size, 1))  # Initialize as a column vector

        # Iterate over each joint
        for i in range(len(self.q_init)):
            # Polynomial position: q_d = q_init + sum(coeff[i] * time^n)
            q_d[i] = self.q_init[i] + np.polyval(self.coefficients[i], time)

            # Polynomial velocity is the derivative of the polynomial (np.polyder)
            qd_d[i] = np.polyval(np.polyder(self.coefficients[i]), time)

        return q_d, qd_d
