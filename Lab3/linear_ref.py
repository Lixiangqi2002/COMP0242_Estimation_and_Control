import numpy as np

class LinearReference:
    def __init__(self, slope, intercept, q_init):
        self.slope = np.array(slope)
        self.intercept = np.array(intercept)
        self.q_init = np.array(q_init)

        # Check if all arrays have the same length
        if not (self.slope.size == self.intercept.size == self.q_init.size):
            expected_num_elements = self.q_init.size
            raise ValueError(f"All arrays must have the same number of elements. "
                             f"Expected number of elements (joints): {expected_num_elements}, "
                             f"Received - Slope: {self.slope.size}, "
                             f"Intercept: {self.intercept.size}, "
                             f"Q_init: {expected_num_elements}.")

    def get_values(self, time):
        # Linear position around the initial position: q_d = q_init + slope * time
        q_d = self.q_init + self.slope * time + self.intercept
        # The velocity is constant: qd_d = slope
        qd_d = self.slope
        return q_d, qd_d
