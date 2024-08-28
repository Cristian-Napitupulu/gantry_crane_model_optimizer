from numba import jit, float64
import numpy as np
import pandas as pd
import warnings


# Define the function with JIT compilation
@jit(nopython=True)
def simulate_core(
    num_steps,
    dt,
    L_tm,
    L_hm,
    rp_tm,
    rp_hm,
    m_t,
    m_c,
    J_tm,
    J_hm,
    Kt_tm,
    Kt_hm,
    Kemf_tm,
    Kemf_hm,
    R_tm,
    R_hm,
    b_t,
    b_c,
    b_tm,
    b_hm,
    g,
    bias_tm,
    bias_hm,
    supply_voltage,
    max_pwm,
    trolley_motor_input,
    hoist_motor_input,
    initial_conditions,
):
    x = np.zeros(num_steps + 1)
    l = np.zeros(num_steps + 1)
    theta = np.zeros(num_steps + 1)
    x_dot = np.zeros(num_steps + 1)
    l_dot = np.zeros(num_steps + 1)
    theta_dot = np.zeros(num_steps + 1)
    x_dot_dot = np.zeros(num_steps + 1)
    l_dot_dot = np.zeros(num_steps + 1)
    theta_dot_dot = np.zeros(num_steps + 1)

    Ux = trolley_motor_input * supply_voltage / max_pwm
    Ul = hoist_motor_input * supply_voltage / max_pwm

    if initial_conditions is not None:
        x[0] = initial_conditions[0]
        l[0] = initial_conditions[1]
        theta[0] = initial_conditions[2]
    else:
        raise ValueError("Initial conditions are not provided")

    for i in range(num_steps):
        # print(i)
        # Update matrix A
        a_1_1 = L_tm * rp_tm * (
            m_t + m_c * np.sin(theta[i]) ** 2
        ) / Kt_tm + J_tm * L_tm / (Kt_tm * rp_tm)

        a_1_2 = -L_tm * m_c * rp_tm * np.sin(theta[i]) / Kt_tm

        a_2_1 = -L_hm * m_c * rp_hm * np.sin(theta[i]) / Kt_hm
        a_2_2 = L_hm * m_c * rp_hm / Kt_hm + J_hm * L_hm / (Kt_hm * rp_hm)

        # Update matrix B
        b_1_1 = (
            +L_tm * m_c * rp_tm * np.sin(2 * theta[i]) * theta_dot[i] / Kt_tm
            + R_tm * rp_tm * (m_t + m_c * np.sin(theta[i]) ** 2) / Kt_tm
            + L_tm * b_t * rp_tm / Kt_tm
            + J_tm * R_tm / (Kt_tm * rp_tm)
            + L_tm * b_tm / (Kt_tm * rp_tm)
        )

        b_1_2 = (
            -L_tm * m_c * rp_tm * np.cos(theta[i]) * theta_dot[i] / Kt_tm
            - R_tm * m_c * rp_tm * np.sin(theta[i]) / Kt_tm
        )

        b_2_1 = (
            -L_hm * m_c * rp_hm * np.cos(theta[i]) * theta_dot[i] / Kt_hm
            - R_hm * m_c * rp_hm * np.sin(theta[i]) / Kt_hm
        )

        b_2_2 = (
            +L_hm * b_c * rp_hm / Kt_hm
            + R_hm * m_c * rp_hm / Kt_hm
            + J_hm * R_hm / (Kt_hm * rp_hm)
            + L_hm * b_hm / (Kt_hm * rp_hm)
        )

        # Update matrix C
        c_1_1 = (
            R_tm * b_t * rp_tm / Kt_tm + Kemf_tm / rp_tm + R_tm * b_tm / (Kt_tm * rp_tm)
        )

        c_2_2 = (
            R_hm * b_c * rp_hm / Kt_hm + Kemf_hm / rp_hm + R_hm * b_hm / (Kt_hm * rp_hm)
        )

        # Update matrix D
        d_1 = 2 * L_tm * m_c * rp_tm * l[i] * np.sin(theta[i]) * theta_dot[i] / Kt_tm
        d_2 = -2 * L_hm * m_c * rp_hm * l[i] * theta_dot[i] / Kt_hm

        # Update matrix E
        e_1 = (
            L_tm * m_c * rp_tm * l[i] * np.cos(theta[i]) * theta_dot[i] ** 2 / Kt_tm
            + L_tm * g * m_c * rp_tm * np.cos(2 * theta[i]) / Kt_tm
            + L_tm * m_c * rp_tm * np.sin(theta[i]) * l_dot[i] * theta_dot[i] / Kt_tm
            + R_tm * m_c * rp_tm * l[i] * np.sin(theta[i]) * theta_dot[i] / Kt_tm
        )

        e_2 = (
            +L_hm * g * m_c * rp_hm * np.sin(theta[i]) / Kt_hm
            - L_hm * m_c * rp_hm * l_dot[i] * theta_dot[i] / Kt_hm
            - R_hm * m_c * rp_hm * l[i] * theta_dot[i] / Kt_hm
        )

        # Update matrix F
        f_1 = R_tm * g * m_c * rp_tm * np.sin(theta[i]) * np.cos(theta[i]) / Kt_tm

        f_2 = -R_hm * g * m_c * rp_hm * np.cos(theta[i]) / Kt_hm

        temp_x_triple_dot = (
            Ux[i]
            - b_1_1 * x_dot_dot[i]
            - b_1_2 * l_dot_dot[i]
            - c_1_1 * x_dot[i]
            - d_1 * theta_dot_dot[i]
            - e_1 * theta_dot[i]
            - f_1
        )
        temp_l_triple_dot = (
            Ul[i]
            - b_2_1 * x_dot_dot[i]
            - b_2_2 * l_dot_dot[i]
            - c_2_2 * l_dot[i]
            - d_2 * theta_dot_dot[i]
            - e_2 * theta_dot[i]
            - f_2
        )

        determinant = a_1_1 * a_2_2 - a_1_2 * a_2_1

        x_triple_dot = (
            (a_2_2 * temp_x_triple_dot - a_1_2 * temp_l_triple_dot) * 1 / determinant
        )
        l_triple_dot = (
            (a_1_1 * temp_l_triple_dot - a_2_1 * temp_x_triple_dot) * 1 / determinant
        )

        if (
            (abs(Ux[i]) < bias_tm)
            and (abs(x_dot_dot[i]) < 1e-6)
            and (abs(x_dot[i]) < 1e-6)
        ):
            x_triple_dot = 0

        if (
            (abs(Ul[i]) < bias_hm)
            and (abs(l_dot_dot[i]) < 1e-6)
            and (abs(l_dot[i]) < 1e-6)
        ):
            l_triple_dot = 0

        x_dot_dot[i + 1] = x_dot_dot[i] + x_triple_dot * dt

        l_dot_dot[i + 1] = l_dot_dot[i] + l_triple_dot * dt

        theta_dot_dot_temp_ = (
            np.cos(theta[i]) * x_dot_dot[i]
            - 2 * l_dot[i] * theta_dot[i]
            - np.sin(theta[i]) * g
        ) / l[i]
        theta_dot_dot[i + 1] = theta_dot_dot_temp_

        x_dot[i + 1] = x_dot[i] + x_dot_dot[i + 1] * dt

        l_dot[i + 1] = l_dot[i] + l_dot_dot[i + 1] * dt

        theta_dot[i + 1] = theta_dot[i] + theta_dot_dot[i + 1] * dt

        x[i + 1] = x[i] + x_dot[i + 1] * dt

        l[i + 1] = l[i] + l_dot[i + 1] * dt

        theta[i + 1] = theta[i] + theta_dot[i + 1] * dt

    return (
        x,
        l,
        theta,
        x_dot,
        l_dot,
        theta_dot,
        x_dot_dot,
        l_dot_dot,
        theta_dot_dot,
        Ux,
        Ul,
    )


class Simulator:
    def __init__(self, dt, num_steps):
        self.simulation_successful = False
        self.dt = dt
        self.num_steps = num_steps

    def set_parameters(self, parameters):
        self.m_c = 1e-3
        self.g = 1e-3
        self.L_tm = 1e-3
        self.L_hm = 1e-3
        self.R_tm = 1e-3
        self.R_hm = 1e-3
        self.rp_tm = 1e-3
        self.rp_hm = 1e-3
        self.m_t = 1e-3
        self.b_t = 1e-3
        self.b_c = 1e-3
        self.b_tm = 1e-3
        self.b_hm = 1e-3
        self.J_tm = 1e-3
        self.J_hm = 1e-3
        self.Kemf_tm = 1e-3
        self.Kemf_hm = 1e-3
        self.Kt_tm = 1e-3
        self.Kt_hm = 1e-3
        self.bias_tm = 1e-3
        self.bias_hm = 1e-3
        self.supply_voltage = 12
        self.max_pwm = 1023

        if parameters is not None:
            for parameter in parameters:
                # Measured parameters
                if parameter == "container_mass":
                    self.m_c = parameters["container_mass"]["value"]

                if parameter == "gravity_acceleration":
                    self.g = parameters["gravity_acceleration"]["value"]

                if parameter == "trolley_motor_inductance":
                    self.L_tm = parameters["trolley_motor_inductance"]["value"]

                if parameter == "trolley_motor_resistance":
                    self.R_tm = parameters["trolley_motor_resistance"]["value"]

                if parameter == "trolley_motor_pulley_radius":
                    self.rp_tm = parameters["trolley_motor_pulley_radius"]["value"]

                if parameter == "hoist_motor_inductance":
                    self.L_hm = parameters["hoist_motor_inductance"]["value"]

                if parameter == "hoist_motor_resistance":
                    self.R_hm = parameters["hoist_motor_resistance"]["value"]

                if parameter == "hoist_motor_pulley_radius":
                    self.rp_hm = parameters["hoist_motor_pulley_radius"]["value"]

                # Not measured parameters
                if parameter == "trolley_mass":
                    self.m_t = parameters["trolley_mass"]["value"]

                if parameter == "trolley_damping_coefficient":
                    self.b_t = parameters["trolley_damping_coefficient"]["value"]

                if parameter == "cable_damping_coefficient":
                    self.b_c = parameters["cable_damping_coefficient"]["value"]

                if parameter == "trolley_motor_rotator_inertia":
                    self.J_tm = parameters["trolley_motor_rotator_inertia"]["value"]

                if parameter == "trolley_motor_damping_coefficient":
                    self.b_tm = parameters["trolley_motor_damping_coefficient"]["value"]

                if parameter == "trolley_motor_back_emf_constant":
                    self.Kemf_tm = parameters["trolley_motor_back_emf_constant"][
                        "value"
                    ]

                if parameter == "trolley_motor_torque_constant":
                    self.Kt_tm = parameters["trolley_motor_torque_constant"]["value"]

                if parameter == "hoist_motor_rotator_inertia":
                    self.J_hm = parameters["hoist_motor_rotator_inertia"]["value"]

                if parameter == "hoist_motor_damping_coefficient":
                    self.b_hm = parameters["hoist_motor_damping_coefficient"]["value"]

                if parameter == "hoist_motor_back_emf_constant":
                    self.Kemf_hm = parameters["hoist_motor_back_emf_constant"]["value"]

                if parameter == "hoist_motor_torque_constant":
                    self.Kt_hm = parameters["hoist_motor_torque_constant"]["value"]

                if parameter == "trolley_motor_bias":
                    self.bias_tm = parameters["trolley_motor_bias"]["value"]

                if parameter == "hoist_motor_bias":
                    self.bias_hm = parameters["hoist_motor_bias"]["value"]

        else:
            raise ValueError("Parameters are not provided")

    def set_sliding_mode_parameters(self, parameters):
        self.alpha1 = 1e-3
        self.alpha2 = 1e-3
        self.beta1 = 1e-3
        self.beta2 = 1e-3
        self.lambda1 = 1e-3
        self.lambda2 = 1e-3
        self.k1 = 1e-3
        self.k2 = 1e-3

        if parameters is not None:
            for parameter in parameters:
                if parameter == "alpha1":
                    self.alpha1 = parameters["alpha1"]["value"]

                if parameter == "alpha2":
                    self.alpha2 = parameters["alpha2"]["value"]

                if parameter == "beta1":
                    self.beta1 = parameters["beta1"]["value"]

                if parameter == "beta2":
                    self.beta2 = parameters["beta2"]["value"]

                if parameter == "lambda1":
                    self.lambda1 = parameters["lambda1"]["value"]

                if parameter == "lambda2":
                    self.lambda2 = parameters["lambda2"]["value"]

                if parameter == "k1":
                    self.k1 = parameters["k1"]["value"]

                if parameter == "k2":
                    self.k2 = parameters["k2"]["value"]
    
    def set_variables(self, initial_conditions):
        # Initial conditions
        self.x = np.zeros(self.num_steps)
        self.x_dot = np.zeros(self.num_steps)
        self.x_dot_dot = np.zeros(self.num_steps)
        self.l = np.zeros(self.num_steps)
        self.l_dot = np.zeros(self.num_steps)
        self.l_dot_dot = np.zeros(self.num_steps)
        self.theta = np.zeros(self.num_steps)
        self.theta_dot = np.zeros(self.num_steps)
        self.theta_dot_dot = np.zeros(self.num_steps)

        self.PWMx = np.zeros(self.num_steps)
        self.PWMl = np.zeros(self.num_steps)
        self.Ux = np.zeros(self.num_steps)
        self.Ul = np.zeros(self.num_steps)

        if initial_conditions is not None:
            for variable in initial_conditions:
                if variable == "x":
                    self.x = [initial_conditions[variable]]
                if variable == "l":
                    self.l = [initial_conditions[variable]]
                if variable == "theta":
                    self.theta = [initial_conditions[variable]]
        else:
            print("Initial conditions are not provided")
            print("Setting initial conditions to default values...")
            self.l[0] = 1.0  # Cable length can never be zero

    def simulate(
        self,
        parameters,
        trolley_motor_input_PWM,
        hoist_motor_input_PWM,
        initial_conditions=None,
    ):
        self.set_variables(initial_conditions)
        self.set_parameters(parameters)

        self.PWMx = trolley_motor_input_PWM
        self.PWMl = hoist_motor_input_PWM

        variable_initial_conditions = [self.x[0], self.l[0], self.theta[0]]

        self.simulation_successful = True
        try:
            (
                self.x,
                self.l,
                self.theta,
                self.x_dot,
                self.l_dot,
                self.theta_dot,
                self.x_dot_dot,
                self.l_dot_dot,
                self.theta_dot_dot,
                self.Ux,
                self.Ul,
            ) = simulate_core(
                self.num_steps - 1,
                self.dt,
                self.L_tm,
                self.L_hm,
                self.rp_tm,
                self.rp_hm,
                self.m_t,
                self.m_c,
                self.J_tm,
                self.J_hm,
                self.Kt_tm,
                self.Kt_hm,
                self.Kemf_tm,
                self.Kemf_hm,
                self.R_tm,
                self.R_hm,
                self.b_t,
                self.b_c,
                self.b_tm,
                self.b_hm,
                self.g,
                self.bias_tm,
                self.bias_hm,
                self.supply_voltage,
                self.max_pwm,
                trolley_motor_input_PWM,
                hoist_motor_input_PWM,
                variable_initial_conditions,
            )

        except Exception as e:
            print(f"An error occurred: {e}")
            self.simulation_successful = False

    def simulate_legacy(self, parameters, inputs, initial_conditions=None):
        self.set_variables(initial_conditions)
        self.set_parameters(parameters)
        dt = self.dt
        if inputs is not None:
            for input in inputs:
                if input == "trolley_motor_voltage":
                    self.Ux = inputs[input]
                if input == "hoist_motor_voltage":
                    self.Ul = inputs[input]

        self.matrix_A = np.matrix([[0.0, 0.0], [0.0, 0.0]])
        self.matrix_B = np.matrix([[0.0, 0.0], [0.0, 0.0]])
        self.matrix_C = np.matrix([[0.0, 0.0], [0.0, 0.0]])
        self.matrix_D = np.matrix([[0.0], [0.0]])
        self.matrix_E = np.matrix([[0.0], [0.0]])
        self.matrix_F = np.matrix([[0.0], [0.0]])

        self.simulation_successful = True
        for i in range(self.num_steps - 1):
            # print(f"Iteration: {i}", end="\r", flush=True)
            try:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("error", RuntimeWarning)

                    # Update matrix A
                    self.matrix_A[0, 0] = self.L_tm * self.rp_tm * (
                        self.m_t + self.m_c * np.sin(self.theta[i]) ** 2
                    ) / self.Kt_tm + self.J_tm * self.L_tm / (self.Kt_tm * self.rp_tm)
                    self.matrix_A[0, 1] = (
                        -self.L_tm
                        * self.m_c
                        * self.rp_tm
                        * np.sin(self.theta[i])
                        / self.Kt_tm
                    )
                    self.matrix_A[1, 0] = (
                        -self.L_hm
                        * self.m_c
                        * self.rp_hm
                        * np.sin(self.theta[i])
                        / self.Kt_hm
                    )
                    self.matrix_A[1, 1] = (
                        self.L_hm * self.m_c * self.rp_hm / self.Kt_hm
                        + self.J_hm * self.L_hm / (self.Kt_hm * self.rp_hm)
                    )

                    # Update matrix B
                    self.matrix_B[0, 0] = (
                        +self.L_tm
                        * self.m_c
                        * self.rp_tm
                        * np.sin(2 * self.theta[i])
                        * self.theta_dot[i]
                        / self.Kt_tm
                        + self.R_tm
                        * self.rp_tm
                        * (self.m_t + self.m_c * np.sin(self.theta[i]) ** 2)
                        / self.Kt_tm
                        + self.L_tm * self.b_t * self.rp_tm / self.Kt_tm
                        + self.J_tm * self.R_tm / (self.Kt_tm * self.rp_tm)
                        + self.L_tm * self.b_tm / (self.Kt_tm * self.rp_tm)
                    )
                    self.matrix_B[0, 1] = (
                        -self.L_tm
                        * self.m_c
                        * self.rp_tm
                        * np.cos(self.theta[i])
                        * self.theta_dot[i]
                        / self.Kt_tm
                        - self.R_tm
                        * self.m_c
                        * self.rp_tm
                        * np.sin(self.theta[i])
                        / self.Kt_tm
                    )
                    self.matrix_B[1, 0] = (
                        -self.L_hm
                        * self.m_c
                        * self.rp_hm
                        * np.cos(self.theta[i])
                        * self.theta_dot[i]
                        / self.Kt_hm
                        - self.R_hm
                        * self.m_c
                        * self.rp_hm
                        * np.sin(self.theta[i])
                        / self.Kt_hm
                    )
                    self.matrix_B[1, 1] = (
                        +self.L_hm * self.b_c * self.rp_hm / self.Kt_hm
                        + self.R_hm * self.m_c * self.rp_hm / self.Kt_hm
                        + self.J_hm * self.R_hm / (self.Kt_hm * self.rp_hm)
                        + self.L_hm * self.b_hm / (self.Kt_hm * self.rp_hm)
                    )

                    # Update matrix C
                    self.matrix_C[0, 0] = (
                        self.R_tm * self.b_t * self.rp_tm / self.Kt_tm
                        + self.Kemf_tm / self.rp_tm
                        + self.R_tm * self.b_tm / (self.Kt_tm * self.rp_tm)
                    )
                    self.matrix_C[1, 1] = (
                        self.R_hm * self.b_c * self.rp_hm / self.Kt_hm
                        + self.Kemf_hm / self.rp_hm
                        + self.R_hm * self.b_hm / (self.Kt_hm * self.rp_hm)
                    )

                    # Update matrix D
                    self.matrix_D[0, 0] = (
                        2
                        * self.L_tm
                        * self.m_c
                        * self.rp_tm
                        * self.l[i]
                        * np.sin(self.theta[i])
                        * self.theta_dot[i]
                        / self.Kt_tm
                    )
                    self.matrix_D[1, 0] = (
                        -2
                        * self.L_hm
                        * self.m_c
                        * self.rp_hm
                        * self.l[i]
                        * self.theta_dot[i]
                        / self.Kt_hm
                    )

                    # Update matrix E
                    self.matrix_E[0, 0] = (
                        self.L_tm
                        * self.m_c
                        * self.rp_tm
                        * self.l[i]
                        * np.cos(self.theta[i])
                        * self.theta_dot[i] ** 2
                        / self.Kt_tm
                        + self.L_tm
                        * self.g
                        * self.m_c
                        * self.rp_tm
                        * np.cos(2 * self.theta[i])
                        / self.Kt_tm
                        + self.L_tm
                        * self.m_c
                        * self.rp_tm
                        * np.sin(self.theta[i])
                        * self.l_dot[i]
                        * self.theta_dot[i]
                        / self.Kt_tm
                        + self.R_tm
                        * self.m_c
                        * self.rp_tm
                        * self.l[i]
                        * np.sin(self.theta[i])
                        * self.theta_dot[i]
                        / self.Kt_tm
                    )
                    self.matrix_E[1, 0] = (
                        +self.L_hm
                        * self.g
                        * self.m_c
                        * self.rp_hm
                        * np.sin(self.theta[i])
                        / self.Kt_hm
                        - self.L_hm
                        * self.m_c
                        * self.rp_hm
                        * self.l_dot[i]
                        * self.theta_dot[i]
                        / self.Kt_hm
                        - self.R_hm
                        * self.m_c
                        * self.rp_hm
                        * self.l[i]
                        * self.theta_dot[i]
                        / self.Kt_hm
                    )

                    # Update matrix F
                    self.matrix_F[0, 0] = (
                        self.R_tm
                        * self.g
                        * self.m_c
                        * self.rp_tm
                        * np.sin(self.theta[i])
                        * np.cos(self.theta[i])
                        / self.Kt_tm
                    )
                    self.matrix_F[1, 0] = (
                        -self.R_hm
                        * self.g
                        * self.m_c
                        * self.rp_hm
                        * np.cos(self.theta[i])
                        / self.Kt_hm
                    )

                    # Create variable for q, q_dot, q_dot_dot, and q_desired
                    q_now = np.matrix([[self.x[i]], [self.l[i]]])
                    q_dot_now = np.matrix([[self.x_dot[i]], [self.l_dot[i]]])
                    q_dot_dot_now = np.matrix(
                        [[self.x_dot_dot[i]], [self.l_dot_dot[i]]]
                    )

                    # Create variable for control input
                    control_now = np.matrix([[self.Ux[i]], [self.Ul[i]]])

                    # Update self.x_dot_dot, self.l_dot_dot, self.theta_dot_dot using state space equations
                    q_triple_dot_now = np.matmul(
                        np.linalg.inv(self.matrix_A),
                        (
                            control_now
                            - (
                                self.matrix_B * q_dot_dot_now
                                + self.matrix_C * q_dot_now
                                + self.matrix_D * self.theta_dot_dot[i]
                                + self.matrix_E * self.theta_dot[i]
                                + self.matrix_F
                            )
                        ),
                    )

                    if (control_now[0, 0] < self.bias_tm) and (
                        abs(self.x_dot[i]) < 1e-6
                    ):
                        q_triple_dot_now[0, 0] = 0

                    if (control_now[1, 0] < self.bias_hm) and (
                        abs(self.l_dot[i]) < 1e-6
                    ):
                        q_triple_dot_now[1, 0] = 0

                    self.x_dot_dot[i + 1] = (
                        self.x_dot_dot[i] + q_triple_dot_now[0, 0] * dt
                    )

                    self.l_dot_dot[i + 1] = (
                        self.l_dot_dot[i] + q_triple_dot_now[1, 0] * dt
                    )

                    theta_dot_dot_temp_ = (
                        np.cos(self.theta[i]) * self.x_dot_dot[i]
                        - 2 * self.l_dot[i] * self.theta_dot[i]
                        - np.sin(self.theta[i]) * self.g
                    ) / self.l[i]
                    self.theta_dot_dot[i + 1] = theta_dot_dot_temp_

                    self.x_dot[i + 1] = self.x_dot[i] + self.x_dot_dot[i + 1] * dt

                    self.l_dot[i + 1] = self.l_dot[i] + self.l_dot_dot[i + 1] * dt

                    self.theta_dot[i + 1] = (
                        self.theta_dot[i] + self.theta_dot_dot[i + 1] * dt
                    )

                    self.x[i + 1] = self.x[i] + self.x_dot[i + 1] * dt

                    self.l[i + 1] = self.l[i] + self.l_dot[i + 1] * dt

                    self.theta[i + 1] = self.theta[i] + self.theta_dot[i + 1] * dt

                    if self.check_divergence():
                        print("Divergence detected at step", i)
                        self.simulation_successful = False
                        break

            except RuntimeWarning as e:
                print(f"Caught a runtime warning at step {i}: {e}")
                self.simulation_successful = False
                break
            except Exception as e:
                print(f"An error occurred at step {i}: {e}")
                self.simulation_successful = False
                break

    def create_dataframe(self):
        if self.simulation_successful:
            print("Creating dataframe...")
            print(
                f"Data length: {self.num_steps}, {len(self.x)}, {len(self.x_dot)}, {len(self.x_dot_dot)}, {len(self.l)}, {len(self.l_dot)}, {len(self.l_dot_dot)}, {len(self.theta)}, {len(self.theta_dot)}, {len(self.theta_dot_dot)}, {len(self.PWMx)}, {len(self.PWMl)}, {len(self.Ux)}, {len(self.Ul)}"
            )
            data = {
                "time": np.arange(0, self.num_steps * self.dt, self.dt),
                "trolley_position": self.x,
                "trolley_position_first_derivative": self.x_dot,
                "trolley_position_second_derivative": self.x_dot_dot,
                "cable_length": self.l,
                "cable_length_first_derivative": self.l_dot,
                "cable_length_second_derivative": self.l_dot_dot,
                "sway_angle": self.theta,
                "sway_angle_first_derivative": self.theta_dot,
                "sway_angle_second_derivative": self.theta_dot_dot,
                "trolley_motor_pwm": self.PWMx,
                "hoist_motor_pwm": self.PWMl,
                "trolley_motor_voltage": self.Ux,
                "hoist_motor_voltage": self.Ul,
            }
            return pd.DataFrame(data)
        else:
            return None

    def get_results(self):
        if self.simulation_successful:
            data = {
                "time": np.arange(0, self.num_steps * self.dt, self.dt),
                "trolley_position": np.array(self.x),
                "cable_length": np.array(self.l),
                "sway_angle": np.array(self.theta),
                "trolley_speed": np.array(self.x_dot),
                "cable_speed": np.array(self.l_dot),
                "trolley_motor_pwm": np.array(self.PWMx),
                "hoist_motor_pwm": np.array(self.PWMl),
                "trolley_motor_voltage": np.array(self.Ux),
                "hoist_motor_voltage": np.array(self.Ul),
            }
            return data
        else:
            data = {}
            return data

    def check_divergence(self):
        if (
            np.isnan(self.x).any()
            or np.isnan(self.x_dot).any()
            or np.isnan(self.x_dot_dot).any()
            or np.isnan(self.l).any()
            or np.isnan(self.l_dot).any()
            or np.isnan(self.l_dot_dot).any()
            or np.isnan(self.theta).any()
            or np.isnan(self.theta_dot).any()
            or np.isnan(self.theta_dot_dot).any()
        ):
            return True
        else:
            return False
