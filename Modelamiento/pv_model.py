import numpy as np

class PVModel:
    """
    Clase para el modelo de un panel fotovoltaico.
    """

    def __init__(self, num_panels_series=1, num_panels_parallel=1):
        self.R_sh = 545.82  # Resistencia en paralelo
        self.k_i = 0.037 # Coeficiente de temperatura de corriente de cortocircuito
        self.T_n = 298 # Temperatura de referencia
        self.q = 1.60217646e-19 # Carga del electrón
        self.n = 1.0 # Factor de idealidad
        self.K = 1.3806503e-23 # Constante de Boltzmann
        self.E_g0 = 1.1 # Energía de banda del silicio
        self.R_s = 0.39 # Resistencia en serie
        self.num_panels_series = num_panels_series # Número de paneles en serie
        self.num_panels_parallel = num_panels_parallel # Número de paneles en paralelo
        self.I_sc = 9.35 * num_panels_parallel # Corriente de cortocircuito
        self.V_oc = 47.4 * num_panels_series # Voltaje de circuito abierto
        self.N_s = 72 * num_panels_series # Número de celdas en serie

    def pv_model(self, G, T):
        """
        Función que calcula el modelo de un panel fotovoltaico. Modelo de los 5 parámetros.
        :param G: Irradiancia
        :param T: Temperatura
        :return: Función del modelo PV
        """

        # Cálculo de I_rs: corriente de saturación inversa
        I_rs = self.I_sc / (np.exp((self.q * self.V_oc) / (self.n * self.N_s * self.K * T)) - 1)
        # Cálculo de I_o: corriente de saturación inversa
        I_o = I_rs * (T / self.T_n) * np.exp((self.q * self.E_g0 * (1 / self.T_n - 1 / T)) / (self.n * self.K))
        # Cálculo de I_ph: corriente fotogenerada
        I_ph = (self.I_sc + self.k_i * (T - 298)) * (G / 1000)

        # Función para la ecuación del modelo PV
        def func(V, I):
            """
            Función que calcula la ecuación del modelo PV.
            :param V: Tensión del panel
            :param I: Corriente del panel
            :return: Ecuación del modelo PV
            """
            return (I_ph - I_o * (np.exp((self.q * (V + I * self.R_s)) / (self.n * self.K * self.N_s * T)) - 1) -
                    (V + I * self.R_s) / self.R_sh - I)

        return func
