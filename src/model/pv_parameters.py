import numpy as np

class PVModel:
    """
    Clase para el modelo de un panel fotovoltaico.
    """
    def __init__(self, I_sc, V_oc, N_s, R_s=0.39, R_sh=545.82, k_i=0.037, T_n=298, q=1.60217646e-19, n=1.0, K=1.3806503e-23, E_g0=1.1):
        self.I_sc = I_sc # Corriente de cortocircuito en A
        self.V_oc = V_oc # Voltaje de circuito abierto en V
        self.N_s = N_s # Número de celdas en serie
        self.R_s = R_s # Resistencia serie en ohmios
        self.R_sh = R_sh # Resistencia paralelo en ohmios
        self.k_i = k_i # Coeficiente de corriente de cortocircuito en A/K
        self.T_n = T_n # Temperatura nominal en K
        self.q = q # Carga del electrón en C
        self.n = n # Factor de idealidad
        self.K = K # Constante de Boltzmann en J/K
        self.E_g0 = E_g0 # Energía de banda a 0 K en eV

    def pv_model(self, G, T):
        """
        Modelo de panel fotovoltaico.
        :param G:  Irradiancia en W/m²
        :param T:  Temperatura en K
        :return:  Función de corriente en función del voltaje
        """
        I_rs = self.I_sc / (np.exp((self.q * self.V_oc) / (self.n * self.N_s * self.K * T)) - 1)
        I_o = I_rs * (T / self.T_n) ** 3 * np.exp((self.q * self.E_g0 * (1 / T - 1 / self.T_n)) / (self.n * self.K))
        I_ph = (self.I_sc + self.k_i * (T - self.T_n)) * (G / 1000)

        def func(V, I):
            return I_ph - I_o * (np.exp((self.q * (V + I * self.R_s)) / (self.n * self.K * self.N_s * T)) - 1) - (V + I * self.R_s) / self.R_sh - I

        return func
