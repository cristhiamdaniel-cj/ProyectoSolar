import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, newton, minimize, root, brentq, bisect
from pv_model import PVModel
import time

class PVModelComplexity(PVModel):
    def pv_model_method(self, G, T, method='fsolve', n=100):
        I_rs = self.I_sc / (np.exp((self.q * self.V_oc) / (self.n * self.N_s * self.K * T)) - 1)
        I_o = I_rs * (T / self.T_n) ** 3 * np.exp((self.q * self.E_g0 * (1 / T - 1 / self.T_n)) / (self.n * self.K))
        I_ph = (self.I_sc + self.k_i * (T - self.T_n)) * (G / 1000)
        Vpv = np.linspace(0, self.V_oc, n)
        Ipv = np.zeros_like(Vpv)
        Ppv = np.zeros_like(Vpv)

        for i, V in enumerate(Vpv):
            func = lambda I: self.f(I, V, I_ph, I_o, T)
            if method == 'fsolve':
                Ipv[i] = fsolve(func, self.I_sc)[0]
            elif method == 'newton':
                Ipv[i] = newton(func, self.I_sc)
            elif method == 'minimize':
                res = minimize(lambda I: func(I)**2, self.I_sc)
                Ipv[i] = res.x[0]
            elif method == 'root':
                sol = root(func, self.I_sc)
                if sol.success:
                    Ipv[i] = sol.x[0]
            elif method == 'brentq':
                try:
                    Ipv[i] = brentq(func, -self.I_sc, self.I_sc)
                except ValueError:
                    Ipv[i] = np.nan
            elif method == 'bisect':
                try:
                    Ipv[i] = bisect(func, -self.I_sc, self.I_sc)
                except ValueError:
                    Ipv[i] = np.nan
            Ppv[i] = V * Ipv[i]

        max_power_idx = np.argmax(Ppv)
        Vmpp = Vpv[max_power_idx]
        Impp = Ipv[max_power_idx]
        P_max = Ppv[max_power_idx]

        return Vmpp, Impp, P_max

    def f(self, I, V, I_ph, I_o, T):
        return I_ph - I_o * (np.exp((self.q * (V + I * self.R_s)) / (self.n * self.K * self.N_s * T)) - 1) - (V + I * self.R_s) / self.R_sh - I

# Ejemplo de uso y análisis de complejidad algorítmica
if __name__ == "__main__":
    # Verificar si el directorio existe
    graficas_dir = './Graficas'
    if not os.path.exists(graficas_dir):
        os.makedirs(graficas_dir)

    modelo = PVModelComplexity()
    G = 1000  # Irradiancia en W/m²
    T = 298  # Temperatura en K
    metodos = ['fsolve', 'newton', 'minimize', 'root', 'brentq', 'bisect']
    n_values = []
    for i in range(1, 101):
        n_values.append(10*i)

    tiempos = {metodo: [] for metodo in metodos}

    for n in n_values:
        print(f"Analizando para n={n}")
        for metodo in metodos:
            inicio = time.time()
            modelo.pv_model_method(G, T, metodo, n)
            fin = time.time()
            tiempo = fin - inicio
            tiempos[metodo].append(tiempo)
            print(f"Metodo: {metodo}, Tiempo: {tiempo}, Punto de maxima potencia: {modelo.pv_model_method(G, T, metodo, n)}")

    # Graficar los resultados
    plt.figure(figsize=(10, 6))
    for metodo in metodos:
        plt.plot(n_values, tiempos[metodo], label=metodo)
    plt.xlabel('Número de puntos (n)')
    plt.ylabel('Tiempo de ejecución (s)')
    plt.title('Complejidad Algorítmica de Métodos Numéricos')
    plt.legend()
    plt.grid()
    plt.savefig(f'{graficas_dir}/complejidad_algoritmica.png')
    plt.show()

# Output:
