import os
import time
import numpy as np
from scipy.optimize import fsolve, newton, root, brentq, bisect
import matplotlib.pyplot as plt
from src.model.pv_parameters import PVModel

class PVModelComplexityAnalysis:
    def __init__(self, model):
        self.model = model

    def analyze_methods(self, G, T, methods, n_points):
        results = {}
        max_powers = {}  # Almacenar los puntos de máxima potencia para cada método

        for method in methods:
            times = []
            max_power_per_method = []

            for n in n_points:
                V = np.linspace(0, self.model.V_oc, n)
                I = np.zeros(n)
                P = np.zeros(n)
                start_time = time.perf_counter()

                for i, v in enumerate(V):
                    func = self.model.pv_model(G, T)
                    try:
                        if method == 'fsolve':
                            I[i] = fsolve(lambda I: func(v, I), 0)[0]
                        elif method == 'newton':
                            I[i] = newton(lambda I: func(v, I), 0)
                        elif method == 'root':
                            I[i] = root(lambda I: func(v, I), 0).x[0]
                        # elif method == 'minimize':
                        #     res = minimize(lambda I: func(v, I)**2, 0)
                        #     I[i] = res.x[0]
                        elif method == 'brentq':
                            I[i] = brentq(lambda I: func(v, I), -self.model.I_sc, self.model.I_sc)
                        elif method == 'bisect':
                            I[i] = bisect(lambda I: func(v, I), -self.model.I_sc, self.model.I_sc)
                        P[i] = v * I[i]
                    except Exception as e:
                        print(f"Error en el método {method} para V={v}: {e}")

                max_power_idx = np.argmax(P)
                max_power_per_method.append((V[max_power_idx], I[max_power_idx], P[max_power_idx]))

                times.append(time.perf_counter() - start_time)

            results[method] = times
            max_powers[method] = max_power_per_method

        return results, max_powers

if __name__ == "__main__":
    model = PVModel(9.35, 47.4, 72)
    analysis = PVModelComplexityAnalysis(model)
    G = 1000  # Irradiancia en W/m^2
    T = 298  # Temperatura en K
    methods = ['fsolve', 'newton', 'root', 'brentq', 'bisect']  # 'minimize' se omitió por simplicidad
    n_points = list(range(10, 1001, 10))

    # Ruta para guardar la imagen
    if not os.path.exists('../../Graficas'):
        os.makedirs('../../Graficas')
    os.chdir('../../Graficas')

    results, max_powers = analysis.analyze_methods(G, T, methods, n_points)

    # Plotting
    plt.figure(figsize=(10, 6))
    for method, times in results.items():
        plt.plot(n_points, times, label=method)
        print(f"Max power for {method}: {max_powers[method][-1]}")  # Imprimir el último punto de máxima potencia calculado para cada método

    plt.xlabel('Número de puntos (n)')
    plt.ylabel('Tiempo de ejecución (s)')
    plt.title('Análisis de Complejidad de Métodos Numéricos')
    plt.legend()
    plt.grid()
    plt.savefig('complejidad_metodos.png')
    plt.show()

