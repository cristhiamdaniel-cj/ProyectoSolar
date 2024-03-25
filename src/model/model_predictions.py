import os
import numpy as np
import pandas as pd
from scipy.optimize import brentq
import warnings
from src.model.pv_parameters import PVModel
import logging

# Configuración del logger para incluir timestamp.
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
log = logging.getLogger(__name__)

# Ignorar warnings de overflow y operaciones inválidas durante los cálculos.
warnings.filterwarnings('ignore', category=RuntimeWarning)

class PVDataGenerator(PVModel):
    def generate_data(self, G, T, n_points=1000):
        T_k = T + 273.15  # Conversión de Celsius a Kelvin
        Vpv = np.linspace(0, self.V_oc, n_points)
        Ipv = np.zeros_like(Vpv)
        Ppv = np.zeros_like(Vpv)

        for i, V in enumerate(Vpv):
            f = lambda I: self.pv_model(G, T_k)(V, I)
            Ipv[i] = brentq(f, -self.I_sc * 2, self.I_sc * 2)
            Ppv[i] = V * Ipv[i]

        max_power_idx = np.argmax(Ppv)
        Vmpp = Vpv[max_power_idx]
        Impp = Ipv[max_power_idx]
        P_max = Ppv[max_power_idx]
        log.info(f"Punto de máxima potencia: Vmpp={Vmpp}, Impp={Impp}, Pmax={P_max}")

        df = pd.DataFrame({'T': np.full_like(Vpv, T_k), 'G': np.full_like(Vpv, G), 'V': Vpv, 'I': Ipv, 'P': Ppv})

        # Asegurar la creación del directorio si no existe.
        data_dir = "../Datos/raw_data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        filename = os.path.join(data_dir, f"datos_pv_G{G}_T{T}.csv")
        df.to_csv(filename, index=False)
        log.info(f"Archivo '{filename}' creado con éxito.")

if __name__ == "__main__":
    temperatures = range(15, 46)  # De 15°C a 45°C
    irradiances = range(300, 1001)  # De 300 W/m² a 1000 W/m²

    modelo = PVDataGenerator(I_sc=9.35, V_oc=47.4, N_s=72)

    for T in temperatures:
        for G in irradiances:
            modelo.generate_data(G, T, n_points=1000)
            log.info(f"Datos generados para G={G} W/m² y T={T}°C.")
