from sqlalchemy import create_engine
import pandas as pd
import os
import logging

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define globalmente la ruta base
BASE_PATH = "C:/Users/cristhiamcampos/Documents/Datos/"
DB_PATH = f"sqlite:///{BASE_PATH}/pv_data.db"

def consultar_potencia_maxima(T, G):
    """
    Consulta la potencia máxima para una temperatura T (en Kelvin) y una irradiancia G.
    """
    engine = create_engine(DB_PATH)
    query = f"""
    SELECT T, G, V, I, P
    FROM pv_data
    WHERE T = {T} AND G = {G}
    ORDER BY P DESC
    LIMIT 1;
    """
    df = pd.read_sql_query(query, engine)
    if not df.empty:
        logger.info(f"Potencia máxima para T={T} K y G={G} W/m²: {df.iloc[0]['P']} W")
        logger.info(f"Correspondiente a V={df.iloc[0]['V']} V e I={df.iloc[0]['I']} A")
    else:
        logger.info("No se encontraron datos para los parámetros especificados.")

if __name__ == "__main__":
    T = 25  # 25°C en Kelvin
    G = 1000  # Irradiancia en W/m²
    consultar_potencia_maxima(T, G)
