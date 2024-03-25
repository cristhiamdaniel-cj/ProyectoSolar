import os
import re

import polars as pl
import logging
from sqlalchemy import create_engine

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directorio donde se encuentran tus archivos CSV
input_directory = 'C:/Users/cristhiamcampos/Documents/Datos/raw_data'

# Ruta para el archivo de salida
output_parquet = 'C:/Users/cristhiamcampos/Documents/Datos/combined_data.parquet'
output_db = 'sqlite:///C:/Users/cristhiamcampos/Documents/Datos/pv_data.db'

# Lista para almacenar los DataFrames
dataframes = []

# Iterar sobre todos los archivos CSV en el directorio
for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        # Extraer G y T del nombre del archivo usando expresiones regulares
        match = re.match(r"datos_pv_G(\d+)_T(\d+).csv", filename)
        if match:
            G, T = map(int, match.groups())

            # Leer el archivo CSV con Polars
            df = pl.read_csv(os.path.join(input_directory, filename))
            # Añadir las columnas G y T al DataFrame
            df = df.with_columns([
                pl.lit(T).alias("T"),
                pl.lit(G).alias("G")
            ])
            logger.info(f"Leído archivo {filename} con T={T} y G={G}")

            # Añadir el DataFrame a la lista
            dataframes.append(df)

# Concatenar todos los DataFrames en uno solo
if dataframes:
    combined_df = pl.concat(dataframes)
    logger.info("Todos los DataFrames han sido concatenados.")

    # Guardar el DataFrame combinado en un archivo Parquet
    combined_df.write_parquet(output_parquet)
    logger.info(f"Los datos combinados han sido guardados en {output_parquet}.")

    # Guardar el DataFrame combinado en SQLite
    engine = create_engine(output_db)
    combined_df.to_pandas().to_sql('pv_data', con=engine, if_exists='replace', index=False)
    logger.info(f"Los datos combinados han sido guardados en SQLite en {output_db}.")
else:
    logger.info("No se encontraron archivos CSV que coincidan con el patrón.")
