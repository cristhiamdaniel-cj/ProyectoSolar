import os
import tensorflow as tf
import logging
import json

# Configurar el nivel de logging
logging.basicConfig(level=logging.INFO)


class ModelLoaderSaver:
    def __init__(self, model_filename):
        # Ruta al directorio donde se encuentra este script
        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        # Define la ruta completa del archivo del modelo en el directorio externo
        self.model_path = os.path.join('C:/Users/cristhiamcampos/Documents/Datos', model_filename)
        # Define la ruta dentro del proyecto PyCharm para guardar el archivo JSON
        self.save_path = os.path.join(self.script_dir, 'Data')

    def load_model(self):
        # Cargar el modelo desde el archivo .h5
        try:
            model = tf.keras.models.load_model(self.model_path)
            logging.info("Modelo cargado correctamente desde " + self.model_path)
            return model
        except Exception as e:
            logging.error(f"Error al cargar el modelo: {e}")
            exit()

    def save_model_parameters(self, model):
        # Extraer y guardar los pesos y sesgos de cada capa del modelo
        parameters = {}
        for i, layer in enumerate(model.layers):
            weights, biases = layer.get_weights()
            parameters[f'W{i + 1}'] = weights.tolist()
            parameters[f'b{i + 1}'] = biases.tolist()

        # Ruta completa del archivo JSON donde se guardarán los parámetros
        json_file_path = os.path.join(self.save_path, "model_parameters.json")

        # Crear el directorio 'Data' si no existe
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            logging.info(f"Creado el directorio {self.save_path}.")

        # Guardar los parámetros del modelo en un archivo JSON
        try:
            with open(json_file_path, "w") as json_file:
                json.dump(parameters, json_file)
            logging.info(f"Parámetros del modelo guardados en {json_file_path}.")
        except Exception as e:
            logging.error(f"Error al guardar los parámetros del modelo: {e}")


if __name__ == "__main__":
    model_filename = 'pv_model.h5'
    model_loader_saver = ModelLoaderSaver(model_filename)
    model = model_loader_saver.load_model()
    model_loader_saver.save_model_parameters(model)
