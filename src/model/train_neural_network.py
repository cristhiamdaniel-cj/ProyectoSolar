import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging

class PVModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.history = None
        self.scaler = StandardScaler()
        logging.basicConfig(level=logging.INFO)

    def load_data(self):
        try:
            data = pd.read_parquet(self.data_path)
            logging.info("Datos cargados correctamente.")
            return data
        except Exception as e:
            logging.error(f"Error al cargar los datos: {e}")
            exit()

    def prepare_data(self, data):
        try:
            X = data[['T', 'G', 'V']].values
            y = data['I'].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled, y_train, y_test
        except Exception as e:
            logging.error(f"Error al preparar los datos: {e}")
            exit()

    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_model(self, X_train_scaled, y_train):
        try:
            self.history = self.model.fit(X_train_scaled, y_train, epochs=10, validation_split=0.2)
            logging.info("Modelo entrenado correctamente.")
        except Exception as e:
            logging.error(f"Error durante el entrenamiento: {e}")
            exit()

    def evaluate_model(self, X_test_scaled, y_test):
        try:
            loss = self.model.evaluate(X_test_scaled, y_test)
            logging.info(f"Error de test: {loss}")
        except Exception as e:
            logging.error(f"Error al evaluar el modelo: {e}")
            exit()

    def plot_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label='Error de entrenamiento')
        plt.plot(self.history.history['val_loss'], label='Error de validación')
        plt.title('Error del modelo a lo largo de las épocas')
        plt.xlabel('Épocas')
        plt.ylabel('Error')
        plt.legend()
        plt.savefig('error_plot.png')
        plt.show()

    def save_model(self):
        try:
            # Actualiza aquí la ruta donde deseas guardar el modelo
            model_path = 'C:/Users/cristhiamcampos/Documents/Datos/pv_model.h5'
            self.model.save(model_path)
            logging.info(f"Modelo guardado correctamente en {model_path}.")
        except Exception as e:
            logging.error(f"Error al guardar el modelo: {e}")

if __name__ == "__main__":
    # Asegúrate de actualizar esta ruta con la ubicación exacta de tu archivo .parquet
    trainer = PVModelTrainer('C:/Users/cristhiamcampos/Documents/Datos/combined_data.parquet')
    data = trainer.load_data()
    X_train_scaled, X_test_scaled, y_train, y_test = trainer.prepare_data(data)
    trainer.build_model()
    trainer.train_model(X_train_scaled, y_train)
    trainer.evaluate_model(X_test_scaled, y_test)
    trainer.plot_history()
    trainer.save_model()
