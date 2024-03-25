# Sistema de Predicción para Paneles Fotovoltaicos

Este proyecto desarrolla y utiliza modelos para predecir el comportamiento de paneles fotovoltaicos bajo diversas condiciones. Se utiliza un enfoque basado en el modelo de los cinco parámetros para generar datos simulados, que luego alimentan una red neuronal para realizar predicciones precisas.

## Estructura del Proyecto

El proyecto está organizado en varios subdirectorios, cada uno con un propósito específico:

```txt
src/
│
├── data_preparation/
│ ├── compile_data.py - Compila datos CSV en formatos .parquet y SQLite.
│ └── sqlite_query.py - Realiza consultas a la base de datos SQLite.
│
├── model/
│ ├── pv_parameters.py - Define parámetros y modelos del panel fotovoltaico.
│ ├── model_analysis.py - Analiza diferentes métodos de modelado.
│ ├── model_predictions.py - Genera predicciones o cálculos del modelo.
│ ├── train_neural_network.py - Entrena la red neuronal para predicciones.
│ └── generate_model_json.py - Genera una representación JSON del modelo.
│
└── visualization/
└── visualization_tools.py - Herramientas para visualizar datos y resultados (sugerido).
```


### Descripción de los Scripts

- **compile_data.py**: Combina múltiples archivos CSV de datos de paneles fotovoltaicos en una sola base de datos SQLite y un archivo .parquet para análisis y entrenamiento del modelo.

- **sqlite_query.py**: Facilita la extracción de datos específicos de la base de datos SQLite para análisis adicional o visualización.

- **pv_parameters.py**: Contiene la definición del modelo de los cinco parámetros del panel fotovoltaico y funciones asociadas para calcular la corriente y potencia bajo diferentes condiciones de irradiación y temperatura.

- **model_analysis.py**: Realiza un análisis comparativo de diferentes métodos numéricos para resolver el modelo de los cinco parámetros, ayudando a seleccionar el método más eficiente para generar los datos de entrenamiento.

- **model_predictions.py**: Utiliza los parámetros del panel fotovoltaico para generar un conjunto de datos de simulación que representa el comportamiento del panel bajo variadas condiciones.

- **train_neural_network.py**: Entrena una red neuronal con los datos generados para predecir la corriente de salida del panel fotovoltaico basándose en la irradiación, temperatura y voltaje.

- **generate_model_json.py**: Exporta la arquitectura entrenada del modelo, junto con sus pesos y sesgos, a un archivo JSON para su uso en aplicaciones web o en otros entornos que requieran del modelo.

## Uso

Describir brevemente cómo utilizar cada script o realizar tareas comunes dentro del proyecto, como preparar datos, entrenar el modelo y realizar predicciones.

## Requisitos

- Python 3.x
- TensorFlow 2.x
- Scikit-learn
- Pandas
- Polars
- Matplotlib
- SQLite3

Instalar dependencias:

```bash
pip install -r requirements.txt
```


## Contribución

Para contribuir al proyecto, sigue estos pasos:

1. Crea un fork del proyecto.
2. Crea una rama para tu contribución (`git checkout -b feature/feature-name`).
3. Realiza tus cambios y haz commit de ellos (`git commit -am 'Add new feature'`).
4. Realiza un push de tu rama (`git push origin feature/feature-name`).
5. Crea un nuevo Pull Request.
6. Espera a que tu Pull Request sea revisado y aceptado.

## Licencia

Universidad Nacional de Colombia - Sede Manizales. Todos los derechos reservados.

---
