from tensorflow.keras.models import model_from_json
import tensorflow as tf
import numpy as np

# Configuración para limitar el uso de memoria de la GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Configurar TensorFlow para limitar la memoria GPU a un porcentaje
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])  # Ajusta el límite de memoria en MB
    except RuntimeError as e:
        print(e)

class ERModel:
    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        # Cargar la estructura del modelo desde un archivo JSON
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # Cargar los pesos del modelo
        self.loaded_model.load_weights(model_weights_file)

    def predict_emotion(self, img):
        # Realizar la predicción
        self.preds = self.loaded_model.predict(img)
        # Retornar la emoción con mayor probabilidad
        return ERModel.EMOTIONS_LIST[np.argmax(self.preds)]