from pathlib import Path
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf

from secuencial_gru_utils import (
    CLASS_TO_ARDUINO_COMMAND,
    CONFIDENCE_THRESHOLD,
    predict_audio,
)


MODEL_PATH = Path("modelos/modelo_secuencial_gru.keras")
CLASSES_PATH = Path("modelos/secuencial_gru_classes.npy")


def main():
    if len(sys.argv) < 2:
        print("Uso:")
        print("python scripts/probar_modelo_secuencial_gru.py ruta_del_audio.wav")
        return

    audio_path = Path(sys.argv[1])

    if not audio_path.exists():
        print(f"No existe el archivo: {audio_path}")
        return

    if not MODEL_PATH.exists():
        print(f"No existe el modelo: {MODEL_PATH}")
        return

    if not CLASSES_PATH.exists():
        print(f"No existe el archivo de clases: {CLASSES_PATH}")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    classes = np.load(CLASSES_PATH, allow_pickle=True)

    predicted_class, confidence, probabilities = predict_audio(model, classes, audio_path)

    print()
    print(f"Audio: {audio_path}")
    print(f"Clase predicha: {predicted_class}")
    print(f"Confianza: {confidence:.4f}")

    print()
    print("Probabilidades por clase:")
    for class_name, probability in zip(classes, probabilities):
        print(f"{class_name}: {probability:.4f}")

    print()
    if confidence < CONFIDENCE_THRESHOLD:
        print("Decision: COMANDO_RECHAZADO")
        print("Accion Arduino: SIN_ACCION")
        return

    arduino_command = CLASS_TO_ARDUINO_COMMAND.get(predicted_class, "SIN_ACCION")
    print("Decision: COMANDO_ACEPTADO")
    print(f"Accion Arduino: {arduino_command}")


if __name__ == "__main__":
    main()
