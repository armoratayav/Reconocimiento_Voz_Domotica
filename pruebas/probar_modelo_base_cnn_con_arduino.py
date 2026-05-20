from pathlib import Path
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf

from base_cnn_utils import CONFIDENCE_THRESHOLD, predict_audio
from enviar_comando_arduino import BAUD_RATE, READ_SECONDS, enviar_comando_serial
from mapear_comando_ia import mapear_base_a_arduino


SERIAL_PORT = "COM3"

MODEL_PATH = Path("modelos/modelo_base_cnn.keras")
CLASSES_PATH = Path("modelos/base_cnn_classes.npy")


def main():
    if len(sys.argv) < 2:
        print("Uso:")
        print("python scripts/probar_modelo_base_cnn_con_arduino.py ruta_del_audio.wav")
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

    predicted_class, confidence, _ = predict_audio(model, classes, audio_path)

    print()
    print(f"Audio: {audio_path}")
    print(f"Clase predicha: {predicted_class}")
    print(f"Confianza: {confidence:.4f}")

    arduino_command = mapear_base_a_arduino(
        predicted_class,
        confidence,
        umbral=CONFIDENCE_THRESHOLD,
    )

    if arduino_command is None:
        if confidence < CONFIDENCE_THRESHOLD:
            print("Decision: COMANDO_RECHAZADO")
        elif predicted_class == "ruido_fondo":
            print("Decision: SIN_ACCION")
        else:
            print("Decision: SIN_ACCION")
        print("No se envio nada al Arduino.")
        return

    print(f"Comando Arduino: {arduino_command}")
    sent, responses, error = enviar_comando_serial(
        arduino_command,
        serial_port=SERIAL_PORT,
        baud_rate=BAUD_RATE,
        read_seconds=READ_SECONDS,
    )

    if not sent:
        print("No se pudo enviar el comando al Arduino.")
        print(error)
        return

    print("Comando enviado al Arduino.")
    if responses:
        print()
        print("Respuesta Arduino:")
        for line in responses:
            print(line)
    else:
        print("Arduino no envio respuesta durante el tiempo de espera.")


if __name__ == "__main__":
    main()
