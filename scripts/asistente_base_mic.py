from __future__ import annotations

from pathlib import Path
import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import librosa
import numpy as np
import sounddevice as sd
import tensorflow as tf

from enviar_comando_arduino import enviar_comando_serial


# ==========================
# CONFIGURACION
# ==========================

SERIAL_PORT = "COM3"
BAUD_RATE = 9600
USAR_ARDUINO = False

MODEL_PATH = Path("modelos/modelo_base_cnn_robusto.keras")
CLASSES_PATH = Path("modelos/base_cnn_robusto_classes.npy")

SAMPLE_RATE = 16000
DURATION_SECONDS = 3.0
CHANNELS = 1
MAX_SAMPLES = int(SAMPLE_RATE * DURATION_SECONDS)

N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 256
MAX_FRAMES = 188
CONFIDENCE_THRESHOLD = 0.70

CLASS_TO_ARDUINO_COMMAND = {
    "enciende": "LUZ_ON",
    "apaga": "LUZ_OFF",
    "ventilador": "VENT_TOGGLE",
    "puerta": "PUERTA_TOGGLE",
    "alarma": "ALARMA_TOGGLE",
    "seguro": "SEGURO",
}


def capturar_audio_mic():
    print("Grabando...")
    audio = sd.rec(
        MAX_SAMPLES,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
    )
    sd.wait()
    print("Grabación terminada.")

    return audio.reshape(-1).astype(np.float32)


def ajustar_audio(audio):
    if len(audio) > MAX_SAMPLES:
        return audio[:MAX_SAMPLES]

    if len(audio) < MAX_SAMPLES:
        padding = MAX_SAMPLES - len(audio)
        return np.pad(audio, (0, padding), mode="constant")

    return audio


def ajustar_frames_mfcc(mfcc):
    frames_actuales = mfcc.shape[1]

    if frames_actuales > MAX_FRAMES:
        return mfcc[:, :MAX_FRAMES]

    if frames_actuales < MAX_FRAMES:
        padding = MAX_FRAMES - frames_actuales
        return np.pad(mfcc, ((0, 0), (0, padding)), mode="constant")

    return mfcc


def preprocesar_audio(audio):
    audio = ajustar_audio(audio)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    mfcc = ajustar_frames_mfcc(mfcc)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    mfcc = np.expand_dims(mfcc, axis=-1)

    return np.expand_dims(mfcc.astype(np.float32), axis=0)


def predecir(model, classes, audio):
    features = preprocesar_audio(audio)
    probabilities = model.predict(features, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = str(classes[predicted_index])
    confidence = float(probabilities[predicted_index])

    return predicted_class, confidence, probabilities


def mostrar_probabilidades(classes, probabilities):
    print("Probabilidades por clase:")
    for class_name, probability in zip(classes, probabilities):
        print(f"- {str(class_name):18s}: {float(probability):.4f}")


def decidir_comando(predicted_class, confidence):
    clase = predicted_class.strip().lower()

    if confidence < CONFIDENCE_THRESHOLD:
        return None, "COMANDO_RECHAZADO"

    if clase == "ruido_fondo":
        return None, "SIN_ACCION"

    command = CLASS_TO_ARDUINO_COMMAND.get(clase)
    if command is None:
        return None, "SIN_ACCION"

    return command, "COMANDO_VALIDO"


def enviar_o_simular(command):
    if command is None:
        print("No se envia comando al Arduino.")
        return

    if not USAR_ARDUINO:
        print("Arduino no conectado. Modo simulación.")
        print(f"Comando generado: {command}")
        return

    sent, responses, error = enviar_comando_serial(
        command,
        serial_port=SERIAL_PORT,
        baud_rate=BAUD_RATE,
    )

    if not sent:
        print("Arduino no conectado. Modo simulación.")
        print(f"Comando generado: {command}")
        if error:
            print(f"Detalle: {error}")
        return

    print(f"Comando enviado: {command}")
    for response in responses:
        print(f"Arduino: {response}")


def ejecutar_ciclo(model, classes):
    while True:
        user_input = input("\nPresiona Enter para grabar comando o escribe salir: ").strip()
        if user_input.lower() == "salir":
            print("Asistente finalizado.")
            break

        start_time = time.perf_counter()
        audio = capturar_audio_mic()
        predicted_class, confidence, probabilities = predecir(model, classes, audio)
        command, decision = decidir_comando(predicted_class, confidence)
        enviar_o_simular(command)
        latency_ms = (time.perf_counter() - start_time) * 1000

        print()
        print(f"Clase predicha: {predicted_class}")
        print(f"Confianza: {confidence:.4f}")
        mostrar_probabilidades(classes, probabilities)
        print(f"Decisión: {decision}")
        print(f"Latencia total: {latency_ms:.0f} ms")


def main():
    if not MODEL_PATH.exists():
        print(f"No existe el modelo: {MODEL_PATH}")
        return

    if not CLASSES_PATH.exists():
        print(f"No existe el archivo de clases: {CLASSES_PATH}")
        return

    print("Cargando modelo base CNN...")
    model = tf.keras.models.load_model(MODEL_PATH)
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    print("Modelo cargado.")

    ejecutar_ciclo(model, classes)


if __name__ == "__main__":
    main()
