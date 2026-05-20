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

SERIAL_PORT = "COM6"
BAUD_RATE = 9600
USAR_ARDUINO = True

MODEL_PATH = Path("modelos/modelo_secuencial_gru.keras")
CLASSES_PATH = Path("modelos/secuencial_gru_classes.npy")

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
    "enciende_luz": "LUZ_ON",
    "apaga_luz": "LUZ_OFF",
    "enciende_ventilador": "VENT_ON",
    "apaga_ventilador": "VENT_OFF",
    "abre_puerta": "PUERTA_ABRIR",
    "cierra_puerta": "PUERTA_CERRAR",
    "activa_alarma": "ALARMA_ON",
    "apaga_alarma": "ALARMA_OFF",
    "apaga_todo": "TODO_OFF",
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


def ajustar_frames_secuencia(mfcc_sequence):
    frames_actuales = mfcc_sequence.shape[0]

    if frames_actuales > MAX_FRAMES:
        return mfcc_sequence[:MAX_FRAMES, :]

    if frames_actuales < MAX_FRAMES:
        padding = MAX_FRAMES - frames_actuales
        return np.pad(mfcc_sequence, ((0, padding), (0, 0)), mode="constant")

    return mfcc_sequence


def preprocesar_audio(audio):
    audio = ajustar_audio(audio)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    mfcc_sequence = mfcc.T
    mfcc_sequence = ajustar_frames_secuencia(mfcc_sequence)
    mfcc_sequence = (mfcc_sequence - np.mean(mfcc_sequence)) / (
        np.std(mfcc_sequence) + 1e-8
    )

    return np.expand_dims(mfcc_sequence.astype(np.float32), axis=0)


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
        print(f"- {str(class_name):22s}: {float(probability):.4f}")


def decidir_comando(predicted_class, confidence):
    if confidence < CONFIDENCE_THRESHOLD:
        return None, "COMANDO_RECHAZADO"

    command = CLASS_TO_ARDUINO_COMMAND.get(predicted_class.strip().lower())
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

    print("Cargando modelo secuencial GRU...")
    model = tf.keras.models.load_model(MODEL_PATH)
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    print("Modelo cargado.")

    ejecutar_ciclo(model, classes)


if __name__ == "__main__":
    main()
