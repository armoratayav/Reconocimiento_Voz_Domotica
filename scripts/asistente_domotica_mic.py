from __future__ import annotations

from dataclasses import dataclass
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
USAR_ARDUINO = False

SAMPLE_RATE = 16000
DURATION_SECONDS = 3.0
CHANNELS = 1
MAX_SAMPLES = int(SAMPLE_RATE * DURATION_SECONDS)

N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 256
MAX_FRAMES = 188
CONFIDENCE_THRESHOLD = 0.70

BASE_MAP = {
    "enciende": "LUZ_ON",
    "apaga": "LUZ_OFF",
    "ventilador": "VENT_TOGGLE",
    "puerta": "PUERTA_TOGGLE",
    "alarma": "ALARMA_TOGGLE",
    "seguro": "SEGURO",
    "ruido_fondo": None,
}

SEQUENTIAL_MAP = {
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


@dataclass
class ModeConfig:
    name: str
    model_path: Path
    classes_path: Path
    class_to_command: dict[str, str | None]


MODE_CONFIGS = {
    "base": ModeConfig(
        name="base",
        model_path=Path("modelos/modelo_base_cnn.keras"),
        classes_path=Path("modelos/base_cnn_classes.npy"),
        class_to_command=BASE_MAP,
    ),
    "secuencial": ModeConfig(
        name="secuencial",
        model_path=Path("modelos/modelo_secuencial_gru.keras"),
        classes_path=Path("modelos/secuencial_gru_classes.npy"),
        class_to_command=SEQUENTIAL_MAP,
    ),
}


def elegir_modo():
    while True:
        mode = input("Elige modo inicial (base/secuencial): ").strip().lower()
        if mode in MODE_CONFIGS:
            return mode

        print("Modo no valido. Escribe base o secuencial.")


def cargar_modo(mode):
    config = MODE_CONFIGS[mode]

    if not config.model_path.exists():
        raise FileNotFoundError(f"No existe el modelo: {config.model_path}")

    if not config.classes_path.exists():
        raise FileNotFoundError(f"No existe el archivo de clases: {config.classes_path}")

    print(f"Cargando modelo en modo {mode}...")
    model = tf.keras.models.load_model(config.model_path)
    classes = np.load(config.classes_path, allow_pickle=True)
    print("Modelo cargado.")

    return config, model, classes


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


def calcular_mfcc(audio):
    audio = ajustar_audio(audio)
    return librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )


def ajustar_frames_cnn(mfcc):
    frames_actuales = mfcc.shape[1]

    if frames_actuales > MAX_FRAMES:
        return mfcc[:, :MAX_FRAMES]

    if frames_actuales < MAX_FRAMES:
        padding = MAX_FRAMES - frames_actuales
        return np.pad(mfcc, ((0, 0), (0, padding)), mode="constant")

    return mfcc


def ajustar_frames_secuencia(mfcc_sequence):
    frames_actuales = mfcc_sequence.shape[0]

    if frames_actuales > MAX_FRAMES:
        return mfcc_sequence[:MAX_FRAMES, :]

    if frames_actuales < MAX_FRAMES:
        padding = MAX_FRAMES - frames_actuales
        return np.pad(mfcc_sequence, ((0, padding), (0, 0)), mode="constant")

    return mfcc_sequence


def normalizar_muestra(features):
    return (features - np.mean(features)) / (np.std(features) + 1e-8)


def preprocesar_base(audio):
    mfcc = calcular_mfcc(audio)
    mfcc = ajustar_frames_cnn(mfcc)
    mfcc = normalizar_muestra(mfcc)
    mfcc = np.expand_dims(mfcc, axis=-1)

    return np.expand_dims(mfcc.astype(np.float32), axis=0)


def preprocesar_secuencial(audio):
    mfcc = calcular_mfcc(audio)
    mfcc_sequence = ajustar_frames_secuencia(mfcc.T)
    mfcc_sequence = normalizar_muestra(mfcc_sequence)

    return np.expand_dims(mfcc_sequence.astype(np.float32), axis=0)


def preprocesar_audio(mode, audio):
    if mode == "base":
        return preprocesar_base(audio)

    return preprocesar_secuencial(audio)


def predecir(model, classes, features):
    probabilities = model.predict(features, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = str(classes[predicted_index])
    confidence = float(probabilities[predicted_index])

    return predicted_class, confidence, probabilities


def decidir_comando(config, predicted_class, confidence):
    clase = predicted_class.strip().lower()

    if confidence < CONFIDENCE_THRESHOLD:
        return None, "COMANDO_RECHAZADO"

    command = config.class_to_command.get(clase)
    if command is None:
        if clase == "ruido_fondo":
            return None, "SIN_ACCION"
        return None, "SIN_COMANDO_VALIDO"

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


def ejecutar_prediccion(config, model, classes):
    start_time = time.perf_counter()
    audio = capturar_audio_mic()
    features = preprocesar_audio(config.name, audio)
    predicted_class, confidence, _ = predecir(model, classes, features)
    command, decision = decidir_comando(config, predicted_class, confidence)
    enviar_o_simular(command)
    latency_ms = (time.perf_counter() - start_time) * 1000

    print()
    print(f"Modo usado: {config.name}")
    print(f"Clase predicha: {predicted_class}")
    print(f"Confianza: {confidence:.4f}")
    print(f"Comando Arduino generado: {command if command else 'NINGUNO'}")
    print(f"Decisión: {decision}")
    print(f"Latencia total: {latency_ms:.0f} ms")


def main():
    mode = elegir_modo()
    try:
        config, model, classes = cargar_modo(mode)
    except FileNotFoundError as exc:
        print(exc)
        return

    while True:
        user_input = input(
            "\nEnter para grabar, modo para cambiar, salir para terminar: "
        ).strip().lower()

        if user_input == "salir":
            print("Asistente finalizado.")
            break

        if user_input == "modo":
            mode = "secuencial" if mode == "base" else "base"
            try:
                config, model, classes = cargar_modo(mode)
            except FileNotFoundError as exc:
                print(exc)
            continue

        ejecutar_prediccion(config, model, classes)


if __name__ == "__main__":
    main()
