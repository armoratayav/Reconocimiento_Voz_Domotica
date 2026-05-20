from __future__ import annotations

from pathlib import Path
import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import librosa
import numpy as np
import serial
import sounddevice as sd
import tensorflow as tf


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

WINDOW_SECONDS = 0.25
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
MIN_VOICE_WINDOWS = 3
MAX_COMMAND_SECONDS = 3.0
SILENCE_WINDOWS_TO_END = 4
ENERGY_THRESHOLD = 0.015
MIN_COMMAND_SECONDS = 0.5
COOLDOWN_SECONDS = 1.5
MODO_DEBUG_VAD = False

LAST_COMMAND = None
LAST_COMMAND_TIME = 0.0

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


class ArduinoSerialManager:
    def __init__(self, serial_port: str, baud_rate: int, usar_arduino: bool = True):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.usar_arduino = usar_arduino
        self.serial_conn: serial.Serial | None = None
        self.simulacion = not usar_arduino

    def conectar(self):
        if not self.usar_arduino:
            self.simulacion = True
            print("Arduino desactivado. Modo simulacion.")
            return

        try:
            self.serial_conn = serial.Serial(
                port=self.serial_port,
                baudrate=self.baud_rate,
                timeout=0.2,
                write_timeout=1.0,
            )
            print(f"Puerto serial abierto en {self.serial_port} a {self.baud_rate} baudios.")
            print("Esperando reinicio inicial del Arduino...")
            time.sleep(2.0)
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            self.simulacion = False
            print("Conexion serial lista.")
        except serial.SerialException as error:
            self.serial_conn = None
            self.simulacion = True
            print("Advertencia: Arduino no conectado. Continuando en modo simulacion.")
            print(f"Detalle: {error}")

    def enviar_comando(self, command: str):
        if self.simulacion or self.serial_conn is None or not self.serial_conn.is_open:
            print("Modo simulacion. No se envia comando al Arduino fisico.")
            print(f"Comando generado: {command}")
            return False

        try:
            self.serial_conn.write((command.strip() + "\n").encode("utf-8"))
            self.serial_conn.flush()
            print(f"Comando enviado: {command}")
            return True
        except serial.SerialException as error:
            print("Advertencia: no se pudo enviar el comando. Cambiando a modo simulacion.")
            print(f"Detalle: {error}")
            self.simulacion = True
            return False

    def leer_respuestas(self, timeout: float = 1.0):
        if self.simulacion or self.serial_conn is None or not self.serial_conn.is_open:
            return []

        respuestas = []
        inicio = time.perf_counter()
        while time.perf_counter() - inicio < timeout:
            try:
                if self.serial_conn.in_waiting <= 0:
                    time.sleep(0.05)
                    continue

                line = self.serial_conn.readline().decode("utf-8", errors="replace").strip()
                if line:
                    respuestas.append(line)
            except serial.SerialException as error:
                print(f"Advertencia leyendo respuesta del Arduino: {error}")
                self.simulacion = True
                break

        return respuestas

    def cerrar(self):
        if self.serial_conn is not None and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Puerto serial cerrado.")


def capturar_audio_mic():
    print("Grabando...")
    audio = sd.rec(
        MAX_SAMPLES,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
    )
    sd.wait()
    print("Grabacion terminada.")

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


def enviar_o_simular(manager: ArduinoSerialManager, command):
    if command is None:
        print("No se envia comando al Arduino.")
        return

    manager.enviar_comando(command)
    for response in manager.leer_respuestas(timeout=1.0):
        print(f"Arduino: {response}")


def imprimir_resultado(predicted_class, confidence, probabilities, classes, command, decision):
    print()
    print(f"Clase predicha: {predicted_class}")
    print(f"Confianza: {confidence:.4f}")
    mostrar_probabilidades(classes, probabilities)
    print(f"Comando Arduino: {command if command is not None else 'N/A'}")
    print(f"Decision: {decision}")


def ejecutar_modo_manual(model, classes, manager: ArduinoSerialManager):
    while True:
        user_input = input(
            "\nPresiona Enter para grabar, escribe activo para escucha activa o salir: "
        ).strip().lower()

        if user_input == "salir":
            return "salir"

        if user_input == "activo":
            return "activo"

        start_time = time.perf_counter()
        audio = capturar_audio_mic()
        predicted_class, confidence, probabilities = predecir(model, classes, audio)
        command, decision = decidir_comando(predicted_class, confidence)
        enviar_o_simular(manager, command)
        latency_ms = (time.perf_counter() - start_time) * 1000

        imprimir_resultado(predicted_class, confidence, probabilities, classes, command, decision)
        print(f"Latencia total: {latency_ms:.0f} ms")


def calcular_rms(window):
    return float(np.sqrt(np.mean(np.square(window))))


def escuchar_comando_por_vad():
    recording = False
    voice_windows = 0
    silence_windows = 0
    audio_windows = []
    max_windows = int(MAX_COMMAND_SECONDS / WINDOW_SECONDS)

    while True:
        window = sd.rec(
            WINDOW_SAMPLES,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
        )
        sd.wait()
        window = window.reshape(-1).astype(np.float32)
        rms = calcular_rms(window)
        has_voice = rms > ENERGY_THRESHOLD

        if MODO_DEBUG_VAD:
            print(f"RMS: {rms:.4f}")

        if has_voice:
            if not recording:
                print("Voz detectada...")
                recording = True
                voice_windows = 0
                silence_windows = 0
                audio_windows = []

            voice_windows += 1
            silence_windows = 0
            audio_windows.append(window)
        elif recording:
            silence_windows += 1
            audio_windows.append(window)

        if recording and len(audio_windows) >= max_windows:
            print("Duracion maxima alcanzada.")
            break

        if recording and silence_windows >= SILENCE_WINDOWS_TO_END:
            print("Fin de comando detectado.")
            break

    if voice_windows < MIN_VOICE_WINDOWS:
        print("Audio ignorado: voz demasiado corta.")
        return None

    audio = np.concatenate(audio_windows)
    if len(audio) < int(MIN_COMMAND_SECONDS * SAMPLE_RATE):
        print("Audio ignorado: duracion menor al minimo.")
        return None

    return audio[:MAX_SAMPLES]


def aplicar_cooldown(command):
    global LAST_COMMAND, LAST_COMMAND_TIME

    if command is None:
        return False

    now = time.perf_counter()
    if LAST_COMMAND == command and now - LAST_COMMAND_TIME < COOLDOWN_SECONDS:
        print("Comando repetido ignorado por cooldown.")
        return True

    LAST_COMMAND = command
    LAST_COMMAND_TIME = now
    return False


def ejecutar_modo_activo(model, classes, manager: ArduinoSerialManager):
    print("\nModo escucha activa iniciado. Presiona Ctrl+C para volver al menu.")
    try:
        while True:
            audio = escuchar_comando_por_vad()
            if audio is None:
                continue

            decision_start = time.perf_counter()
            predicted_class, confidence, probabilities = predecir(model, classes, audio)
            command, decision = decidir_comando(predicted_class, confidence)
            imprimir_resultado(predicted_class, confidence, probabilities, classes, command, decision)

            if command is not None and not aplicar_cooldown(command):
                enviar_o_simular(manager, command)
            elif command is None:
                print("No se envia comando al Arduino.")

            latency_ms = (time.perf_counter() - decision_start) * 1000
            print(f"Latencia total: {latency_ms:.0f} ms")
            time.sleep(COOLDOWN_SECONDS / 2)
    except KeyboardInterrupt:
        print("\nVolviendo al menu principal.")
        return "menu"


def mostrar_menu():
    print("\n========== Asistente secuencial activo ==========")
    print("1. Modo manual por Enter")
    print("2. Modo escucha activa")
    print("3. Salir")
    return input("Selecciona una opcion: ").strip()


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

    manager = ArduinoSerialManager(SERIAL_PORT, BAUD_RATE, USAR_ARDUINO)
    manager.conectar()

    try:
        while True:
            opcion = mostrar_menu()

            if opcion == "1":
                resultado = ejecutar_modo_manual(model, classes, manager)
                if resultado == "salir":
                    break
                if resultado == "activo":
                    ejecutar_modo_activo(model, classes, manager)
            elif opcion == "2":
                ejecutar_modo_activo(model, classes, manager)
            elif opcion == "3":
                break
            else:
                print("Opcion no valida.")
    finally:
        manager.cerrar()
        print("Asistente finalizado.")


if __name__ == "__main__":
    main()
