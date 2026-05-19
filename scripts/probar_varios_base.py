from pathlib import Path
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import librosa
import tensorflow as tf


# ==========================
# CONFIGURACIÓN
# ==========================

DATASET_DIR = Path("dataset_procesado/Base")
MODEL_PATH = Path("modelos/modelo_base.keras")
CLASSES_PATH = Path("modelos/base_classes.npy")

SAMPLE_RATE = 16000
N_MFCC = 13
MAX_DURATION = 3.0
MAX_SAMPLES = int(SAMPLE_RATE * MAX_DURATION)

# Cantidad de audios a probar por clase
AUDIOS_POR_CLASE = 5


def load_audio_fixed_length(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    if len(audio) > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]
    else:
        padding = MAX_SAMPLES - len(audio)
        audio = np.pad(audio, (0, padding), mode="constant")

    return audio


def extract_mfcc(file_path):
    audio = load_audio_fixed_length(file_path)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    mfcc_mean = np.mean(mfcc, axis=1)

    return mfcc_mean.astype(np.float32)


def main():
    if not MODEL_PATH.exists():
        print(f"No existe el modelo: {MODEL_PATH}")
        return

    if not CLASSES_PATH.exists():
        print(f"No existe el archivo de clases: {CLASSES_PATH}")
        return

    if not DATASET_DIR.exists():
        print(f"No existe el dataset: {DATASET_DIR}")
        return

    print("Cargando modelo...")
    model = tf.keras.models.load_model(MODEL_PATH)
    classes = np.load(CLASSES_PATH, allow_pickle=True)

    total = 0
    correctos = 0

    print()
    print("Probando audios por clase...")
    print("=" * 80)

    for class_dir in sorted(DATASET_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        clase_real = class_dir.name
        wav_files = sorted(class_dir.glob("*.wav"))[:AUDIOS_POR_CLASE]

        print()
        print(f"Clase real: {clase_real}")
        print("-" * 80)

        for wav_file in wav_files:
            features = extract_mfcc(wav_file)
            features = np.expand_dims(features, axis=0)

            prediction = model.predict(features, verbose=0)[0]

            predicted_index = np.argmax(prediction)
            clase_predicha = classes[predicted_index]
            confianza = prediction[predicted_index]

            acierto = clase_real == clase_predicha

            total += 1
            if acierto:
                correctos += 1

            estado = "OK" if acierto else "ERROR"

            print(
                f"{estado} | Real: {clase_real:12s} | "
                f"Pred: {clase_predicha:12s} | "
                f"Confianza: {confianza:.4f} | "
                f"Archivo: {wav_file.name}"
            )

    print()
    print("=" * 80)

    if total > 0:
        accuracy = correctos / total
        print(f"Total probados: {total}")
        print(f"Correctos: {correctos}")
        print(f"Accuracy manual: {accuracy:.4f}")
    else:
        print("No se encontraron audios para probar.")


if __name__ == "__main__":
    main()