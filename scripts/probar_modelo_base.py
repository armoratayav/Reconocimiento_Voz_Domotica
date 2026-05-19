from pathlib import Path
import sys
import numpy as np
import librosa
import tensorflow as tf


MODEL_PATH = Path("modelos/modelo_base.keras")
CLASSES_PATH = Path("modelos/base_classes.npy")

SAMPLE_RATE = 16000
N_MFCC = 13
MAX_DURATION = 3.0
MAX_SAMPLES = int(SAMPLE_RATE * MAX_DURATION)


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
    if len(sys.argv) < 2:
        print("Uso:")
        print("python scripts/probar_modelo_base.py ruta_del_audio.wav")
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

    features = extract_mfcc(audio_path)
    features = np.expand_dims(features, axis=0)

    prediction = model.predict(features)[0]

    predicted_index = np.argmax(prediction)
    predicted_class = classes[predicted_index]
    confidence = prediction[predicted_index]

    print()
    print(f"Audio: {audio_path}")
    print(f"Predicción: {predicted_class}")
    print(f"Confianza: {confidence:.4f}")

    print()
    print("Probabilidades por clase:")
    for class_name, prob in zip(classes, prediction):
        print(f"{class_name}: {prob:.4f}")


if __name__ == "__main__":
    main()