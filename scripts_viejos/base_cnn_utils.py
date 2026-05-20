from pathlib import Path

import librosa
import numpy as np


SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 256
DURATION_SECONDS = 3.0
MAX_SAMPLES = int(SAMPLE_RATE * DURATION_SECONDS)

# Con librosa y center=True, un audio de 48000 muestras produce 188 frames.
TARGET_FRAMES = 1 + (MAX_SAMPLES // HOP_LENGTH)
CONFIDENCE_THRESHOLD = 0.70


def load_audio_fixed_length(file_path):
    """Carga audio mono a 16 kHz y lo ajusta exactamente a 3 segundos."""
    audio, _ = librosa.load(Path(file_path), sr=SAMPLE_RATE, mono=True)

    if len(audio) > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]
    else:
        padding = MAX_SAMPLES - len(audio)
        audio = np.pad(audio, (0, padding), mode="constant")

    return audio.astype(np.float32)


def fix_mfcc_frames(mfcc, target_frames=TARGET_FRAMES):
    """Recorta o rellena la matriz MFCC para mantener frames constantes."""
    current_frames = mfcc.shape[1]

    if current_frames > target_frames:
        return mfcc[:, :target_frames]

    if current_frames < target_frames:
        padding = target_frames - current_frames
        return np.pad(mfcc, ((0, 0), (0, padding)), mode="constant")

    return mfcc


def normalize_mfcc(mfcc):
    """Normaliza cada matriz MFCC de forma independiente."""
    mean = np.mean(mfcc)
    std = np.std(mfcc)

    if std < 1e-8:
        std = 1e-8

    return (mfcc - mean) / std


def extract_mfcc_cnn(file_path):
    """
    Extrae MFCC completo en el tiempo.
    Devuelve forma (13, TARGET_FRAMES, 1), lista para CNN 2D.
    """
    audio = load_audio_fixed_length(file_path)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    mfcc = fix_mfcc_frames(mfcc)
    mfcc = normalize_mfcc(mfcc)
    mfcc = np.expand_dims(mfcc, axis=-1)

    return mfcc.astype(np.float32)


def list_class_dirs(dataset_dir):
    """Devuelve subcarpetas ordenadas que representan clases."""
    dataset_dir = Path(dataset_dir)
    return sorted([path for path in dataset_dir.iterdir() if path.is_dir()])


def predict_audio(model, classes, audio_path):
    """Predice un audio individual y devuelve clase, confianza y probabilidades."""
    features = extract_mfcc_cnn(audio_path)
    features = np.expand_dims(features, axis=0)
    probabilities = model.predict(features, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = str(classes[predicted_index])
    confidence = float(probabilities[predicted_index])

    return predicted_class, confidence, probabilities
