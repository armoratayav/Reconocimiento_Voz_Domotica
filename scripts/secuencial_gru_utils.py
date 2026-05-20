from pathlib import Path

import librosa
import numpy as np


SAMPLE_RATE = 16000
MAX_DURATION = 3.0
MAX_SAMPLES = int(SAMPLE_RATE * MAX_DURATION)
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


def load_audio_fixed_length(file_path):
    """Carga audio mono a 16 kHz y lo ajusta exactamente a 3 segundos."""
    audio, _ = librosa.load(Path(file_path), sr=SAMPLE_RATE, mono=True)

    if len(audio) > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]
    else:
        padding = MAX_SAMPLES - len(audio)
        audio = np.pad(audio, (0, padding), mode="constant")

    return audio.astype(np.float32)


def fix_sequence_frames(mfcc_sequence):
    """Recorta o rellena una secuencia MFCC para obtener forma (188, 13)."""
    current_frames = mfcc_sequence.shape[0]

    if current_frames > MAX_FRAMES:
        return mfcc_sequence[:MAX_FRAMES, :]

    if current_frames < MAX_FRAMES:
        padding = MAX_FRAMES - current_frames
        return np.pad(mfcc_sequence, ((0, padding), (0, 0)), mode="constant")

    return mfcc_sequence


def extract_mfcc_sequence(file_path):
    """
    Extrae MFCC completo como secuencia temporal.
    Devuelve forma (MAX_FRAMES, N_MFCC), es decir (188, 13).
    """
    audio = load_audio_fixed_length(file_path)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    # librosa devuelve (n_mfcc, frames). GRU/LSTM necesita (frames, n_mfcc).
    mfcc_sequence = mfcc.T
    mfcc_sequence = fix_sequence_frames(mfcc_sequence)
    mfcc_sequence = (mfcc_sequence - np.mean(mfcc_sequence)) / (
        np.std(mfcc_sequence) + 1e-8
    )

    return mfcc_sequence.astype(np.float32)


def list_class_dirs(dataset_dir):
    """Devuelve subcarpetas ordenadas que representan clases."""
    dataset_dir = Path(dataset_dir)
    return sorted([path for path in dataset_dir.iterdir() if path.is_dir()])


def predict_audio(model, classes, audio_path):
    """Predice un audio individual y devuelve clase, confianza y probabilidades."""
    features = extract_mfcc_sequence(audio_path)
    features = np.expand_dims(features, axis=0)
    probabilities = model.predict(features, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = str(classes[predicted_index])
    confidence = float(probabilities[predicted_index])

    return predicted_class, confidence, probabilities
