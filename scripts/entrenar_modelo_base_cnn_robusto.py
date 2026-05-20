from pathlib import Path
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import librosa
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


# ==========================
# CONFIGURACION
# ==========================

DATASET_DIR = Path("dataset_procesado/Base")
MODEL_DIR = Path("modelos")
METRICS_DIR = Path("metricas")

MODEL_PATH = MODEL_DIR / "modelo_base_cnn_robusto.keras"
CLASSES_PATH = MODEL_DIR / "base_cnn_robusto_classes.npy"
REPORT_PATH = METRICS_DIR / "base_cnn_robusto_classification_report.txt"
CONFUSION_MATRIX_PATH = METRICS_DIR / "base_cnn_robusto_confusion_matrix.png"
ACCURACY_PATH = METRICS_DIR / "base_cnn_robusto_accuracy.png"
LOSS_PATH = METRICS_DIR / "base_cnn_robusto_loss.png"

EXPECTED_CLASSES = [
    "alarma",
    "apaga",
    "enciende",
    "puerta",
    "ruido_fondo",
    "seguro",
    "ventilador",
]

SAMPLE_RATE = 16000
DURATION_SECONDS = 3.0
MAX_SAMPLES = int(SAMPLE_RATE * DURATION_SECONDS)
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 256
MAX_FRAMES = 188

RANDOM_STATE = 42
TEST_SIZE = 0.20
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15


def ensure_output_dirs():
    MODEL_DIR.mkdir(exist_ok=True)
    METRICS_DIR.mkdir(exist_ok=True)


def load_audio_with_vad(file_path):
    """Carga audio mono a 16 kHz y recorta silencios con VAD basico."""
    audio, _ = librosa.load(Path(file_path), sr=SAMPLE_RATE, mono=True)
    audio = audio.astype(np.float32)

    if len(audio) == 0:
        return np.zeros(MAX_SAMPLES, dtype=np.float32)

    trimmed_audio, _ = librosa.effects.trim(audio, top_db=30)
    if len(trimmed_audio) == 0:
        trimmed_audio = audio

    return trimmed_audio.astype(np.float32)


def fix_audio_length(audio):
    """Ajusta cualquier audio exactamente a 3 segundos."""
    if len(audio) > MAX_SAMPLES:
        return audio[:MAX_SAMPLES]

    if len(audio) < MAX_SAMPLES:
        padding = MAX_SAMPLES - len(audio)
        return np.pad(audio, (0, padding), mode="constant")

    return audio


def fix_mfcc_frames(mfcc):
    """Recorta o rellena frames para obtener forma (13, 188)."""
    current_frames = mfcc.shape[1]

    if current_frames > MAX_FRAMES:
        return mfcc[:, :MAX_FRAMES]

    if current_frames < MAX_FRAMES:
        padding = MAX_FRAMES - current_frames
        return np.pad(mfcc, ((0, 0), (0, padding)), mode="constant")

    return mfcc


def audio_to_mfcc_cnn(audio):
    """Convierte audio ya cargado a tensor CNN con forma (13, 188, 1)."""
    audio = fix_audio_length(audio)
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    mfcc = fix_mfcc_frames(mfcc)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    mfcc = np.expand_dims(mfcc, axis=-1)

    return mfcc.astype(np.float32)


def extract_mfcc_cnn_robusto(file_path):
    """Preprocesamiento robusto compartido por entrenamiento y prueba."""
    audio = load_audio_with_vad(file_path)
    return audio_to_mfcc_cnn(audio)


def augment_audio(audio, rng):
    """Genera variantes suaves para mejorar generalizacion solo en entrenamiento."""
    augmented = []

    shift = int(rng.integers(-int(0.20 * SAMPLE_RATE), int(0.20 * SAMPLE_RATE) + 1))
    augmented.append(np.roll(audio, shift))

    volume = float(rng.uniform(0.80, 1.20))
    augmented.append(np.clip(audio * volume, -1.0, 1.0))

    noise_std = float(rng.uniform(0.002, 0.008))
    noise = rng.normal(0.0, noise_std, size=len(audio)).astype(np.float32)
    augmented.append(np.clip(audio + noise, -1.0, 1.0))

    pitch_steps = float(rng.uniform(-1.0, 1.0))
    augmented.append(
        librosa.effects.pitch_shift(y=audio, sr=SAMPLE_RATE, n_steps=pitch_steps)
    )

    stretch_rate = float(rng.uniform(0.90, 1.10))
    augmented.append(librosa.effects.time_stretch(y=audio, rate=stretch_rate))

    return [item.astype(np.float32) for item in augmented]


def list_expected_class_dirs():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta: {DATASET_DIR}")

    class_dirs = []
    missing_classes = []

    for class_name in EXPECTED_CLASSES:
        class_dir = DATASET_DIR / class_name
        if class_dir.exists() and class_dir.is_dir():
            class_dirs.append(class_dir)
        else:
            missing_classes.append(class_name)

    if missing_classes:
        missing = ", ".join(missing_classes)
        raise FileNotFoundError(f"Faltan carpetas de clases en {DATASET_DIR}: {missing}")

    return class_dirs


def collect_file_paths():
    file_paths = []
    labels = []
    counts = {}

    for class_dir in list_expected_class_dirs():
        wav_files = sorted(class_dir.glob("*.wav"))
        counts[class_dir.name] = len(wav_files)

        for wav_file in wav_files:
            file_paths.append(wav_file)
            labels.append(class_dir.name)

    if not file_paths:
        raise ValueError("No se encontraron audios .wav para entrenar.")

    return np.array(file_paths, dtype=object), np.array(labels), counts


def build_feature_set(file_paths, labels, augment=False):
    X = []
    y = []
    rng = np.random.default_rng(RANDOM_STATE)

    for file_path, label in zip(file_paths, labels):
        try:
            audio = load_audio_with_vad(file_path)
            X.append(audio_to_mfcc_cnn(audio))
            y.append(label)

            if augment:
                for augmented_audio in augment_audio(audio, rng):
                    X.append(audio_to_mfcc_cnn(augmented_audio))
                    y.append(label)

        except Exception as exc:
            print(f"Error procesando {file_path}: {exc}")

    if not X:
        raise ValueError("No se pudo procesar ningun audio valido.")

    return np.array(X, dtype=np.float32), np.array(y)


def shuffle_features_and_labels(X, labels):
    """Baraja muestras antes de usar validation_split de Keras."""
    rng = np.random.default_rng(RANDOM_STATE)
    indexes = rng.permutation(len(X))
    return X[indexes], labels[indexes]


def build_model(input_shape, num_classes):
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.20),
            Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.30),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.30),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def plot_training_history(history):
    plt.figure()
    plt.plot(history.history["accuracy"], label="accuracy entrenamiento")
    plt.plot(history.history["val_accuracy"], label="accuracy validacion")
    plt.xlabel("Epoca")
    plt.ylabel("Accuracy")
    plt.title("Accuracy - Modelo Base CNN Robusto")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ACCURACY_PATH)
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="loss entrenamiento")
    plt.plot(history.history["val_loss"], label="loss validacion")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.title("Loss - Modelo Base CNN Robusto")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PATH)
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap="Blues", xticks_rotation=45, ax=ax)
    ax.set_title("Matriz de confusion - Modelo Base CNN Robusto")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH)
    plt.close(fig)


def main():
    ensure_output_dirs()
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    print("Recolectando audios del dataset base...")
    file_paths, labels, counts = collect_file_paths()

    print()
    print("Cantidad de audios por clase:")
    for class_name in EXPECTED_CLASSES:
        print(f"- {class_name}: {counts.get(class_name, 0)}")

    train_files, test_files, train_labels, test_labels = train_test_split(
        file_paths,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print()
    print("Procesando entrenamiento con data augmentation...")
    X_train, y_train_labels = build_feature_set(train_files, train_labels, augment=True)
    X_train, y_train_labels = shuffle_features_and_labels(X_train, y_train_labels)

    print("Procesando prueba sin data augmentation...")
    X_test, y_test_labels = build_feature_set(test_files, test_labels, augment=False)

    label_encoder = LabelEncoder()
    label_encoder.fit(EXPECTED_CLASSES)
    np.save(CLASSES_PATH, label_encoder.classes_)

    y_train_encoded = label_encoder.transform(y_train_labels)
    y_test_encoded = label_encoder.transform(y_test_labels)
    y_train = to_categorical(y_train_encoded, num_classes=len(label_encoder.classes_))
    y_test = to_categorical(y_test_encoded, num_classes=len(label_encoder.classes_))

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(label_encoder.classes_)),
        y=label_encoder.transform(train_labels),
    )
    class_weight = {
        class_index: float(weight)
        for class_index, weight in enumerate(class_weights_array)
    }

    print()
    print(f"Forma X_train: {X_train.shape}")
    print(f"Forma X_test: {X_test.shape}")
    print("Class weight:")
    for class_index, weight in class_weight.items():
        print(f"- {label_encoder.classes_[class_index]}: {weight:.4f}")

    model = build_model(input_shape=X_train.shape[1:], num_classes=len(label_encoder.classes_))
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
    )

    print()
    print("Entrenando modelo base CNN robusto...")
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stopping],
        class_weight=class_weight,
        verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print()
    print(f"Accuracy final de prueba: {test_accuracy:.4f}")
    print(f"Loss final de prueba: {test_loss:.4f}")

    probabilities = model.predict(X_test, verbose=0)
    y_pred_encoded = np.argmax(probabilities, axis=1)
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

    report = classification_report(
        y_test_labels,
        y_pred_labels,
        labels=label_encoder.classes_,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    print()
    print("Classification report:")
    print(report)

    REPORT_PATH.write_text(report, encoding="utf-8")
    save_confusion_matrix(y_test_labels, y_pred_labels, label_encoder.classes_)
    plot_training_history(history)
    model.save(MODEL_PATH)

    print()
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Clases guardadas en: {CLASSES_PATH}")
    print(f"Reporte guardado en: {REPORT_PATH}")
    print(f"Matriz de confusion guardada en: {CONFUSION_MATRIX_PATH}")
    print(f"Grafica de accuracy guardada en: {ACCURACY_PATH}")
    print(f"Grafica de loss guardada en: {LOSS_PATH}")


if __name__ == "__main__":
    main()
