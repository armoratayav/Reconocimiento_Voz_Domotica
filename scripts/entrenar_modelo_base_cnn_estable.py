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
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


# ==========================
# CONFIGURACION
# ==========================

DATASET_DIR = Path("dataset_procesado/Base")
MODEL_DIR = Path("modelos")
METRICS_DIR = Path("metricas")

MODEL_PATH = MODEL_DIR / "modelo_base_cnn.keras"
CLASSES_PATH = MODEL_DIR / "base_cnn_classes.npy"
REPORT_PATH = METRICS_DIR / "base_cnn_estable_classification_report.txt"
CONFUSION_MATRIX_PATH = METRICS_DIR / "base_cnn_estable_confusion_matrix.png"
ACCURACY_PATH = METRICS_DIR / "base_cnn_estable_accuracy.png"
LOSS_PATH = METRICS_DIR / "base_cnn_estable_loss.png"

SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 256
DURATION_SECONDS = 3.0
MAX_SAMPLES = int(SAMPLE_RATE * DURATION_SECONDS)
MAX_FRAMES = 188

RANDOM_STATE = 42
TEST_SIZE = 0.20
EPOCHS = 80
BATCH_SIZE = 16
LEARNING_RATE = 0.001


def ensure_output_dirs():
    MODEL_DIR.mkdir(exist_ok=True)
    METRICS_DIR.mkdir(exist_ok=True)


def load_audio_fixed_length(file_path):
    """Carga audio mono a 16 kHz y lo ajusta exactamente a 3 segundos."""
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    if len(audio) > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]
    else:
        padding = MAX_SAMPLES - len(audio)
        audio = np.pad(audio, (0, padding), mode="constant")

    return audio.astype(np.float32)


def fix_mfcc_frames(mfcc):
    """Recorta o rellena los frames para obtener forma (13, 188)."""
    current_frames = mfcc.shape[1]

    if current_frames > MAX_FRAMES:
        return mfcc[:, :MAX_FRAMES]

    if current_frames < MAX_FRAMES:
        padding = MAX_FRAMES - current_frames
        return np.pad(mfcc, ((0, 0), (0, padding)), mode="constant")

    return mfcc


def extract_mfcc_cnn(file_path):
    """Extrae MFCC completo en el tiempo y devuelve forma (13, 188, 1)."""
    audio = load_audio_fixed_length(file_path)

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


def list_class_dirs():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta: {DATASET_DIR}")

    class_dirs = sorted([path for path in DATASET_DIR.iterdir() if path.is_dir()])
    if not class_dirs:
        raise ValueError(f"No se encontraron carpetas de clases en: {DATASET_DIR}")

    return class_dirs


def load_dataset():
    """Carga todos los .wav del dataset y conserva la ruta de cada muestra."""
    X = []
    labels = []
    file_paths = []
    counts = {}

    for class_dir in list_class_dirs():
        class_name = class_dir.name
        wav_files = sorted(class_dir.glob("*.wav"))
        counts[class_name] = len(wav_files)

        for wav_file in wav_files:
            try:
                X.append(extract_mfcc_cnn(wav_file))
                labels.append(class_name)
                file_paths.append(wav_file)
            except Exception as exc:
                print(f"Error procesando {wav_file}: {exc}")

    if not X:
        raise ValueError("No se pudo cargar ningun audio .wav valido.")

    return np.array(X, dtype=np.float32), np.array(labels), file_paths, counts


def build_model(input_shape, num_classes):
    """CNN pequena y estable, entrenada desde cero."""
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.20),
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
    """Guarda graficas simples del entrenamiento, sin validacion separada."""
    plt.figure()
    plt.plot(history.history["accuracy"], label="accuracy entrenamiento")
    plt.xlabel("Epoca")
    plt.ylabel("Accuracy")
    plt.title("Accuracy - Modelo Base CNN Estable")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ACCURACY_PATH)
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="loss entrenamiento")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.title("Loss - Modelo Base CNN Estable")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PATH)
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap="Blues", xticks_rotation=45, ax=ax)
    ax.set_title("Matriz de confusion - Modelo Base CNN Estable")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH)
    plt.close(fig)


def print_first_dataset_predictions(model, X, labels, file_paths, class_names):
    print()
    print("Prediccion de los primeros 10 audios del dataset:")
    print("=" * 110)

    limit = min(10, len(X))
    probabilities = model.predict(X[:limit], verbose=0)
    predicted_indexes = np.argmax(probabilities, axis=1)

    for file_path, real_class, predicted_index, probs in zip(
        file_paths[:limit],
        labels[:limit],
        predicted_indexes,
        probabilities,
    ):
        predicted_class = str(class_names[predicted_index])
        confidence = float(probs[predicted_index])

        print(
            f"Archivo: {file_path.name:30s} | "
            f"Real: {real_class:12s} | "
            f"Pred: {predicted_class:12s} | "
            f"Confianza: {confidence:.4f}"
        )


def main():
    ensure_output_dirs()
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    print("Cargando dataset completo para CNN estable...")
    X, labels, file_paths, counts = load_dataset()

    print()
    print("Cantidad de audios por clase:")
    for class_name, count in counts.items():
        print(f"- {class_name}: {count}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    y_categorical = to_categorical(y_encoded)

    np.save(CLASSES_PATH, label_encoder.classes_)

    print()
    print(f"Forma de X: {X.shape}")
    print(f"Forma de y: {y_categorical.shape}")
    print()
    print("Clases:")
    for index, class_name in enumerate(label_encoder.classes_):
        print(f"{index}: {class_name}")

    X_train, X_test, y_train, y_test, encoded_train, encoded_test = train_test_split(
        X,
        y_categorical,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    print()
    print(f"Muestras de entrenamiento: {len(X_train)}")
    print(f"Muestras de prueba: {len(X_test)}")

    model = build_model(input_shape=X.shape[1:], num_classes=len(label_encoder.classes_))
    model.summary()

    print()
    print("Entrenando modelo CNN estable...")
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    print()
    print(f"Accuracy final entrenamiento: {train_accuracy:.4f}")
    print(f"Loss final entrenamiento: {train_loss:.4f}")
    print(f"Accuracy final prueba: {test_accuracy:.4f}")
    print(f"Loss final prueba: {test_loss:.4f}")

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    print()
    print("Classification report:")
    print(report)

    REPORT_PATH.write_text(report, encoding="utf-8")
    save_confusion_matrix(y_true, y_pred, label_encoder.classes_)
    plot_training_history(history)
    print_first_dataset_predictions(model, X, labels, file_paths, label_encoder.classes_)

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
