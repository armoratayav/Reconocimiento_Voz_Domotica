from pathlib import Path
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import (
    Bidirectional,
    Concatenate,
    Dense,
    Dropout,
    GRU,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

from secuencial_gru_utils import MAX_FRAMES, N_MFCC, extract_mfcc_sequence, list_class_dirs


DATASET_DIR = Path("dataset_procesado/Secuencial")
MODEL_DIR = Path("modelos")
METRICS_DIR = Path("metricas")

MODEL_PATH = MODEL_DIR / "modelo_secuencial_gru.keras"
CLASSES_PATH = MODEL_DIR / "secuencial_gru_classes.npy"
REPORT_PATH = METRICS_DIR / "secuencial_gru_classification_report.txt"
CONFUSION_MATRIX_PATH = METRICS_DIR / "secuencial_gru_confusion_matrix.png"
ACCURACY_PATH = METRICS_DIR / "secuencial_gru_accuracy.png"
LOSS_PATH = METRICS_DIR / "secuencial_gru_loss.png"

RANDOM_STATE = 42
TEST_SIZE = 0.20
EPOCHS = 80
BATCH_SIZE = 16
LEARNING_RATE = 0.001


def ensure_output_dirs():
    MODEL_DIR.mkdir(exist_ok=True)
    METRICS_DIR.mkdir(exist_ok=True)


def load_dataset():
    """Carga todos los audios .wav del dataset secuencial."""
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta: {DATASET_DIR}")

    class_dirs = list_class_dirs(DATASET_DIR)
    if not class_dirs:
        raise ValueError(f"No se encontraron carpetas de clases en: {DATASET_DIR}")

    X = []
    labels = []
    file_paths = []
    counts = {}

    print("Clases encontradas:")
    for class_dir in class_dirs:
        print(f"- {class_dir.name}")

    print()
    for class_dir in class_dirs:
        class_name = class_dir.name
        wav_files = sorted(class_dir.glob("*.wav"))
        counts[class_name] = len(wav_files)
        print(f"Cargando clase '{class_name}' con {len(wav_files)} audios...")

        for wav_file in wav_files:
            try:
                X.append(extract_mfcc_sequence(wav_file))
                labels.append(class_name)
                file_paths.append(wav_file)
            except Exception as exc:
                print(f"Error procesando {wav_file}: {exc}")

    if not X:
        raise ValueError("No se pudo cargar ningun audio .wav valido.")

    return np.array(X, dtype=np.float32), np.array(labels), file_paths, counts


def build_model(num_classes):
    """
    Red GRU entrenada desde cero para comandos compuestos.
    Usa GRU bidireccional y pooling temporal para no depender solo del ultimo
    frame, que suele contener silencio o relleno en audios de 3 segundos.
    """
    inputs = Input(shape=(MAX_FRAMES, N_MFCC))
    x = Bidirectional(GRU(64, return_sequences=True))(inputs)
    x = Dropout(0.20)(x)
    x = Bidirectional(GRU(32, return_sequences=True))(x)

    max_pool = GlobalMaxPooling1D()(x)
    avg_pool = GlobalAveragePooling1D()(x)
    x = Concatenate()([max_pool, avg_pool])

    x = Dense(96, activation="relu")(x)
    x = Dropout(0.20)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
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
    plt.title("Accuracy - Modelo Secuencial GRU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ACCURACY_PATH)
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="loss entrenamiento")
    plt.plot(history.history["val_loss"], label="loss validacion")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.title("Loss - Modelo Secuencial GRU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PATH)
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(11, 9))
    disp.plot(cmap="Blues", xticks_rotation=45, ax=ax)
    ax.set_title("Matriz de confusion - Modelo Secuencial GRU")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH)
    plt.close(fig)


def print_first_dataset_predictions(model, X, labels, file_paths, class_names):
    print()
    print("Prediccion de los primeros 10 audios del dataset:")
    print("=" * 120)

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
            f"Archivo: {file_path.name:35s} | Real: {real_class:20s} | "
            f"Pred: {predicted_class:20s} | Confianza: {confidence:.4f}"
        )


def main():
    ensure_output_dirs()
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    print("Cargando dataset secuencial...")
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
    print("Mapeo de clases:")
    for index, class_name in enumerate(label_encoder.classes_):
        print(f"{index}: {class_name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_categorical,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    print()
    print(f"Muestras de entrenamiento: {len(X_train)}")
    print(f"Muestras de prueba: {len(X_test)}")

    model = build_model(num_classes=len(label_encoder.classes_))
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
    )

    print()
    print("Entrenando modelo secuencial GRU desde cero...")
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        callbacks=[early_stopping],
        verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print()
    print(f"Accuracy final de prueba: {test_accuracy:.4f}")
    print(f"Loss final de prueba: {test_loss:.4f}")

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
    print(f"Metricas guardadas en: {METRICS_DIR}")


if __name__ == "__main__":
    main()
