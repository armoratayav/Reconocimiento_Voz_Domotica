from pathlib import Path
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from base_cnn_utils import TARGET_FRAMES, extract_mfcc_cnn, list_class_dirs


DATASET_DIR = Path("dataset_procesado/Base")
MODEL_DIR = Path("modelos")
METRICS_DIR = Path("metricas")

MODEL_PATH = MODEL_DIR / "modelo_base_cnn.keras"
CLASSES_PATH = MODEL_DIR / "base_cnn_classes.npy"
REPORT_PATH = METRICS_DIR / "base_cnn_classification_report.txt"
CONFUSION_MATRIX_PATH = METRICS_DIR / "base_cnn_confusion_matrix.png"
ACCURACY_PATH = METRICS_DIR / "base_cnn_accuracy.png"
LOSS_PATH = METRICS_DIR / "base_cnn_loss.png"

RANDOM_STATE = 42
EPOCHS = 80
BATCH_SIZE = 16


def ensure_output_dirs():
    MODEL_DIR.mkdir(exist_ok=True)
    METRICS_DIR.mkdir(exist_ok=True)


def load_dataset():
    """Lee audios .wav por clase y devuelve X, y y conteos por clase."""
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta: {DATASET_DIR}")

    class_dirs = list_class_dirs(DATASET_DIR)
    if not class_dirs:
        raise ValueError(f"No se encontraron carpetas de clases en: {DATASET_DIR}")

    X = []
    y = []
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
                X.append(extract_mfcc_cnn(wav_file))
                y.append(class_name)
            except Exception as exc:
                print(f"Error procesando {wav_file}: {exc}")

    if not X:
        raise ValueError("No se pudo cargar ningun audio .wav valido.")

    return np.array(X, dtype=np.float32), np.array(y), counts


def build_model(input_shape, num_classes):
    """CNN 2D entrenada desde cero sobre MFCC completo."""
    model = Sequential(
        [
            Input(shape=input_shape),

            Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.30),

            Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            GlobalAveragePooling2D(),

            Dense(64, activation="relu"),
            Dropout(0.40),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def split_dataset(X, y_encoded, y_categorical):
    """Divide en 70% entrenamiento, 15% validacion y 15% prueba."""
    X_train, X_temp, y_train, y_temp, encoded_train, encoded_temp = train_test_split(
        X,
        y_categorical,
        y_encoded,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=encoded_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_training_history(history):
    """Guarda graficas de accuracy y loss."""
    plt.figure()
    plt.plot(history.history["accuracy"], label="accuracy entrenamiento")
    plt.plot(history.history["val_accuracy"], label="accuracy validacion")
    plt.xlabel("Epoca")
    plt.ylabel("Accuracy")
    plt.title("Accuracy - Modelo Base CNN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ACCURACY_PATH)
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="loss entrenamiento")
    plt.plot(history.history["val_loss"], label="loss validacion")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.title("Loss - Modelo Base CNN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PATH)
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    plt.figure(figsize=(10, 8))
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Matriz de confusion - Modelo Base CNN")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()


def main():
    ensure_output_dirs()

    print("Cargando dataset base para CNN...")
    X, y, counts = load_dataset()

    print()
    print("Cantidad de audios por clase:")
    for class_name, count in counts.items():
        print(f"- {class_name}: {count}")

    print()
    print(f"Forma de X: {X.shape}")
    print(f"Frames MFCC usados: {TARGET_FRAMES}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    np.save(CLASSES_PATH, label_encoder.classes_)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X,
        y_encoded,
        y_categorical,
    )

    print()
    print(f"Muestras de entrenamiento: {len(X_train)}")
    print(f"Muestras de validacion: {len(X_val)}")
    print(f"Muestras de prueba: {len(X_test)}")

    model = build_model(input_shape=X.shape[1:], num_classes=y_categorical.shape[1])
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
    )

    print()
    print("Entrenando CNN 2D desde cero...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1,
    )

    print()
    print("Evaluando con conjunto de prueba interno...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
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

    model.save(MODEL_PATH)

    print()
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Clases guardadas en: {CLASSES_PATH}")
    print(f"Metricas guardadas en: {METRICS_DIR}")


if __name__ == "__main__":
    main()
