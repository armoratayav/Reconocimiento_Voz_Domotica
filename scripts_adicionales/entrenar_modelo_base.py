from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical


# ==========================
# CONFIGURACIÓN
# ==========================

DATASET_DIR = Path("dataset_procesado/Base")
MODEL_DIR = Path("modelos")
METRICS_DIR = Path("metricas")

MODEL_DIR.mkdir(exist_ok=True)
METRICS_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000
N_MFCC = 13
MAX_DURATION = 3.0

# Si cada audio está a 16 kHz y dura máximo 3 segundos:
# 16000 * 3 = 48000 muestras
MAX_SAMPLES = int(SAMPLE_RATE * MAX_DURATION)


def load_audio_fixed_length(file_path):
    """
    Carga un audio en mono a 16 kHz.
    Si dura menos de 3 segundos, rellena con ceros.
    Si dura más de 3 segundos, lo recorta.
    """
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    if len(audio) > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]
    else:
        padding = MAX_SAMPLES - len(audio)
        audio = np.pad(audio, (0, padding), mode="constant")

    return audio


def extract_mfcc(file_path):
    """
    Extrae MFCC del audio y devuelve un vector promedio.
    """
    audio = load_audio_fixed_length(file_path)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    # Promediamos cada coeficiente MFCC en el tiempo.
    # Resultado final: vector de 13 números.
    mfcc_mean = np.mean(mfcc, axis=1)

    return mfcc_mean


def load_dataset():
    """
    Lee todas las carpetas de clases dentro de dataset_procesado/Base.
    Cada carpeta representa una clase.
    """
    X = []
    y = []

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta: {DATASET_DIR}")

    class_dirs = sorted([p for p in DATASET_DIR.iterdir() if p.is_dir()])

    print("Clases encontradas:")
    for class_dir in class_dirs:
        print(f"- {class_dir.name}")

    for class_dir in class_dirs:
        class_name = class_dir.name
        wav_files = sorted(class_dir.glob("*.wav"))

        print(f"Cargando clase '{class_name}' con {len(wav_files)} audios...")

        for wav_file in wav_files:
            try:
                features = extract_mfcc(wav_file)
                X.append(features)
                y.append(class_name)
            except Exception as e:
                print(f"Error procesando {wav_file}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    return X, y


def build_model(input_shape, num_classes):
    """
    Modelo base simple tipo MLP.
    Recibe 13 características MFCC promedio.
    """
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation="relu"),
        Dropout(0.2),

        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def plot_training_history(history):
    """
    Guarda gráficas de accuracy y loss.
    """
    plt.figure()
    plt.plot(history.history["accuracy"], label="accuracy entrenamiento")
    plt.plot(history.history["val_accuracy"], label="accuracy validación")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.title("Accuracy - Modelo Base")
    plt.legend()
    plt.savefig(METRICS_DIR / "base_accuracy.png")
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="loss entrenamiento")
    plt.plot(history.history["val_loss"], label="loss validación")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Loss - Modelo Base")
    plt.legend()
    plt.savefig(METRICS_DIR / "base_loss.png")
    plt.close()


def main():
    print("Cargando dataset base...")
    X, y = load_dataset()

    print()
    print(f"Total de muestras: {len(X)}")
    print(f"Forma de X: {X.shape}")
    print(f"Forma de y: {y.shape}")

    # Convertir etiquetas de texto a números
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    print()
    print("Mapeo de clases:")
    for index, class_name in enumerate(label_encoder.classes_):
        print(f"{index}: {class_name}")

    # Guardar clases para usarlas después en predicción
    np.save(MODEL_DIR / "base_classes.npy", label_encoder.classes_)

    # Separar entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_categorical,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print()
    print(f"Entrenamiento: {len(X_train)} muestras")
    print(f"Prueba: {len(X_test)} muestras")

    model = build_model(
        input_shape=X.shape[1],
        num_classes=y_categorical.shape[1]
    )

    model.summary()

    print()
    print("Entrenando modelo...")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=60,
        batch_size=16,
        verbose=1
    )

    print()
    print("Evaluando modelo...")

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Accuracy en prueba: {test_accuracy:.4f}")
    print(f"Loss en prueba: {test_loss:.4f}")

    # Predicciones
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_
    )

    print()
    print("Reporte de clasificación:")
    print(report)

    with open(METRICS_DIR / "base_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_
    )

    plt.figure(figsize=(10, 8))
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Matriz de Confusión - Modelo Base")
    plt.tight_layout()
    plt.savefig(METRICS_DIR / "base_confusion_matrix.png")
    plt.close()

    plot_training_history(history)

    # Guardar modelo
    model.save(MODEL_DIR / "modelo_base.keras")

    print()
    print("Modelo guardado en:")
    print(MODEL_DIR / "modelo_base.keras")

    print()
    print("Clases guardadas en:")
    print(MODEL_DIR / "base_classes.npy")

    print()
    print("Métricas guardadas en:")
    print(METRICS_DIR)


if __name__ == "__main__":
    main()