from pathlib import Path
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from base_cnn_utils import extract_mfcc_cnn, list_class_dirs


DATASET_DIR = Path("dataset_procesado/Base")
MODEL_DIR = Path("modelos")
MODEL_PATH = MODEL_DIR / "debug_overfit_cnn_base.keras"

AUDIOS_POR_CLASE = 5
EPOCHS = 150
BATCH_SIZE = 8


class StopAtAccuracy(tf.keras.callbacks.Callback):
    """Detiene el entrenamiento cuando accuracy llega al valor objetivo."""

    def __init__(self, target_accuracy=1.0):
        super().__init__()
        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        accuracy = logs.get("accuracy")

        if accuracy is not None and accuracy >= self.target_accuracy:
            print()
            print(f"Accuracy {accuracy:.4f} alcanzada en epoca {epoch + 1}.")
            self.model.stop_training = True


def load_debug_subset():
    """Carga exactamente 5 audios .wav por clase."""
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta: {DATASET_DIR}")

    class_dirs = list_class_dirs(DATASET_DIR)
    if not class_dirs:
        raise ValueError(f"No se encontraron clases en: {DATASET_DIR}")

    X = []
    y = []
    used_files = []

    print("Clases encontradas:")
    for class_dir in class_dirs:
        print(f"- {class_dir.name}")

    print()
    print("Archivos usados:")
    for class_dir in class_dirs:
        class_name = class_dir.name
        wav_files = sorted(class_dir.glob("*.wav"))[:AUDIOS_POR_CLASE]

        if len(wav_files) < AUDIOS_POR_CLASE:
            raise ValueError(
                f"La clase '{class_name}' tiene {len(wav_files)} audios .wav, "
                f"pero se necesitan {AUDIOS_POR_CLASE}."
            )

        for wav_file in wav_files:
            print(f"- {class_name:12s} | {wav_file}")
            X.append(extract_mfcc_cnn(wav_file))
            y.append(class_name)
            used_files.append(wav_file)

    return np.array(X, dtype=np.float32), np.array(y), used_files


def build_debug_model(input_shape, num_classes):
    """CNN pequena para comprobar si puede memorizar 35 muestras."""
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def print_predictions(model, X, y_true, used_files, class_names):
    probabilities = model.predict(X, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    true_indexes = np.argmax(y_true, axis=1)

    print()
    print("Prediccion por archivo:")
    print("=" * 100)
    for file_path, true_index, predicted_index, probs in zip(
        used_files,
        true_indexes,
        predictions,
        probabilities,
    ):
        real_class = class_names[true_index]
        predicted_class = class_names[predicted_index]
        confidence = float(probs[predicted_index])
        status = "OK" if real_class == predicted_class else "ERROR"

        print(
            f"{status:5s} | Real: {real_class:12s} | "
            f"Pred: {predicted_class:12s} | "
            f"Confianza: {confidence:.4f} | Archivo: {file_path.name}"
        )


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    X, labels, used_files = load_debug_subset()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    y_categorical = to_categorical(y_encoded)

    print()
    print(f"Forma de X: {X.shape}")
    print(f"Forma de y: {y_categorical.shape}")
    print()
    print("Mapeo de clases:")
    for index, class_name in enumerate(label_encoder.classes_):
        print(f"{index}: {class_name}")

    model = build_debug_model(
        input_shape=X.shape[1:],
        num_classes=len(label_encoder.classes_),
    )

    model.summary()

    print()
    print("Entrenando y evaluando sobre el mismo subconjunto...")
    history = model.fit(
        X,
        y_categorical,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[StopAtAccuracy(target_accuracy=1.0)],
        verbose=1,
    )

    final_loss, final_accuracy = model.evaluate(X, y_categorical, verbose=0)

    print()
    print(f"Accuracy final sobre el mismo conjunto: {final_accuracy:.4f}")
    print(f"Loss final sobre el mismo conjunto: {final_loss:.4f}")

    print_predictions(
        model,
        X,
        y_categorical,
        used_files,
        label_encoder.classes_,
    )

    model.save(MODEL_PATH)

    print()
    print(f"Modelo temporal guardado en: {MODEL_PATH}")

    if final_accuracy < 0.95:
        print()
        print(
            "ALERTA: la CNN no logra memorizar un dataset pequeño. "
            "Revisar preprocesamiento, etiquetas, modelo o datos."
        )


if __name__ == "__main__":
    main()
