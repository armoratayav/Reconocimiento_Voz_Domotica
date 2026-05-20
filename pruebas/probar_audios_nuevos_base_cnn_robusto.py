from pathlib import Path
import csv
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from entrenar_modelo_base_cnn_robusto import EXPECTED_CLASSES, extract_mfcc_cnn_robusto


DATASET_DIR = Path("audios_prueba_nuevos/Base")
MODEL_PATH = Path("modelos/modelo_base_cnn_robusto.keras")
CLASSES_PATH = Path("modelos/base_cnn_robusto_classes.npy")
METRICS_DIR = Path("metricas")

RESULTS_CSV_PATH = METRICS_DIR / "base_cnn_robusto_audios_nuevos_resultados.csv"
REPORT_PATH = METRICS_DIR / "base_cnn_robusto_audios_nuevos_report.txt"
CONFUSION_MATRIX_PATH = METRICS_DIR / "base_cnn_robusto_audios_nuevos_confusion_matrix.png"

UMBRAL = 0.85


def find_class_audio_dirs(dataset_root):
    """
    Busca carpetas de clase desde audios_prueba_nuevos/Base.
    Si hay anidamientos accidentales, usa la coincidencia con mas .wav directos.
    """
    class_dirs = {}

    for class_name in EXPECTED_CLASSES:
        matches = []

        direct_dir = dataset_root / class_name
        if direct_dir.exists():
            wav_files = sorted(direct_dir.glob("*.wav"))
            if wav_files:
                matches.append((direct_dir, wav_files))

        for directory in dataset_root.rglob("*"):
            if not directory.is_dir():
                continue

            if directory.name.lower() != class_name:
                continue

            wav_files = sorted(directory.glob("*.wav"))
            if wav_files:
                matches.append((directory, wav_files))

        if matches:
            matches.sort(key=lambda item: len(item[1]), reverse=True)
            class_dirs[class_name] = matches[0]

    return class_dirs


def predict_audio(model, classes, audio_path):
    features = extract_mfcc_cnn_robusto(audio_path)
    features = np.expand_dims(features, axis=0)
    probabilities = model.predict(features, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = str(classes[predicted_index])
    confidence = float(probabilities[predicted_index])

    return predicted_class, confidence, probabilities


def save_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap="Blues", xticks_rotation=45, ax=ax)
    ax.set_title("Matriz de confusion - Audios nuevos CNN robusto")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH)
    plt.close(fig)


def write_results_csv(rows):
    fieldnames = [
        "archivo",
        "clase_real",
        "clase_predicha",
        "confianza",
        "estado",
        "decision",
    ]

    with RESULTS_CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    METRICS_DIR.mkdir(exist_ok=True)

    if not MODEL_PATH.exists():
        print(f"No existe el modelo: {MODEL_PATH}")
        print("Entrena primero con: python scripts/entrenar_modelo_base_cnn_robusto.py")
        return

    if not CLASSES_PATH.exists():
        print(f"No existe el archivo de clases: {CLASSES_PATH}")
        return

    if not DATASET_DIR.exists():
        print(f"No existe la carpeta de audios nuevos: {DATASET_DIR}")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    labels = [str(class_name) for class_name in classes]
    class_audio_dirs = find_class_audio_dirs(DATASET_DIR)

    rows = []
    y_true = []
    y_pred = []

    print()
    print("Evaluando audios nuevos con modelo base CNN robusto...")
    print(f"Carpeta: {DATASET_DIR}")
    print(f"Umbral: {UMBRAL:.2f}")
    print("=" * 110)

    for class_name in EXPECTED_CLASSES:
        if class_name not in class_audio_dirs:
            print(f"Advertencia: no se encontraron audios .wav para clase {class_name}")
            continue

        class_dir, wav_files = class_audio_dirs[class_name]
        print()
        print(f"Clase real: {class_name}")
        print(f"Carpeta: {class_dir}")
        print("-" * 110)

        for wav_file in wav_files:
            predicted_class, confidence, _ = predict_audio(model, classes, wav_file)
            accepted = confidence >= UMBRAL
            correct = class_name == predicted_class
            status = "OK" if correct else "ERROR"

            if not accepted:
                decision = "RECHAZADO"
            elif predicted_class == "ruido_fondo":
                decision = "SIN_ACCION"
            else:
                decision = "ACEPTADO"

            y_true.append(class_name)
            y_pred.append(predicted_class)
            rows.append(
                {
                    "archivo": str(wav_file),
                    "clase_real": class_name,
                    "clase_predicha": predicted_class,
                    "confianza": f"{confidence:.6f}",
                    "estado": status,
                    "decision": decision,
                }
            )

            print(
                f"{status:5s} | Real: {class_name:12s} | "
                f"Pred: {predicted_class:12s} | Confianza: {confidence:.4f} | "
                f"{decision:10s} | Archivo: {wav_file.name}"
            )

    if not rows:
        print("No se encontraron audios .wav en audios_prueba_nuevos/Base.")
        return

    total = len(rows)
    correct = sum(row["estado"] == "OK" for row in rows)
    accuracy = correct / total

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=labels,
        zero_division=0,
    )

    write_results_csv(rows)
    REPORT_PATH.write_text(report, encoding="utf-8")
    save_confusion_matrix(y_true, y_pred, labels)

    print()
    print("=" * 110)
    print(f"Total de audios nuevos: {total}")
    print(f"Correctos: {correct}")
    print(f"Accuracy sobre audios nuevos: {accuracy:.4f}")
    print()
    print("Classification report:")
    print(report)
    print()
    print(f"Resultados CSV guardados en: {RESULTS_CSV_PATH}")
    print(f"Reporte guardado en: {REPORT_PATH}")
    print(f"Matriz de confusion guardada en: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
