from pathlib import Path
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from base_cnn_utils import CONFIDENCE_THRESHOLD, predict_audio


DATASET_CANDIDATES = [
    Path("audios_prueba_nuevos/Base"),
    Path("audios_prueba_nuevos/base"),
    Path("dataset_procesado/audios_prueba_nuevos/Base"),
    Path("dataset_procesado/audios_prueba_nuevos/base"),
]
RAW_DATASET_DIR = Path("dataset_original/audios_prueba_nuevos/base")
MODEL_PATH = Path("modelos/modelo_base_cnn.keras")
CLASSES_PATH = Path("modelos/base_cnn_classes.npy")
METRICS_DIR = Path("metricas")

REPORT_PATH = METRICS_DIR / "base_cnn_audios_nuevos_report.txt"
CONFUSION_MATRIX_PATH = METRICS_DIR / "base_cnn_audios_nuevos_confusion_matrix.png"
RESULTS_CSV_PATH = METRICS_DIR / "base_cnn_audios_nuevos_resultados.csv"


def find_dataset_root():
    """Busca la carpeta procesada de audios nuevos segun la estructura actual."""
    for candidate in DATASET_CANDIDATES:
        if candidate.exists():
            return candidate

    return None


def find_class_audio_dirs(dataset_root, class_names):
    """
    Busca carpetas de clase de forma recursiva.
    Esto tolera estructuras anidadas como:
    dataset_procesado/audios_prueba_nuevos/base/Base/Base/alarma
    """
    class_dirs = {}

    for class_name in class_names:
        matches = []

        for directory in dataset_root.rglob("*"):
            if not directory.is_dir():
                continue

            if directory.name.lower() != class_name.lower():
                continue

            wav_files = sorted(directory.glob("*.wav"))
            if wav_files:
                matches.append((directory, wav_files))

        if matches:
            # Si hay duplicados, usamos la carpeta con mas audios directos.
            matches.sort(key=lambda item: len(item[1]), reverse=True)
            class_dirs[class_name] = matches[0]

    return class_dirs


def save_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    plt.figure(figsize=(10, 8))
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Matriz de confusion - Audios nuevos CNN")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()


def main():
    METRICS_DIR.mkdir(exist_ok=True)

    if not MODEL_PATH.exists():
        print(f"No existe el modelo: {MODEL_PATH}")
        return

    if not CLASSES_PATH.exists():
        print(f"No existe el archivo de clases: {CLASSES_PATH}")
        return

    dataset_root = find_dataset_root()
    if dataset_root is None:
        print("No existe una carpeta procesada de audios nuevos en rutas conocidas:")
        for candidate in DATASET_CANDIDATES:
            print(f"- {candidate}")
        if RAW_DATASET_DIR.exists():
            print(f"Si tus audios estan en: {RAW_DATASET_DIR}")
            print("primero procesalos con:")
            print(r".\.venv\Scripts\python.exe scripts\procesar_audios.py")
        else:
            print("Coloca audios nuevos en dataset_original/audios_prueba_nuevos/base")
            print("y luego ejecuta scripts/procesar_audios.py")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    labels = [str(class_name) for class_name in classes]
    class_audio_dirs = find_class_audio_dirs(dataset_root, labels)

    print(f"Carpeta detectada de audios nuevos: {dataset_root}")

    missing_classes = [class_name for class_name in labels if class_name not in class_audio_dirs]
    if missing_classes:
        print()
        print("Advertencia: no se encontraron audios .wav para estas clases:")
        for class_name in missing_classes:
            print(f"- {class_name}")

    rows = []
    y_true = []
    y_pred = []

    print()
    print("Evaluando audios nuevos...")
    print("=" * 110)

    for clase_real in labels:
        if clase_real not in class_audio_dirs:
            continue

        class_dir, wav_files = class_audio_dirs[clase_real]

        print()
        print(f"Clase real: {clase_real}")
        print(f"Carpeta: {class_dir}")
        print("-" * 110)

        for wav_file in wav_files:
            clase_predicha, confianza, _ = predict_audio(model, classes, wav_file)
            rechazado = confianza < CONFIDENCE_THRESHOLD
            acierto = clase_real == clase_predicha
            estado = "OK" if acierto else "ERROR"

            y_true.append(clase_real)
            y_pred.append(clase_predicha)

            rows.append(
                {
                    "archivo": str(wav_file),
                    "clase_real": clase_real,
                    "clase_predicha": clase_predicha,
                    "confianza": confianza,
                    "estado": estado,
                    "rechazado_baja_confianza": rechazado,
                }
            )

            rechazo_texto = "RECHAZADO" if rechazado else "ACEPTADO"
            print(
                f"{estado:5s} | Real: {clase_real:12s} | "
                f"Pred: {clase_predicha:12s} | Confianza: {confianza:.4f} | "
                f"{rechazo_texto:9s} | Archivo: {wav_file.name}"
            )

    if not rows:
        print("No se encontraron audios .wav en la carpeta de audios nuevos.")
        return

    total = len(rows)
    correctos = sum(row["estado"] == "OK" for row in rows)
    accuracy = correctos / total

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=labels,
        zero_division=0,
    )

    pd.DataFrame(rows).to_csv(RESULTS_CSV_PATH, index=False, encoding="utf-8")
    REPORT_PATH.write_text(report, encoding="utf-8")
    save_confusion_matrix(y_true, y_pred, labels)

    print()
    print("=" * 110)
    print(f"Total de audios nuevos: {total}")
    print(f"Correctos: {correctos}")
    print(f"Accuracy sobre audios nuevos: {accuracy:.4f}")
    print()
    print("Classification report:")
    print(report)
    print()
    print(f"Reporte guardado en: {REPORT_PATH}")
    print(f"Matriz de confusion guardada en: {CONFUSION_MATRIX_PATH}")
    print(f"Resultados CSV guardados en: {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()
