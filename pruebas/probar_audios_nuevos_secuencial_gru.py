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

from secuencial_gru_utils import CONFIDENCE_THRESHOLD, list_class_dirs, predict_audio


DATASET_DIR = Path("audios_prueba_nuevos/Secuencial")
MODEL_PATH = Path("modelos/modelo_secuencial_gru.keras")
CLASSES_PATH = Path("modelos/secuencial_gru_classes.npy")
METRICS_DIR = Path("metricas")

REPORT_PATH = METRICS_DIR / "secuencial_gru_audios_nuevos_report.txt"
CONFUSION_MATRIX_PATH = METRICS_DIR / "secuencial_gru_audios_nuevos_confusion_matrix.png"
RESULTS_CSV_PATH = METRICS_DIR / "secuencial_gru_audios_nuevos_resultados.csv"


def save_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(11, 9))
    disp.plot(cmap="Blues", xticks_rotation=45, ax=ax)
    ax.set_title("Matriz de confusion - Audios nuevos GRU")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH)
    plt.close(fig)


def main():
    METRICS_DIR.mkdir(exist_ok=True)

    if not DATASET_DIR.exists():
        print(f"Todavia no hay audios nuevos para evaluar.")
        print(f"Crea la carpeta con clases en: {DATASET_DIR}")
        return

    if not MODEL_PATH.exists():
        print(f"No existe el modelo: {MODEL_PATH}")
        return

    if not CLASSES_PATH.exists():
        print(f"No existe el archivo de clases: {CLASSES_PATH}")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    labels = [str(class_name) for class_name in classes]

    rows = []
    y_true = []
    y_pred = []

    print()
    print("Evaluando audios nuevos secuenciales...")
    print("=" * 130)

    for class_dir in list_class_dirs(DATASET_DIR):
        clase_real = class_dir.name
        wav_files = sorted(class_dir.glob("*.wav"))

        print()
        print(f"Clase real: {clase_real}")
        print("-" * 130)

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
                f"{estado:5s} | Real: {clase_real:20s} | Pred: {clase_predicha:20s} | "
                f"Confianza: {confianza:.4f} | {rechazo_texto:9s} | Archivo: {wav_file.name}"
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
    print("=" * 130)
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
