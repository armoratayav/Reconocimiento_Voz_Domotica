from pathlib import Path
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf

from secuencial_gru_utils import list_class_dirs, predict_audio


DATASET_DIR = Path("dataset_procesado/Secuencial")
MODEL_PATH = Path("modelos/modelo_secuencial_gru.keras")
CLASSES_PATH = Path("modelos/secuencial_gru_classes.npy")
METRICS_DIR = Path("metricas")
CSV_PATH = METRICS_DIR / "secuencial_gru_prueba_varios.csv"

AUDIOS_POR_CLASE = 10


def main():
    METRICS_DIR.mkdir(exist_ok=True)

    if not MODEL_PATH.exists():
        print(f"No existe el modelo: {MODEL_PATH}")
        return

    if not CLASSES_PATH.exists():
        print(f"No existe el archivo de clases: {CLASSES_PATH}")
        return

    if not DATASET_DIR.exists():
        print(f"No existe el dataset: {DATASET_DIR}")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    classes = np.load(CLASSES_PATH, allow_pickle=True)

    rows = []
    errores = []
    total = 0
    correctos = 0

    print()
    print("Probando audios del dataset secuencial con GRU...")
    print("=" * 120)

    for class_dir in list_class_dirs(DATASET_DIR):
        clase_real = class_dir.name
        wav_files = sorted(class_dir.glob("*.wav"))[:AUDIOS_POR_CLASE]

        print()
        print(f"Clase real: {clase_real}")
        print("-" * 120)

        for wav_file in wav_files:
            clase_predicha, confianza, _ = predict_audio(model, classes, wav_file)
            acierto = clase_real == clase_predicha
            estado = "OK" if acierto else "ERROR"

            total += 1
            if acierto:
                correctos += 1
            else:
                errores.append((clase_real, clase_predicha, confianza, wav_file.name))

            rows.append(
                {
                    "archivo": str(wav_file),
                    "clase_real": clase_real,
                    "clase_predicha": clase_predicha,
                    "confianza": confianza,
                    "estado": estado,
                }
            )

            print(
                f"{estado:5s} | Real: {clase_real:20s} | "
                f"Pred: {clase_predicha:20s} | "
                f"Confianza: {confianza:.4f} | Archivo: {wav_file.name}"
            )

    pd.DataFrame(rows).to_csv(CSV_PATH, index=False, encoding="utf-8")

    print()
    print("=" * 120)

    if total == 0:
        print("No se encontraron audios para probar.")
        return

    accuracy = correctos / total
    print(f"Total de audios probados: {total}")
    print(f"Correctos: {correctos}")
    print(f"Accuracy manual: {accuracy:.4f}")
    print(f"CSV guardado en: {CSV_PATH}")

    print()
    print(f"Errores encontrados: {len(errores)}")
    for clase_real, clase_predicha, confianza, filename in errores:
        print(
            f"Real: {clase_real:20s} | Pred: {clase_predicha:20s} | "
            f"Confianza: {confianza:.4f} | Archivo: {filename}"
        )


if __name__ == "__main__":
    main()
