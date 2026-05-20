from pathlib import Path

import pandas as pd


DATASET_DIR = Path("dataset_procesado/Base")
METRICS_DIR = Path("metricas")
CSV_PATH = METRICS_DIR / "revision_dataset_base.csv"


def revisar_archivo(file_path, class_name):
    """Devuelve una fila de revision para un archivo del dataset."""
    extension = file_path.suffix.lower()
    es_wav = extension == ".wav"
    nombre_sospechoso = es_wav and class_name.lower() not in file_path.stem.lower()

    advertencias = []
    if not es_wav:
        advertencias.append("NO_WAV")
    if nombre_sospechoso:
        advertencias.append("NOMBRE_NO_CONTIENE_CLASE")

    return {
        "clase": class_name,
        "archivo": file_path.name,
        "ruta": str(file_path),
        "extension": extension,
        "es_wav": es_wav,
        "nombre_sospechoso": nombre_sospechoso,
        "advertencias": ";".join(advertencias),
    }


def main():
    METRICS_DIR.mkdir(exist_ok=True)

    if not DATASET_DIR.exists():
        print(f"No existe la carpeta: {DATASET_DIR}")
        return

    rows = []
    counts = {}

    for class_dir in sorted(DATASET_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        files = sorted([path for path in class_dir.iterdir() if path.is_file()])
        wav_count = sum(1 for path in files if path.suffix.lower() == ".wav")
        counts[class_name] = wav_count

        for file_path in files:
            rows.append(revisar_archivo(file_path, class_name))

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    print("Revision de dataset base")
    print("=" * 80)
    print("Audios .wav por clase:")
    for class_name, count in counts.items():
        print(f"- {class_name}: {count}")

    if df.empty:
        print()
        print("No se encontraron archivos dentro de las carpetas de clases.")
        print(f"CSV generado en: {CSV_PATH}")
        return

    no_wav = df[df["es_wav"] == False]
    sospechosos = df[df["nombre_sospechoso"] == True]

    print()
    if no_wav.empty:
        print("No se detectaron archivos con extension diferente de .wav.")
    else:
        print("Advertencia: archivos que no son .wav:")
        for _, row in no_wav.iterrows():
            print(f"- Clase: {row['clase']} | Archivo: {row['archivo']}")

    print()
    if sospechosos.empty:
        print("No se detectaron nombres sospechosos.")
    else:
        print("Advertencia: archivos .wav cuyo nombre no contiene la clase:")
        for _, row in sospechosos.iterrows():
            print(f"- Clase: {row['clase']} | Archivo: {row['archivo']}")

    print()
    print(f"CSV generado en: {CSV_PATH}")
    print("No se modifico ni borro ningun archivo.")


if __name__ == "__main__":
    main()
