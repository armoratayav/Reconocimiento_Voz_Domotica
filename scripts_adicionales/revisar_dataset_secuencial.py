from pathlib import Path

import pandas as pd


DATASET_DIR = Path("dataset_procesado/Secuencial")
METRICS_DIR = Path("metricas")
CSV_PATH = METRICS_DIR / "revision_dataset_secuencial.csv"

MIN_FILE_SIZE_BYTES = 1024


def revisar_archivo(file_path, class_name):
    """Devuelve una fila de revision para un archivo del dataset."""
    extension = file_path.suffix.lower()
    es_wav = extension == ".wav"
    file_size = file_path.stat().st_size
    nombre_sospechoso = es_wav and class_name.lower() not in file_path.stem.lower()
    archivo_muy_pequeno = file_size < MIN_FILE_SIZE_BYTES

    advertencias = []
    if not es_wav:
        advertencias.append("NO_WAV")
    if nombre_sospechoso:
        advertencias.append("NOMBRE_NO_CONTIENE_CLASE")
    if archivo_muy_pequeno:
        advertencias.append("ARCHIVO_VACIO_O_MUY_PEQUENO")

    return {
        "clase": class_name,
        "archivo": file_path.name,
        "ruta": str(file_path),
        "extension": extension,
        "tamano_bytes": file_size,
        "es_wav": es_wav,
        "nombre_sospechoso": nombre_sospechoso,
        "archivo_muy_pequeno": archivo_muy_pequeno,
        "carpeta_sin_audios": False,
        "advertencias": ";".join(advertencias),
    }


def main():
    METRICS_DIR.mkdir(exist_ok=True)

    if not DATASET_DIR.exists():
        print(f"No existe la carpeta: {DATASET_DIR}")
        return

    rows = []
    counts = {}

    class_dirs = sorted([path for path in DATASET_DIR.iterdir() if path.is_dir()])

    for class_dir in class_dirs:
        class_name = class_dir.name
        files = sorted([path for path in class_dir.iterdir() if path.is_file()])
        wav_count = sum(1 for path in files if path.suffix.lower() == ".wav")
        counts[class_name] = wav_count

        if wav_count == 0:
            rows.append(
                {
                    "clase": class_name,
                    "archivo": "",
                    "ruta": str(class_dir),
                    "extension": "",
                    "tamano_bytes": 0,
                    "es_wav": False,
                    "nombre_sospechoso": False,
                    "archivo_muy_pequeno": False,
                    "carpeta_sin_audios": True,
                    "advertencias": "CARPETA_SIN_AUDIOS",
                }
            )

        for file_path in files:
            rows.append(revisar_archivo(file_path, class_name))

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    print("Revision de dataset secuencial")
    print("=" * 90)
    print("Audios .wav por clase:")
    for class_name, count in counts.items():
        print(f"- {class_name}: {count}")

    if df.empty:
        print()
        print("No se encontraron archivos dentro de las carpetas de clases.")
        print(f"CSV generado en: {CSV_PATH}")
        return

    problemas = df[df["advertencias"].astype(str) != ""]

    print()
    if problemas.empty:
        print("No se detectaron advertencias.")
    else:
        print("Advertencias encontradas:")
        for _, row in problemas.iterrows():
            print(
                f"- Clase: {row['clase']} | Archivo: {row['archivo']} | "
                f"Advertencias: {row['advertencias']}"
            )

    print()
    print(f"CSV generado en: {CSV_PATH}")
    print("No se modifico ni borro ningun archivo.")


if __name__ == "__main__":
    main()
