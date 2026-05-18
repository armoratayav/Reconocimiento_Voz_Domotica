from pathlib import Path
import pandas as pd

DATASET_DIR = Path("dataset_procesado")

def main():
    if not DATASET_DIR.exists():
        print("No existe dataset_procesado.")
        return

    rows = []

    for model_dir in DATASET_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        for class_dir in model_dir.iterdir():
            if not class_dir.is_dir():
                continue

            wav_files = list(class_dir.glob("*.wav"))

            rows.append({
                "modelo": model_dir.name,
                "clase": class_dir.name,
                "cantidad_audios": len(wav_files)
            })

    df = pd.DataFrame(rows)

    if df.empty:
        print("No se encontraron audios WAV.")
        return

    print(df.sort_values(["modelo", "clase"]).to_string(index=False))

    output_report = DATASET_DIR / "resumen_cantidades.csv"
    df.to_csv(output_report, index=False, encoding="utf-8")

    print()
    print(f"Resumen guardado en: {output_report}")

if __name__ == "__main__":
    main()