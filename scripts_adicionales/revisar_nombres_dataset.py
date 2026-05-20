from pathlib import Path

DATASET_DIR = Path("dataset_procesado/Base")

def main():
    if not DATASET_DIR.exists():
        print(f"No existe la carpeta: {DATASET_DIR}")
        return

    problemas = []

    for class_dir in sorted(DATASET_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        clase = class_dir.name.lower()

        for wav_file in sorted(class_dir.glob("*.wav")):
            nombre = wav_file.stem.lower()

            # Regla básica:
            # El nombre del archivo debería contener el nombre de la clase.
            if clase not in nombre:
                problemas.append((clase, wav_file.name, wav_file))

    if not problemas:
        print("No se detectaron archivos sospechosos por nombre.")
        return

    print("Archivos sospechosos encontrados:")
    print("=" * 80)

    for clase, nombre, ruta in problemas:
        print(f"Carpeta/clase: {clase:15s} | Archivo: {nombre}")
        print(f"Ruta: {ruta}")
        print("-" * 80)

    print()
    print(f"Total sospechosos: {len(problemas)}")
    print("Revisa manualmente si están en la carpeta correcta.")

if __name__ == "__main__":
    main()