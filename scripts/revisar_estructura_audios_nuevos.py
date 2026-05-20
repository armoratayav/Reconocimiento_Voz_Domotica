from pathlib import Path
import csv


DATASET_DIR = Path("audios_prueba_nuevos/Base")
METRICS_DIR = Path("metricas")
REPORT_PATH = METRICS_DIR / "revision_audios_nuevos_base.csv"

EXPECTED_CLASSES = [
    "alarma",
    "apaga",
    "enciende",
    "puerta",
    "ruido_fondo",
    "seguro",
    "ventilador",
]


def has_nested_base(directory):
    parts = [part.lower() for part in directory.parts]

    for index in range(len(parts) - 2):
        if parts[index : index + 3] == ["base", "base", "base"]:
            return True

    return False


def write_report(rows):
    METRICS_DIR.mkdir(exist_ok=True)
    fieldnames = ["tipo", "ruta", "clase", "cantidad_wav", "detalle"]

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = []

    if not DATASET_DIR.exists():
        print(f"No existe la carpeta: {DATASET_DIR}")
        rows.append(
            {
                "tipo": "ERROR",
                "ruta": str(DATASET_DIR),
                "clase": "",
                "cantidad_wav": 0,
                "detalle": "No existe la carpeta de audios nuevos.",
            }
        )
        write_report(rows)
        return

    print(f"Revisando estructura: {DATASET_DIR}")
    print()

    direct_dirs = sorted([path for path in DATASET_DIR.iterdir() if path.is_dir()])
    direct_files = sorted([path for path in DATASET_DIR.iterdir() if path.is_file()])
    direct_dir_names = {path.name.lower() for path in direct_dirs}
    expected_set = set(EXPECTED_CLASSES)

    missing = sorted(expected_set - direct_dir_names)
    extra = sorted(direct_dir_names - expected_set)

    if not missing and not extra:
        print("Carpetas principales: OK")
    else:
        print("Carpetas principales: revisar")

    for class_name in missing:
        print(f"- Falta carpeta: {class_name}")
        rows.append(
            {
                "tipo": "FALTA_CARPETA",
                "ruta": str(DATASET_DIR / class_name),
                "clase": class_name,
                "cantidad_wav": 0,
                "detalle": "No existe carpeta esperada.",
            }
        )

    for directory in direct_dirs:
        if directory.name.lower() not in expected_set:
            print(f"- Carpeta no esperada: {directory}")
            rows.append(
                {
                    "tipo": "CARPETA_NO_ESPERADA",
                    "ruta": str(directory),
                    "clase": "",
                    "cantidad_wav": 0,
                    "detalle": "Carpeta directa fuera de las clases esperadas.",
                }
            )

    for file_path in direct_files:
        print(f"- Archivo fuera de lugar en raiz: {file_path}")
        rows.append(
            {
                "tipo": "ARCHIVO_FUERA_DE_LUGAR",
                "ruta": str(file_path),
                "clase": "",
                "cantidad_wav": 0,
                "detalle": "Archivo ubicado directamente en audios_prueba_nuevos/Base.",
            }
        )

    print()
    print("Conteo por clase:")
    for class_name in EXPECTED_CLASSES:
        class_dir = DATASET_DIR / class_name
        wav_files = sorted(class_dir.glob("*.wav")) if class_dir.exists() else []

        print(f"- {class_name}: {len(wav_files)} audios .wav")
        rows.append(
            {
                "tipo": "CONTEO_CLASE",
                "ruta": str(class_dir),
                "clase": class_name,
                "cantidad_wav": len(wav_files),
                "detalle": "Conteo de audios .wav directos.",
            }
        )

        if class_dir.exists():
            nested_files = [
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.parent != class_dir
            ]
            for file_path in nested_files:
                rows.append(
                    {
                        "tipo": "ARCHIVO_ANIDADO",
                        "ruta": str(file_path),
                        "clase": class_name,
                        "cantidad_wav": 0,
                        "detalle": "Archivo dentro de subcarpeta de una clase.",
                    }
                )

    nested_base_dirs = [
        directory
        for directory in DATASET_DIR.rglob("*")
        if directory.is_dir() and has_nested_base(directory)
    ]

    if nested_base_dirs:
        print()
        print("Advertencia: se encontraron carpetas anidadas tipo Base/Base/Base:")
        for directory in nested_base_dirs:
            print(f"- {directory}")
            rows.append(
                {
                    "tipo": "BASE_ANIDADA",
                    "ruta": str(directory),
                    "clase": "",
                    "cantidad_wav": 0,
                    "detalle": "Posible estructura anidada Base/Base/Base.",
                }
            )

    expected_class_paths = {DATASET_DIR / class_name for class_name in EXPECTED_CLASSES}
    for file_path in DATASET_DIR.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.parent in expected_class_paths:
            continue

        if file_path.parent == DATASET_DIR:
            continue

        if any(file_path == Path(row["ruta"]) for row in rows):
            continue

        rows.append(
            {
                "tipo": "ARCHIVO_FUERA_DE_LUGAR",
                "ruta": str(file_path),
                "clase": "",
                "cantidad_wav": 0,
                "detalle": "Archivo fuera de las carpetas directas esperadas.",
            }
        )

    write_report(rows)
    print()
    print(f"Revision guardada en: {REPORT_PATH}")


if __name__ == "__main__":
    main()
