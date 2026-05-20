from pathlib import Path
import argparse
import csv
import subprocess
import sys


# ==========================
# CONFIGURACION PRINCIPAL
# ==========================

SAMPLE_RATE = 16000
CHANNELS = 1
TARGET_LUFS = -20
MAX_DURATION_SECONDS = 3.0

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".flac",
    ".webm",
}

METRICS_DIR = Path("metricas")
REPORT_PATH = METRICS_DIR / "reporte_procesamiento_audios.csv"
SUMMARY_PATH = METRICS_DIR / "resumen_procesamiento_audios.csv"

# Entradas y salidas principales.
# Los audios de prueba nuevos quedan fuera de dataset_procesado para evitar
# mezclarlos accidentalmente con el entrenamiento.
PROCESS_TARGETS = [
    {
        "name": "Base entrenamiento",
        "input_dir": Path("dataset_original/Base"),
        "output_dir": Path("dataset_procesado/Base"),
    },
    {
        "name": "Secuencial entrenamiento",
        "input_dir": Path("dataset_original/Secuencial"),
        "output_dir": Path("dataset_procesado/Secuencial"),
    },
    {
        "name": "Base audios nuevos",
        "input_dir": Path("dataset_original/audios_prueba_nuevos/base"),
        "output_dir": Path("audios_prueba_nuevos/Base"),
    },
    {
        "name": "Secuencial audios nuevos",
        "input_dir": Path("dataset_original/audios_prueba_nuevos/secuencial"),
        "output_dir": Path("audios_prueba_nuevos/Secuencial"),
    },
]


def run_command(command):
    """Ejecuta un comando externo y lanza error si falla."""
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result


def check_ffmpeg_tools():
    """Comprueba que ffmpeg y ffprobe existan en el PATH."""
    for tool in ("ffmpeg", "ffprobe"):
        try:
            run_command([tool, "-version"])
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo ejecutar '{tool}'. Instala ffmpeg/ffprobe "
                f"y verifica que esten en el PATH. Detalle: {exc}"
            )


def get_duration_seconds(audio_path):
    """Obtiene duracion del audio usando ffprobe."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]

    result = run_command(command)

    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def find_audio_files(input_dir):
    """Lista audios validos dentro de una carpeta."""
    if not input_dir.exists():
        return []

    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def process_audio(input_path, output_path):
    """
    Convierte audio a WAV mono 16 kHz, normalizado y con maximo 3 segundos.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_filter = f"loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11"

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-t",
        str(MAX_DURATION_SECONDS),
        "-ac",
        str(CHANNELS),
        "-ar",
        str(SAMPLE_RATE),
        "-af",
        audio_filter,
        "-sample_fmt",
        "s16",
        str(output_path),
    ]

    run_command(command)


def build_plan():
    """Construye la lista de archivos a procesar para las carpetas configuradas."""
    plan = []

    for target in PROCESS_TARGETS:
        input_dir = target["input_dir"]
        output_dir = target["output_dir"]
        audio_files = find_audio_files(input_dir)

        for input_path in audio_files:
            relative_path = input_path.relative_to(input_dir)
            output_path = output_dir / relative_path.with_suffix(".wav")
            plan.append(
                {
                    "grupo": target["name"],
                    "input_dir": input_dir,
                    "output_dir": output_dir,
                    "input_path": input_path,
                    "output_path": output_path,
                    "relative_path": relative_path,
                }
            )

    return plan


def print_summary(plan):
    """Muestra conteos por grupo antes de procesar."""
    print("Resumen de audios detectados:")
    print("=" * 90)

    total = 0
    for target in PROCESS_TARGETS:
        count = sum(1 for item in plan if item["grupo"] == target["name"])
        total += count
        status = "OK" if target["input_dir"].exists() else "NO EXISTE"
        print(
            f"{target['name']:25s} | {count:4d} audios | "
            f"Entrada: {target['input_dir']} | {status}"
        )
        print(f"{'':25s} | Salida:  {target['output_dir']}")

    print("-" * 90)
    print(f"Total: {total} audios")


def write_reports(rows):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "grupo",
        "archivo_original",
        "archivo_procesado",
        "estado",
        "duracion_original",
        "duracion_procesada",
        "observacion",
    ]

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for target in PROCESS_TARGETS:
        group_rows = [row for row in rows if row["grupo"] == target["name"]]
        ok_count = sum(1 for row in group_rows if row["estado"] == "OK")
        error_count = sum(1 for row in group_rows if row["estado"] == "ERROR")
        summary_rows.append(
            {
                "grupo": target["name"],
                "total": len(group_rows),
                "ok": ok_count,
                "error": error_count,
                "salida": str(target["output_dir"]),
            }
        )

    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["grupo", "total", "ok", "error", "salida"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def process_plan(plan):
    rows = []

    for index, item in enumerate(plan, start=1):
        input_path = item["input_path"]
        output_path = item["output_path"]

        try:
            original_duration = get_duration_seconds(input_path)
            process_audio(input_path, output_path)
            processed_duration = get_duration_seconds(output_path)

            rows.append(
                {
                    "grupo": item["grupo"],
                    "archivo_original": str(input_path),
                    "archivo_procesado": str(output_path),
                    "estado": "OK",
                    "duracion_original": original_duration,
                    "duracion_procesada": processed_duration,
                    "observacion": "",
                }
            )

            print(f"[{index}/{len(plan)}] OK: {item['grupo']} / {item['relative_path']}")

        except Exception as exc:
            rows.append(
                {
                    "grupo": item["grupo"],
                    "archivo_original": str(input_path),
                    "archivo_procesado": str(output_path),
                    "estado": "ERROR",
                    "duracion_original": "",
                    "duracion_procesada": "",
                    "observacion": str(exc).replace("\n", " ")[:300],
                }
            )

            print(f"[{index}/{len(plan)}] ERROR: {item['grupo']} / {item['relative_path']}")

    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Procesa audios originales y audios nuevos a WAV mono 16 kHz.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo muestra que se procesaria, sin crear ni sobrescribir archivos.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    plan = build_plan()
    print_summary(plan)

    if not plan:
        print("No se encontraron audios para procesar.")
        sys.exit(1)

    if args.dry_run:
        print()
        print("Dry-run activo: no se proceso ningun archivo.")
        return

    try:
        check_ffmpeg_tools()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print()
    print("Procesando audios...")
    rows = process_plan(plan)
    write_reports(rows)

    ok_count = sum(1 for row in rows if row["estado"] == "OK")
    error_count = sum(1 for row in rows if row["estado"] == "ERROR")

    print()
    print("Proceso terminado.")
    print(f"OK: {ok_count}")
    print(f"Errores: {error_count}")
    print(f"Reporte generado en: {REPORT_PATH}")
    print(f"Resumen generado en: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
