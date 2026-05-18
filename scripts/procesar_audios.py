from pathlib import Path
import subprocess
import csv
import sys

# ==========================
# CONFIGURACIÓN PRINCIPAL
# ==========================

INPUT_DIR = Path("dataset_original")
OUTPUT_DIR = Path("dataset_procesado")

# Formato final recomendado para entrenamiento
SAMPLE_RATE = 16000
CHANNELS = 1

# Normalización aproximada de loudness.
# -20 LUFS suele funcionar bien para comandos de voz.
TARGET_LUFS = -20

# Duración máxima recomendada.
# Para comandos simples y compuestos cortos, 3 segundos es un límite razonable.
MAX_DURATION_SECONDS = 3.0

# Extensiones permitidas de entrada
AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".webm"
}


def run_command(command):
    """Ejecuta un comando de consola y maneja errores."""
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result


def get_duration_seconds(audio_path):
    """Obtiene duración del audio usando ffprobe."""
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path)
    ]

    result = run_command(command)

    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def process_audio(input_path, output_path):
    """
    Convierte audio a:
    - WAV
    - mono
    - 16 kHz
    - normalizado en volumen
    - duración máxima controlada
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_filter = (
        f"loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11"
    )

    command = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-t", str(MAX_DURATION_SECONDS),
        "-ac", str(CHANNELS),
        "-ar", str(SAMPLE_RATE),
        "-af", audio_filter,
        "-sample_fmt", "s16",
        str(output_path)
    ]

    run_command(command)


def main():
    if not INPUT_DIR.exists():
        print(f"ERROR: No existe la carpeta {INPUT_DIR}")
        print("Crea la carpeta dataset_original y coloca allí los audios.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = OUTPUT_DIR / "reporte_procesamiento.csv"

    audio_files = [
        path for path in INPUT_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]

    if not audio_files:
        print("No se encontraron audios en dataset_original.")
        sys.exit(1)

    print(f"Audios encontrados: {len(audio_files)}")
    print("Procesando...")

    rows = []

    for index, input_path in enumerate(audio_files, start=1):
        relative_path = input_path.relative_to(INPUT_DIR)

        # Mantiene la misma estructura de carpetas,
        # pero convierte todo a .wav
        output_relative = relative_path.with_suffix(".wav")
        output_path = OUTPUT_DIR / output_relative

        try:
            original_duration = get_duration_seconds(input_path)
            process_audio(input_path, output_path)
            processed_duration = get_duration_seconds(output_path)

            rows.append({
                "archivo_original": str(input_path),
                "archivo_procesado": str(output_path),
                "estado": "OK",
                "duracion_original": original_duration,
                "duracion_procesada": processed_duration,
                "observacion": ""
            })

            print(f"[{index}/{len(audio_files)}] OK: {relative_path}")

        except Exception as e:
            rows.append({
                "archivo_original": str(input_path),
                "archivo_procesado": str(output_path),
                "estado": "ERROR",
                "duracion_original": "",
                "duracion_procesada": "",
                "observacion": str(e).replace("\n", " ")[:300]
            })

            print(f"[{index}/{len(audio_files)}] ERROR: {relative_path}")

    with open(report_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "archivo_original",
            "archivo_procesado",
            "estado",
            "duracion_original",
            "duracion_procesada",
            "observacion"
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Proceso terminado.")
    print(f"Audios procesados en: {OUTPUT_DIR}")
    print(f"Reporte generado en: {report_path}")


if __name__ == "__main__":
    main()