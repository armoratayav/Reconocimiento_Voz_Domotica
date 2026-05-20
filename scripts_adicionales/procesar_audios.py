from pathlib import Path
import runpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ACTUAL = PROJECT_ROOT / "scripts" / "procesar_audios.py"


def main():
    print("Este script fue archivado en scripts_adicionales.")
    print("Usando la version actual:")
    print(SCRIPT_ACTUAL)
    print()
    runpy.run_path(str(SCRIPT_ACTUAL), run_name="__main__")


if __name__ == "__main__":
    main()
