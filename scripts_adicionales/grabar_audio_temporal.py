from pathlib import Path

import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 16000
DURATION_SECONDS = 3
CHANNELS = 1
OUTPUT_PATH = Path("temp/audio_mic.wav")


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    print("Grabando...")
    audio = sd.rec(
        int(SAMPLE_RATE * DURATION_SECONDS),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
    )
    sd.wait()
    print("Grabación terminada.")

    sf.write(OUTPUT_PATH, audio, SAMPLE_RATE)
    print("Audio guardado en temp/audio_mic.wav")


if __name__ == "__main__":
    main()
