from __future__ import annotations

import sys
import time

import serial
from serial import SerialException


SERIAL_PORT = "COM6"
BAUD_RATE = 9600
WAIT_AFTER_OPEN_SECONDS = 2.0
READ_SECONDS = 3.0

VALID_COMMANDS = {
    "LUZ_ON",
    "LUZ_OFF",
    "VENT_ON",
    "VENT_OFF",
    "VENT_TOGGLE",
    "PUERTA_ABRIR",
    "PUERTA_CERRAR",
    "PUERTA_TOGGLE",
    "ALARMA_ON",
    "ALARMA_OFF",
    "ALARMA_TOGGLE",
    "TODO_OFF",
    "SEGURO",
    "ESTADO",
}


def normalizar_comando(command: str) -> str:
    return command.strip().upper()


def leer_respuestas(arduino: serial.Serial, read_seconds: float = READ_SECONDS) -> list[str]:
    """Lee lineas del Arduino durante unos segundos."""
    responses = []
    deadline = time.time() + read_seconds

    while time.time() < deadline:
        try:
            raw_line = arduino.readline()
        except SerialException:
            break

        if not raw_line:
            continue

        line = raw_line.decode("utf-8", errors="replace").strip()
        if line:
            responses.append(line)

    return responses


def enviar_comando_serial(
    command: str,
    serial_port: str = SERIAL_PORT,
    baud_rate: int = BAUD_RATE,
    wait_after_open_seconds: float = WAIT_AFTER_OPEN_SECONDS,
    read_seconds: float = READ_SECONDS,
) -> tuple[bool, list[str], str | None]:
    """
    Envia un comando terminado en salto de linea al Arduino.
    Devuelve: (enviado, respuestas, error).
    """
    command = normalizar_comando(command)

    if not command:
        return False, [], "Comando vacio."

    if command not in VALID_COMMANDS:
        return False, [], f"Comando no reconocido por este proyecto: {command}"

    try:
        with serial.Serial(
            port=serial_port,
            baudrate=baud_rate,
            timeout=0.2,
            write_timeout=2,
        ) as arduino:
            print(f"Puerto abierto: {serial_port} a {baud_rate} baudios")
            print("Esperando reinicio del Arduino...")
            time.sleep(wait_after_open_seconds)

            arduino.reset_input_buffer()
            payload = f"{command}\n".encode("utf-8")
            arduino.write(payload)
            arduino.flush()

            responses = leer_respuestas(arduino, read_seconds=read_seconds)
            return True, responses, None

    except SerialException as exc:
        return False, [], f"Error serial: {exc}"
    except OSError as exc:
        return False, [], f"Error del sistema al abrir/enviar por serial: {exc}"


def main():
    if len(sys.argv) < 2:
        print("Uso:")
        print("python scripts/enviar_comando_arduino.py LUZ_ON")
        print()
        print("Comandos validos:")
        for command in sorted(VALID_COMMANDS):
            print(f"- {command}")
        return

    command = normalizar_comando(sys.argv[1])
    sent, responses, error = enviar_comando_serial(command)

    if not sent:
        print(f"No se pudo enviar el comando: {command}")
        print(error)
        return

    print(f"Comando enviado: {command}")

    if responses:
        print()
        print("Respuesta Arduino:")
        for line in responses:
            print(line)
    else:
        print("Arduino no envio respuesta durante el tiempo de espera.")


if __name__ == "__main__":
    main()
