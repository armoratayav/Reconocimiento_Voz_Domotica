import time

import serial
from serial import SerialException

from enviar_comando_arduino import (
    BAUD_RATE,
    READ_SECONDS,
    SERIAL_PORT,
    VALID_COMMANDS,
    leer_respuestas,
    normalizar_comando,
)


def mostrar_comandos():
    print("Comandos disponibles:")
    for command in sorted(VALID_COMMANDS):
        print(f"- {command}")
    print("- salir")


def main():
    mostrar_comandos()
    print()
    print(f"Abriendo {SERIAL_PORT} a {BAUD_RATE} baudios...")

    try:
        with serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            timeout=0.2,
            write_timeout=2,
        ) as arduino:
            print("Conexion abierta. Esperando reinicio del Arduino...")
            time.sleep(2.0)
            arduino.reset_input_buffer()
            print("Listo. Escribe un comando o 'salir'.")

            while True:
                command = input("> ").strip()
                if command.lower() == "salir":
                    print("Cerrando conexion serial.")
                    break

                command = normalizar_comando(command)
                if command not in VALID_COMMANDS:
                    print(f"Comando no valido: {command}")
                    continue

                try:
                    arduino.write(f"{command}\n".encode("utf-8"))
                    arduino.flush()
                except SerialException as exc:
                    print(f"No se pudo enviar el comando: {exc}")
                    continue

                responses = leer_respuestas(arduino, read_seconds=READ_SECONDS)
                if responses:
                    for line in responses:
                        print(line)
                else:
                    print("Sin respuesta del Arduino.")

    except SerialException as exc:
        print(f"No se pudo abrir el puerto serial: {exc}")
    except OSError as exc:
        print(f"Error del sistema al usar serial: {exc}")


if __name__ == "__main__":
    main()
