MAPEO_BASE = {
    "enciende": "LUZ_ON",
    "apaga": "LUZ_OFF",
    "ventilador": "VENT_TOGGLE",
    "puerta": "PUERTA_TOGGLE",
    "alarma": "ALARMA_TOGGLE",
    "seguro": "SEGURO",
    "ruido_fondo": None,
}

MAPEO_SECUENCIAL = {
    "enciende_luz": "LUZ_ON",
    "apaga_luz": "LUZ_OFF",
    "enciende_ventilador": "VENT_ON",
    "apaga_ventilador": "VENT_OFF",
    "abre_puerta": "PUERTA_ABRIR",
    "cierra_puerta": "PUERTA_CERRAR",
    "activa_alarma": "ALARMA_ON",
    "apaga_alarma": "ALARMA_OFF",
    "apaga_todo": "TODO_OFF",
}


def mapear_base_a_arduino(
    clase_predicha: str,
    confianza: float,
    umbral: float = 0.70,
) -> str | None:
    """Convierte una clase del modelo base CNN a comando Arduino."""
    if confianza < umbral:
        return None

    clase = clase_predicha.strip().lower()
    if clase == "ruido_fondo":
        return None

    return MAPEO_BASE.get(clase)


def mapear_secuencial_a_arduino(
    clase_predicha: str,
    confianza: float,
    umbral: float = 0.70,
) -> str | None:
    """Convierte una clase del modelo secuencial GRU a comando Arduino."""
    if confianza < umbral:
        return None

    clase = clase_predicha.strip().lower()
    return MAPEO_SECUENCIAL.get(clase)
