# Panel de Domotica Controlado por Voz

Proyecto final de Inteligencia Artificial para reconocer comandos de voz en
espanol y controlar un panel fisico de domotica mediante Arduino UNO.

El sistema esta pensado para funcionar localmente y sin internet. La laptop
captura audio con un microfono externo, ejecuta un modelo entrenado desde cero
con audios propios y envia el comando reconocido al Arduino por USB serial.

## Restricciones del Proyecto

Este proyecto no usa APIs externas ni modelos preentrenados de reconocimiento
de voz. En particular, no se usa Whisper, Vosk, Google Speech-to-Text, Azure
Speech, Amazon Transcribe, Picovoice, Wav2Vec2, Hugging Face ni servicios en la
nube.

Se permite usar:

- Python
- TensorFlow/Keras
- scikit-learn
- librosa
- NumPy
- SciPy
- Matplotlib
- PySerial

El modelo principal se entrena desde cero usando MFCC completo en el tiempo y
una CNN 2D.

## Dataset

El dataset procesado se encuentra en:

```text
dataset_procesado/Base/
```

Cada subcarpeta representa una clase:

```text
alarma        50 audios
apaga         75 audios
enciende      60 audios
puerta        50 audios
ruido_fondo   20 audios
seguro        50 audios
ventilador    50 audios
```

Los audios deben estar en formato WAV, mono, 16 kHz y normalizados.

Para evaluar audios externos al entrenamiento se puede crear:

```text
audios_prueba_nuevos/Base/
```

con la misma estructura de clases.

## Modelo CNN Estable

El entrenamiento recomendado actualmente es:

```text
scripts/entrenar_modelo_base_cnn_estable.py
```

Preprocesamiento usado:

- Carga con `librosa.load` a 16 kHz y mono.
- Duracion fija de 3 segundos.
- Recorte si el audio es mas largo.
- Relleno con ceros si el audio es mas corto.
- MFCC con:
  - `n_mfcc = 13`
  - `n_fft = 512`
  - `hop_length = 256`
  - `MAX_FRAMES = 188`
- Normalizacion por muestra:

```python
mfcc = (mfcc - mean) / (std + 1e-8)
```

Forma final para la CNN:

```text
(13, 188, 1)
```

Arquitectura base:

- `Conv2D(16)`
- `MaxPooling2D`
- `Conv2D(32)`
- `MaxPooling2D`
- `Flatten`
- `Dense(64)`
- `Dropout(0.20)`
- `Dense(num_classes, softmax)`

El modelo se entrena con Adam, `learning_rate=0.001`, `batch_size=16` y 80
epocas, usando `train_test_split` estratificado con `test_size=0.20`.

## Resultados Actuales

Con el entrenamiento estable se obtuvo:

```text
Accuracy entrenamiento: 1.0000
Accuracy prueba interna: 0.9718
Accuracy manual sobre 70 audios del dataset: 0.9714
```

Estos resultados confirman que el modelo CNN ya no colapsa prediciendo una sola
clase y que la tuberia de MFCC + CNN esta aprendiendo correctamente.

## Informacion Tecnica

La descripcion del preprocesamiento de audio, la justificacion de los
hiperparametros, las metricas de ambos modelos, las matrices de confusion y el
analisis de latencia se encuentran en:

```text
informacion/README.md
```

El notebook Jupyter reproducible para entrenamiento y generacion de metricas se
encuentra en:

```text
notebooks/entrenamiento_metricas_reproducible.ipynb
```

## Instalacion

Crear o activar el entorno virtual:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar dependencias:

```powershell
pip install -r requirements.txt
```

Dependencias principales:

```text
numpy==1.26.4
pandas==2.2.3
librosa==0.10.2.post1
scikit-learn==1.5.2
matplotlib==3.9.2
soundfile==0.12.1
tensorflow-cpu==2.17.1
pyserial==3.5
```

Nota: `scripts/procesar_audios.py` tambien requiere `ffmpeg` y `ffprobe`
instalados en el sistema y disponibles en el `PATH`.

## Uso Principal

Procesar audios originales y audios nuevos:

```powershell
.\.venv\Scripts\python.exe scripts\procesar_audios.py
```

Revisar que se detectara antes de procesar:

```powershell
.\.venv\Scripts\python.exe scripts\procesar_audios.py --dry-run
```

Entrenar el modelo CNN estable:

```powershell
.\.venv\Scripts\python.exe scripts\entrenar_modelo_base_cnn_estable.py
```

Probar un audio individual:

```powershell
.\.venv\Scripts\python.exe scripts\probar_modelo_base_cnn.py ruta\del\audio.wav
```

Probar varios audios del dataset procesado:

```powershell
.\.venv\Scripts\python.exe scripts\probar_varios_base_cnn.py
```

Evaluar audios nuevos no usados en entrenamiento:

```powershell
.\.venv\Scripts\python.exe scripts\probar_audios_nuevos_base_cnn.py
```

Enviar un comando directo al Arduino:

```powershell
.\.venv\Scripts\python.exe scripts\enviar_comando_arduino.py LUZ_ON
```

Abrir consola interactiva serial:

```powershell
.\.venv\Scripts\python.exe scripts\test_serial_interactivo.py
```

Probar modelo base y enviar accion al Arduino:

```powershell
.\.venv\Scripts\python.exe scripts\probar_modelo_base_cnn_con_arduino.py ruta\del\audio.wav
```

Probar modelo secuencial y enviar accion al Arduino:

```powershell
.\.venv\Scripts\python.exe scripts\probar_modelo_secuencial_gru_con_arduino.py ruta\del\audio.wav
```

## Archivos Generados

El entrenamiento CNN estable genera:

```text
modelos/modelo_base_cnn.keras
modelos/base_cnn_classes.npy
metricas/base_cnn_estable_classification_report.txt
metricas/base_cnn_estable_confusion_matrix.png
metricas/base_cnn_estable_accuracy.png
metricas/base_cnn_estable_loss.png
```

La evaluacion de audios nuevos genera:

```text
metricas/base_cnn_audios_nuevos_report.txt
metricas/base_cnn_audios_nuevos_confusion_matrix.png
metricas/base_cnn_audios_nuevos_resultados.csv
```

## Scripts Importantes

- `scripts/procesar_audios.py`: procesa `dataset_original/Base`, `dataset_original/Secuencial` y `dataset_original/audios_prueba_nuevos`.
- `scripts/base_cnn_utils.py`: funciones reutilizables de preprocesamiento MFCC y prediccion.
- `scripts/entrenar_modelo_base_cnn_estable.py`: entrenamiento recomendado del modelo CNN completo.
- `scripts/probar_modelo_base_cnn.py`: prediccion de un audio individual con regla de rechazo por baja confianza.
- `scripts/probar_varios_base_cnn.py`: prueba manual sobre varios audios por clase.
- `scripts/probar_audios_nuevos_base_cnn.py`: evaluacion sobre audios nuevos fuera del entrenamiento.
- `scripts/secuencial_gru_utils.py`: funciones reutilizables de preprocesamiento secuencial y mapeo a comandos Arduino.
- `scripts/entrenar_modelo_secuencial_gru.py`: entrenamiento recomendado del modelo GRU para comandos compuestos.
- `scripts/probar_modelo_secuencial_gru.py`: prediccion de un audio secuencial individual.
- `scripts/probar_varios_secuencial_gru.py`: prueba manual del modelo secuencial por lotes.
- `scripts/probar_audios_nuevos_secuencial_gru.py`: evaluacion secuencial sobre audios nuevos fuera del entrenamiento.
- `scripts/mapear_comando_ia.py`: convierte clases predichas a comandos compatibles con el Arduino.
- `scripts/enviar_comando_arduino.py`: envia un comando serial directo al Arduino.
- `scripts/test_serial_interactivo.py`: consola manual para probar comandos sin reiniciar en cada envio.
- `scripts/probar_modelo_base_cnn_con_arduino.py`: reconoce comando simple y lo envia al Arduino.
- `scripts/probar_modelo_secuencial_gru_con_arduino.py`: reconoce comando compuesto y lo envia al Arduino.

Los scripts de depuracion, versiones anteriores, revisiones auxiliares y
procesamiento historico se conservaron en:

```text
scripts_adicionales/
```

## Regla de Decision

En la prueba de un audio individual:

- Si la confianza es menor que `0.70`, se muestra `COMANDO_RECHAZADO`.
- Si la clase predicha es `ruido_fondo`, se muestra `SIN_ACCION`.
- Si la confianza es mayor o igual a `0.70` y no es `ruido_fondo`, se acepta el comando.

## Comandos Arduino

La comunicacion con Arduino se realiza por USB serial a `9600` baudios usando
comandos terminados en salto de linea. El puerto se configura al inicio de los
scripts seriales con:

```python
SERIAL_PORT = "COM3"
BAUD_RATE = 9600
```

Comandos enviados por el modelo base:

```text
enciende      -> LUZ_ON
apaga         -> LUZ_OFF
ventilador    -> VENT_TOGGLE
puerta        -> PUERTA_TOGGLE
alarma        -> ALARMA_TOGGLE
seguro        -> SEGURO
ruido_fondo   -> no envia nada
```

Comandos enviados por el modelo secuencial:

```text
enciende_luz         -> LUZ_ON
apaga_luz            -> LUZ_OFF
enciende_ventilador  -> VENT_ON
apaga_ventilador     -> VENT_OFF
abre_puerta          -> PUERTA_ABRIR
cierra_puerta        -> PUERTA_CERRAR
activa_alarma        -> ALARMA_ON
apaga_alarma         -> ALARMA_OFF
apaga_todo           -> TODO_OFF
```

## Estructura Recomendada

```text
Proyecto_Voz_Domotica/
+-- dataset_original/
+-- dataset_procesado/
|   +-- Base/
+-- audios_prueba_nuevos/
|   +-- Base/
+-- metricas/
+-- modelos/
+-- scripts/
+-- scripts_adicionales/
+-- requirements.txt
+-- README.md
```

## Estado Actual

La parte de reconocimiento de comandos con CNN 2D sobre MFCC completo ya cuenta
con entrenamiento, pruebas internas, prueba manual por lotes, evaluacion de
audios nuevos y scripts de revision del dataset. La siguiente etapa natural es
integrar la prediccion aceptada con el envio de comandos por USB serial hacia el
Arduino UNO.
