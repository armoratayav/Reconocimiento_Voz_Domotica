# Informacion tecnica del proyecto

Esta carpeta resume la informacion solicitada para el proyecto final: el
preprocesamiento de audio, la justificacion de hiperparametros, las metricas de
los modelos y el analisis de latencia del sistema.

## 1. Preprocesamiento de audio

El sistema trabaja localmente con audios de voz en formato mono a 16 kHz. Antes
de entrar al modelo, cada audio se ajusta a una duracion fija de 3 segundos:

- Si el audio dura mas de 3 segundos, se recorta.
- Si el audio dura menos de 3 segundos, se rellena con ceros.
- La longitud final es `48000` muestras, porque `16000 Hz * 3 s = 48000`.

Despues se calculan coeficientes MFCC con `librosa`, ya que los MFCC resumen la
forma espectral de la voz y son apropiados para comandos cortos entrenados desde
cero. Finalmente, los MFCC se normalizan por muestra:

```python
mfcc = (mfcc - mean) / (std + 1e-8)
```

Esta normalizacion reduce diferencias de volumen entre grabaciones y ayuda a
que el modelo aprenda patrones de voz en vez de depender solo de la amplitud.

## 2. Justificacion de hiperparametros

| Hiperparametro | Valor | Justificacion |
| --- | ---: | --- |
| `SAMPLE_RATE` | `16000` Hz | Es suficiente para voz humana y reduce el costo computacional frente a tasas mayores. |
| `DURATION_SECONDS` | `3.0` s | Permite capturar comandos completos como "enciende ventilador" o "cierra puerta" sin hacer entradas demasiado largas. |
| `N_MFCC` | `13` | Es un valor clasico para reconocimiento de voz con MFCC; conserva informacion relevante sin aumentar demasiado la dimensionalidad. |
| `N_FFT` | `512` | A 16 kHz equivale a una ventana aproximada de 32 ms, adecuada para analizar fonemas y cambios cortos en la voz. |
| `HOP_LENGTH` | `256` | Genera saltos de 16 ms entre frames; da buena resolucion temporal sin producir demasiados frames. |
| `MAX_FRAMES` | `188` | Es la cantidad esperada de frames MFCC para audios de 3 segundos con `hop_length=256`; se recorta o rellena para mantener forma fija. |
| `CONFIDENCE_THRESHOLD` | `0.70` | Evita enviar comandos al Arduino cuando el modelo no tiene suficiente seguridad. |

## 3. Forma de entrada de los modelos

### Modelo base CNN robusto

El modelo base usa MFCC como una imagen 2D:

```text
(13, 188, 1)
```

Este modelo reconoce comandos simples como `enciende`, `apaga`, `ventilador`,
`puerta`, `alarma`, `seguro` y `ruido_fondo`.

### Modelo secuencial GRU

El modelo secuencial conserva el orden temporal de los MFCC:

```text
(188, 13)
```

En inferencia se agrega la dimension de lote:

```text
(1, 188, 13)
```

Este modelo reconoce comandos compuestos como `enciende_luz`,
`apaga_ventilador`, `abre_puerta` y `apaga_todo`.

## 4. Metricas de evaluacion

Las matrices de confusion y los reportes completos estan guardados en la carpeta
`metricas/`.

### Modelo base CNN robusto

- Reporte: `metricas/base_cnn_robusto_classification_report.txt`
- Matriz de confusion: `metricas/base_cnn_robusto_confusion_matrix.png`
- Grafica de accuracy: `metricas/base_cnn_robusto_accuracy.png`
- Grafica de loss: `metricas/base_cnn_robusto_loss.png`

Resumen del reporte de prueba:

| Clase | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| alarma | 0.95 | 0.95 | 0.95 | 20 |
| apaga | 0.91 | 1.00 | 0.95 | 20 |
| enciende | 0.94 | 1.00 | 0.97 | 17 |
| puerta | 1.00 | 0.95 | 0.97 | 20 |
| ruido_fondo | 1.00 | 0.88 | 0.93 | 8 |
| seguro | 1.00 | 1.00 | 1.00 | 20 |
| ventilador | 1.00 | 0.95 | 0.97 | 20 |
| **Accuracy** |  |  | **0.97** | **125** |
| **Macro avg** | **0.97** | **0.96** | **0.97** | **125** |
| **Weighted avg** | **0.97** | **0.97** | **0.97** | **125** |

### Modelo secuencial GRU

- Reporte: `metricas/secuencial_gru_classification_report.txt`
- Matriz de confusion: `metricas/secuencial_gru_confusion_matrix.png`
- Grafica de accuracy: `metricas/secuencial_gru_accuracy.png`
- Grafica de loss: `metricas/secuencial_gru_loss.png`

Resumen del reporte de prueba:

| Clase | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| abre_puerta | 1.00 | 1.00 | 1.00 | 12 |
| activa_alarma | 1.00 | 1.00 | 1.00 | 12 |
| apaga_alarma | 1.00 | 1.00 | 1.00 | 12 |
| apaga_luz | 1.00 | 1.00 | 1.00 | 16 |
| apaga_todo | 1.00 | 1.00 | 1.00 | 12 |
| apaga_ventilador | 1.00 | 1.00 | 1.00 | 12 |
| cierra_puerta | 1.00 | 1.00 | 1.00 | 12 |
| enciende_luz | 1.00 | 1.00 | 1.00 | 16 |
| enciende_ventilador | 1.00 | 1.00 | 1.00 | 12 |
| **Accuracy** |  |  | **1.00** | **116** |
| **Macro avg** | **1.00** | **1.00** | **1.00** | **116** |
| **Weighted avg** | **1.00** | **1.00** | **1.00** | **116** |

## 5. Analisis de latencia

El asistente mide la latencia con `time.perf_counter()` y la muestra como:

```text
Latencia total: X ms
```

En modo manual, la latencia se mide desde que inicia la grabacion hasta que se
decide si se envia o no un comando. En modo activo, se mide desde que termina la
voz detectada por VAD hasta que se decide el comando.

| Componente | Valor actual / medicion | FPS equivalente | Comentario |
| --- | ---: | ---: | --- |
| Captura manual | 3000 ms | 0.33 FPS | Duracion fija de grabacion para comandos completos. |
| Ventana VAD activo | 250 ms | 4 FPS | El sistema escucha en bloques cortos para detectar voz por energia RMS. |
| Ventana MFCC (`N_FFT=512`) | 32 ms | 31.25 FPS | Tamano de ventana espectral usado por `librosa`. |
| Salto MFCC (`HOP_LENGTH=256`) | 16 ms | 62.5 FPS | Resolucion temporal entre frames consecutivos. |
| Inferencia del modelo | Medida durante ejecucion | Depende del equipo | Se incluye dentro de `Latencia total`. |
| Actuacion Arduino | Serial a 9600 baudios | No aplica | El envio del comando es corto; la conexion persistente evita reinicios entre comandos. |

Para registrar valores reales durante la demostracion, se recomienda ejecutar:

```powershell
.\.venv\Scripts\python.exe scripts\asistente_secuencial_activo.py
```

y anotar varias salidas de `Latencia total` por comando. Esto permite reportar
promedio, minimo y maximo sobre el hardware usado en la presentacion.

## 6. Notebook Jupyter reproducible

El entregable de notebook reproducible se encuentra en:

```text
notebooks/entrenamiento_metricas_reproducible.ipynb
```

El notebook reproduce el entrenamiento del modelo base CNN robusto y del modelo
secuencial GRU, y genera:

- Reporte de clasificacion.
- Matriz de confusion.
- Graficas de accuracy y loss.
- Resumen CSV con `test_loss` y `test_accuracy`.

Para no sobrescribir las metricas principales del proyecto, las salidas del
notebook se guardan en:

```text
metricas/notebook/
```

