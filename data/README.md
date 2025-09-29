# Carpeta `data/` — Contrato de datos e inferencia

## Objetivo
Esta carpeta no contiene el dataset completo de Oxford-IIIT Pets. Únicamente incluye los artefactos necesarios para reproducir la normalización de datos, ejemplos de prueba y el código auxiliar que permitió generar las estadísticas usadas en entrenamiento. El propósito es garantizar que la API de inferencia use exactamente el mismo preprocesamiento que se aplicó al entrenar la red ResNet-101.

## Contenido
- **`create_dataset/`**: código auxiliar (`load_data.py` y `utils_data.py`) que se usó para calcular estadísticas del dataset y definir las transformaciones de entrenamiento.  
- **`pet_stats.json`**: archivo de contrato que contiene las estadísticas de normalización (media y desviación estándar por canal RGB) calculadas en el dataset original. Es el insumo que utiliza la API para transformar las imágenes en inferencia.  
- **`processed/examples/`**: carpeta con imágenes de ejemplo (un gato y un perro) que sirven para pruebas rápidas de la API y para generar capturas de pantalla en el README del proyecto.  
- **`processed/fixtures/`**: carpeta destinada a imágenes sintéticas o de prueba que permiten validar el preprocesamiento y los tests sin depender de datos sensibles o con licencia restringida.  

## Dataset original
- **Fuente**: Oxford-IIIT Pet Dataset.  
- **Tarea**: Clasificación binaria (cat vs dog) derivada de las 37 razas originales.  
- **Licencia y cita**: ver repositorio oficial. Este proyecto no redistribuye el dataset completo, únicamente metadatos y ejemplos mínimos para fines de demostración.  

## Contrato de normalización (`pet_stats.json`)
Este archivo define los parámetros de preprocesamiento que deben usarse tanto en entrenamiento como en inferencia. Contiene:
- `robust`: indica si se usó media/STD (`false`) o mediana/MAD (`true`).  
- `loc`: vector con la media (o mediana) por canal RGB.  
- `scale`: vector con la desviación estándar (o MAD ajustado) por canal RGB.  
- `img_size`: tamaño de entrada del modelo (224).  

Ejemplo (valores reales calculados en este proyecto):
- `loc`: [0.4829, 0.4449, 0.3957]  
- `scale`: [0.2592, 0.2533, 0.2598]  

## Nota para la API
- Extensiones aceptadas: `.jpg`, `.jpeg`, `.png`.  
- Entrada esperada: imágenes RGB.  
- Preprocesamiento: `Resize → CenterCrop (224×224) → ToTensor → Normalize (loc/scale)`.  
- Las imágenes con metadatos EXIF son re-orientadas automáticamente antes de procesarse.