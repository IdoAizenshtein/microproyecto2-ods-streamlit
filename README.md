# Clasificador de textos por Objetivos de Desarrollo Sostenible

Aplicación interactiva del **Microproyecto 2** de la Maestría en Inteligencia
Artificial de la Universidad de los Andes.

**Autores:** Ido Aizenshtein y William Armando Vera Serrano.

## Aplicación en vivo

[Abrir la aplicación en Streamlit Community Cloud](https://microproyecto2-ods-aizenshtein-vera.streamlit.app/)

## Método

El archivo del modelo incluye todas las transformaciones ajustadas en el notebook
para aplicar el siguiente flujo:

`texto → BOW con TF-IDF → TruncatedSVD → normalización L2 → regresión logística`

El modelo recibe texto libre en español y muestra el número y el nombre del ODS
predicho. También presenta las tres puntuaciones más altas y comunica las
limitaciones necesarias para interpretar la salida.

Los arreglos de SVD se guardan en `float32` para reducir el tamaño del archivo.
Se conservan los pasos, hiperparámetros y vocabulario del pipeline, y el notebook
verifica las predicciones del archivo exportado sobre todo el conjunto de prueba.
La aplicación no entrena el modelo ni guarda los textos introducidos por el usuario.

## Ejecución local

Se recomienda Python 3.14, la misma versión utilizada por el despliegue. Desde
esta carpeta:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

El archivo `modelo_ods.joblib` debe permanecer en la misma carpeta que `app.py`.

## Despliegue y actualización

En Streamlit Community Cloud se seleccionan este repositorio, la rama `main` y
`app.py` como archivo de entrada. La versión de Python se elige en **Advanced
settings** al crear el despliegue; `.python-version` orienta el entorno local y no
configura por sí solo el runtime de Community Cloud.

Los cambios enviados a `main` actualizan la aplicación. Si cambia el modelo,
primero hay que ejecutar y evaluar el notebook, exportar `modelo_ods.joblib` y
subir el nuevo archivo junto con cualquier cambio de código o dependencias.
Una visita al enlace no vuelve a entrenar el modelo.

Si la aplicación está inactiva, puede aparecer una pantalla para reactivarla.
Los errores de instalación o ejecución se consultan en **Manage app**.

## Archivos

- `app.py`: interfaz y predicción.
- `modelo_ods.joblib`: pipeline ajustado y metadatos mínimos de despliegue.
- `requirements.txt`: versiones reproducibles de las dependencias.
- `.streamlit/config.toml`: configuración visual de la aplicación.

## Alcance

El archivo de entrenamiento suministrado incluye los ODS 1 a 16 y no contiene
observaciones del ODS 17. Por ello, la aplicación no puede predecir esa clase. La
salida es una herramienta de apoyo y no sustituye la revisión humana.
