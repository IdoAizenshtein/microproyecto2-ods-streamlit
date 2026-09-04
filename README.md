# Clasificador de textos por Objetivos de Desarrollo Sostenible

Aplicación interactiva del **Microproyecto 2** de la Maestría en Inteligencia
Artificial de la Universidad de los Andes.

**Autores:** Ido Aizenshtein y William Armando Vera Serrano.

## Aplicación en vivo

[Abrir la aplicación en Streamlit Community Cloud](https://microproyecto2-ods-aizenshtein-vera.streamlit.app/)

## Método

La aplicación carga el estimador completo seleccionado en el notebook y aplica, sin
programar transformaciones paralelas, el siguiente flujo:

`texto → BOW con TF-IDF → TruncatedSVD → normalización L2 → regresión logística`

El modelo recibe texto libre en español y muestra el número y el nombre del ODS
predicho. También presenta las tres puntuaciones más altas y comunica las
limitaciones necesarias para interpretar la salida.

## Ejecución local

Se recomienda Python 3.14, la misma versión utilizada por el despliegue. Desde
esta carpeta:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

El archivo `modelo_ods.joblib` debe permanecer en la misma carpeta que `app.py`.

## Archivos

- `app.py`: interfaz y predicción.
- `modelo_ods.joblib`: pipeline ajustado y metadatos mínimos de despliegue.
- `requirements.txt`: versiones reproducibles de las dependencias.
- `.streamlit/config.toml`: configuración visual de la aplicación.

## Alcance

El archivo de entrenamiento suministrado incluye los ODS 1 a 16 y no contiene
observaciones del ODS 17. Por ello, la aplicación no puede predecir esa clase. La
salida es una herramienta de apoyo y no sustituye la revisión humana.
