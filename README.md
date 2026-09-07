# Clasificador de textos por Objetivos de Desarrollo Sostenible

Aplicación interactiva del **Microproyecto 2** de la Maestría en Inteligencia
Artificial de la Universidad de los Andes.

**Autores:** Ido Aizenshtein y William Armando Vera Serrano.

## Aplicación en vivo

[Abrir la aplicación en Streamlit Community Cloud](https://microproyecto2-ods-aizenshtein-vera.streamlit.app/)

## Método

El archivo del modelo incluye todas las transformaciones ajustadas en el notebook
para aplicar el siguiente flujo:

`texto → tokenización NLTK → palabras vacías → Snowball → BOW con TF-IDF → TruncatedSVD → normalización L2 → regresión logística`

La preparación compartida en `preprocesamiento.py` convierte cada texto a
minúsculas y usa `RegexpTokenizer` para reconocer palabras españolas de al menos
dos letras, conservando tildes y `ñ` en esta etapa. Después elimina la lista
completa de `stopwords.words("spanish")` de NLTK, sin añadir ni retirar entradas,
incluidas `no`, `ni` y `sin`. El filtrado se hace antes de aplicar
`SnowballStemmer("spanish")`.

Snowball agrupa variantes morfológicas para reducir la dispersión del vocabulario.
Sus raíces pueden ser menos legibles que las palabras completas; incorporar esta
etapa no demuestra por sí solo una mejora de exactitud. La eliminación de las
negaciones también puede perder información del texto.

El notebook obtiene la lista de palabras vacías y la guarda en el
`FunctionTransformer` del pipeline. Durante la inferencia, la aplicación usa esa
lista serializada y no ejecuta `nltk.download` ni descarga modelos lingüísticos.
El identificador `nltk-snowball-es-v1` aparece en los metadatos y en el apartado
«Alcance y limitaciones» de la aplicación.

El modelo recibe texto libre en español y muestra el número y el nombre del ODS
predicho. También presenta las tres puntuaciones más altas y comunica las
limitaciones necesarias para interpretar la salida.

Los arreglos de SVD se guardan en `float32` para reducir el tamaño del archivo.
Se conservan los pasos, hiperparámetros y vocabulario del pipeline, y el notebook
verifica las predicciones del archivo exportado sobre todo el conjunto de prueba.
La aplicación no entrena el modelo ni guarda los textos introducidos por el usuario.

## Fuentes de los datos y de los nombres

Los textos de entrenamiento y sus etiquetas numéricas ODS proceden de
`Train_textosODS.xlsx`, suministrado por el docente.

El diccionario que relaciona cada número ODS con su nombre se definió manualmente
en el notebook para la presentación de resultados. Los nombres se verificaron con
el [PNUD — Unidos por las personas y el planeta](https://teamup.undp.org/es/).
Ese sitio se usó como referencia de los nombres; los datos de entrenamiento no se
extrajeron de él.

## Ejecución local

Se recomienda Python 3.14, la misma versión utilizada por el despliegue. Desde
esta carpeta:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Los archivos `modelo_ods.joblib` y `preprocesamiento.py` deben permanecer en la
misma carpeta que `app.py`; el módulo permite cargar la función guardada en el
pipeline.

## Despliegue y actualización

En Streamlit Community Cloud se seleccionan este repositorio, la rama `main` y
`app.py` como archivo de entrada. La versión de Python se elige en **Advanced
settings** al crear el despliegue; `.python-version` orienta el entorno local y no
configura por sí solo el runtime de Community Cloud.

Los cambios enviados a `main` actualizan la aplicación. Si cambia el modelo,
primero hay que ejecutar y evaluar el notebook, exportar `modelo_ods.joblib` y
copiar su módulo `preprocesamiento.py` a la carpeta de la aplicación. Se suben
ambos archivos junto con cualquier cambio de código o dependencias. La caché de
carga utiliza la fecha de modificación y el tamaño del artefacto para recargarlo
cuando se reemplaza.
Una visita al enlace no vuelve a entrenar el modelo.

Si la aplicación está inactiva, puede aparecer una pantalla para reactivarla.
Los errores de instalación o ejecución se consultan en **Manage app**.

## Archivos

- `app.py`: interfaz y predicción.
- `preprocesamiento.py`: tokenización, filtrado de palabras vacías y Snowball;
  copia del módulo compartido con el notebook.
- `modelo_ods.joblib`: pipeline ajustado y metadatos mínimos de despliegue.
- `requirements.txt`: versiones reproducibles de las dependencias.
- `.streamlit/config.toml`: configuración visual de la aplicación.

## Alcance

El archivo de entrenamiento suministrado incluye los ODS 1 a 16 y no contiene
observaciones del ODS 17. Por ello, la aplicación no puede predecir esa clase. La
salida es una herramienta de apoyo y no sustituye la revisión humana.
