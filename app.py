from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load


st.set_page_config(
    page_title="Textos y ODS",
    layout="centered",
)

RUTA_MODELO = Path(__file__).with_name("modelo_ods.joblib")
RUTA_LOGO = Path(__file__).parent / "assets" / "logo_uniandes.png"


# Streamlit usa firma_modelo como clave de caché, aunque no se lea dentro de la función.
@st.cache_resource(show_spinner="Cargando el modelo...")
def cargar_artefacto(firma_modelo):
    # El pipeline guardado necesita preparar_textos, definida en preprocesamiento.py.
    artefacto = load(RUTA_MODELO)
    return artefacto["modelo"], artefacto["ods_nombres"], artefacto["metadatos"]


try:
    estado_modelo = RUTA_MODELO.stat()
    # Si cambia la fecha o el tamaño del archivo, se carga de nuevo.
    firma_modelo = (estado_modelo.st_mtime_ns, estado_modelo.st_size)
    modelo, ods_nombres, metadatos = cargar_artefacto(firma_modelo)
except Exception as error:
    st.error(
        "No se pudo cargar el modelo. Revise los archivos de la aplicación "
        "y las versiones indicadas en requirements.txt."
    )
    with st.expander("Detalle del error"):
        st.exception(error)
    st.stop()


st.image(str(RUTA_LOGO), width=193, link="https://www.uniandes.edu.co/")
st.title("Clasificación de textos por ODS")
st.caption("Microproyecto 2 · Ido Aizenshtein y William Armando Vera Serrano")
st.caption(
    "Curso: [Machine learning no supervisado]"
    "(https://www.coursera.org/learn/maia-machine-learning-no-supervisado/home/welcome)  \n"
    "[Maestría en Inteligencia Artificial (MAIA)](https://sistemas.uniandes.edu.co/maia/)  \n"
    "Universidad de los Andes · Bogotá, Colombia"
)
st.write(
    "Escriba un párrafo en español para consultar con qué Objetivo de Desarrollo "
    "Sostenible se relaciona."
)

with st.form("formulario_clasificacion"):
    texto = st.text_area(
        "Texto",
        height=220,
        placeholder=(
            "Por ejemplo: El municipio mejorará el suministro de agua potable "
            "y el tratamiento de aguas residuales."
        ),
    )
    clasificar = st.form_submit_button(
        "Clasificar texto",
        type="primary",
    )

if clasificar:
    texto = texto.strip()
    if not texto:
        st.warning("Escriba un texto para continuar.")
    else:
        textos_preparados = modelo.named_steps["preparacion"].transform(
            [texto]
        )
        representacion_texto = modelo.named_steps["tfidf"].transform(
            textos_preparados
        )
        if representacion_texto.nnz == 0:
            st.warning(
                "Después del procesamiento no quedan términos que el modelo "
                "reconozca. Pruebe con una descripción más completa."
            )
        else:
            prediccion = int(modelo.predict([texto])[0])
            probabilidades = modelo.predict_proba([texto])[0]
            clases = np.asarray(modelo.named_steps["clasificador"].classes_, dtype=int)
            orden = np.argsort(probabilidades)[::-1][:3]
            confianza = float(probabilidades[orden[0]])

            st.subheader(f"ODS {prediccion}: {ods_nombres[prediccion]}")
            st.write("Estos son los tres ODS con mayor probabilidad estimada:")
            alternativas = pd.DataFrame(
                {
                    "ODS": [f"ODS {int(clases[i])}" for i in orden],
                    "Nombre": [ods_nombres[int(clases[i])] for i in orden],
                    "Probabilidad estimada": [f"{probabilidades[i]:.1%}" for i in orden],
                }
            )
            st.dataframe(alternativas, hide_index=True, width="stretch")

            if confianza < 0.50:
                st.info(
                    "La probabilidad estimada más alta es inferior al 50 %. "
                    "Conviene revisar las alternativas antes de usar el resultado."
                )

st.divider()
with st.expander("Sobre este proyecto"):
    n_entrenamiento = f"{metadatos['n_textos_entrenamiento']:,}".replace(",", ".")
    st.write(
        f"Entrenamos el modelo con {n_entrenamiento} textos del archivo entregado "
        "en el curso y lo evaluamos con un conjunto de prueba separado. "
        "Los datos incluyen los ODS 1 a 16. No hay ejemplos del ODS 17, "
        "por lo que el modelo no puede predecirlo."
    )
    st.write(
        "La aplicación usa el mismo procesamiento del notebook: stopwords de NLTK, "
        "Snowball en español, TF-IDF, SVD y normalización. La clasificación se hace "
        "con regresión logística. El modelo ya está entrenado; aquí solo se usa "
        "para hacer predicciones."
    )
    st.write(
        "Las probabilidades no están calibradas y no garantizan un acierto. "
        "Un texto puede tratar varios ODS, aunque el modelo elige uno. "
        "Por eso conviene leer el resultado junto con el texto original."
    )

st.caption(
    "Proyecto académico de estudiantes. No es una aplicación oficial de la Universidad."
)
st.caption(
    "Nombres de los ODS: [PNUD](https://teamup.undp.org/es/). "
    "[Código del proyecto](https://github.com/IdoAizenshtein/microproyecto2-ods-streamlit)."
)
