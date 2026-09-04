from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load


st.set_page_config(
    page_title="Clasificador de textos por ODS",
    page_icon="🌎",
    layout="centered",
)

RUTA_MODELO = Path(__file__).with_name("modelo_ods.joblib")


@st.cache_resource(show_spinner="Cargando el modelo de clasificación…")
def cargar_artefacto():
    if not RUTA_MODELO.exists():
        raise FileNotFoundError(
            "No se encontró modelo_ods.joblib junto a app.py."
        )
    artefacto = load(RUTA_MODELO)
    return artefacto["modelo"], artefacto["ods_nombres"], artefacto["metadatos"]


try:
    modelo, ods_nombres, metadatos = cargar_artefacto()
except Exception as error:
    st.error(
        "No fue posible cargar el modelo. Revise que el artefacto esté completo "
        "y que las versiones de las dependencias coincidan con requirements.txt."
    )
    st.exception(error)
    st.stop()


st.title("🌎 Clasificador de textos por ODS")
st.caption("Microproyecto 2 · Ido Aizenshtein y William Armando Vera Serrano")
st.write(
    "Ingrese un texto en español. La aplicación utilizará el mismo pipeline "
    "TF-IDF → SVD → normalización → regresión logística construido y evaluado "
    "en el proyecto."
)

with st.form("formulario_clasificacion"):
    texto = st.text_area(
        "Texto para clasificar",
        height=220,
        placeholder=(
            "Ejemplo: El municipio ampliará el acceso a agua potable y mejorará "
            "el tratamiento de las aguas residuales en las zonas rurales."
        ),
        help="Puede pegar un párrafo o un fragmento más extenso en español.",
    )
    clasificar = st.form_submit_button(
        "Identificar ODS",
        type="primary",
        width="stretch",
    )

if clasificar:
    texto_normalizado = texto.strip()
    if not texto_normalizado:
        st.warning("Escriba un texto antes de solicitar la clasificación.")
    else:
        prediccion = int(modelo.predict([texto_normalizado])[0])
        probabilidades = modelo.predict_proba([texto_normalizado])[0]
        clases = np.asarray(modelo.named_steps["clasificador"].classes_, dtype=int)
        orden = np.argsort(probabilidades)[::-1][:3]
        confianza = float(probabilidades[orden[0]])

        st.success(
            f"Predicción: ODS {prediccion} — {ods_nombres[prediccion]}"
        )

        columna_ods, columna_confianza = st.columns(2)
        columna_ods.metric("ODS predicho", prediccion)
        columna_confianza.metric("Confianza estimada", f"{confianza:.1%}")

        st.subheader("Tres resultados con mayor puntuación")
        alternativas = pd.DataFrame(
            {
                "ODS": [f"ODS {int(clases[i])}" for i in orden],
                "Nombre": [ods_nombres[int(clases[i])] for i in orden],
                "Puntuación estimada": [f"{probabilidades[i]:.1%}" for i in orden],
            }
        )
        st.dataframe(alternativas, hide_index=True, width="stretch")

        if confianza < 0.50:
            st.info(
                "La predicción principal tiene una puntuación moderada. El texto "
                "puede mezclar varios objetivos y conviene revisarlo manualmente."
            )

st.divider()
with st.expander("Alcance y limitaciones"):
    st.markdown(
        f"""
        - El modelo fue entrenado con **{metadatos['n_textos_entrenamiento']:,} textos**
          y evaluado sobre datos separados del entrenamiento.
        - El archivo entregado contiene ejemplos de los ODS 1 a 16, pero no del ODS 17;
          por eso la aplicación no puede aprender ni predecir el ODS 17.
        - La puntuación mostrada es una estimación del modelo, no una garantía ni una
          probabilidad calibrada.
        - La salida sirve como apoyo inicial y no reemplaza la revisión de una persona,
          especialmente cuando un texto trata varios ODS.
        """
    )
