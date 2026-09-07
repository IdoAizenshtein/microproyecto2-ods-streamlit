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
def cargar_artefacto(firma_modelo):
    # La firma forma parte de la clave de caché y cambia al reemplazar el modelo.
    if not RUTA_MODELO.exists():
        raise FileNotFoundError(
            "No se encontró modelo_ods.joblib junto a app.py."
        )
    artefacto = load(RUTA_MODELO)
    # ods_nombres se definió manualmente en el notebook para mostrar los nombres.
    # Los nombres se verificaron con el PNUD: https://teamup.undp.org/es/
    # Los textos y las etiquetas numéricas proceden de Train_textosODS.xlsx,
    # suministrado por el docente; no se extrajeron de ese sitio web.
    return artefacto["modelo"], artefacto["ods_nombres"], artefacto["metadatos"]


try:
    estado_modelo = RUTA_MODELO.stat()
    firma_modelo = (estado_modelo.st_mtime_ns, estado_modelo.st_size)
    modelo, ods_nombres, metadatos = cargar_artefacto(firma_modelo)
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
    "tokenización NLTK → filtrado de palabras vacías → Snowball → TF-IDF → "
    "SVD → normalización → regresión logística construido y evaluado en el proyecto."
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
        textos_preparados = modelo.named_steps["preparacion"].transform(
            [texto_normalizado]
        )
        representacion_texto = modelo.named_steps["tfidf"].transform(
            textos_preparados
        )
        if representacion_texto.nnz == 0:
            st.warning(
                "Después de preparar el texto no quedan términos conocidos por "
                "el modelo. Escriba una descripción más completa del tema."
            )
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
            columna_confianza.metric("Puntuación del modelo", f"{confianza:.1%}")

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
                    "La puntuación más alta es inferior al 50 %. Revise también las "
                    "alternativas antes de usar el resultado."
                )

st.divider()
with st.expander("Alcance y limitaciones"):
    st.caption(
        "Versión del preprocesamiento: "
        f"{metadatos.get('version_preprocesamiento', 'no registrada')}"
    )
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

st.caption(
    "Código y archivos de la aplicación: "
    "[GitHub](https://github.com/IdoAizenshtein/microproyecto2-ods-streamlit)"
)
