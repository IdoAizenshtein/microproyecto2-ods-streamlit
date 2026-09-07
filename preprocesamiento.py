"""Preparación de textos en español compartida por el notebook y la aplicación."""

from functools import lru_cache

from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer


# Conservamos letras españolas (educación, niños) antes de aplicar Snowball.
# Pasar a minúsculas no elimina las tildes ni convierte ñ en n.
TOKENIZADOR = RegexpTokenizer(r"\b[a-záéíóúüñ]{2,}\b")
STEMMER = SnowballStemmer("spanish")


@lru_cache(maxsize=50_000)
def _raiz(palabra):
    return STEMMER.stem(palabra)


def preparar_textos(textos, palabras_vacias):
    """Tokeniza, filtra la lista recibida y aplica stemming a cada documento."""
    palabras_vacias = frozenset(palabras_vacias)
    return [
        " ".join(
            _raiz(palabra)
            for palabra in TOKENIZADOR.tokenize(texto.lower())
            if palabra not in palabras_vacias
        )
        for texto in textos
    ]
