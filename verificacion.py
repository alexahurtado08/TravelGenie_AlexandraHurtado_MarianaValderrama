"""
Verificación propia del criterio de cada caso (dimensión 3 del harness).

En vez de depender del juez, cada caso del eval set trae una regla sencilla,
sacada de su campo `criterio`, que se comprueba con código sobre la respuesta:
¿están los datos exactos?, ¿son nombres reales del dataset?, ¿admite que no
tiene información?, ¿inventa algo que no debe?

Reglas disponibles (campo `verificar` de cada caso):
  contiene_todos     : lista de textos que TODOS deben aparecer (meses, origen, precio)
  nombres_de         : {"destino", "categoria", "minimo"}: al menos `minimo` nombres
                       reales de esa lista del dataset (data/06_lugares.json)
  admite             : True -> la respuesta admite que no tiene información o se niega
  admite_o_corrige   : True -> admite, o corrige la premisa ("sí tiene playas")
  sin_cifras_ni_meses: True -> no da meses ni precios (no se los puede inventar)
  sin_nombres_de_lugares: True -> no lista hoteles/restaurantes/atractivos del dataset
  no_contiene        : lista de textos que NO deben aparecer
  destino_y_dato     : True -> nombra un destino cubierto y da un mes o un precio

verificar_criterio(respuesta, caso) devuelve (cumple: bool, motivo: str).
No usa modelos: se puede probar en cualquier computador.
"""

import json
import os
import re
import unicodedata

RUTA_LUGARES = "data/06_lugares.json"

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

# Frases con las que el sistema admite no tener información o se niega a responder.
_ADMITE = [
    "no tengo informacion", "no tengo datos", "no cuento con", "no dispongo",
    "no incluye", "solo cubro", "no cubro", "no esta entre", "fuera de",
    "no puedo", "no es posible",
]
# Frases con las que se corrige una premisa falsa.
_CORRIGE = ["si tiene playas", "tiene playas", "no es cierto", "no es correcta",
            "premisa", "es falso", "no es verdad"]

_cache = {}


def normalizar(texto):
    """minúsculas, sin tildes, solo letras y dígitos separados por espacio."""
    t = unicodedata.normalize("NFD", str(texto).lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return " " + re.sub(r"[^a-z0-9]+", " ", t).strip() + " "


def _sin_vocales_acentuadas(texto):
    """El generador de M1 a veces pierde las letras con tilde ("Medellín" -> "Medelln").
    Esta variante permite reconocer el nombre aun así."""
    t = re.sub(r"[áéíóúÁÉÍÓÚ]", "", str(texto))
    return normalizar(t)


def _aparece(frase_norm, texto_norm):
    return frase_norm in texto_norm


def _lugares():
    if "lugares" not in _cache:
        if not os.path.exists(RUTA_LUGARES):
            raise FileNotFoundError(
                f"Falta {RUTA_LUGARES}. Corre el harness desde la raíz del repositorio.")
        with open(RUTA_LUGARES, encoding="utf-8") as f:
            _cache["lugares"] = json.load(f)
    return _cache["lugares"]


def _coincide_nombre(nombre, respuesta_norm, respuesta_sin_tilde):
    n = normalizar(nombre)
    if len(n.strip()) < 6:                       # nombres muy cortos dan falsos positivos
        return False
    return _aparece(n, respuesta_norm) or _aparece(_sin_vocales_acentuadas(nombre), respuesta_sin_tilde)


def nombres_encontrados(respuesta, destino, categoria):
    r, rs = normalizar(respuesta), _sin_vocales_acentuadas(respuesta)
    nombres = {l["nombre"] for l in _lugares()[destino][categoria]}
    return sorted(n for n in nombres if _coincide_nombre(n, r, rs))


def _admite(respuesta):
    r = normalizar(respuesta)
    return any(f" {f} " in r or f in r for f in _ADMITE)


def _destinos_cubiertos():
    return list(_lugares().keys())


def _regla(nombre, valor, respuesta, r_norm, r_sin):
    """Devuelve (cumple, motivo)."""
    if nombre == "contiene_todos":
        faltan = [x for x in valor
                  if not re.search(r"(?<![a-z0-9])" + re.escape(normalizar(x).strip()) + r"(?![a-z0-9])", r_norm)]
        return (not faltan, "faltan: " + ", ".join(faltan) if faltan else "tiene todos los datos")

    if nombre == "nombres_de":
        hallados = nombres_encontrados(respuesta, valor["destino"], valor["categoria"])
        ok = len(hallados) >= valor.get("minimo", 1)
        return (ok, f"{len(hallados)} nombres reales de {valor['destino']}/{valor['categoria']}"
                    + (f" ({', '.join(hallados[:3])})" if hallados else ""))

    if nombre == "admite":
        return (_admite(respuesta) == bool(valor),
                "admite que no tiene información" if _admite(respuesta) else "no admite falta de datos")

    if nombre == "admite_o_corrige":
        corrige = any(f in r_norm for f in _CORRIGE)
        ok = _admite(respuesta) or corrige
        return (ok, "corrige la premisa" if corrige else
                ("admite que no tiene información" if ok else "asume la premisa o no dice nada útil"))

    if nombre == "sin_cifras_ni_meses":
        meses = [m for m in MESES if f" {m} " in r_norm]
        hay_precio = "$" in respuesta or bool(re.search(r"\b\d{2,3}[.,]?\d{3}\b", respuesta))
        mal = meses + (["un precio"] if hay_precio else [])
        return (not mal, "inventa: " + ", ".join(mal) if mal else "no da meses ni precios")

    if nombre == "sin_nombres_de_lugares":
        encontrados = []
        for d, cats in _lugares().items():
            for c, lista in cats.items():
                for l in lista:
                    n = normalizar(l["nombre"])
                    if len(n.strip()) >= 12 and _aparece(n, r_norm):
                        encontrados.append(l["nombre"])
        return (not encontrados, "lista lugares del dataset: " + ", ".join(sorted(set(encontrados))[:3])
                if encontrados else "no lista lugares")

    if nombre == "no_contiene":
        malos = [x for x in valor if normalizar(x) in r_norm]
        return (not malos, "contiene lo prohibido: " + ", ".join(malos) if malos else "no contiene lo prohibido")

    if nombre == "destino_y_dato":
        destinos = [d for d in _destinos_cubiertos()
                    if _aparece(normalizar(d), r_norm) or _aparece(_sin_vocales_acentuadas(d), r_sin)]
        dato = any(f" {m} " in r_norm for m in MESES) or "$" in respuesta
        ok = bool(destinos) and dato
        return (ok, f"destino: {destinos[0]}" + (" con mes/precio" if dato else " sin mes ni precio")
                if destinos else "no nombra ningún destino de la base")

    raise ValueError(f"Regla de verificación desconocida: {nombre!r}")


def verificar_criterio(respuesta, caso):
    """(cumple, motivo). Cumple si TODAS las reglas del caso se cumplen."""
    reglas = caso.get("verificar")
    if not reglas:
        raise ValueError(f"El caso no tiene campo 'verificar': {caso.get('input')!r}")
    r_norm, r_sin = normalizar(respuesta), _sin_vocales_acentuadas(respuesta)
    motivos = []
    for nombre, valor in reglas.items():
        ok, motivo = _regla(nombre, valor, respuesta, r_norm, r_sin)
        motivos.append(motivo)
        if not ok:
            return False, motivo
    return True, "; ".join(motivos)
