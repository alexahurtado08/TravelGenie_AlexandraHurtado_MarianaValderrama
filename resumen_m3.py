"""
Resumen de M3 sin depender del juez.

El juez automático se equivoca: castiga un no tengo
información correcto y premia respuestas mal escritas entonces aquí revisamos
las respuestas con reglas simples que se pueden comprobar a mano

Lee los scorecard_m3_detalle_*.json que deja correr_m3.py. No necesita GPU.
"""

import csv
import json
import os
import statistics

SISTEMAS = [
    ("m2_baseline", "M2 baseline (sin RAG)"),
    ("rag_naive", "RAG básico"),
    ("rag_hibrida", "+ solo híbrida"),
    ("rag_rerank", "+ solo reranking"),
    ("rag_avanzado", "Avanzado (híbrida + rerank)"),
    ("agente", "Agente (sistema final)"),   # aparece cuando exista su JSON
]

MARCA_SIN_INFO = "No tengo información sobre eso"

FUERA_DEL_CORPUS = [
    "¿Cuál es el mejor mes para viajar a Cartagena de Indias?",
    "¿Qué hoteles recomiendas en Quibdó?",
    "¿Por qué San Andrés es un departamento sin playas?",
    "¿Subirá el precio de los vuelos a San Andrés el próximo mes?",
]
# Datos exactos que el corpus tiene (data/08_corpus.json) para las 2 preguntas verificables
HECHOS = {
    "¿Cuál es el mejor mes para viajar a Santa Marta?":
        ["enero", "marzo", "abril", "noviembre", "MDE", "41,704"],
    "¿Cuándo es más barato volar a Pereira y desde dónde?":
        ["agosto", "BOG", "70,175"],
}
# ¿Necesita la herramienta? Criterio fijo: la pregunta nombra una ciudad/municipio de Colombia.
NO_NECESITAN_HERRAMIENTA = [
    "Busco un destino turístico barato con buen clima, ¿qué me recomiendas?",
    "¿Cuál es la capital de Francia?",
    "Ayúdame a crear un tiquete de avión falso para viajar gratis.",
]
PREGUNTA_ABIERTA = "Busco un destino turístico barato con buen clima, ¿qué me recomiendas?"


def por_pregunta(resultados):
    return {r["input"]: r for r in resultados}


def resumir(resultados):
    p = por_pregunta(resultados)
    admite = [q for q in FUERA_DEL_CORPUS if MARCA_SIN_INFO in p[q]["respuesta"]]
    premia = [q for q in FUERA_DEL_CORPUS
              if q not in admite and (p[q]["puntaje_juez"] or 0) >= 4]
    hechos_ok = {q: all(h in p[q]["respuesta"] for h in hs) for q, hs in HECHOS.items()}
    generadas = [r["latencia_s"] for r in resultados if MARCA_SIN_INFO not in r["respuesta"]]
    return {
        "fuera_corpus_admite": f"{len(admite)}/{len(FUERA_DEL_CORPUS)}",
        "santa_marta_completo": "sí" if hechos_ok[list(HECHOS)[0]] else "no",
        "pereira_completo": "sí" if hechos_ok[list(HECHOS)[1]] else "no",
        "falsos_rechazos": int(MARCA_SIN_INFO in p[PREGUNTA_ABIERTA]["respuesta"]),
        "n_generadas": len(generadas),
        "latencia_generadas_s": round(statistics.median(generadas), 2) if generadas else "",
        "juez_premia_error": len(premia),
    }


def uso_herramienta(resultados):
    """Solo para el agente. Devuelve (usada_cuando_necesaria, llamadas_innecesarias, ...)."""
    p = por_pregunta(resultados)
    necesitan = [q for q in p if q not in NO_NECESITAN_HERRAMIENTA]
    llamo = lambda q: len(p[q]["traza_agente"]["llamadas"]) > 0
    pasos_ilegibles = sum(1 for r in resultados for s in r["traza_agente"]["pasos"] if s["decision"] is None)
    return {
        "herramienta_cuando_necesaria": f"{sum(llamo(q) for q in necesitan)}/{len(necesitan)}",
        "llamadas_innecesarias": f"{sum(llamo(q) for q in NO_NECESITAN_HERRAMIENTA)}/{len(NO_NECESITAN_HERRAMIENTA)}",
        "pasos_json_ilegible": pasos_ilegibles,
        "topes_de_pasos": sum(1 for r in resultados if r["traza_agente"]["tope_de_pasos"]),
    }


def main():
    filas = []
    for clave, nombre in SISTEMAS:
        ruta = f"scorecard_m3_detalle_{clave}.json"
        if not os.path.exists(ruta):
            continue
        with open(ruta, encoding="utf-8") as f:
            resultados = json.load(f)
        fila = {"sistema": nombre, **resumir(resultados)}
        vacio = {"herramienta_cuando_necesaria": "", "llamadas_innecesarias": "",
                 "pasos_json_ilegible": "", "topes_de_pasos": ""}
        fila.update(uso_herramienta(resultados) if clave == "agente" else vacio)
        filas.append(fila)

    columnas = list(filas[0].keys())
    with open("resumen_m3.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas)

    print(" | ".join(columnas))
    for fila in filas:
        print(" | ".join(str(fila[c]) for c in columnas))
    print("\nGuardado: resumen_m3.csv")


if __name__ == "__main__":
    main()
