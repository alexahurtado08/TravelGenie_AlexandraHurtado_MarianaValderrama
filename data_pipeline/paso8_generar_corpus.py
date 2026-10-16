"""
Paso 8 — Generar el corpus de documentos para RAG (M3), a partir de
data/05_dataset_final.json (clima + vuelos) y data/06_lugares.json
(hoteles/restaurantes/atractivos).

Decisión de diseño: un documento por (destino, categoría) — no un
documento gigante por destino. Así el retriever puede traer justo la
categoría relevante (solo hoteles, solo clima) en vez de un bloque con
todo mezclado.

Categorías por destino: clima, vuelos, hoteles, restaurantes, atractivos.
Solo se genera el documento si hay datos reales para esa categoría.

Uso:
    python3 data_pipeline/paso8_generar_corpus.py

Salida:
    data/08_corpus.json
"""

import json
import os

MESES_NOMBRE = {
    "01": "enero", "02": "febrero", "03": "marzo", "04": "abril",
    "05": "mayo", "06": "junio", "07": "julio", "08": "agosto",
    "09": "septiembre", "10": "octubre", "11": "noviembre", "12": "diciembre",
}


def mes_legible(valor):
    if valor and "-" in valor and valor.split("-")[-1] in MESES_NOMBRE:
        return MESES_NOMBRE[valor.split("-")[-1]]
    return valor


def doc_clima(destino):
    meses = destino.get("meses_recomendados")
    if not meses:
        return None
    nombres = ", ".join(m["mes"] for m in meses)
    return (
        f"Clima en {destino['municipio']}, {destino['departamento']}: "
        f"los meses con mejor clima (temperatura agradable y poca lluvia) son {nombres}."
    )


def doc_vuelos(destino):
    combinada = destino.get("mejor_epoca_combinada")
    if not combinada:
        return None
    return (
        f"Vuelos a {destino['municipio']}, {destino['departamento']}: "
        f"el precio más económico encontrado fue en {mes_legible(combinada['mes_mas_barato'])}, "
        f"volando desde {combinada['mejor_origen']}, por ${combinada['precio_mas_barato']:,.0f} COP. "
        f"Ese mes {'sí' if combinada['coincide_barato_y_buen_clima'] else 'no'} coincide "
        f"con los meses de mejor clima ({', '.join(combinada['mejores_meses_clima'])})."
    )


def doc_lugares(destino, lugares, categoria, etiqueta):
    items = lugares.get(categoria, []) if lugares else []
    nombres = [i["nombre"] for i in items if i.get("nombre")]
    if not nombres:
        return None
    return (
        f"{etiqueta} en {destino['municipio']}, {destino['departamento']}: "
        f"{', '.join(nombres)}."
    )


def main():
    with open("data/05_dataset_final.json", encoding="utf-8") as f:
        municipios = json.load(f)

    try:
        with open("data/06_lugares.json", encoding="utf-8") as f:
            lugares_por_municipio = json.load(f)
    except FileNotFoundError:
        lugares_por_municipio = {}

    destinos = [m for m in municipios if m.get("mejor_epoca_combinada")]
    print(f"Generando corpus para {len(destinos)} destinos...")

    corpus = []  # lista de {"id", "texto", "destino", "categoria"}
    for destino in destinos:
        municipio_nombre = destino["municipio"]
        lugares = lugares_por_municipio.get(municipio_nombre, {})

        candidatos = [
            ("clima", doc_clima(destino)),
            ("vuelos", doc_vuelos(destino)),
            ("hoteles", doc_lugares(destino, lugares, "hoteles", "Hoteles")),
            ("restaurantes", doc_lugares(destino, lugares, "restaurantes", "Restaurantes")),
            ("atractivos", doc_lugares(destino, lugares, "atractivos", "Atractivos turísticos")),
        ]

        for categoria, texto in candidatos:
            if texto:
                corpus.append({
                    "id": f"{municipio_nombre}_{categoria}".lower().replace(" ", "_"),
                    "texto": texto,
                    "destino": municipio_nombre,
                    "categoria": categoria,
                })

    os.makedirs("data", exist_ok=True)
    with open("data/08_corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    # resumen por categoría, para la justificación del README
    por_categoria = {}
    for d in corpus:
        por_categoria[d["categoria"]] = por_categoria.get(d["categoria"], 0) + 1

    print(f"\nTotal documentos: {len(corpus)}")
    for cat, n in por_categoria.items():
        print(f"  {cat}: {n}")

    longitudes = [len(d["texto"].split()) for d in corpus]
    print(f"\nLongitud de documento (palabras): min={min(longitudes)}, "
          f"max={max(longitudes)}, promedio={sum(longitudes)/len(longitudes):.1f}")

    print("\nEjemplo:")
    print(json.dumps(corpus[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
