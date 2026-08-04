# ## Paso 7 — Generar el dataset de fine-tuning
# 
# Construye los pares entrada→salida en español que se usan para el fine-tuning, a partir de todo lo anterior (`data/05_dataset_final.json` + `data/06_lugares.json`).
# 
# **Decisión de diseño clave (v2):** la primera versión de este paso ponía en el input *solo* la pregunta del usuario, esperando que el modelo "recordara" de memoria qué hotel/restaurante corresponde a cada ciudad. Con ~300 ejemplos de entrenamiento y decenas de nombres propios únicos, eso resultó ser un problema de memorización que ningún ajuste de hiperparámetros lograba resolver (evidencia de esas corridas fallidas documentada en el README). La versión actual incluye los **hechos ya dados** dentro del input (clima, vuelo, lugares) y le pide al modelo solo que los **redacte** en lenguaje natural — la tarea pasa de "recordar" a "redactar", mucho más aprendible con este tamaño de dataset.

"""
Paso 7 (v2) — Generar el dataset de fine-tuning con una reformulación
importante de la tarea:

ANTES: input = pregunta sobre un destino -> output = recomendación (el
modelo tenía que "recordar" qué hotel/restaurante corresponde a cada
ciudad solo por el nombre; con ~300 ejemplos y decenas de nombres únicos,
eso es un problema de memorización que no se resuelve con más épocas).

AHORA: input = pregunta del usuario + LOS HECHOS YA DADOS (clima, precio,
hoteles, restaurantes, atractivos) -> output = la misma recomendación en
lenguaje natural. La tarea pasa de "recordar hechos" a "redactar hechos
que ya te dieron" — mucho más aprendible con pocos ejemplos, y además es
como funcionaría el sistema real (primero se consultan los datos con
código determinístico, el modelo solo se usa para la redacción final).

Uso:
    python3 paso7_generar_pares_entrenamiento.py
"""

import json
import os
import random

random.seed(42)  # reproducibilidad (la rúbrica la pide explícitamente)

# -----------------------------------------------------------------------
# Plantillas de PREGUNTA (lo que el usuario escribe; se combina con los
# hechos para formar el input completo)
# -----------------------------------------------------------------------
PREGUNTAS_GENERICAS = [
    "Quiero viajar por Colombia con buen clima y vuelos económicos.",
    "Busco un destino turístico barato para visitar pronto.",
    "Recomiéndame un lugar en Colombia con buen clima y precio accesible.",
    "¿A dónde me recomiendas viajar si quiero ahorrar en el tiquete?",
    "Necesito un plan de viaje económico con buen clima.",
    "Dame una recomendación de destino turístico en Colombia.",
    "Estoy planeando un viaje, ¿qué destino en Colombia me recomiendas?",
    "Quiero unas vacaciones baratas en Colombia, ¿alguna sugerencia?",
]

PREGUNTAS_ESPECIFICAS = [
    "¿Cuál es el mejor mes para viajar a {municipio}?",
    "Dame una recomendación de viaje a {municipio}.",
    "Quiero ir a {municipio}, ¿cuándo me conviene viajar?",
    "Háblame de {municipio} como destino turístico.",
    "¿Vale la pena viajar a {municipio}? ¿En qué mes?",
    "¿Qué tan caro es volar a {municipio}?",
    "Cuéntame sobre viajar a {municipio}.",
    "¿Me recomiendas {municipio} para vacacionar?",
    "Estoy pensando en ir a {municipio}, ¿qué me dices?",
    "¿Cómo es viajar a {municipio}?",
]

PREGUNTAS_HOSPEDAJE = [
    "¿Dónde me puedo hospedar en {municipio}?",
    "Recomiéndame hoteles en {municipio}.",
    "¿Qué opciones de alojamiento hay en {municipio}?",
]

PREGUNTAS_COMIDA = [
    "¿Dónde puedo comer en {municipio}?",
    "Recomiéndame restaurantes en {municipio}.",
    "¿Qué comida probar si voy a {municipio}?",
]

PREGUNTAS_ATRACTIVOS = [
    "¿Qué puedo visitar en {municipio}?",
    "¿Cuáles son los atractivos turísticos de {municipio}?",
    "¿Qué no me puedo perder si voy a {municipio}?",
]

# -----------------------------------------------------------------------
# Fragmentos de SALIDA (frases con variación; se combinan con los MISMOS
# datos que se muestran en el input, así el modelo aprende a redactar,
# no a inventar)
# -----------------------------------------------------------------------
APERTURAS = [
    "Te recomiendo viajar a {municipio}, {departamento}.",
    "Una buena opción es {municipio}, en el departamento de {departamento}.",
    "{municipio} ({departamento}) es un destino que te puede interesar.",
]

CLIMA_FRASES = [
    "El mejor mes por clima es {mes_clima}, con temperaturas agradables y poca lluvia.",
    "Por clima, {mes_clima} es la mejor época para visitarlo.",
    "El clima es más favorable en {mes_clima}.",
]

VUELO_FRASES = [
    "El vuelo más económico encontrado fue en {mes_barato}, desde {origen}, por ${precio:,.0f} COP.",
    "Si buscas ahorrar, vuela en {mes_barato} desde {origen}: encontramos tiquetes desde ${precio:,.0f} COP.",
    "El precio más bajo detectado fue ${precio:,.0f} COP volando desde {origen} en {mes_barato}.",
]

COINCIDENCIA_FRASES = {
    True: "Además, ese mes económico coincide con uno de los mejores momentos climáticos del año — combinación ideal.",
    False: "Ten en cuenta que el mes más barato no coincide con el mejor clima, así que es un trade-off entre precio y clima.",
}

VERBO_LUGARES = {
    "hoteles": "Para hospedarte puedes considerar",
    "restaurantes": "Para comer, algunas opciones son",
    "atractivos": "No te pierdas",
}

MESES_NOMBRE = {
    "01": "enero", "02": "febrero", "03": "marzo", "04": "abril",
    "05": "mayo", "06": "junio", "07": "julio", "08": "agosto",
    "09": "septiembre", "10": "octubre", "11": "noviembre", "12": "diciembre",
}


def mes_legible(valor: str) -> str:
    if valor and "-" in valor and valor.split("-")[-1] in MESES_NOMBRE:
        return MESES_NOMBRE[valor.split("-")[-1]]
    return valor


def preparar_datos(destino: dict, lugares: dict) -> dict:
    """Muestrea UNA vez los datos de este destino (hoteles/restaurantes/
    atractivos incluidos). Este mismo diccionario se usa para construir
    el input (hechos dados) y el output (redacción), así quedan
    garantizadamente consistentes entre sí."""
    combinada = destino["mejor_epoca_combinada"]

    def muestra(categoria, k=3):
        items = lugares.get(categoria, []) if lugares else []
        elegidos = random.sample(items, k=min(k, len(items))) if items else []
        return [i["nombre"] for i in elegidos if i.get("nombre")]

    return {
        "municipio": destino["municipio"].title(),
        "departamento": destino["departamento"].title(),
        "mes_clima": combinada["mejores_meses_clima"][0],
        "mes_barato": mes_legible(combinada["mes_mas_barato"]),
        "origen": combinada["mejor_origen"],
        "precio": combinada["precio_mas_barato"],
        "coincide": combinada["coincide_barato_y_buen_clima"],
        "hoteles": muestra("hoteles"),
        "restaurantes": muestra("restaurantes"),
        "atractivos": muestra("atractivos"),
    }


def construir_bloque_hechos(datos: dict) -> str:
    """Serializa los datos como texto simple para meter en el INPUT."""
    partes = [
        f"Destino: {datos['municipio']}, {datos['departamento']}.",
        f"Mejor mes por clima: {datos['mes_clima']}.",
        f"Vuelo más económico: {datos['mes_barato']} desde {datos['origen']}, "
        f"${datos['precio']:,.0f} COP.",
    ]
    for categoria in ("hoteles", "restaurantes", "atractivos"):
        nombres = datos[categoria]
        if nombres:
            partes.append(f"{categoria.capitalize()}: {', '.join(nombres)}.")
    return " ".join(partes)


def construir_salida(datos: dict, categorias_a_mencionar: list = None) -> str:
    """Redacta la recomendación en lenguaje natural usando los MISMOS
    datos que ya se mostraron en el input. Si categorias_a_mencionar es
    None, menciona todas las categorías con datos disponibles."""
    partes = [
        random.choice(APERTURAS).format(municipio=datos["municipio"], departamento=datos["departamento"]),
        random.choice(CLIMA_FRASES).format(mes_clima=datos["mes_clima"]),
        random.choice(VUELO_FRASES).format(
            mes_barato=datos["mes_barato"], origen=datos["origen"], precio=datos["precio"],
        ),
        COINCIDENCIA_FRASES[datos["coincide"]],
    ]

    categorias = categorias_a_mencionar or ["hoteles", "restaurantes", "atractivos"]
    for categoria in categorias:
        nombres = datos[categoria]
        if nombres:
            partes.append(f"{VERBO_LUGARES[categoria]}: {', '.join(nombres)}.")

    return " ".join(partes)


def construir_par(pregunta: str, datos: dict, categorias_a_mencionar: list = None) -> dict:
    hechos = construir_bloque_hechos(datos)
    entrada = f"Datos: {hechos} Pregunta: {pregunta}"
    salida = construir_salida(datos, categorias_a_mencionar)
    return {"input": entrada, "output": salida}


def main():
    with open("data/05_dataset_final.json", encoding="utf-8") as f:
        municipios = json.load(f)

    try:
        with open("data/06_lugares.json", encoding="utf-8") as f:
            lugares_por_municipio = json.load(f)
    except FileNotFoundError:
        print("Aviso: no encontré data/06_lugares.json, sigo sin esa info.")
        lugares_por_municipio = {}

    destinos = [m for m in municipios if m.get("mejor_epoca_combinada")]
    print(f"Generando pares para {len(destinos)} destinos...")

    pares = []
    for destino in destinos:
        municipio_nombre = destino["municipio"]
        lugares = lugares_por_municipio.get(municipio_nombre, {})
        municipio_titulo = municipio_nombre.title()

        # preguntas generales (info completa)
        for plantilla in PREGUNTAS_ESPECIFICAS:
            datos = preparar_datos(destino, lugares)  # muestreo propio por ejemplo
            pregunta = plantilla.format(municipio=municipio_titulo)
            pares.append(construir_par(pregunta, datos))

        # preguntas enfocadas por categoría
        for plantillas, categoria in [
            (PREGUNTAS_HOSPEDAJE, "hoteles"),
            (PREGUNTAS_COMIDA, "restaurantes"),
            (PREGUNTAS_ATRACTIVOS, "atractivos"),
        ]:
            for plantilla in plantillas:
                datos = preparar_datos(destino, lugares)
                if not datos[categoria]:
                    continue  # este destino no tiene datos de esa categoría
                pregunta = plantilla.format(municipio=municipio_titulo)
                pares.append(construir_par(pregunta, datos, categorias_a_mencionar=[categoria]))

    # preguntas genéricas: se empareja con varios destinos al azar
    for pregunta in PREGUNTAS_GENERICAS:
        for _ in range(10):
            destino = random.choice(destinos)
            municipio_nombre = destino["municipio"]
            lugares = lugares_por_municipio.get(municipio_nombre, {})
            datos = preparar_datos(destino, lugares)
            pares.append(construir_par(pregunta, datos))

    random.shuffle(pares)

    corte = int(len(pares) * 0.8)
    train, val = pares[:corte], pares[corte:]

    os.makedirs("data", exist_ok=True)

    with open("data/07_train.jsonl", "w", encoding="utf-8") as f:
        for p in train:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with open("data/07_val.jsonl", "w", encoding="utf-8") as f:
        for p in val:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nTotal pares: {len(pares)} (train: {len(train)}, val: {len(val)})")
    print("-> data/07_train.jsonl")
    print("-> data/07_val.jsonl")

    print("\nEjemplo:")
    print(json.dumps(pares[0], ensure_ascii=False, indent=2))


main()
