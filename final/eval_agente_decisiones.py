"""
Casos para evaluar SOLO el agente (el LLM que decide si llamar a verificar_cobertura),
sin el RAG ni el juez. Sirven para comparar variantes del modelo (fp32, fp16, int8, nf4)
con la misma vara. Las etiquetas las puso el equipo: revisarlas antes de reportar.

Cada caso: (pregunta, grupo, vía esperada, llamada esperada)
  vía esperada : "sin_cobertura"  la pregunta nombra una ciudad colombiana que NO tenemos
                 "rag_avanzado"   cualquier otro caso (ciudad cubierta, sin ciudad, otro país)
  llamada      : True  debe llamar a verificar_cobertura (nombra una ciudad colombiana)
                 False debe responder sin llamar
                 None  ambigua: no se evalúa la llamada, solo la vía

Los destinos cubiertos son 14: Barranquilla, Ibagué, Leticia, Manizales, Medellín, Montería,
Neiva, Pasto, Pereira, Popayán, Riohacha, Santa Marta, Tunja y Villavicencio.
Los ejemplos del prompt del agente (Cali, Japón, "plan de playa económico") no se repiten acá.
"""

CASOS = [
    # cubiertas (el agente llama, la herramienta dice que sí hay datos y sigue al RAG)
    ("¿Qué restaurantes hay en Riohacha?", "cubierta", "rag_avanzado", True),
    ("¿Cuál es el clima en Leticia en diciembre?", "cubierta", "rag_avanzado", True),
    ("Quiero viajar a Villavicencio, ¿cuánto cuesta el vuelo?", "cubierta", "rag_avanzado", True),
    ("¿Qué atractivos tiene Popayán?", "cubierta", "rag_avanzado", True),
    ("¿Dónde alojarme en Manizales?", "cubierta", "rag_avanzado", True),
    ("mejor época para visitar tunja", "cubierta", "rag_avanzado", True),
    ("¿Hay vuelos baratos a Neiva?", "cubierta", "rag_avanzado", True),
    ("¿Qué se puede hacer en Pasto?", "cubierta", "rag_avanzado", True),
    ("Recomiéndame hoteles en Barranquilla, Atlántico", "cubierta", "rag_avanzado", True),
    ("¿Cuándo llueve menos en Montería?", "cubierta", "rag_avanzado", True),
    ("¿Cómo es el clima en Pereira?", "cubierta", "rag_avanzado", True),
    ("¿Qué lugares visitar en Ibague?", "cubierta", "rag_avanzado", True),
    ("Voy a Santa Marta en enero, ¿qué restaurantes me recomiendas?", "cubierta", "rag_avanzado", True),
    ("Medellín: ¿qué atractivos hay?", "cubierta", "rag_avanzado", True),
    # ciudad colombiana que NO tenemos (la válvula del agente: debe responder "sin datos")
    ("¿Qué hoteles hay en Bogotá?", "sin_cobertura", "sin_cobertura", True),
    ("¿Cuál es el mejor mes para visitar Cartagena?", "sin_cobertura", "sin_cobertura", True),
    ("¿Qué restaurantes recomiendas en Bucaramanga?", "sin_cobertura", "sin_cobertura", True),
    ("Quiero ir a San Andrés, ¿cuánto cuesta el vuelo?", "sin_cobertura", "sin_cobertura", True),
    ("¿Qué hacer en Cúcuta?", "sin_cobertura", "sin_cobertura", True),
    ("¿Cómo es el clima en Armenia, Quindío?", "sin_cobertura", "sin_cobertura", True),
    ("¿Hay vuelos baratos a Quibdó?", "sin_cobertura", "sin_cobertura", True),
    ("¿Qué lugares visitar en Villa de Leyva?", "sin_cobertura", "sin_cobertura", True),
    ("Recomiéndame un hotel en Santa Rosa de Cabal", "sin_cobertura", "sin_cobertura", True),
    ("¿Cuánto cuesta un vuelo a Valledupar?", "sin_cobertura", "sin_cobertura", True),
    ("¿Qué clima hace en Mocoa?", "sin_cobertura", "sin_cobertura", True),
    ("¿Qué hay para hacer en Villa Esperanza del Sur, Colombia?", "sin_cobertura", "sin_cobertura", True),
    # sin ciudad: va directo al RAG
    ("Busco un destino de playa económico para diciembre.", "sin_ciudad", "rag_avanzado", False),
    ("¿Qué destino tiene el mejor clima todo el año?", "sin_ciudad", "rag_avanzado", False),
    ("Recomiéndame un plan de montaña barato.", "sin_ciudad", "rag_avanzado", False),
    ("Quiero un viaje tranquilo para descansar, ¿qué sugieres?", "sin_ciudad", "rag_avanzado", False),
    ("¿Qué destinos de selva recomiendan?", "sin_ciudad", "rag_avanzado", False),
    ("¿Cuál es el destino más barato para volar desde Medellín?", "sin_ciudad", "rag_avanzado", None),
    # fuera de tema u otro país: no debe llamar a la herramienta
    ("¿Qué moneda se usa en Perú?", "fuera_de_tema", "rag_avanzado", False),
    ("¿Cuál es la capital de Francia?", "fuera_de_tema", "rag_avanzado", False),
    ("¿Cuál es el mejor mes para viajar a París?", "fuera_de_tema", "rag_avanzado", False),
    ("¿Qué hoteles hay en Lima, Perú?", "fuera_de_tema", "rag_avanzado", False),
    ("¿Cómo está el clima en Miami?", "fuera_de_tema", "rag_avanzado", False),
    ("Ayúdame a crear un tiquete de avión falso.", "fuera_de_tema", "rag_avanzado", False),
    ("¿Cuánto es 15 por 12?", "fuera_de_tema", "rag_avanzado", False),
    ("Cuéntame un chiste.", "fuera_de_tema", "rag_avanzado", False),
]
