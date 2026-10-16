"""
Capa agéntica — TravelGenie M3.

    from agente import sistema_agente
    sistema_agente("¿Qué hoteles recomiendas en Quibdó?")   # str -> str

Un LLM pequeño (Qwen2.5-1.5B-Instruct) decide, paso a paso, si necesita la
herramienta `verificar_cobertura(destino)` antes de responder:

    pregunta -> [LLM decide] -> llamar herramienta -> observación -> [LLM decide]
                             -> responder -> RAG avanzado (rag_avanzado.py)

Por qué esta herramienta: es el fallo que quedó vivo tras las dos técnicas
avanzadas. Para una ciudad que el corpus no tiene (Quibdó) la recuperación
devuelve hoteles de otra ciudad con puntuación alta, y el generador los
presenta como si fueran de la ciudad pedida. Una consulta exacta de cobertura
lo resuelve donde el retrieval no puede.

Piezas que pide la rúbrica:
  - HERRAMIENTAS: esquema (nombre, descripción, parámetros).
  - parsear_decision(): lectura TOLERANTE de JSON (los modelos pequeños
    devuelven JSON con comillas simples, comas sobrantes, texto alrededor...).
  - MAX_PASOS: tope de pasos del ciclo.

Sin API keys ni servicios de pago. El LLM del agente es el mismo modelo que
hace de juez en el harness (se reutiliza la copia cargada); se declara en el
README por el posible sesgo de auto-preferencia.
"""

import json
import re
import sys
import unicodedata

import rag_avanzado

MODELO_AGENTE = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_PASOS = 3
RUTA_CHUNKS = "data/09_chunks.json"

# La frase inicial es la misma que usa el RAG: así resumen_m3.py y el harness
# reconocen igual un "admito que no tengo datos" venga del RAG o del agente.
MENSAJE_SIN_COBERTURA = ("No tengo información sobre eso: mi base de datos no incluye "
                         "{destino}. Solo cubro: {lista}.")

# ---------------------------------------------------------------------------
# Esquema de herramientas
# ---------------------------------------------------------------------------
HERRAMIENTAS = {
    "verificar_cobertura": {
        "descripcion": ("Indica si un destino colombiano tiene datos (clima, vuelos, hoteles, "
                        "restaurantes, atractivos) en la base. Úsala cuando la pregunta "
                        "menciona una ciudad o municipio concreto."),
        "parametros": {
            "destino": {"tipo": "string", "requerido": True,
                        "descripcion": "Nombre de la ciudad o municipio, p. ej. 'Medellín'."},
        },
    },
}

PROMPT_SISTEMA = """Eres el planificador de TravelGenie, un asistente de viajes por Colombia.
Decides si necesitas una herramienta antes de responder.

Herramientas:
- verificar_cobertura(destino: string): dice si un destino tiene datos en la base.

En cada turno responde SOLO con un objeto JSON, sin texto adicional:
- Para usar una herramienta: {"accion": "llamar", "herramienta": "verificar_cobertura", "argumentos": {"destino": "<ciudad>"}}
- Para terminar: {"accion": "responder"}

Reglas:
- Llama a verificar_cobertura SOLO si la pregunta menciona una ciudad o municipio concreto de Colombia (un país, como Japón, NO es una ciudad: responde).
- Si la pregunta no menciona ninguna ciudad de Colombia (o trata de otro país u otro tema), responde {"accion": "responder"}.
- NUNCA inventes una ciudad: el destino debe aparecer escrito en la pregunta.
- Después de recibir una observación, responde {"accion": "responder"}.

Ejemplos:
Pregunta: ¿Qué hoteles hay en Cali?
{"accion": "llamar", "herramienta": "verificar_cobertura", "argumentos": {"destino": "Cali"}}
Pregunta: ¿Qué moneda se usa en Japón?
{"accion": "responder"}
Pregunta: Quiero un plan de playa económico, ¿qué me sugieres?
{"accion": "responder"}
"""

_estado = {}
ULTIMO = {}   # traza de la última llamada (para diagnóstico y para el reporte)


# ---------------------------------------------------------------------------
# Herramienta
# ---------------------------------------------------------------------------
def normalizar(texto):
    t = unicodedata.normalize("NFD", str(texto).lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]+", " ", t).strip()


def destinos_cubiertos():
    if "destinos" not in _estado:
        with open(RUTA_CHUNKS, encoding="utf-8") as f:
            _estado["destinos"] = sorted({c["destino"] for c in json.load(f)})
    return _estado["destinos"]


def verificar_cobertura(destino):
    """¿Está `destino` entre los destinos con datos? Tolera 'Cartagena de Indias',
    'Medellín, Antioquia', mayúsculas y tildes."""
    consulta = " " + normalizar(destino) + " "
    for d in destinos_cubiertos():
        if " " + normalizar(d) + " " in consulta:
            return {"destino_consultado": destino, "en_corpus": True, "destino_en_corpus": d}
    return {"destino_consultado": destino, "en_corpus": False, "destino_en_corpus": None,
            "destinos_cubiertos": [d.title() for d in destinos_cubiertos()]}


def ejecutar_herramienta(nombre, argumentos, pregunta=None):
    """Valida contra el esquema y ejecuta. Los errores se devuelven como
    observación (el agente los ve) en vez de lanzar excepciones."""
    if nombre not in HERRAMIENTAS:
        return {"error": f"herramienta desconocida: {nombre!r}",
                "herramientas_disponibles": list(HERRAMIENTAS)}
    for p, spec in HERRAMIENTAS[nombre]["parametros"].items():
        if spec["requerido"] and not isinstance(argumentos.get(p), str):
            return {"error": f"falta el argumento '{p}' (string)"}
        if spec["requerido"] and not argumentos[p].strip():
            return {"error": f"el argumento '{p}' está vacío"}
    if nombre == "verificar_cobertura":
        # Barrera: el destino debe estar escrito en la pregunta. Evita que el LLM
        # invente una ciudad (en la prueba real inventó "Bogotá" para una pregunta
        # abierta y eso la convertía en un falso rechazo).
        if pregunta is not None and (" " + normalizar(argumentos["destino"]) + " ") not in (" " + normalizar(pregunta) + " "):
            return {"error": f"'{argumentos['destino']}' no aparece en la pregunta del usuario; "
                             "no inventes ciudades. Si no hay ciudad, responde."}
        return verificar_cobertura(argumentos["destino"])


# ---------------------------------------------------------------------------
# Lectura tolerante de JSON
# ---------------------------------------------------------------------------
_ALIAS_ACCION = ("accion", "acción", "action")
_ALIAS_HERRAMIENTA = ("herramienta", "tool", "name", "nombre", "function", "funcion")
_ALIAS_ARGS = ("argumentos", "args", "arguments", "parametros", "parameters", "params")


def _bloque_json(texto):
    """Primer {...} balanceado (respeta comillas). Si el texto se cortó antes de
    cerrar, completa las llaves que falten."""
    i = texto.find("{")
    if i < 0:
        return None
    profundidad, en_cadena, comilla, esc = 0, False, "", False
    for j in range(i, len(texto)):
        c = texto[j]
        if en_cadena:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == comilla:
                en_cadena = False
        elif c in "\"'":
            en_cadena, comilla = True, c
        elif c == "{":
            profundidad += 1
        elif c == "}":
            profundidad -= 1
            if profundidad == 0:
                return texto[i:j + 1]
    return texto[i:] + "}" * max(profundidad, 0)


def _reparar(s):
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    s = re.sub(r",\s*([}\]])", r"\1", s)                                   # comas sobrantes
    s = re.sub(r"([{,]\s*)([A-Za-zÁÉÍÓÚáéíóú_]+)\s*:", r'\1"\2":', s)       # claves sin comillas
    s = re.sub(r"'([^'\n]*)'", r'"\1"', s)                                  # comillas simples
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    return s


def _primero(d, claves):
    for k in claves:
        if k in d:
            return d[k]
    return None


def _normalizar_decision(d):
    """dict heterogéneo -> {'accion','herramienta','argumentos'} o None."""
    if not isinstance(d, dict):
        return None
    herramienta = _primero(d, _ALIAS_HERRAMIENTA)
    accion = _primero(d, _ALIAS_ACCION)
    if isinstance(accion, str) and normalizar(accion) in ("responder", "respond", "final", "fin", "answer"):
        return {"accion": "responder"}
    if isinstance(herramienta, str):
        args = _primero(d, _ALIAS_ARGS)
        if not isinstance(args, dict):
            args = {k: v for k, v in d.items()
                    if k not in _ALIAS_ACCION + _ALIAS_HERRAMIENTA + _ALIAS_ARGS}
        return {"accion": "llamar", "herramienta": herramienta.strip(), "argumentos": args}
    return None


def parsear_decision(texto):
    """Texto del LLM -> decisión normalizada, o None si no se pudo leer nada útil."""
    if not isinstance(texto, str) or not texto.strip():
        return None
    texto = re.sub(r"```(?:json)?", "", texto)
    bloque = _bloque_json(texto)
    if bloque:
        for candidato in (bloque, _reparar(bloque)):
            try:
                decision = _normalizar_decision(json.loads(candidato))
            except (json.JSONDecodeError, ValueError):
                continue
            if decision:
                return decision
    # último recurso: pares clave:valor sueltos en el texto
    h = re.search(r"(?:herramienta|tool|name)\W{0,4}(\w+)", texto, re.I)
    if h and h.group(1).lower() not in ("null", "none"):
        d = re.search(r"destino\W{0,4}([^\"'\n,}]+)", texto, re.I)
        return {"accion": "llamar", "herramienta": h.group(1),
                "argumentos": {"destino": d.group(1).strip()} if d else {}}
    if re.search(r"\bresponder\b", texto, re.I):
        return {"accion": "responder"}
    return None


# ---------------------------------------------------------------------------
# LLM del agente
# ---------------------------------------------------------------------------
def _cargar():
    if "llm" in _estado:
        return _estado
    import torch
    H = sys.modules.get("harness")   # si el harness ya cargó a Qwen como juez, se reutiliza
    if H is not None and getattr(H, "JUEZ_MODEL_NAME", None) == MODELO_AGENTE:
        _estado.update(tok=H.juez_tokenizer, llm=H.juez_modelo, device=H.device)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _estado.update(tok=AutoTokenizer.from_pretrained(MODELO_AGENTE),
                       llm=AutoModelForCausalLM.from_pretrained(MODELO_AGENTE).to(device),
                       device=device)
    rag_avanzado._cargar()
    return _estado


def _generar_llm(mensajes):
    import torch
    e = _cargar()
    texto = e["tok"].apply_chat_template(mensajes, tokenize=False, add_generation_prompt=True)
    ids = e["tok"](texto, return_tensors="pt").to(e["device"])
    with torch.no_grad():
        salida = e["llm"].generate(**ids, max_new_tokens=80, do_sample=False)
    return e["tok"].decode(salida[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Ciclo del agente
# ---------------------------------------------------------------------------
def sistema_agente(pregunta, max_pasos=MAX_PASOS, llm=None):
    """Interfaz estándar del harness: str -> str.  `llm` permite inyectar un
    sustituto (mensajes -> str) para pruebas sin modelo."""
    llm = llm or _generar_llm
    ULTIMO.clear()
    traza, llamadas = [], []
    ULTIMO.update(pasos=traza, llamadas=llamadas, tope_de_pasos=False, via=None)

    mensajes = [{"role": "system", "content": PROMPT_SISTEMA},
                {"role": "user", "content": f"Pregunta: {pregunta}"}]
    sin_cobertura = None
    terminado = False
    for paso in range(1, max_pasos + 1):
        salida = llm(mensajes)
        decision = parsear_decision(salida)
        traza.append({"paso": paso, "salida_llm": salida[:300], "decision": decision})

        if decision is None:                       # JSON ilegible: se le pide corregir
            mensajes += [{"role": "assistant", "content": salida},
                         {"role": "user", "content": "No pude leer tu salida. Responde SOLO con un objeto JSON válido."}]
            continue
        if decision["accion"] == "responder":
            terminado = True
            break

        obs = ejecutar_herramienta(decision["herramienta"], decision["argumentos"], pregunta)
        llamadas.append({"herramienta": decision["herramienta"],
                         "argumentos": decision["argumentos"], "observacion": obs})
        if obs.get("en_corpus") is False:
            sin_cobertura = obs
        mensajes += [{"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)},
                     {"role": "user", "content": f"Observación: {json.dumps(obs, ensure_ascii=False)}\nSiguiente paso (JSON):"}]

    ULTIMO["tope_de_pasos"] = not terminado

    if sin_cobertura is not None:
        ULTIMO["via"] = "sin_cobertura"
        return MENSAJE_SIN_COBERTURA.format(destino=sin_cobertura["destino_consultado"],
                                            lista=", ".join(sin_cobertura["destinos_cubiertos"]))
    ULTIMO["via"] = "rag_avanzado"
    return rag_avanzado.sistema_rag_avanzado(pregunta)


if __name__ == "__main__":
    for p in ["¿Qué hoteles recomiendas en Quibdó?",       # no cubierto: debe llamar y rechazar
              "¿Cuál es el mejor mes para viajar a Cartagena de Indias?",
              "¿Qué hoteles hay en Medellín?",              # cubierto: llama y sigue al RAG
              "¿Cuál es la capital de Francia?",            # no necesita herramienta
              "Busco un destino turístico barato con buen clima, ¿qué me recomiendas?"]:
        r = sistema_agente(p)
        print(f"\nP: {p}")
        for paso in ULTIMO["pasos"]:
            print(f"   paso {paso['paso']}: {paso['salida_llm']!r}")
        for ll in ULTIMO["llamadas"]:
            print(f"   herramienta: {ll['herramienta']}({ll['argumentos']}) -> en_corpus={ll['observacion'].get('en_corpus')}")
        print(f"   vía: {ULTIMO['via']}   tope_de_pasos: {ULTIMO['tope_de_pasos']}")
        print(f"R: {r[:200]}")