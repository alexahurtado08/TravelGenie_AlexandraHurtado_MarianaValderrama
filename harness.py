"""
Harness de evaluación — TravelGenie (M2)

Junta 3 dimensiones sobre el mismo eval_set:
  1. Métrica automática: similitud por embeddings (ya construida en S05/M1)
  2. LLM-as-a-judge: Qwen2.5 pequeño, rúbrica anclada 1-5 (ver rubrica_juez.md)
  3. Medida de dominio: % de ejemplos que cumplen el criterio (juez >= 4)

harness(eval_set, sistema) es genérica: 'sistema' es cualquier función
input(str) -> output(str). Así se puede re-usar tal cual en M3 sobre el
sistema con RAG, sin tocar el harness.
"""

import re
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------
# Dimensión 1 — métrica automática (similitud por embeddings)
# -----------------------------------------------------------------------
st = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


def sim_embeddings(a: str, b: str) -> float:
    ea, eb = st.encode([a, b])
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb)))


# -----------------------------------------------------------------------
# Dimensión 2 — LLM-as-a-judge (Qwen2.5 pequeño, abierto, cabe en Colab gratis)
# -----------------------------------------------------------------------
JUEZ_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # 0.5B mostró sesgo de indulgencia severo:
# calificó 9/10 ejemplos con 5/5, incluyendo respuestas que eran eco literal de la
# pregunta sin ningún rechazo/corrección real (ver README, sección de sesgo del juez).

with open("rubrica_juez.md", encoding="utf-8") as f:
    RUBRICA_TEXTO = f.read()

juez_tokenizer = AutoTokenizer.from_pretrained(JUEZ_MODEL_NAME)
juez_modelo = AutoModelForCausalLM.from_pretrained(JUEZ_MODEL_NAME).to(device)

PROMPT_JUEZ = """Eres un evaluador estricto. Usa esta rúbrica para calificar una respuesta del 1 al 5:

{rubrica}

---
PREGUNTA DEL USUARIO: {pregunta}

RESPUESTA A EVALUAR: {respuesta}

Califica la respuesta según la rúbrica. Responde ÚNICAMENTE en este formato exacto,
sin explicación adicional:
PUNTAJE: <número del 1 al 5>
"""


def juez_puntuar(pregunta: str, respuesta: str) -> int:
    """Llama al juez y parsea el puntaje de forma robusta (regex, no parsing frágil)."""
    prompt = PROMPT_JUEZ.format(rubrica=RUBRICA_TEXTO, pregunta=pregunta, respuesta=respuesta)
    mensajes = [{"role": "user", "content": prompt}]
    texto_entrada = juez_tokenizer.apply_chat_template(
        mensajes, tokenize=False, add_generation_prompt=True
    )
    ids = juez_tokenizer(texto_entrada, return_tensors="pt").to(device)

    with torch.no_grad():
        salida = juez_modelo.generate(
            **ids, max_new_tokens=20, do_sample=False, temperature=None, top_p=None,
        )
    texto_salida = juez_tokenizer.decode(
        salida[0][ids["input_ids"].shape[1]:], skip_special_tokens=True
    )

    match = re.search(r"PUNTAJE:\s*([1-5])", texto_salida)
    if match:
        return int(match.group(1))

    # fallback: busca cualquier dígito 1-5 aislado si el formato exacto falló
    match_laxo = re.search(r"\b([1-5])\b", texto_salida)
    if match_laxo:
        return int(match_laxo.group(1))

    print(f"  !! No se pudo parsear puntaje del juez, salida cruda: {texto_salida!r}")
    return None


# -----------------------------------------------------------------------
# Dimensión 3 — medida de cumplimiento del criterio de dominio
# -----------------------------------------------------------------------
UMBRAL_CUMPLE = 4  # puntaje del juez >= este umbral cuenta como "cumple el criterio"


def cumple_criterio_dominio(puntaje_juez):
    if puntaje_juez is None:
        return False
    return puntaje_juez >= UMBRAL_CUMPLE


# -----------------------------------------------------------------------
# harness() — junta las 3 dimensiones sobre cualquier 'sistema'
# -----------------------------------------------------------------------
def harness(eval_set: list, sistema, nombre_sistema: str = "sistema") -> dict:
    """
    eval_set: lista de dicts con 'input', 'esperado', 'criterio' (ver eval_set.py)
    sistema: función str -> str (recibe el input, devuelve la respuesta generada)
    """
    resultados = []

    for ej in eval_set:
        respuesta = sistema(ej["input"])
        similitud = sim_embeddings(respuesta, ej["esperado"])
        puntaje = juez_puntuar(ej["input"], respuesta)
        cumple = cumple_criterio_dominio(puntaje)

        resultados.append({
            "input": ej["input"],
            "criterio": ej["criterio"],
            "respuesta": respuesta,
            "similitud_embeddings": round(similitud, 3),
            "puntaje_juez": puntaje,
            "cumple_criterio_dominio": cumple,
            "longitud_respuesta_palabras": len(respuesta.split()),
        })

    puntajes_validos = [r["puntaje_juez"] for r in resultados if r["puntaje_juez"] is not None]

    scorecard = {
        "sistema": nombre_sistema,
        "n_ejemplos": len(eval_set),
        "dim1_similitud_embeddings_promedio": round(
            np.mean([r["similitud_embeddings"] for r in resultados]), 3
        ),
        "dim2_puntaje_juez_promedio": round(np.mean(puntajes_validos), 2) if puntajes_validos else None,
        "dim3_pct_cumple_criterio_dominio": round(
            100 * sum(r["cumple_criterio_dominio"] for r in resultados) / len(resultados), 1
        ),
    }

    return scorecard, resultados


# -----------------------------------------------------------------------
# Detección de sesgo del juez: verbosidad/longitud
# -----------------------------------------------------------------------
def detectar_sesgo_longitud(resultados: list) -> float:
    """Correlación de Pearson entre longitud de la respuesta y puntaje del juez.
    Si es alta y positiva, el juez está premiando extensión, no calidad."""
    longitudes = [r["longitud_respuesta_palabras"] for r in resultados if r["puntaje_juez"] is not None]
    puntajes = [r["puntaje_juez"] for r in resultados if r["puntaje_juez"] is not None]
    if len(longitudes) < 2:
        return None
    correlacion = float(np.corrcoef(longitudes, puntajes)[0, 1])
    return correlacion


# Versión de la rúbrica SIN la nota anti-longitud, para medir el efecto real de
# la mitigación (no basta con declarar que se mitigó, hay que mostrarlo)
RUBRICA_SIN_MITIGACION = re.sub(
    r"- .*?<!-- LENGTH-BIAS GUARD -->\n?", "", RUBRICA_TEXTO, flags=re.DOTALL
)


def juez_puntuar_con_rubrica(pregunta: str, respuesta: str, texto_rubrica: str) -> int:
    prompt = PROMPT_JUEZ.format(rubrica=texto_rubrica, pregunta=pregunta, respuesta=respuesta)
    mensajes = [{"role": "user", "content": prompt}]
    texto_entrada = juez_tokenizer.apply_chat_template(
        mensajes, tokenize=False, add_generation_prompt=True
    )
    ids = juez_tokenizer(texto_entrada, return_tensors="pt").to(device)
    with torch.no_grad():
        salida = juez_modelo.generate(**ids, max_new_tokens=20, do_sample=False)
    texto_salida = juez_tokenizer.decode(
        salida[0][ids["input_ids"].shape[1]:], skip_special_tokens=True
    )
    match = re.search(r"PUNTAJE:\s*([1-5])", texto_salida) or re.search(r"\b([1-5])\b", texto_salida)
    return int(match.group(1)) if match else None


def comparar_mitigacion_longitud(eval_set: list, sistema, resultados_ya_calculados: list = None) -> dict:
    """Corre el juez SOLO con la rúbrica sin mitigación (la versión con
    mitigación reutiliza lo que harness() ya calculó, para no repetir
    llamadas costosas al modelo/juez dos veces)."""
    if resultados_ya_calculados is not None:
        con_mitigacion = [
            (r["longitud_respuesta_palabras"], r["puntaje_juez"]) for r in resultados_ya_calculados
        ]
        pares_entrada = [(ej, r["respuesta"]) for ej, r in zip(eval_set, resultados_ya_calculados)]
    else:
        pares_entrada = [(ej, sistema(ej["input"])) for ej in eval_set]
        con_mitigacion = []
        for i, (ej, respuesta) in enumerate(pares_entrada, 1):
            print(f"  [con mitigación] {i}/{len(pares_entrada)}...")
            p = juez_puntuar_con_rubrica(ej["input"], respuesta, RUBRICA_TEXTO)
            con_mitigacion.append((len(respuesta.split()), p))

    sin_mitigacion = []
    for i, (ej, respuesta) in enumerate(pares_entrada, 1):
        print(f"  [sin mitigación] {i}/{len(pares_entrada)}...")
        p_sin = juez_puntuar_con_rubrica(ej["input"], respuesta, RUBRICA_SIN_MITIGACION)
        sin_mitigacion.append((len(respuesta.split()), p_sin))

    def correlacion(pares):
        pares_validos = [(l, p) for l, p in pares if p is not None]
        if len(pares_validos) < 2:
            return None
        longs, punts = zip(*pares_validos)
        return float(np.corrcoef(longs, punts)[0, 1])

    return {
        "correlacion_SIN_mitigacion": correlacion(sin_mitigacion),
        "correlacion_CON_mitigacion": correlacion(con_mitigacion),
    }
