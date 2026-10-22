"""
Harness de evaluación — TravelGenie (M2, v2 tras la retroalimentación)

Junta 3 dimensiones sobre el mismo eval_set:
  1. Similitud por embeddings entre la respuesta y la respuesta de referencia (`esperado`).
  2. LLM-as-a-judge: Qwen2.5 pequeño, rúbrica anclada 1-5 (ver rubrica_juez.md).
     v2: el juez recibe la pregunta, la respuesta de referencia y el criterio del
     caso, además de la respuesta (antes solo veía pregunta y respuesta).
  3. Criterio de dominio: % de casos que cumplen la regla propia de su `criterio`
     (ver verificacion.py). v2: ya no depende del juez.

harness(eval_set, sistema) es genérica: 'sistema' es cualquier función
input(str) -> output(str). Así se reutiliza tal cual en M3 sobre el sistema con RAG.
"""

import re
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

from verificacion import verificar_criterio

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

RESPUESTA DE REFERENCIA (lo que diría un buen sistema, con los datos reales): {esperado}

CRITERIO ESPECÍFICO DE ESTE CASO: {criterio}

RESPUESTA A EVALUAR: {respuesta}

Sigue estos pasos:
1. Mira el CRITERIO: ¿qué tendría que hacer la respuesta (dar datos, admitir que no
   tiene información, rechazar, corregir una premisa)?
2. ¿La respuesta a evaluar hace justo eso? Si el criterio pide admitir la falta de
   datos, rechazar o corregir, una respuesta que lo hace merece 5 aunque sea corta,
   y una que responde como si nada merece 1 o 2.
3. Si repite la pregunta, no dice nada útil o mezcla datos de otra ciudad, es 1 o 2.
4. En listas de lugares, la referencia es solo un ejemplo: otros nombres son válidos
   si suenan a lugares reales de esa ciudad. Son inventados si no tienen sentido.
5. Si da meses, precios u orígenes distintos a los de la referencia, es 2.
Responde ÚNICAMENTE en este formato exacto, sin explicación adicional:
PUNTAJE: <número del 1 al 5>
"""


def _prompt_juez(rubrica: str, pregunta: str, respuesta: str, esperado: str, criterio: str) -> str:
    return PROMPT_JUEZ.format(rubrica=rubrica, pregunta=pregunta, respuesta=respuesta,
                              esperado=esperado, criterio=criterio)


def juez_puntuar(pregunta: str, respuesta: str, esperado: str, criterio: str) -> int:
    """Llama al juez y parsea el puntaje de forma robusta (regex, no parsing frágil)."""
    prompt = _prompt_juez(RUBRICA_TEXTO, pregunta, respuesta, esperado, criterio)
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
# Dimensión 3 — cumplimiento del criterio de dominio (verificación propia)
# -----------------------------------------------------------------------
# Cada caso trae en `verificar` una regla comprobable sacada de su `criterio`
# (datos exactos presentes, nombres reales, admite falta de datos...). Ver
# verificacion.py. El puntaje del juez >= 4 se guarda solo como dato de apoyo.
UMBRAL_JUEZ = 4


def juez_ge_umbral(puntaje_juez):
    return puntaje_juez is not None and puntaje_juez >= UMBRAL_JUEZ


# -----------------------------------------------------------------------
# harness() — junta las 3 dimensiones sobre cualquier 'sistema'
# -----------------------------------------------------------------------
def harness(eval_set: list, sistema, nombre_sistema: str = "sistema") -> dict:
    """
    eval_set: lista de dicts con 'input', 'esperado', 'criterio', 'verificar' (ver eval_set.py)
    sistema: función str -> str (recibe el input, devuelve la respuesta generada)
    """
    resultados = []

    for ej in eval_set:
        respuesta = sistema(ej["input"])
        similitud = sim_embeddings(respuesta, ej["esperado"])
        puntaje = juez_puntuar(ej["input"], respuesta, ej["esperado"], ej["criterio"])
        cumple, motivo = verificar_criterio(respuesta, ej)

        resultados.append({
            "input": ej["input"],
            "criterio": ej["criterio"],
            "respuesta": respuesta,
            "similitud_embeddings": round(similitud, 3),
            "puntaje_juez": puntaje,
            "cumple_criterio_dominio": cumple,
            "motivo_verificacion": motivo,
            "juez_ge_4": juez_ge_umbral(puntaje),
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


def juez_puntuar_con_rubrica(pregunta: str, respuesta: str, texto_rubrica: str,
                             esperado: str, criterio: str) -> int:
    prompt = _prompt_juez(texto_rubrica, pregunta, respuesta, esperado, criterio)
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
            p = juez_puntuar_con_rubrica(ej["input"], respuesta, RUBRICA_TEXTO, ej["esperado"], ej["criterio"])
            con_mitigacion.append((len(respuesta.split()), p))

    sin_mitigacion = []
    for i, (ej, respuesta) in enumerate(pares_entrada, 1):
        print(f"  [sin mitigación] {i}/{len(pares_entrada)}...")
        p_sin = juez_puntuar_con_rubrica(ej["input"], respuesta, RUBRICA_SIN_MITIGACION, ej["esperado"], ej["criterio"])
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
