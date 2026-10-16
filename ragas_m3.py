"""
RAGAS de M3 — fidelidad, relevancia de la respuesta, precisión y exhaustividad
del contexto, para RAG básico, RAG avanzado y Agente (el sistema final).

IMPLEMENTACIÓN: propia, siguiendo las definiciones del paper de RAGAS
(Es et al., 2023), NO la librería `ragas`. Motivo: la librería exige que el
modelo juez devuelva JSON estructurado sin errores; un juez de 1.5B parámetros
falla ahí con frecuencia y la librería devuelve NaN. Aquí cada pregunta al
juez es una decisión Sí/No, que un modelo pequeño sí sabe contestar.
JUEZ: Qwen2.5-1.5B-Instruct local (sin llaves de pago). Es el mismo modelo del
juez de M2 y del agente: sesgo de auto-preferencia posible y juez débil; las
cifras son indicativas, no precisas.

Definiciones (todas en [0, 1], más alto = mejor):
  - faithfulness       : fracción de afirmaciones de la respuesta (oraciones)
                         que el contexto recuperado respalda.  Detecta INVENCIÓN.
  - answer_relevancy   : similitud coseno media entre la pregunta y 3 preguntas
                         que el juez deduce de la respuesta (embeddings).
                         Detecta respuestas fuera de tema.
  - context_precision  : precisión promedio (AP@k) de los chunks recuperados,
                         según si el juez los considera útiles para la respuesta
                         de referencia.  Detecta RUIDO en la recuperación.
  - context_recall     : fracción de oraciones de la respuesta de referencia
                         (campo `esperado` del eval set) que el contexto cubre.
                         Detecta datos que la recuperación NO trajo.

Lectura: faithfulness baja + recall alto = falla de GENERACIÓN (el generador
inventa aunque tenía los datos). Recall bajo = falla de RECUPERACIÓN.

Solo se evalúan las respuestas que pasaron por el generador (con contexto).
Las rechazadas ("No tengo información…") se cuentan aparte: no tienen
afirmaciones que verificar.  Los contextos se reconstruyen con la misma
recuperación determinista que usó cada sistema.

Uso (Colab, después de correr correr_m3.py de los tres sistemas):
    python ragas_m3.py
Salida: ragas_m3.csv (por sistema) y ragas_m3_detalle.csv (por pregunta).
"""

import csv
import json
import os
import re
import statistics

MARCA_SIN_INFO = "No tengo información sobre eso"
SISTEMAS = [
    ("rag_naive", "RAG básico"),
    ("rag_avanzado", "RAG avanzado"),
    ("agente", "Agente (sistema final)"),
]
METRICAS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


# ---------------------------------------------------------------------------
# Piezas puras (reciben las funciones del juez, así se pueden probar sin modelo)
# ---------------------------------------------------------------------------
def oraciones(texto, min_palabras=3):
    partes = re.split(r"(?<=[.!?])\s+|\n+", texto.strip())
    return [p.strip() for p in partes if len(p.split()) >= min_palabras]


def faithfulness(respuesta, contexto, si_no):
    afirmaciones = oraciones(respuesta)
    if not afirmaciones:
        return None
    ok = sum(si_no(
        f"Contexto:\n{contexto}\n\nAfirmación: {a}\n\n"
        "¿La afirmación está respaldada por el contexto? Responde solo Sí o No.")
        for a in afirmaciones)
    return ok / len(afirmaciones)


def context_recall(referencia, contexto, si_no):
    frases = oraciones(referencia, min_palabras=2)
    if not frases:
        return None
    ok = sum(si_no(
        f"Contexto:\n{contexto}\n\nFrase: {f}\n\n"
        "¿La información de esta frase aparece en el contexto? Responde solo Sí o No.")
        for f in frases)
    return ok / len(frases)


def context_precision(pregunta, referencia, chunks, si_no):
    """Precisión promedio: sum_k (precision@k * relevante_k) / n_relevantes."""
    rel = [int(si_no(
        f"Pregunta: {pregunta}\nRespuesta de referencia: {referencia}\n\nFragmento: {c}\n\n"
        "¿El fragmento es útil para llegar a la respuesta de referencia? Responde solo Sí o No."))
        for c in chunks]
    if not chunks:
        return None
    if sum(rel) == 0:
        return 0.0
    acum, total = 0, 0.0
    for k, r in enumerate(rel, start=1):
        acum += r
        if r:
            total += acum / k
    return total / sum(rel)


def answer_relevancy(pregunta, respuesta, generar_preguntas, similitud, n=3):
    preguntas = generar_preguntas(respuesta, n)
    if not preguntas:
        return 0.0
    return statistics.mean(similitud(pregunta, q) for q in preguntas)


def media(valores):
    v = [x for x in valores if x is not None]
    return round(statistics.mean(v), 3) if v else ""


# ---------------------------------------------------------------------------
# Cableado con modelos reales (solo se usa al ejecutar el script)
# ---------------------------------------------------------------------------
def _juez_funciones():
    import torch
    from harness import juez_modelo, juez_tokenizer, device

    def charlar(prompt, max_new_tokens):
        txt = juez_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        ids = juez_tokenizer(txt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = juez_modelo.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        return juez_tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def si_no(prompt):
        r = charlar(prompt, 4).strip().lower()
        return r.startswith("s")            # "sí", "si"

    def generar_preguntas(respuesta, n):
        r = charlar(f"Respuesta: {respuesta}\n\nEscribe {n} preguntas distintas, una por línea y "
                    "sin numerar, a las que esta respuesta podría servir de contestación.", 90)
        qs = [re.sub(r"^[\-\d\.\)\s]+", "", l).strip() for l in r.splitlines()]
        return [q for q in qs if len(q) > 5][:n]

    return si_no, generar_preguntas


def _similitud():
    import rag_naive
    emb = rag_naive._cargar()["emb"]

    def sim(a, b):
        va, vb = emb.encode([a, b], normalize_embeddings=True)
        return float(va @ vb)
    return sim


def contextos(clave, pregunta):
    """Textos que el sistema pasó al generador (misma recuperación determinista)."""
    if clave == "rag_naive":
        import rag_naive
        rec = rag_naive.recuperar(pregunta)
        return [t for t, _, _ in rec]
    import rag_avanzado
    final, _ = rag_avanzado.recuperar_avanzado(pregunta)
    return [c["texto"] for c in final if c["rerank"] is None or c["rerank"] >= rag_avanzado.UMBRAL_RERANK]


def main():
    from eval_set_m3 import eval_set_m3
    referencia = {e["input"]: e["esperado"] for e in eval_set_m3}
    si_no, generar_preguntas = _juez_funciones()
    sim = _similitud()

    resumen, detalle = [], []
    for clave, nombre in SISTEMAS:
        ruta = f"scorecard_m3_detalle_{clave}.json"
        if not os.path.exists(ruta):
            print(f"(falta {ruta}, se omite)")
            continue
        with open(ruta, encoding="utf-8") as f:
            resultados = json.load(f)
        filas, rechazadas = [], 0
        for r in resultados:
            if MARCA_SIN_INFO in r["respuesta"]:
                rechazadas += 1
                continue
            p, resp, ref = r["input"], r["respuesta"], referencia[r["input"]]
            chunks = contextos(clave, p)
            ctx = "\n".join(f"- {c}" for c in chunks)
            fila = {
                "sistema": nombre, "pregunta": p,
                "faithfulness": faithfulness(resp, ctx, si_no),
                "answer_relevancy": answer_relevancy(p, resp, generar_preguntas, sim),
                "context_precision": context_precision(p, ref, chunks, si_no),
                "context_recall": context_recall(ref, ctx, si_no),
            }
            filas.append(fila)
            detalle.append(fila)
            print(f"[{nombre}] {p[:50]:50} " + " ".join(f"{m[:5]}={fila[m]:.2f}" if fila[m] is not None else f"{m[:5]}=—" for m in METRICAS))
        resumen.append({"sistema": nombre, "n_evaluadas": len(filas), "n_rechazadas": rechazadas,
                        **{m: media([f[m] for f in filas]) for m in METRICAS}})

    with open("ragas_m3.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(resumen[0].keys()))
        w.writeheader()
        w.writerows(resumen)
    with open("ragas_m3_detalle.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sistema", "pregunta"] + METRICAS)
        w.writeheader()
        w.writerows(detalle)

    print("\n=== RAGAS (juez: Qwen2.5-1.5B local, implementación propia) ===")
    cols = list(resumen[0].keys())
    print(" | ".join(cols))
    for fila in resumen:
        print(" | ".join(str(fila[c]) for c in cols))
    print("\nGuardado: ragas_m3.csv y ragas_m3_detalle.csv")


if __name__ == "__main__":
    main()
