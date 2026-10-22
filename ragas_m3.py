"""
RAGAS de M3: mide qué tan bien funcionan la búsqueda y la redacción.

Usamos el modelo Qwen2.5-1.5B (local, sin llaves de pago) como juez, y
calculamos las cuatro métricas de RAGAS con nuestro propio código en vez de la
librería `ragas`, que con un juez tan pequeño suele fallar y devolver vacíos.

  - faithfulness: lo que dice la respuesta está en los datos? o esta inventando?
  - answer_relevancy: la respuesta trata de lo que se preguntó?
  - context_precision: los fragmentos que trajo la búsqueda si sirven?
  - context_recall: trajo los datos que nos  hacían falta?

Todas van de 0 a 1 (más alto es mejor). Si faithfulness es baja pero recall es
alto, el fallo está en la redacción; si recall es bajo, está en la búsqueda.

Solo se evalúan las respuestas que pasaron por el generador; las rechazadas
"No tengo información" se cuentan aparte

Uso:  python ragas_m3.py después de correr_m3.py
Salida: ragas_m3.csv por sistema y ragas_m3_detalle.csv 
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



#  reciben las funciones del juez para probar sin modelo

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



# lo hacemos con modelos reales y solo se usa al ejecutar el script

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
        return r.startswith("s")            

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
