"""
RAG avanzado — TravelGenie M3.  Dos técnicas sobre el RAG básico (rag_naive.py):

  1) BÚSQUEDA HÍBRIDA (BM25 + embeddings, fusionadas con RRF).
     Diagnóstico: en el RAG básico el nombre de la ciudad pesa poco en el
     embedding, y entraban chunks de otras ciudades (Santa Marta -> Barranquilla;
     "mejor mes en Santa Marta" -> clima de Villavicencio y Riohacha). El nombre
     de una ciudad es una palabra rara y exacta: justo lo que BM25 premia.
  2) RERANKING con un cross-encoder multilingüe.
     Diagnóstico: la distancia coseno no distingue "ciudad que el corpus no
     tiene" (Quibdó daba 0.321, más cerca que preguntas válidas). El
     cross-encoder lee pregunta + chunk JUNTOS y da una puntuación de relevancia
     que, se espera, discrimina mejor; se usa también como umbral para decir
     "no tengo información". (Hipótesis: se valida con la ablación.)

Misma firma que los demás sistemas (str -> str):
    sistema_rag_avanzado("¿Qué restaurantes hay en Medellín?")

Pipeline:  pregunta -> [denso top-20] + [BM25 top-20] -> RRF -> top-10
           -> cross-encoder -> top-3 -> mismo prompt y generador que rag_naive.

Los interruptores `hibrida` y `rerank` permiten medir cada técnica por separado
(ablación) con correr_m3.py.

Requisito: haber corrido data_pipeline/paso9_indexar_chroma.py (chroma_db/).
"""

import json
import re
import unicodedata

import rag_naive

RUTA_CHUNKS = "data/09_chunks.json"
MODELO_RERANK = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # multilingüe, ~118M params

N_CANDIDATOS = 20   # candidatos por buscador (denso y BM25)
N_FUSION = 10       # cuántos pasan al reranker tras la fusión RRF
K = 3               # chunks finales al generador (igual que el RAG básico)
RRF_K = 60          # constante estándar de RRF (Cormack et al., 2009)

# Umbral de la puntuación del reranker para decir "no tengo información".
# None = desactivado hasta calibrarlo (python rag_avanzado.py imprime la
# puntuación del mejor chunk para cada pregunta del eval set).
UMBRAL_RERANK = None

_STOP = {"de", "la", "el", "en", "y", "a", "los", "las", "un", "una", "que", "es",
         "por", "para", "con", "del", "al", "hay", "me", "mi", "se", "lo", "como",
         "cual", "cuales", "donde", "puedo", "qu", "cu"}

_estado = {}
ULTIMO = {}   # contexto recuperado en la última llamada (para diagnóstico)


def tokenizar(texto):
    """minúsculas, sin tildes, solo letras/dígitos, sin palabras vacías."""
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return [w for w in re.findall(r"[a-z0-9]+", t) if w not in _STOP]


def rrf(listas, k=RRF_K):
    """Reciprocal Rank Fusion: score(id) = suma de 1/(k + posición) en cada lista."""
    puntaje = {}
    for lista in listas:
        for pos, _id in enumerate(lista, start=1):
            puntaje[_id] = puntaje.get(_id, 0.0) + 1.0 / (k + pos)
    return sorted(puntaje, key=puntaje.get, reverse=True)


def _cargar():
    if "bm25" in _estado:
        return _estado
    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder

    rag_naive._cargar()
    with open(RUTA_CHUNKS, encoding="utf-8") as f:
        chunks = json.load(f)
    _estado["chunks"] = {c["id"]: c for c in chunks}
    _estado["ids"] = [c["id"] for c in chunks]
    _estado["bm25"] = BM25Okapi([tokenizar(c["texto"]) for c in chunks])
    _estado["ce"] = CrossEncoder(MODELO_RERANK, max_length=256,
                                 device=rag_naive._estado["device"])
    return _estado


def buscar_denso(pregunta, n=N_CANDIDATOS):
    e = rag_naive._cargar()
    q = e["emb"].encode([pregunta], normalize_embeddings=True).tolist()
    r = e["col"].query(query_embeddings=q, n_results=n)
    return list(zip(r["ids"][0], r["distances"][0]))


def buscar_bm25(pregunta, n=N_CANDIDATOS):
    e = _cargar()
    puntajes = e["bm25"].get_scores(tokenizar(pregunta))
    orden = sorted(range(len(puntajes)), key=lambda i: puntajes[i], reverse=True)[:n]
    return [e["ids"][i] for i in orden if puntajes[i] > 0]


def reordenar(pregunta, ids):
    """Cross-encoder: devuelve [(id, puntaje)] de mayor a menor relevancia."""
    e = _cargar()
    pares = [(pregunta, e["chunks"][i]["texto"]) for i in ids]
    puntajes = [float(s) for s in e["ce"].predict(pares)]
    return sorted(zip(ids, puntajes), key=lambda x: x[1], reverse=True)


def recuperar_avanzado(pregunta, hibrida=True, rerank=True, k=K):
    """Devuelve (lista_final, mejor_distancia_densa).
    lista_final = [{'id','texto','destino','categoria','rerank'}]"""
    e = _cargar()
    denso = buscar_denso(pregunta)
    mejor_dist = denso[0][1]
    ids_denso = [i for i, _ in denso]

    if hibrida:
        candidatos = rrf([ids_denso, buscar_bm25(pregunta)])[:N_FUSION]
    else:
        candidatos = ids_denso[:N_FUSION]

    if rerank:
        orden = reordenar(pregunta, candidatos)[:k]
    else:
        orden = [(i, None) for i in candidatos[:k]]

    final = []
    for _id, pts in orden:
        c = e["chunks"][_id]
        final.append({"id": _id, "texto": c["texto"], "destino": c["destino"],
                      "categoria": c["categoria"], "rerank": pts})
    return final, mejor_dist


def sistema_rag_avanzado(pregunta, hibrida=True, rerank=True, k=K,
                         umbral_dist=rag_naive.UMBRAL_DISTANCIA,
                         umbral_rerank=UMBRAL_RERANK, con_instruccion=True):
    """Interfaz estándar del harness: str -> str."""
    final, mejor_dist = recuperar_avanzado(pregunta, hibrida, rerank, k)
    ULTIMO.clear()
    ULTIMO["recuperados"] = [(c["destino"], c["categoria"],
                              None if c["rerank"] is None else round(c["rerank"], 3))
                             for c in final]
    ULTIMO["mejor_dist_densa"] = round(mejor_dist, 3)

    rechazada = False
    if umbral_dist is not None and mejor_dist > umbral_dist:
        rechazada = True                      # fuera de dominio (igual que el RAG básico)
    if rerank and umbral_rerank is not None and final[0]["rerank"] < umbral_rerank:
        rechazada = True                      # el reranker no ve nada relevante
    ULTIMO["rechazada_por_umbral"] = rechazada
    if rechazada:
        return rag_naive.MENSAJE_SIN_INFO

    prompt = rag_naive.construir_prompt(pregunta, [c["texto"] for c in final], con_instruccion)
    return rag_naive.generar(prompt)


def calibrar(preguntas):
    """Compara qué recupera el RAG básico vs el avanzado, y muestra la
    puntuación del reranker para fijar UMBRAL_RERANK."""
    print(f"{'dist_densa':>10} {'rerank_top1':>11}  {'top1 avanzado':<28} {'top-3 básico (destinos)':<44} pregunta")
    for p in preguntas:
        final, mejor = recuperar_avanzado(p)
        basico = [m["destino"][:6] for _, _, m in rag_naive.recuperar(p, 3)]
        print(f"{mejor:10.3f} {final[0]['rerank']:11.3f}  "
              f"{final[0]['destino'] + '/' + final[0]['categoria']:<28} {str(basico):<44} {p}")


if __name__ == "__main__":
    from eval_set import eval_set

    print("=== Calibración: básico vs avanzado, y puntuación del reranker ===")
    calibrar([ej["input"] for ej in eval_set])

    print("\n=== Ejemplos completos ===")
    for p in ["¿Cuál es el mejor mes para viajar a Santa Marta?",
              "¿Qué hoteles recomiendas en Quibdó?"]:
        print(f"\nP: {p}")
        final, mejor = recuperar_avanzado(p)
        for c in final:
            print(f"   [rerank={c['rerank']:.3f}] {c['destino']}/{c['categoria']}: {c['texto'][:80]}...")
        print(f"R: {sistema_rag_avanzado(p)}")
