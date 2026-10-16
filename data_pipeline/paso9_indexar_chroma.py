"""
Paso 9 — Chunking, embeddings e indexado en Chroma (M3, RAG).

Lee data/08_corpus.json y produce:
  - data/09_chunks.json  : los chunks finales (útil para inspección y BM25 después)
  - chroma_db/           : base vectorial persistente (se puede reconstruir
                           corriendo este script; NO hace falta subirla al repo)

Decisiones de diseño (justificadas con los datos de ESTE corpus):

1) Chunking por palabras con ventana deslizante (CHUNK_PALABRAS / SOLAPE).
   Clima y vuelos miden 18-32 palabras: caben enteros, no se parten.
   Hoteles/restaurantes/atractivos de ciudades grandes llegan a 2127 palabras
   (listas de nombres separados por comas): esos sí se parten.

2) El ENCABEZADO ("Restaurantes en MEDELLÍN, ANTIOQUIA:") se repite al inicio
   de cada chunk. Sin eso, el 2.º trozo de una lista larga no diría de qué
   ciudad es y el retriever no podría asociarlo a la pregunta.

3) Modelo de embeddings: paraphrase-multilingual-MiniLM-L12-v2, el mismo de
   la dimensión 1 del harness M2 (multilingüe, pequeño, corre en Colab gratis).
   Lee como máximo `max_seq_length` tokens (128 por defecto); el resto se
   descarta en silencio. Por eso el script mide cuántos chunks lo exceden.

Uso (desde la raíz del repo):
    pip install -q sentence-transformers==6.1.0 chromadb==1.5.9
    python data_pipeline/paso9_indexar_chroma.py
"""

import json
import os
import random

SEED = 42
CHUNK_PALABRAS = 35   # palabras de contenido por chunk (sin contar encabezado); 60 truncaba 151/247 chunks
SOLAPE = 7            # palabras compartidas entre chunks consecutivos
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RUTA_CORPUS = "data/08_corpus.json"
RUTA_CHUNKS = "data/09_chunks.json"
RUTA_CHROMA = "chroma_db"
NOMBRE_COLECCION = "travelgenie"


def separar_encabezado(texto):
    """'Restaurantes en X, Y: a, b, c.' -> ('Restaurantes en X, Y:', 'a, b, c.')"""
    if ":" in texto:
        enc, cuerpo = texto.split(":", 1)
        return enc.strip() + ":", cuerpo.strip()
    return "", texto.strip()


def partir_documento(texto, chunk_palabras=CHUNK_PALABRAS, solape=SOLAPE):
    """Devuelve la lista de chunks de un documento. Cada chunk lleva el encabezado."""
    assert 0 <= solape < chunk_palabras
    encabezado, cuerpo = separar_encabezado(texto)
    palabras = cuerpo.split()
    if len(palabras) <= chunk_palabras:
        return [texto]  # cabe entero: no se toca

    paso = chunk_palabras - solape
    chunks = []
    for inicio in range(0, len(palabras), paso):
        trozo = palabras[inicio:inicio + chunk_palabras]
        chunks.append(f"{encabezado} {' '.join(trozo)}".strip())
        if inicio + chunk_palabras >= len(palabras):
            break
    return chunks


def construir_chunks(corpus):
    chunks = []
    for doc in corpus:
        partes = partir_documento(doc["texto"])
        for i, texto in enumerate(partes):
            chunks.append({
                "id": f"{doc['id']}_c{i}",
                "texto": texto,
                "destino": doc["destino"],
                "categoria": doc["categoria"],
                "doc_id": doc["id"],
                "chunk_idx": i,
            })
    return chunks


def main():
    random.seed(SEED)

    with open(RUTA_CORPUS, encoding="utf-8") as f:
        corpus = json.load(f)
    chunks = construir_chunks(corpus)

    with open(RUTA_CHUNKS, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"{len(corpus)} documentos -> {len(chunks)} chunks "
          f"(chunk={CHUNK_PALABRAS} palabras, solape={SOLAPE})")
    por_cat = {}
    for c in chunks:
        por_cat[c["categoria"]] = por_cat.get(c["categoria"], 0) + 1
    for cat, n in por_cat.items():
        print(f"  {cat}: {n} chunks")

    # ---- embeddings ----
    from sentence_transformers import SentenceTransformer
    modelo = SentenceTransformer(MODELO_EMBEDDINGS)
    limite = modelo.max_seq_length
    tokens = [len(modelo.tokenizer.encode(c["texto"])) for c in chunks]
    excedidos = sum(t > limite for t in tokens)
    print(f"\nTokens por chunk: max={max(tokens)}, promedio={sum(tokens)/len(tokens):.1f}, "
          f"límite del modelo={limite}")
    print(f"Chunks que EXCEDEN el límite (se truncan): {excedidos}/{len(chunks)}")
    if excedidos:
        print("  -> baja CHUNK_PALABRAS y vuelve a correr.")

    vectores = modelo.encode([c["texto"] for c in chunks],
                             normalize_embeddings=True, show_progress_bar=True)

    # ---- Chroma ----
    import chromadb
    cliente = chromadb.PersistentClient(path=RUTA_CHROMA)
    try:
        cliente.delete_collection(NOMBRE_COLECCION)  # reconstrucción limpia
    except Exception:
        pass
    col = cliente.create_collection(NOMBRE_COLECCION, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[c["id"] for c in chunks],
        documents=[c["texto"] for c in chunks],
        embeddings=vectores.tolist(),
        metadatas=[{"destino": c["destino"], "categoria": c["categoria"],
                    "doc_id": c["doc_id"], "chunk_idx": c["chunk_idx"]} for c in chunks],
    )
    print(f"\nIndexados {col.count()} chunks en '{RUTA_CHROMA}/' "
          f"(colección '{NOMBRE_COLECCION}').")

    # ---- prueba de sanidad: ¿recupera lo esperado? ----
    for pregunta in ["¿Cuándo es mejor época para viajar a Cartagena?",
                     "restaurantes en Medellín",
                     "vuelos baratos a Santa Marta"]:
        q = modelo.encode([pregunta], normalize_embeddings=True).tolist()
        r = col.query(query_embeddings=q, n_results=3)
        print(f"\nP: {pregunta}")
        for doc_id, dist in zip(r["ids"][0], r["distances"][0]):
            print(f"   {doc_id}  (dist coseno={dist:.3f})")


if __name__ == "__main__":
    main()