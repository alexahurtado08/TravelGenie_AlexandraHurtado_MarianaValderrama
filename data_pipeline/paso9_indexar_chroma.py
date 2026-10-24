"""
Paso 9 — Prepara el corpus para que TravelGenie pueda buscar en él.

Parte los documentos en chunks, convierte cada trozo en un
vector con un modelo de embeddings y lo guarda en Chroma. Lee
data/08_corpus.json y deja data/09_chunks.json y chroma_db/ (esta carpeta no
se sube al repo: se reconstruye corriendo el script).

Tres decisiones, pensadas para este corpus:
1) Trozos de 35 palabras con solape de 7. Clima y vuelos son textos cortos y
   quedan enteros; las listas de hoteles y restaurantes llegan a más de 2.000
   palabras y sí se parten.
2) Cada trozo repite el encabezado ("Restaurantes en MEDELLÍN…"). Sin él, la
   segunda parte de una lista no diría de qué ciudad es.
3) Embeddings con paraphrase-multilingual-MiniLM-L12-v2: entiende español, es
   liviano y corre en Colab gratis. Solo lee los primeros 128 tokens de cada
   trozo, por eso el script revisa que ninguno se pase.

Uso, desde la raíz del repo:
    pip install -q -r requirements.txt
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
