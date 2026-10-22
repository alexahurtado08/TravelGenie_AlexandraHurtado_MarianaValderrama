"""
RAG básico: primero busca datos y luego redacta la respuesta.

    from rag_naive import sistema_rag
    sistema_rag Qué restaurantes hay en Medellín? entoncess recibe texto, devuelve texto

Funciona igual que el sistema de M1  así que  usamos el harness de M2

Cómo funciona:
  1. Busca en la base los 3 fragmentos (chunks) más parecidos a la pregunta.
  2. Se los pasa al modelo de M1 (flan-t5-small + LoRA) junto con la pregunta.
  3. El modelo redacta la respuesta con esos datos.

Antes de usarlo hay que crear la base con data_pipeline/paso9_indexar_chroma.py.
Los modelos se cargan la primera vez que se llama.

Por qué 3 fragmentos: el modelo de M1 aprendió con los datos de un solo destino;
más fragmentos le meten texto que no conoce.

tiene dos frenos:
  - En el prompt: se le pide decir "no tengo esa información" si los datos no
    alcanzan.
  - En el código: si el fragmento más cercano está demasiado lejos
    ni siquiera se llama al modelo y se responde que no hay
    información. Esto frena lo que no tiene que ver con viajes, pero NO frena una
    ciudad que nuestra base no tiene (por ejemplo, Quibdó): ese es justo el
    fallo que en técnicas avanzadas debemos manejar.
"""

import os

BASE_MODEL = "google/flan-t5-small"
ADAPTER_DIR = "travelgenie-lora-adapter"
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RUTA_CHROMA = "chroma_db"
NOMBRE_COLECCION = "travelgenie"

# usamos prefijo con el que se entrenó el Lora en M1.
PREFIJO_TAREA = "redacta una recomendación de viaje con estos datos: "
INSTRUCCION_ESCAPE = ("Usa solo estos datos. Si la respuesta no está en los datos, "
                      "di que no tienes esa información. ")

K = 3
UMBRAL_DISTANCIA = 0.53
MENSAJE_SIN_INFO = ("No tengo información sobre eso en mi base de datos de destinos "
                    "(solo cubro algunos destinos colombianos con datos de clima, "
                    "vuelos, hoteles, restaurantes y atractivos).")

MAX_INPUT_TOKENS = 512   # en m1 truncaba a 220; aquí cortaría la pregunta 
_estado = {}             # modelos cargados una sola vez
ULTIMO = {}              # contexto recuperado en la última llamada 


def _cargar():
    if _estado:
        return _estado
    import torch
    import chromadb
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from peft import PeftModel

    if not os.path.isdir(RUTA_CHROMA):
        raise FileNotFoundError(
            f"No existe '{RUTA_CHROMA}/'. Corre antes: "
            "python data_pipeline/paso9_indexar_chroma.py")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    _estado["device"] = device
    _estado["emb"] = SentenceTransformer(MODELO_EMBEDDINGS)
    _estado["col"] = chromadb.PersistentClient(path=RUTA_CHROMA).get_collection(NOMBRE_COLECCION)
    _estado["tok"] = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)
    _estado["gen"] = PeftModel.from_pretrained(base, ADAPTER_DIR).to(device).eval()
    return _estado


def recuperar(pregunta, k=K):
    """Devuelve [(texto, distancia_coseno, metadata)] ordenado de más a menos cercano."""
    e = _cargar()
    q = e["emb"].encode([pregunta], normalize_embeddings=True).tolist()
    r = e["col"].query(query_embeddings=q, n_results=k)
    return list(zip(r["documents"][0], r["distances"][0], r["metadatas"][0]))


def construir_prompt(pregunta, textos, con_instruccion=True):
    """Mismo esquema que vio el LoRA en M1: 'Datos: ... Pregunta: ...'."""
    datos = " ".join(textos)
    instruccion = INSTRUCCION_ESCAPE if con_instruccion else ""
    return f"{PREFIJO_TAREA}{instruccion}Datos: {datos} Pregunta: {pregunta}"


def generar(prompt):
    import torch
    e = _cargar()
    ids = e["tok"](prompt, return_tensors="pt", max_length=MAX_INPUT_TOKENS,
                   truncation=True).to(e["device"])
    with torch.no_grad():
        salida = e["gen"].generate(
            **ids, max_length=256, min_new_tokens=10, num_beams=4,
            no_repeat_ngram_size=3, repetition_penalty=1.3, early_stopping=True)
    return e["tok"].decode(salida[0], skip_special_tokens=True)


def sistema_rag(pregunta, k=K, umbral=UMBRAL_DISTANCIA, con_instruccion=True):
    """Interfaz estándar del harness: str -> str."""
    recuperados = recuperar(pregunta, k)
    ULTIMO.clear()
    ULTIMO["recuperados"] = [(m["destino"], m["categoria"], round(d, 3)) for _, d, m in recuperados]
    if umbral is not None and recuperados[0][1] > umbral:
        ULTIMO["rechazada_por_umbral"] = True
        return MENSAJE_SIN_INFO        # válvula de escape
    ULTIMO["rechazada_por_umbral"] = False
    prompt = construir_prompt(pregunta, [t for t, _, _ in recuperados], con_instruccion)
    return generar(prompt)


def calibrar_umbral(preguntas):
    """Imprime la distancia del mejor chunk para cada pregunta, para fijar el umbral."""
    print(f"{'dist_top1':>9}  {'destino/categoría del top1':<34} pregunta")
    for p in preguntas:
        texto, dist, meta = recuperar(p, 1)[0]
        print(f"{dist:9.3f}  {meta['destino'] + ' / ' + meta['categoria']:<34} {p}")


if __name__ == "__main__":
    from eval_set_m3 import eval_set_m3 as eval_set

    print("=== 1) Calibración del umbral (distancia del mejor chunk) ===")
    calibrar_umbral([ej["input"] for ej in eval_set])

    print("\n=== 2) Ejemplos completos ===")
    for p in ["¿Qué restaurantes hay en Medellín?",
              "¿Cuál es el mejor mes para viajar a Santa Marta?",
              "¿Cuál es el mejor mes para viajar a Cartagena de Indias?"]:
        print(f"\nP: {p}")
        for texto, dist, meta in recuperar(p):
            print(f"   [{dist:.3f}] {meta['destino']}/{meta['categoria']}: {texto[:90]}...")
        print(f"R: {sistema_rag(p)}")
