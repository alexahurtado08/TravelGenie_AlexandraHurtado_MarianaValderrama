"""
Evaluación M3 — corre el harness de M2 (SIN modificarlo) sobre un sistema y
guarda su fila en scorecard_m3.csv, con la latencia mediana por pregunta.

Uso (desde la raíz del repo, después de haber corrido paso9 para crear chroma_db/):
    python correr_m3.py m2_baseline
    python correr_m3.py rag_naive
(más adelante se añadirán 'rag_avanzado' y 'agente' al diccionario SISTEMAS)

Cada ejecución reemplaza la fila de ese sistema, así que se puede repetir sin
duplicar. Mismo eval_set y mismo juez que en M2 -> los números son comparables.

Instalación previa en Colab (y reiniciar la sesión si es la primera vez):
    !pip install -q -U transformers peft accelerate sentence-transformers torchao chromadb==1.5.9
"""

import csv
import json
import os
import statistics
import sys
import time

from eval_set import eval_set
from harness import harness, device   # carga el juez Qwen y los embeddings (M2)

ARCHIVO_CSV = "scorecard_m3.csv"
COLUMNAS = ["sistema", "n_ejemplos", "dim1_similitud_embeddings_promedio",
            "dim2_puntaje_juez_promedio", "dim3_pct_cumple_criterio_dominio",
            "latencia_mediana_s", "n_respuestas_sin_info"]


# -----------------------------------------------------------------------
# Sistemas (todos: str -> str)
# -----------------------------------------------------------------------
def construir_m2_baseline():
    """Sistema de M1/M2: flan-t5-small + LoRA, pregunta cruda, sin RAG.
    Copia de sistema_m1 de correr_harness.py (ese archivo ejecuta todo el
    harness al importarse, por eso no se puede importar de allí)."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained("travelgenie-lora-adapter")
    base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)
    modelo = PeftModel.from_pretrained(base, "travelgenie-lora-adapter").to(device).eval()
    prefijo = "redacta una recomendación de viaje con estos datos: "

    def sistema(pregunta):
        ids = tok(prefijo + pregunta, return_tensors="pt", max_length=220,
                  truncation=True).to(device)
        with torch.no_grad():
            salida = modelo.generate(**ids, max_length=256, min_new_tokens=10, num_beams=4,
                                     no_repeat_ngram_size=3, repetition_penalty=1.3,
                                     early_stopping=True)
        return tok.decode(salida[0], skip_special_tokens=True)
    return sistema


def construir_rag_naive():
    import rag_naive
    rag_naive._cargar()                      # carga modelos ANTES de medir tiempos
    rag_naive.sistema_rag("¿Qué hoteles hay en Medellín?")   # calentamiento
    return rag_naive.sistema_rag


SISTEMAS = {
    "m2_baseline": ("M2 baseline (LoRA sin RAG)", construir_m2_baseline),
    "rag_naive": ("RAG básico (top-3 + válvula de escape)", construir_rag_naive),
}


# -----------------------------------------------------------------------
def guardar_fila(fila):
    filas = []
    if os.path.exists(ARCHIVO_CSV):
        with open(ARCHIVO_CSV, newline="", encoding="utf-8") as f:
            filas = [r for r in csv.DictReader(f) if r["sistema"] != fila["sistema"]]
    filas.append(fila)
    with open(ARCHIVO_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        w.writeheader()
        w.writerows(filas)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in SISTEMAS:
        sys.exit(f"Uso: python correr_m3.py {{{'|'.join(SISTEMAS)}}}")
    clave = sys.argv[1]
    nombre, constructor = SISTEMAS[clave]
    print(f"Construyendo sistema: {nombre} (device={device})")
    sistema = constructor()

    tiempos, extras = [], []

    def medido(pregunta):
        t0 = time.perf_counter()
        r = sistema(pregunta)
        tiempos.append(time.perf_counter() - t0)
        if clave == "rag_naive":
            import rag_naive
            extras.append(dict(rag_naive.ULTIMO))
        return r

    print("Corriendo harness...")
    scorecard, resultados = harness(eval_set, medido, nombre_sistema=nombre)

    sin_info = 0
    for i, r in enumerate(resultados):
        r["latencia_s"] = round(tiempos[i], 3)
        if extras:
            r["contexto_recuperado"] = extras[i].get("recuperados")
            r["rechazada_por_umbral"] = extras[i].get("rechazada_por_umbral")
            sin_info += bool(extras[i].get("rechazada_por_umbral"))

    fila = {
        "sistema": nombre,
        "n_ejemplos": scorecard["n_ejemplos"],
        "dim1_similitud_embeddings_promedio": scorecard["dim1_similitud_embeddings_promedio"],
        "dim2_puntaje_juez_promedio": scorecard["dim2_puntaje_juez_promedio"],
        "dim3_pct_cumple_criterio_dominio": scorecard["dim3_pct_cumple_criterio_dominio"],
        "latencia_mediana_s": round(statistics.median(tiempos), 3),
        "n_respuestas_sin_info": sin_info,
    }
    guardar_fila(fila)
    with open(f"scorecard_m3_detalle_{clave}.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    print("\n=== SCORECARD ===")
    for k, v in fila.items():
        print(f"  {k}: {v}")
    print("\n=== POR PREGUNTA (puntaje juez | latencia | respuesta) ===")
    for r in resultados:
        print(f"[{r['puntaje_juez']}] {r['latencia_s']:.1f}s  {r['input']}")
        if "contexto_recuperado" in r:
            print(f"     contexto: {r['contexto_recuperado']}")
        print(f"     -> {r['respuesta'][:220]}")
    print(f"\nGuardado: {ARCHIVO_CSV} y scorecard_m3_detalle_{clave}.json")


if __name__ == "__main__":
    main()
