"""
Correr el harness sobre el sistema de M1 (flan-t5-small + adaptador LoRA) y
generar scorecard_baseline.csv.

Requiere: rubrica_juez.md, eval_set.py, harness.py, y la carpeta del
adaptador LoRA guardado (travelgenie-lora-adapter/) en el mismo directorio.

Instalación (correr ANTES de este script, y reiniciar el entorno de
ejecución después si es la primera vez en esta sesión):
    !pip install -q -U transformers peft sentence-transformers torchao
"""

import csv
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

from eval_set import eval_set
from harness import harness, detectar_sesgo_longitud, comparar_mitigacion_longitud, device

# -----------------------------------------------------------------------
# Cargar el sistema de M1 (base + adaptador LoRA guardado)
# -----------------------------------------------------------------------
BASE_MODEL = "google/flan-t5-small"
ADAPTER_DIR = "travelgenie-lora-adapter"
PREFIJO_TAREA = "redacta una recomendación de viaje con estos datos: "

tokenizer_m1 = AutoTokenizer.from_pretrained(ADAPTER_DIR)
modelo_base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)
modelo_m1 = PeftModel.from_pretrained(modelo_base, ADAPTER_DIR).to(device)
modelo_m1.eval()


def sistema_m1(pregunta: str) -> str:
    """Interfaz estándar del harness: str -> str.

    Nota de diseño: se le pasa la pregunta CRUDA, sin los hechos que el
    modelo de M1 necesita para responder bien (ver README) — esto es
    intencional: mide la capacidad real del sistema HOY, sin RAG. La
    brecha que esto revele es justamente lo que RAG debe cerrar en M3."""
    entrada = PREFIJO_TAREA + pregunta
    ids = tokenizer_m1(entrada, return_tensors="pt", max_length=220, truncation=True).to(device)
    with __import__("torch").no_grad():
        salida = modelo_m1.generate(
            **ids, max_length=256, min_new_tokens=10, num_beams=4,
            no_repeat_ngram_size=3, repetition_penalty=1.3, early_stopping=True,
        )
    return tokenizer_m1.decode(salida[0], skip_special_tokens=True)


# -----------------------------------------------------------------------
# Correr el harness
# -----------------------------------------------------------------------
print("Corriendo harness sobre el sistema de M1...")
scorecard, resultados = harness(eval_set, sistema_m1, nombre_sistema="M1 (flan-t5-small + LoRA, sin RAG)")

print("\n=== SCORECARD ===")
for k, v in scorecard.items():
    print(f"  {k}: {v}")

# -----------------------------------------------------------------------
# Detectar y mitigar sesgo de longitud
# -----------------------------------------------------------------------
print("\nMidiendo sesgo de verbosidad del juez...")
correlacion_actual = detectar_sesgo_longitud(resultados)
print(f"Correlación longitud-puntaje (con mitigación ya aplicada en la rúbrica): {correlacion_actual}")

print("\nComparando CON vs SIN la instrucción anti-longitud en la rúbrica...")
comparacion = comparar_mitigacion_longitud(eval_set, sistema_m1, resultados_ya_calculados=resultados)
print(json.dumps(comparacion, indent=2))

# -----------------------------------------------------------------------
# Guardar scorecard_baseline.csv
# -----------------------------------------------------------------------
with open("scorecard_baseline.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["dimension", "valor"])
    writer.writerow(["sistema", scorecard["sistema"]])
    writer.writerow(["n_ejemplos", scorecard["n_ejemplos"]])
    writer.writerow(["dim1_similitud_embeddings_promedio", scorecard["dim1_similitud_embeddings_promedio"]])
    writer.writerow(["dim2_puntaje_juez_promedio", scorecard["dim2_puntaje_juez_promedio"]])
    writer.writerow(["dim3_pct_cumple_criterio_dominio", scorecard["dim3_pct_cumple_criterio_dominio"]])
    writer.writerow(["sesgo_correlacion_longitud_SIN_mitigacion", comparacion["correlacion_SIN_mitigacion"]])
    writer.writerow(["sesgo_correlacion_longitud_CON_mitigacion", comparacion["correlacion_CON_mitigacion"]])

with open("scorecard_resultados_detallados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print("\nListo -> scorecard_baseline.csv, scorecard_resultados_detallados.json")
