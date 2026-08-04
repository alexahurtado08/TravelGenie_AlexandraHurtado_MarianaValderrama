# TravelGenie

> **TravelGenie recomienda a dónde viajar en Colombia y en qué mes conviene más, combinando clima real, precios de vuelos y lugares turísticos — para que cualquier persona pueda planear un viaje sin tener que investigar todo por su cuenta.**

*integrantes*: Alexandra Hurtado, Mariana Valderrama

---

## 1. Modelo base y familia elegida

**Modelo:** flan-t5-small (encoder-decoder)

**Por qué esta familia:** la tarea central del proyecto es *data-to-text* — convertir datos estructurados (clima, precio de vuelo, hoteles/restaurantes/atractivos) en una recomendación en lenguaje natural. Esto es exactamente lo que el objetivo de preentrenamiento de T5  lo que hace es mapear una secuencia de entrada a una secuencia de salida de longitud arbitraria. Un modelo *encoder-only* (como BERT) no puede generar texto libre; un modelo *decoder-only* (como GPT) también podría, pero el encoder-decoder separa mejor la comprensión de la consulta (encoder) de la generación de la respuesta (decoder), lo cual es más natural para una tarea con un input corto y estructurado y un output largo.



## 2. Dataset de dominio

tenemos **384 pares** entrada→salida (307 train / 77 validation), generados por plantilla a partir de datos reales de clima, vuelos y lugares turísticos de 16 destinos colombianos. Para infromación más detallada Ver **[DATASET.md](./DATASET.md)**

## 3. Baseline y comparación

**Baseline:** el mismo `flan-t5-small`, **sin fine-tuning**, con el mismo prompt y el mismo conjunto de validación.

**Por qué este baseline:** aísla el efecto del fine-tuning — como es el mismo modelo, cualquier diferencia en las métricas se explica por el LoRA, no por diferencias de arquitectura o tamaño. Es la opción que la guía de la asignación señala como "lo más recomendable".

## 4. Configuración de LoRA

| Hiperparámetro | Valor | Por qué |
|---|---|---|
| `r` (rank) | 8 | Rank bajo, apropiado para un dataset de entrenamiento pequeño (307 ejemplos) — un rank alto arriesga sobreajuste. |
| `lora_alpha` | 16 | Heurística estándar (2× el rank). |
| `target_modules` | `["q", "v"]` | Proyecciones de atención query/value del T5 — recomendado por el paper original de LoRA para mantener pocos parámetros entrenables sin perder capacidad de adaptación. |
| `learning_rate` | 5e-4 | Encontrado empíricamente: 1e-3 causaba colapso en loops de repetición; 3e-4 subaprendía (salidas demasiado cortas/genéricas). 5e-4 es el punto medio. |
| `num_train_epochs` | 12 | Mismo motivo — 15 sobreajustaba, 8 subaprendía. |
| Parámetros entrenables | 344,064 (0.45% del total) | Confirmado con `print_trainable_parameters()`. |

## 5. Resultados

| Métrica | Baseline (zero-shot) | Con fine-tuning LoRA | Delta |
|---|---|---|---|
| ROUGE-1 | 0.3526 | 0.6041 | **+0.2515** |
| ROUGE-2 | 0.1992 | 0.4579 | **+0.2587** |
| ROUGE-L | 0.2863 | 0.5156 | **+0.2293** |
| ROUGE-Lsum | 0.2863 | 0.5153 | **+0.2290** |



### Conclusiones

El fine-tuning con LoRA mejoró sustancialmente todas las variantes de ROUGE (+0.23 a +0.26). Más importante que el número: la mejora se ve también cualitativamente. Después de reformular la tarea, el modelo afinado **copia correctamente** los hechos que se le dan en el input (nombres de hoteles, restaurantes, atractivos, meses, precios) en la gran mayoría de los ejemplos de validación, algo que el baseline zero-shot no hace de forma confiable.

Sigue fallando en dos formas puntuales, ninguna relacionada con "inventar" información:
- **Mezcla de categorías en algunos destinos** (ej. reporta hoteles bajo la etiqueta "para comer" en 1 de 3 ejemplos cualitativos) — error de organización del texto, no de contenido falso.
- **Pierde la tilde en el carácter "í"** (ej. "Medellín" → "Medelln"). Esto ocurre **también en el baseline sin fine-tuning**, así que es una característica del tokenizer de flan-t5-small con ese carácter específico, no algo introducido por el entrenamiento.



## 7. Cómo correr esto

1. Abre `notebook_finetuning_lora.py` en Google Colab (o cópialo celda por celda a un notebook nuevo).
2. Activa GPU: `Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)`.
3. Corre las celdas 1 a 5 en orden. Semilla fijada en `SEED = 42` para reproducibilidad.
4. El pipeline de recolección/preparación de datos (scraping/APIs, generación de pares) está en la carpeta `data_pipeline/` — ver su propio README si necesitas regenerar el dataset desde cero.

## 8. Atribución de fuentes de datos

Este proyecto usa datos de terceros bajo sus respectivas licencias — ver [DATASET.md](./DATASET.md) para el detalle completo. En particular, incluye datos de © OpenStreetMap contributors, disponibles bajo la Open Database License (ODbL) — más información en [openstreetmap.org/copyright](https://www.openstreetmap.org/copyright).
