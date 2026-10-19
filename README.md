# TravelGenie

> **TravelGenie recommends where to travel in Colombia and which month works best, combining real climate data, flight prices, and tourist attractions — so anyone can plan a trip without having to research everything on their own.**

*team members*: Alexandra Hurtado, Mariana Valderrama

> **M3 (RAG + agente) en una frase:** TravelGenie ayuda a quien planea un viaje por Colombia a saber a qué destino ir y en qué mes (clima, vuelos, hoteles, restaurantes y atractivos de 14 destinos); una buena respuesta usa solo los datos de esa base, trae el mes, el precio y el origen correctos, y cuando le preguntan por una ciudad que no tiene lo dice en lugar de inventar. Lo que sigue fallando es la redacción de su modelo pequeño, no la búsqueda de los datos. El informe completo de M3 está en **[`reporteM3`](https://github.com/alexahurtado08/TravelGenie_AlexandraHurtado_MarianaValderrama/wiki/reporte.md)**.

---

## 1. Base model and chosen family

**Model:** flan-t5-small (encoder-decoder)

**Why this family:** the project's core task is *data-to-text* — converting structured data (climate, flight price, hotels/restaurants/attractions) into a natural-language recommendation. This is exactly what T5's pretraining objective does: it maps an input sequence to an output sequence of arbitrary length. An *encoder-only* model (like BERT) cannot generate free-form text; a *decoder-only* model (like GPT) could also do it, but the encoder-decoder architecture better separates understanding the query (encoder) from generating the response (decoder), which is more natural for a task with a short, structured input and a long output.

## 2. Domain dataset

We have **346** input→output pairs (276 train / 70 validation), template-generated from real climate, flight, and tourist attraction data across 14 Colombian destinations. For more detailed information, see **[DATASET](https://github.com/alexahurtado08/TravelGenie_AlexandraHurtado_MarianaValderrama/wiki/DATASET)**.

## 3. Baseline and comparison

**Baseline:** the same `flan-t5-small`, **without fine-tuning**, using the same prompt and the same validation set.

**Why this baseline:** it isolates the effect of fine-tuning — since it's the same model, any difference in the metrics is explained by LoRA, not by differences in architecture or size. This is the option the assignment guide flags as "most recommended."

## 4. LoRA configuration

| Hyperparameter | Value | Why |
|---|---|---|
| `r` (rank) | 8 | Low rank, appropriate for a small training dataset (276 examples) — a high rank risks overfitting. |
| `lora_alpha` | 16 | Standard heuristic (2× the rank). |
| `target_modules` | `["q", "v"]` | T5's attention query/value projections — recommended by the original LoRA paper to keep few trainable parameters without losing adaptation capacity. |
| `learning_rate` | 5e-4 | Found empirically: 1e-3 caused collapse into repetition loops; 3e-4 underfit (outputs too short/generic). 5e-4 is the middle ground. |
| `num_train_epochs` | 12 | Same reasoning — 15 overfit, 8 underfit. |
| Trainable parameters | 344,064 (0.45% of total) | Confirmed with `print_trainable_parameters()`. |

## 5. Results

| Metric | Baseline (zero-shot) | With LoRA fine-tuning | Delta |
|---|---|---|---|
| ROUGE-1 | 0.3526 | 0.6041 | **+0.2515** |
| ROUGE-2 | 0.1992 | 0.4579 | **+0.2587** |
| ROUGE-L | 0.2863 | 0.5156 | **+0.2293** |
| ROUGE-Lsum | 0.2863 | 0.5153 | **+0.2290** |

### Conclusions

LoRA fine-tuning substantially improved every ROUGE variant (+0.23 to +0.26). More important than the number itself: the improvement also shows up qualitatively. After reformulating the task, the fine-tuned model **correctly copies** the facts given in the input (hotel, restaurant, and attraction names, months, prices) in the large majority of validation examples — something the zero-shot baseline does not do reliably.

It still fails in two specific ways, neither related to "inventing" information:
- **Category mixing in some destinations** (e.g., it lists hotels under the "for food" label in 1 of 3 qualitative examples) — a text-organization error, not fabricated content.
- **Drops the accent on the character "í"** (e.g., "Medellín" → "Medelln"). This also happens **in the baseline without fine-tuning**, so it's a characteristic of the `flan-t5-small` tokenizer with that specific character, not something introduced by training.

## 6. Evaluation Harness (M2)

> **What this harness measures and what counts as a good answer:** TravelGenie's harness scores how well the system answers a travel question **using only the facts it was actually given** — a good answer uses the provided data correctly (no invented hotels, prices, or dates), specifically addresses what was asked, and, for edge cases (false premises, out-of-domain or unsafe requests), **acknowledges the issue instead of answering as if nothing were wrong**.

### The three dimensions

1. **Automatic metric — embedding similarity.** Cosine similarity (multilingual MiniLM) between the generated response and the reference answer.
2. **LLM-as-a-judge.** `Qwen/Qwen2.5-1.5B-Instruct`, scoring 1-5 against an explicit, anchored rubric — see [`rubrica_juez.md`](./rubrica_juez.md) (versioned in this repo).
3. **Domain criterion.** % of eval-set examples with judge score ≥ 4.

### Baseline scorecard (M1 model, no RAG)

| Dimension | Score |
|---|---|
| 1 — Embedding similarity (avg.) | 0.396 |
| 2 — Judge score (avg., 1-5) | 2.4 |
| 3 — % meets domain criterion (judge ≥ 4) | 20% |

*(full per-example results in `scorecard_resultados_detallados.json`; raw scorecard in `scorecard_baseline.csv`)*

### Honest reading

The baseline scores low across all three dimensions, **and this is the expected, correct finding, not a bug.** The M1 model was fine-tuned to *write up* facts already given in its input (see the M1 design note above), not to recall facts about a city from its name alone. This eval set intentionally passes it raw questions with no facts supplied — exactly the situation it will face **without RAG**. A 20% pass rate confirms the system genuinely needs retrieval before it's useful; that gap is precisely what M3 needs to close.

**Where it fails, concretely:** the 3 worst-scoring examples (judge = 1) are the false-premise case (San Andrés "without beaches" — the model just echoes the question, no correction), the safety case (fake ticket request — no refusal at all, question echoed back), and a generic recommendation request that produced no destination whatsoever. These are the clearest, most severe failures, and the judge caught all three correctly.

**Adversarial cases specifically:** 4 of our 5 adversarial examples scored ≤ 3 — the system does not yet detect false premises, does not refuse unsafe requests, and does not flag out-of-domain questions. This is the single most important gap for M3: RAG alone (grounding on real destination facts) will **not** fix the safety/false-premise/out-of-domain failures — those need explicit handling in the prompt or a guardrail step, separate from retrieval.

**A caveat on the "passing" 20%:** even the 2 examples that cleared the judge ≥ 4 threshold aren't fully clean. One of them (hotels in Medellín) includes fabricated, domain-irrelevant content ("a partir de Estados Unidos, en Buenos Aires") that the judge missed — meaning the true failure rate is likely higher than 80%. We're reporting this rather than rounding it away, since an honest scorecard is worth more than a flattering one.

### Judge bias: detected and mitigated

**Bias found — leniency (small-judge over-scoring).** Our first judge, `Qwen2.5-0.5B-Instruct`, scored 9 of 10 examples at 5/5 — including responses that were just the question echoed back verbatim, with no correction, no refusal, no recommendation. It essentially wasn't discriminating between good and bad answers, a known failure mode of very small LLM judges.

**Mitigation — upgrade judge size, verified empirically, not just asserted.** Switching to `Qwen2.5-1.5B-Instruct` (same rubric, same prompt) produced a dramatic, measurable change:

| | 0.5B judge | 1.5B judge |
|---|---|---|
| Avg. judge score | 4.6 | 2.4 |
| % meets criterion | 90% | 20% |
| Correctly scores echoed-question / no-refusal cases as 1 | No (scored 5) | **Yes** |

**A second bias we tested but did NOT successfully mitigate — verbosity.** We also anchored the rubric with an explicit instruction not to reward length ("LENGTH-BIAS GUARD"), and measured the length↔score correlation with and without that instruction present in the prompt. It made almost no difference (Pearson r = 0.78 without the instruction vs. 0.75 with it) — the textual instruction alone did not neutralize this bias. We're reporting this as an open issue rather than claiming a fix that didn't actually work; a more robust mitigation for M3 would need length-controlled response pairs or a separate length-normalization step, not just a rubric note.

## 7. How to run this

**Fine-tuning (M1):**
1. Open `notebook_finetuning.ipynb` in Google Colab.
2. Run the first cell to clone this repository (gives you access to `data/`).
3. Enable GPU: `Runtime → Change runtime type → GPU (T4)`.
4. Run the remaining cells in order, including the "save LoRA adapter" cell — this produces `travelgenie-lora-adapter/`, already included in this repo. Seed fixed at `SEED = 42` for reproducibility.
5. The data collection/preparation pipeline (APIs, pair generation) lives in the `data_pipeline/` folder — see its own README if you need to regenerate the dataset from scratch.

**Evaluation harness (M2):**
1. In a Colab session with this repo cloned, install dependencies: `!pip install -q -U transformers peft sentence-transformers torchao`.
2. Run `%run correr_harness.py` — loads the M1 model (base + `travelgenie-lora-adapter/`), runs it through `eval_set.py`, and produces `scorecard_baseline.csv` and `scorecard_resultados_detallados.json`.

## 8. M3 — RAG + capa agéntica

Informe completo (corpus e índice, diagnóstico del RAG ingenuo, técnicas avanzadas, agente, tablas evolutiva y de RAGAS, lectura honesta): **[`reporteM3`](https://github.com/alexahurtado08/TravelGenie_AlexandraHurtado_MarianaValderrama/wiki/reporte.md)**. Resultados guardados en `scorecard_m3.csv`, `resumen_m3.csv`, `ragas_m3.csv` y `ragas_m3_detalle.csv`.

| Componente | Archivo |
|---|---|
| Corpus (70 docs) e índice (406 chunks, Chroma) | `data_pipeline/paso8_generar_corpus.py`, `data_pipeline/paso9_indexar_chroma.py` |
| RAG ingenuo `sistema_rag(pregunta)` | `rag_naive.py` |
| RAG avanzado `sistema_rag_avanzado(pregunta)` (híbrida BM25 + densa con RRF, reranking) | `rag_avanzado.py` |
| Agente (herramienta `verificar_cobertura`, JSON tolerante, tope de 3 pasos) | `agente.py` |
| Evaluación (harness de M2 sin cambios, eval set de 12 casos) | `correr_m3.py`, `eval_set_m3.py` |
| Métricas objetivas y RAGAS | `resumen_m3.py`, `ragas_m3.py` |

**Cómo ejecutarlo (Colab gratis, GPU T4, sin llaves de pago):**
```
!git clone https://github.com/alexahurtado08/TravelGenie_AlexandraHurtado_MarianaValderrama
%cd TravelGenie_AlexandraHurtado_MarianaValderrama
!pip install -q -r requirements.txt
!python data_pipeline/paso9_indexar_chroma.py   # reconstruye chroma_db/ (no se versiona)
!python correr_m3.py m2_baseline   # y también: rag_naive, rag_hibrida, rag_rerank, rag_avanzado, agente
!python resumen_m3.py
!python ragas_m3.py
```
Si Colab lo pide tras el `pip install`, reiniciar la sesión. Semilla fija en 42. No hace falta ninguna API externa: el agente, el juez de M2 y el juez de RAGAS son el mismo Qwen2.5-1.5B-Instruct local.

**Reproducibilidad verificada.** Se clonó el repositorio en una sesión nueva de Colab (Python 3.13, GPU T4, versiones de `requirements.txt`), se reconstruyó el índice con `paso9` y se volvieron a correr los 6 sistemas sobre el mismo eval set. Resultado: el índice sale igual (406 chunks, 0 truncados, máximo 111 tokens); el scorecard (similitud, juez, % cumple, respuestas sin información) es **idéntico** en los 6 sistemas; y las 72 respuestas (12 preguntas × 6 sistemas) coinciden, texto completo, con las guardadas en el repo, y el puntaje del juez también. Lo único que cambia entre corridas es la latencia, que depende de la máquina (la mediana varió entre −10 % y +0.5 % respecto a la publicada). Los modelos (`flan-t5-small`, `Qwen2.5-1.5B-Instruct`, `paraphrase-multilingual-MiniLM-L12-v2`, `mmarco-mMiniLMv2-L12-H384-v1`) se descargan de Hugging Face y todos corren en Colab gratis. `chroma_db/` no se versiona: se reconstruye con `paso9`. Los archivos de M3 dependen de `eval_set_m3.py`, que a su vez importa `eval_set.py` de M2 sin modificarlo.

## 9. Uso de IA

**Herramienta:** Claude Code (Anthropic, modelo Claude Sonnet), en una sesión de chat, durante todo el trabajo de M3.

**Para qué se usó:**


**Qué hicimos y decidimos nosotras:**


## 10. Data source attribution

This project uses third-party data under their respective licenses — see [DATASET](https://github.com/alexahurtado08/TravelGenie_AlexandraHurtado_MarianaValderrama/wiki/DATASET) for full detail. In particular, it includes © OpenStreetMap contributors data, available under the Open Database License (ODbL) — more information at [openstreetmap.org/copyright](https://www.openstreetmap.org/copyright).
