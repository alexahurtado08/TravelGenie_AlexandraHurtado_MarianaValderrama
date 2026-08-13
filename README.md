# TravelGenie

> **TravelGenie recommends where to travel in Colombia and which month works best, combining real climate data, flight prices, and tourist attractions — so anyone can plan a trip without having to research everything on their own.**

*team members*: Alexandra Hurtado, Mariana Valderrama

---

## 1. Base model and chosen family

**Model:** flan-t5-small (encoder-decoder)

**Why this family:** the project's core task is *data-to-text* — converting structured data (climate, flight price, hotels/restaurants/attractions) into a natural-language recommendation. This is exactly what T5's pretraining objective does: it maps an input sequence to an output sequence of arbitrary length. An *encoder-only* model (like BERT) cannot generate free-form text; a *decoder-only* model (like GPT) could also do it, but the encoder-decoder architecture better separates understanding the query (encoder) from generating the response (decoder), which is more natural for a task with a short, structured input and a long output.

## 2. Domain dataset

We have **384** input→output pairs (307 train / 77 validation), template-generated from real climate, flight, and tourist attraction data across 16 Colombian destinations. For more detailed information, see **[DATASET](https://github.com/alexahurtado08/TravelGenie_AlexandraHurtado_MarianaValderrama/wiki/DATASET)**.

## 3. Baseline and comparison

**Baseline:** the same `flan-t5-small`, **without fine-tuning**, using the same prompt and the same validation set.

**Why this baseline:** it isolates the effect of fine-tuning — since it's the same model, any difference in the metrics is explained by LoRA, not by differences in architecture or size. This is the option the assignment guide flags as "most recommended."

## 4. LoRA configuration

| Hyperparameter | Value | Why |
|---|---|---|
| `r` (rank) | 8 | Low rank, appropriate for a small training dataset (307 examples) — a high rank risks overfitting. |
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

## 7. How to run this

1. Open `notebook_finetuning_lora.py` in Google Colab (or copy it cell by cell into a new notebook).
2. Enable GPU: `Runtime → Change runtime type → GPU (T4)`.
3. Run cells 1 through 5 in order. Seed fixed at `SEED = 42` for reproducibility.
4. The data collection/preparation pipeline (scraping/APIs, pair generation) lives in the `data_pipeline/` folder — see its own README if you need to regenerate the dataset from scratch.

## 8. Data source attribution

This project uses third-party data under their respective licenses — see [DATASET](https://github.com/alexahurtado08/TravelGenie_AlexandraHurtado_MarianaValderrama/wiki/DATASET) for full detail. In particular, it includes © OpenStreetMap contributors data, available under the Open Database License (ODbL) — more information at [openstreetmap.org/copyright](https://www.openstreetmap.org/copyright).
