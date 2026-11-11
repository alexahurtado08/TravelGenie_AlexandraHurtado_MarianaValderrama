# Fotos de prueba del componente visual

Hacen falta **26 fotos** (la lista exacta está en `imagenes_prueba.csv`, una fila por foto).

| Grupo | Cuántas | Qué son |
|---|---|---|
| Paisajes | 18 | 3 por cada tipo: playa, selva, montaña, cafetero, ciudad_colonial, llanura. La foto 1 de cada tipo es **dev** (sirve para elegir el umbral); las 2 y 3 son **test** (no se miran hasta el final). |
| Adversariales | 8 | 2 por cada subtipo: documento (captura o página de texto), comida, interior (una sala o habitación), nieve (montañas con nieve o glaciar). El sistema debe **abstenerse**. |

## Dónde buscarlas
Wikimedia Commons (commons.wikimedia.org). Filtra por licencia libre: **CC0, dominio público, CC BY o CC BY-SA**. No uses fotos "fair use" ni de otras páginas.

Ideas de búsqueda (en inglés o español, pruebe varias):
- **playa:** "Parque Tayrona playa", "Palomino playa", "Riohacha playa", "Bahía Concha".
- **selva:** "Río Amazonas Leticia", "selva amazónica Colombia".
- **montaña:** "páramo Colombia", "Laguna de la Cocha Pasto", "páramo de Ocetá".
- **cafetero:** "Paisaje Cultural Cafetero", "Valle de Cocora", "finca cafetera Colombia".
- **ciudad_colonial:** "Popayán centro histórico", "Villa de Leyva plaza", "Tunja centro".
- **llanura:** "Llanos Orientales atardecer", "Desierto de la Tatacoa", "Desierto de La Guajira".
- **documento:** "scanned page of a book", "screenshot" (una captura de pantalla de texto).
- **comida:** "plate of food", "bandeja paisa".
- **interior:** "living room interior", "habitación de hotel".
- **nieve:** "Alps snow mountains", "glaciar".

## Reglas
1. **Sin personas identificables** (ni caras claras ni matrículas). Si sale gente muy pequeña y lejos, está bien.
2. Cada foto debe ser **de ese tipo a simple vista**, sin trampas. Las difíciles se dejan para el análisis de errores, no para este set.
3. Reduzcan cada foto a **800 px de ancho** (el repo no debe pesar) y guárdenla como `.jpg` con el nombre exacto de `imagenes_prueba.csv` dentro de `final/imagenes/`.
4. Llenen en el CSV: `fuente_url` (enlace a la página de Commons), `autor`, `licencia`. Las licencias CC BY y CC BY-SA **exigen citar al autor**: esa tabla es esa cita, y el README la enlaza.

## Cómo se usa (Colab, GPU o CPU)
Desde la raíz del repo, con las fotos ya en `final/imagenes/`:
```
!pip install -q -r requirements.txt
!python final/eval_vision.py --tune   # elige el umbral SOLO con las fotos dev
!python final/eval_vision.py --test   # mide con las fotos test, umbral ya fijo
```
No cambien el umbral a mano después de ver los resultados de test.

## Fotos difíciles (8 más, `dificil_1.jpg` a `dificil_8.jpg`)

Sirven para mostrar **dónde falla** el componente. No se usan para elegir el umbral ni se corrigen
si el sistema falla: el fallo es el resultado. Mismas reglas de licencia y sin personas identificables.

| Archivo | Qué buscar en Commons | Se espera |
|---|---|---|
| `dificil_1` | Playa de otro país (Tailandia, Filipinas...) | CLIP no sabe el país: la tomará por playa |
| `dificil_2` | Prado alpino verde, sin nieve | La tomará por montaña |
| `dificil_3` | Retrato pintado de una persona (obra de dominio público, p. ej. un cuadro antiguo) | Debe abstenerse |
| `dificil_4` | Flor o planta de cerca | Debe abstenerse |
| `dificil_5` | Mapa de Colombia o de una región | Debe abstenerse |
| `dificil_6` | Un animal (gato, perro, ave) | Debe abstenerse |
| `dificil_7` | Playa de noche o con luz muy baja | Es playa, pero difícil |
| `dificil_8` | Pueblo colonial con montañas al fondo | Vale ciudad_colonial o montaña |

Cómo se corre (después de `--tune` y `--test`, con el umbral ya fijado):
```
!python final/eval_vision.py --dificiles
```
