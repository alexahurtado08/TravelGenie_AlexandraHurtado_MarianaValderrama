## Análisis de errores

Fuente: `final/resultados_eficiencia.json` (40 casos del agente, fp32/fp16 idénticos), `final/resultados_vision_dificiles.json`
y `final/resultados_vision.csv`. Las etiquetas de los 40 casos del agente se escribieron antes de correr y no se cambiaron después.
Una nota previa: "fallo de vía" significa que la vía fue distinta a la esperada, no siempre que el usuario recibió una mala
respuesta. Por eso la tabla separa el **impacto para el usuario**.

| # | Caso | Qué pasó | Causa (lo observado / hipótesis) | Impacto para el usuario | Mitigación posible |
|---|---|---|---|---|---|
| 1 | "Recomiéndame un plan de montaña barato." | El agente llamó a `verificar_cobertura("Montaña")`, la herramienta dijo que no hay datos y la respuesta fue "no incluye Montaña". | **Observado:** tomó un tipo de paisaje por una ciudad. La barrera "el destino debe estar escrito en la pregunta" no lo frena, porque "montaña" sí está escrita. | **Rechazo falso** de una pregunta válida. Es el fallo más dañino. | Aceptar solo destinos que sean municipios colombianos (lista DIVIPOLA ya presente en `data/`). |
| 2 | "¿Qué destinos de selva recomiendan?" | Igual que el 1, con el argumento `selva`. | Igual que el 1. | Rechazo falso. | Igual que el 1. |
| 3 | Cinco preguntas con lugares extranjeros: Perú, Francia, París, Lima, Miami | El agente llamó a la herramienta con ese lugar y respondió "no tengo información, mi base no incluye Francia". La etiqueta esperaba pasar al RAG. | **Hipótesis:** el modelo de 1.5B no generaliza la regla del prompt ("un país no es una ciudad") a partir de un solo ejemplo (Japón). | **Casi ninguno:** el usuario recibe una abstención honesta. Es un desajuste con la etiqueta, aunque el mensaje suena raro ("no incluye Francia"). Explica el 38% en fuera de tema. | Más ejemplos en el prompt, o la validación DIVIPOLA del caso 1. |
| 4 | "Busco un destino de playa económico para diciembre." (fp32) | El agente inventó "Playa del Carmen". | **Observado:** la barrera lo bloqueó porque no está en la pregunta, y el agente siguió al RAG. | **Ninguno:** es un **acierto de la defensa**. | — |
| 5 | Mismo caso, con nf4 | Llamó con el argumento `playa`, que sí está en la pregunta, y la respuesta fue `sin_cobertura`. | **Observado:** la cuantización a 4 bits degrada el argumento y esta vez la barrera no lo detiene. | Rechazo falso. Es el motivo por el que nf4 pierde un caso de vía. | No usar nf4 (ya descartado por la tolerancia). |
| 6 | "mejor época para visitar tunja" (minúsculas) | No llamó a la herramienta. | **Hipótesis:** no reconoce el nombre sin mayúscula inicial. | Ninguno aquí (Tunja está cubierta y va al RAG). **Riesgo:** con una ciudad no cubierta escrita en minúsculas no se activaría la válvula. | Normalizar el texto antes de pasarlo al agente. |

**Visión.** Dos fallos en las 8 fotos difíciles (sin ajustar nada tras verlos):
7. *Playa de Tailandia → playa, p=0.983.* CLIP reconoce el tipo de paisaje, no el país. La válvula no puede arreglarlo con ningún umbral. Mitigación: la respuesta aclara que se recomienda por tipo de paisaje y no verifica el lugar.
8. *Prado de Bulgaria → montaña, p=0.619.* Pasó el umbral (0.60) por 0.019. En cambio, la planta (p=0.493) y el mapa (p=0.542) se rechazaron por quedar bajo el umbral, no porque CLIP los identificara como no paisaje: la zona de 0.54 a 0.62 es frágil.

**Lo que no se arregló.** No se modificó el agente ni el umbral después de ver estos resultados: los casos 1 a 3 son hallazgos para el reporte,
no correcciones pendientes. Arreglarlos exigiría volver a evaluar todo el conjunto, y la validación DIVIPOLA tiene su propio costo
(una ciudad inventada como "Villa Esperanza del Sur" dejaría de ser `sin_cobertura`).
