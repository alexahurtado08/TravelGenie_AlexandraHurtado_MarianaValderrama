"""
Verificamos si la vara mide bien probando el juez y la verificación con una respuesta BUENA y
una POBRE por caso (casos_validacion.py), antes de usarlos para evaluar sistemas.

Para cada caso se espera:
  - la verificación propia: buena = cumple, pobre = no cumple;
  - el juez nuevo (con respuesta de referencia y criterio): buena >= 4, pobre <= 2.
Además se prueba el juez "sin referencia" (como era en la v1: solo pregunta y
respuesta) sobre los mismos pares, para ver cuánto mejoró.

Uso (Colab, GPU):  python validar_juez.py     ->  validacion_juez.csv
"""

import csv
import re

import torch

from eval_set_m3 import eval_set_m3 as eval_set
from casos_validacion import CASOS_VALIDACION
from verificacion import verificar_criterio
from harness import juez_puntuar, juez_modelo, juez_tokenizer, device, RUBRICA_TEXTO

PROMPT_SIN_REFERENCIA = """Eres un evaluador estricto. Usa esta rúbrica para calificar una respuesta del 1 al 5:

{rubrica}

---
PREGUNTA DEL USUARIO: {pregunta}

RESPUESTA A EVALUAR: {respuesta}

Califica la respuesta según la rúbrica. Responde ÚNICAMENTE en este formato exacto,
sin explicación adicional:
PUNTAJE: <número del 1 al 5>
"""


def juez_sin_referencia(pregunta, respuesta):
    """El juez como era en la v1: solo ve la pregunta y la respuesta."""
    prompt = PROMPT_SIN_REFERENCIA.format(rubrica=RUBRICA_TEXTO, pregunta=pregunta, respuesta=respuesta)
    txt = juez_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    ids = juez_tokenizer(txt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = juez_modelo.generate(**ids, max_new_tokens=20, do_sample=False)
    salida = juez_tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    m = re.search(r"PUNTAJE:\s*([1-5])", salida) or re.search(r"\b([1-5])\b", salida)
    return int(m.group(1)) if m else None


def evaluar(caso, puntuar_con_ref, puntuar_sin_ref):
    par = CASOS_VALIDACION[caso["input"]]
    fila = {"caso": caso["input"]}
    for tipo in ("buena", "pobre"):
        resp = par[tipo]
        fila[f"verif_{tipo}"] = verificar_criterio(resp, caso)[0]
        fila[f"juez_nuevo_{tipo}"] = puntuar_con_ref(caso, resp)
        fila[f"juez_sinref_{tipo}"] = puntuar_sin_ref(caso, resp)
    return fila


def resumen(filas):
    def ok_juez(f, k):
        b, p = f[f"{k}_buena"], f[f"{k}_pobre"]
        return b is not None and p is not None and b >= 4 and p <= 2
    return {
        "casos": len(filas),
        "verificacion_correcta": sum(f["verif_buena"] and not f["verif_pobre"] for f in filas),
        "juez_nuevo_correcto": sum(ok_juez(f, "juez_nuevo") for f in filas),
        "juez_sin_referencia_correcto": sum(ok_juez(f, "juez_sinref") for f in filas),
        "juez_nuevo_separa": sum(f["juez_nuevo_buena"] is not None and f["juez_nuevo_pobre"] is not None
                                 and f["juez_nuevo_buena"] > f["juez_nuevo_pobre"] for f in filas),
        "juez_sin_referencia_separa": sum(f["juez_sinref_buena"] is not None and f["juez_sinref_pobre"] is not None
                                          and f["juez_sinref_buena"] > f["juez_sinref_pobre"] for f in filas),
    }


def main():
    filas = [evaluar(c,
                     lambda cs, r: juez_puntuar(cs["input"], r, cs["esperado"], cs["criterio"]),
                     lambda cs, r: juez_sin_referencia(cs["input"], r))
             for c in eval_set]

    print(f"{'caso':44} verif(b/p)  juez nuevo(b/p)  sin ref.(b/p)")
    for f in filas:
        print(f"{f['caso'][:43]:44} {str(f['verif_buena'])[0]}/{str(f['verif_pobre'])[0]}        "
              f"{f['juez_nuevo_buena']}/{f['juez_nuevo_pobre']}            {f['juez_sinref_buena']}/{f['juez_sinref_pobre']}")

    r = resumen(filas)
    print("\n=== RESUMEN (de 12 casos) ===")
    print(f"  Verificación propia: buena cumple y pobre no .......... {r['verificacion_correcta']}/{r['casos']}")
    print(f"  Juez NUEVO: buena >= 4 y pobre <= 2 ................... {r['juez_nuevo_correcto']}/{r['casos']}")
    print(f"  Juez SIN referencia (v1): buena >= 4 y pobre <= 2 ..... {r['juez_sin_referencia_correcto']}/{r['casos']}")
    print(f"  Juez NUEVO le da más a la buena que a la pobre ........ {r['juez_nuevo_separa']}/{r['casos']}")
    print(f"  Juez SIN referencia le da más a la buena que a la pobre {r['juez_sin_referencia_separa']}/{r['casos']}")

    with open("validacion_juez.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print("\nGuardado: validacion_juez.csv")


if __name__ == "__main__":
    main()
