"""
Evaluación del componente visual con las fotos de final/imagenes_prueba.csv.

Dos pasos, en este orden (así el umbral no se ajusta con las fotos de prueba):
  python final/eval_vision.py --tune   usa SOLO las fotos 'dev': elige el umbral de la
                                       válvula y lo guarda en final/umbral_vision.json
  python final/eval_vision.py --test   usa SOLO las fotos 'test' con ese umbral ya fijo y
                                       escribe final/resultados_vision.csv y
                                       final/resultados_vision_detalle.json

Mide: acierto del tipo de paisaje, abstención correcta con imágenes que no son un
paisaje, falsos rechazos, la ablación sin válvula, la prueba de invarianza
(voltear, reducir, comprimir, cambiar brillo, recortar: la decisión no debería
cambiar) y la latencia p50 y p95 del clasificador.

Correr desde la raíz del repo, en Colab con las fotos en final/imagenes/.
"""

import csv
import io
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from vision import ClasificadorPaisaje, decidir, RUTA_UMBRAL  # noqa: E402
from perfil_paisaje import TIPOS  # noqa: E402

CSV_FOTOS = os.path.join(AQUI, "imagenes_prueba.csv")
GRID = [round(float(x), 2) for x in np.arange(0.30, 0.91, 0.05)]


def _arreglar(texto):
    """Repara 'montaÃ±a' (UTF-8 leído como cp1252, típico de Excel) y normaliza tildes."""
    import unicodedata
    texto = (texto or "").strip()
    if "Ã" in texto:
        try:
            texto = texto.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return unicodedata.normalize("NFC", texto)


def _leer_csv():
    """Lee el CSV en UTF-8 (con o sin BOM); si Excel lo guardó en cp1252, lo lee así."""
    for cod in ("utf-8-sig", "cp1252"):
        try:
            with open(CSV_FOTOS, encoding=cod, newline="") as f:
                return [{k: _arreglar(v) for k, v in r.items()} for r in csv.DictReader(f)]
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"No pude leer {CSV_FOTOS}: guárdalo como CSV UTF-8.")


def leer_filas(split):
    filas = [r for r in _leer_csv() if r["split"] == split and r["archivo"].strip()]
    for r in filas:
        r["ruta"] = os.path.join(AQUI, "imagenes", r["archivo"])
    return filas


def transformaciones(img):
    """Cambios que NO deberían cambiar el tipo de paisaje."""
    from PIL import Image, ImageEnhance
    w, h = img.size
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=30)
    buf.seek(0)
    return {
        "voltear": img.transpose(Image.FLIP_LEFT_RIGHT),
        "reducir_50": img.resize((max(1, w // 2), max(1, h // 2))),
        "jpeg_q30": Image.open(buf),
        "brillo_0.8": ImageEnhance.Brightness(img.convert("RGB")).enhance(0.8),
        "brillo_1.2": ImageEnhance.Brightness(img.convert("RGB")).enhance(1.2),
        "recorte_90": img.crop((int(w * .05), int(h * .05), int(w * .95), int(h * .95))),
    }


def es_correcta(esperada, dec):
    """Paisaje: debe decidir 'ok' con el tipo correcto. No-paisaje: debe abstenerse."""
    if esperada == "ninguna":
        return dec["decision"] == "abstiene"
    return dec["decision"] == "ok" and dec["tipo"] == esperada


def tune(clf, filas):
    """Elige el umbral con las fotos dev.

    Si varios umbrales empatan en el mejor acierto (pasa cuando las fotos dev son fáciles),
    se toma el del MEDIO de ese tramo: ni el más permisivo ni el más estricto.
    También devuelve la probabilidad de cada foto dev para poder revisarla.
    """
    probs = [clf.puntajes(_abrir(r["ruta"])) for r in filas]
    tabla = {}
    for u in GRID:
        ok = sum(es_correcta(r["clase_esperada"], decidir(p, u)) for r, p in zip(filas, probs))
        tabla[u] = ok / len(filas)
    mejor_score = max(tabla.values())
    empatados = sorted(u for u, s in tabla.items() if s == mejor_score)
    elegido = empatados[len(empatados) // 2]
    detalle = []
    for r, p in zip(filas, probs):
        top = max(p, key=p.get)
        detalle.append({"archivo": r["archivo"], "esperada": r["clase_esperada"],
                        "top": top, "prob_top": round(p[top], 3)})
    return {"umbral": elegido, "n_dev": len(filas), "score_dev": mejor_score,
            "tramo_empatado": [empatados[0], empatados[-1]], "grid": tabla, "dev": detalle}


def _abrir(ruta):
    from PIL import Image
    return Image.open(ruta)


def evaluar(clf, filas, abrir=_abrir):
    import time
    detalle, lat = [], []
    pares_inv, iguales = 0, 0
    for r in filas:
        img = abrir(r["ruta"])
        t0 = time.perf_counter()
        p = clf.puntajes(img)
        lat.append(time.perf_counter() - t0)
        dec = decidir(p, clf.umbral)
        top_sin_valvula = max((c for c in p if c in TIPOS), key=p.get)
        fila = {"archivo": r["archivo"], "esperada": r["clase_esperada"], "subtipo": r.get("subtipo", ""),
                "decision": dec["decision"], "tipo": dec["tipo"], "prob": round(dec["prob"], 3),
                "correcta": es_correcta(r["clase_esperada"], dec),
                "top_sin_valvula": top_sin_valvula,
                "top3": sorted(p, key=p.get, reverse=True)[:3]}
        inv = {}
        for nombre, t in transformaciones(img).items():
            d2 = decidir(clf.puntajes(t), clf.umbral)
            igual = (d2["decision"], d2["tipo"]) == (dec["decision"], dec["tipo"])
            inv[nombre] = igual
            pares_inv += 1
            iguales += igual
        fila["invarianza"] = inv
        detalle.append(fila)

    paisajes = [d for d in detalle if d["esperada"] != "ninguna"]
    adversas = [d for d in detalle if d["esperada"] == "ninguna"]
    m = {
        "n_test": len(detalle), "n_paisajes": len(paisajes), "n_adversariales": len(adversas),
        "umbral": clf.umbral,
        "acierto_tipo": sum(d["correcta"] for d in paisajes) / max(1, len(paisajes)),
        "falsos_rechazos": sum(d["decision"] == "abstiene" for d in paisajes) / max(1, len(paisajes)),
        "abstencion_correcta_adversariales": sum(d["correcta"] for d in adversas) / max(1, len(adversas)),
        "ablacion_sin_valvula_acierto_tipo": sum(d["top_sin_valvula"] == d["esperada"] for d in paisajes) / max(1, len(paisajes)),
        "ablacion_sin_valvula_falsas_aceptaciones": 1.0 if adversas else 0.0,  # sin válvula siempre responde
        "invarianza_decision_igual": iguales / max(1, pares_inv),
        "latencia_p50_s": float(np.percentile(lat, 50)),
        "latencia_p95_s": float(np.percentile(lat, 95)),
    }
    return m, detalle


def main():
    modo = sys.argv[1] if len(sys.argv) > 1 else ""
    if modo == "--tune":
        clf = ClasificadorPaisaje(umbral=0.5)
        res = tune(clf, leer_filas("dev"))
        with open(RUTA_UMBRAL, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"Umbral elegido con {res['n_dev']} fotos dev: {res['umbral']} (acierto dev {res['score_dev']:.2f})")
        print(f"Tramo de umbrales empatados: {res['tramo_empatado'][0]} a {res['tramo_empatado'][1]} (se toma el del medio)")
        for u, s in res["grid"].items():
            print(f"  umbral {u}: {s:.2f}")
        print("\nFotos dev (esperada -> lo más parecido, probabilidad):")
        for d in res["dev"]:
            print(f"  {d['archivo']:24} {d['esperada']:16} -> {d['top']:16} {d['prob_top']:.3f}")
    elif modo == "--test":
        clf = ClasificadorPaisaje()
        m, det = evaluar(clf, leer_filas("test"))
        with open(os.path.join(AQUI, "resultados_vision.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metrica", "valor"])
            for k, v in m.items():
                w.writerow([k, round(v, 4) if isinstance(v, float) else v])
        with open(os.path.join(AQUI, "resultados_vision_detalle.json"), "w", encoding="utf-8") as f:
            json.dump(det, f, indent=2, ensure_ascii=False)
        for k, v in m.items():
            print(f"  {k}: {round(v, 3) if isinstance(v, float) else v}")
        print("\nFotos que fallan:")
        for d in det:
            if not d["correcta"]:
                print(f"  {d['archivo']}: esperada={d['esperada']} -> {d['decision']}/{d['tipo']} (p={d['prob']}), top3={d['top3']}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()