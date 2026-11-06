"""
Componente visual de TravelGenie: una foto de un paisaje -> destinos de ese tipo.

Qué hace, en simple:
1) Un modelo (CLIP, zero-shot) compara la foto con descripciones de seis tipos
   de paisaje (playa, selva, montaña, cafetero, ciudad colonial, llanura) y de
   cosas que NO son un paisaje (documento, comida, interior, nieve).
2) Válvula visual: si lo más parecido NO es un paisaje, o el modelo no está lo
   bastante seguro (umbral), el sistema dice que no sabe en vez de adivinar.
3) Si hay paisaje, se buscan los destinos de ese tipo (perfil_paisaje.py) y se
   muestran sus datos reales de clima y vuelos tomados del corpus (sin
   generar texto, así no se inventa nada). Se ordenan del vuelo más barato.

La imagen se procesa en memoria: no se guarda, y los registros solo llevan el
tipo, la confianza y la latencia.

Uso (desde la raíz del repo):  python final/vision.py ruta/foto.jpg
"""

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perfil_paisaje import TIPOS, destinos_de

RUTA_CORPUS = "data/08_corpus.json"
RUTA_UMBRAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "umbral_vision.json")
MODELO_CLIP = "openai/clip-vit-base-patch32"
UMBRAL_POR_DEFECTO = 0.50   # se reemplaza por el calculado con el set dev (eval_vision.py --tune)
MAX_DESTINOS = 3

# Varias descripciones por clase (en inglés: CLIP funciona mejor así); se promedian.
DESCRIPCIONES = {
    "playa": ["a photo of a tropical beach with palm trees and sea",
              "a photo of a Caribbean coast with turquoise water and sand"],
    "selva": ["a photo of the Amazon rainforest and a wide river",
              "a photo of dense tropical jungle"],
    "montaña": ["a photo of high Andean mountains and a cold highland landscape",
                "a photo of green mountains covered with clouds and paramo vegetation"],
    "cafetero": ["a photo of a coffee plantation landscape with green hills",
                 "a photo of a coffee farm in the mountains"],
    "ciudad_colonial": ["a photo of a colonial town with white houses and cobblestone streets",
                        "a photo of an old Spanish colonial church and plaza"],
    "llanura": ["a photo of flat plains and savanna with cattle",
                "a photo of a desert landscape with dry vegetation"],
}
# Lo que NO es un paisaje que podamos ubicar: si gana una de estas, el sistema se abstiene.
NO_PAISAJE = {
    "documento": ["a screenshot of a document or a page of text"],
    "comida": ["a photo of food on a plate"],
    "interior": ["a photo of an indoor room with furniture"],
    "persona": ["a portrait photo of a person"],
    "nieve": ["a photo of snowy mountains and glaciers"],
}
CLASES = list(DESCRIPCIONES) + list(NO_PAISAJE)


def cargar_umbral():
    """Umbral guardado por eval_vision.py --tune (set dev); si no existe, el de por defecto."""
    try:
        with open(RUTA_UMBRAL, encoding="utf-8") as f:
            return float(json.load(f)["umbral"])
    except (OSError, KeyError, ValueError):
        return UMBRAL_POR_DEFECTO


class ClasificadorPaisaje:
    def __init__(self, umbral=None, cargar_modelo=True):
        self.umbral = cargar_umbral() if umbral is None else umbral
        self._modelo = None
        if cargar_modelo:
            self._cargar()

    def _cargar(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        torch.manual_seed(42)
        self._torch = torch
        self._procesador = CLIPProcessor.from_pretrained(MODELO_CLIP)
        self._modelo = CLIPModel.from_pretrained(MODELO_CLIP).eval()
        # embeddings de texto: promedio de las descripciones de cada clase
        vecs = []
        with torch.no_grad():
            for clase in CLASES:
                textos = DESCRIPCIONES.get(clase) or NO_PAISAJE[clase]
                ent = self._procesador(text=textos, return_tensors="pt", padding=True)
                e = self._modelo.get_text_features(**ent)
                e = e / e.norm(dim=-1, keepdim=True)
                m = e.mean(dim=0)
                vecs.append(m / m.norm())
        self._texto = torch.stack(vecs)

    def puntajes(self, imagen):
        """{clase: probabilidad} (softmax sobre todas las clases). `imagen` es un PIL.Image."""
        torch = self._torch
        with torch.no_grad():
            ent = self._procesador(images=imagen.convert("RGB"), return_tensors="pt")
            e = self._modelo.get_image_features(**ent)
            e = e / e.norm(dim=-1, keepdim=True)
            logits = self._modelo.logit_scale.exp() * (e @ self._texto.T)
            p = logits.softmax(dim=-1)[0]
        return {c: float(p[i]) for i, c in enumerate(CLASES)}

    def clasificar(self, imagen):
        """Decisión con válvula. Devuelve dict: decision ('ok'|'abstiene'), tipo, prob, motivo."""
        p = self.puntajes(imagen)
        return decidir(p, self.umbral)


def decidir(p, umbral):
    """Regla de la válvula visual, separada para poder probarla sin el modelo."""
    top = max(p, key=p.get)
    if top in NO_PAISAJE:
        return {"decision": "abstiene", "tipo": None, "prob": p[top],
                "motivo": f"la imagen se parece más a '{top}' que a un paisaje"}
    if p[top] < umbral:
        return {"decision": "abstiene", "tipo": None, "prob": p[top],
                "motivo": f"confianza {p[top]:.2f} menor que el umbral {umbral:.2f}"}
    return {"decision": "ok", "tipo": top, "prob": p[top], "motivo": "paisaje reconocido"}


# ---------------------------------------------------------------- datos del corpus
_corpus = {}


def _hechos():
    """{destino: {'clima': texto, 'vuelos': texto, 'precio': int}} leído del corpus."""
    if not _corpus:
        with open(RUTA_CORPUS, encoding="utf-8") as f:
            for d in json.load(f):
                if d["categoria"] in ("clima", "vuelos"):
                    _corpus.setdefault(d["destino"], {})[d["categoria"]] = d["texto"]
        for d, h in _corpus.items():
            m = re.search(r"\$([\d,]+)", h.get("vuelos", ""))
            h["precio"] = int(m.group(1).replace(",", "")) if m else 10**12
    return _corpus


def responder_con_imagen(imagen, clasificador):
    """Respuesta completa para una foto. Devuelve dict con texto y datos para el registro."""
    t0 = time.perf_counter()
    r = clasificador.clasificar(imagen)
    if r["decision"] == "abstiene":
        texto = ("No puedo ubicar un paisaje en esa imagen, así que prefiero no adivinar "
                 f"({r['motivo']}). Prueba con una foto de un lugar al aire libre: "
                 "una playa, una montaña, la selva, un pueblo colonial, un cafetal o una llanura.")
        return {"texto": texto, "valvula": True, "componentes": ["vision"], "tipo": None,
                "prob": r["prob"], "destinos": [], "latencia_s": time.perf_counter() - t0}
    hechos = _hechos()
    destinos = sorted(destinos_de(r["tipo"]), key=lambda d: hechos[d]["precio"])[:MAX_DESTINOS]
    lineas = [f"Tu foto parece un paisaje de tipo **{r['tipo']}** (confianza {r['prob']:.0%}; "
              "la clasificación es aproximada). Destinos de ese tipo en nuestra base, "
              "del vuelo más barato al más caro:", ""]
    for d in destinos:
        lineas.append(f"- **{d.title()}**: {hechos[d]['clima']} {hechos[d]['vuelos']}")
    return {"texto": "\n".join(lineas), "valvula": False, "componentes": ["vision", "corpus"],
            "tipo": r["tipo"], "prob": r["prob"], "destinos": destinos,
            "latencia_s": time.perf_counter() - t0}


if __name__ == "__main__":
    from PIL import Image
    clf = ClasificadorPaisaje()
    print(responder_con_imagen(Image.open(sys.argv[1]), clf)["texto"])
