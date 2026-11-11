"""
TravelGenie: aplicación final (texto o foto) para Hugging Face Spaces.

Una sola entrada con dos pestañas:
  - Pregunta de viaje (texto): agente (Qwen2.5-1.5B + verificar_cobertura) -> RAG avanzado.
  - Foto de un paisaje: CLIP zero-shot -> destinos de ese tipo con datos reales.
Las dos tienen válvula: si no hay datos o la foto no es un paisaje, dicen que no saben.

Además de la interfaz expone GET /health (200 listo, 503 cargando) y escribe un
registro JSON por solicitud sin datos personales (ver observabilidad.py).

Uso local, desde la raíz del repo:   python final/app.py
Variables de entorno (todas opcionales):
  TRAVELGENIE_TEXTO=rag      usa solo el RAG avanzado (sin el agente Qwen) si falta memoria
  TRAVELGENIE_STUB=1         modelos de mentira, para probar la app sin descargar nada
  WANDB_API_KEY              activa las métricas en W&B
  PORT                       puerto (7860 por defecto, el de Spaces)
"""

import os
import subprocess
import sys
import threading
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
os.chdir(RAIZ)                      # las rutas de data/ y chroma_db/ son relativas a la raíz
for p in (AQUI, RAIZ):
    if p not in sys.path:
        sys.path.insert(0, p)

import observabilidad as obs

VERSION = "final-1.0"
MAX_CHARS_PREGUNTA = 500
MODO_TEXTO = os.environ.get("TRAVELGENIE_TEXTO", "agente")
STUB = os.environ.get("TRAVELGENIE_STUB") == "1"

ESTADO = {"texto": False, "vision": False, "error": None}
_motores = {}


# ----------------------------------------------------------------- carga de modelos
def _asegurar_indice():
    """chroma_db/ no se sube al repo: si falta, se reconstruye (≈1 min en CPU)."""
    if not os.path.isdir("chroma_db"):
        obs.registrar(evento="arranque", estado="indexando")
        subprocess.run([sys.executable, "data_pipeline/paso9_indexar_chroma.py"], check=True)


def _cargar_texto():
    if STUB:
        _motores["texto"] = lambda q: ("[stub] respuesta de prueba", {"via": "stub", "n_llamadas": 0, "tope": False})
        return
    _asegurar_indice()
    if MODO_TEXTO == "rag":
        import rag_avanzado
        rag_avanzado._cargar()
        _motores["texto"] = lambda q: (rag_avanzado.sistema_rag_avanzado(q),
                                       {"via": "rag_avanzado", "n_llamadas": 0, "tope": False})
    else:
        import agente
        agente._cargar()

        def _agente(q):
            r = agente.sistema_agente(q)
            u = agente.ULTIMO
            return r, {"via": u.get("via"), "n_llamadas": len(u.get("llamadas", [])),
                       "tope": bool(u.get("tope_de_pasos"))}
        _motores["texto"] = _agente


def _cargar_vision():
    import vision
    if STUB:
        class Falso:
            umbral = vision.cargar_umbral()
            def clasificar(self, img):
                return vision.decidir({"playa": 0.9, "selva": 0.1}, self.umbral)
        clf = Falso()
    else:
        clf = vision.ClasificadorPaisaje()
    _motores["vision"] = lambda img: vision.responder_con_imagen(img, clf)
    _motores["umbral"] = clf.umbral


def calentar():
    """Carga los dos componentes en segundo plano para que /health responda desde el inicio."""
    t0 = time.perf_counter()
    try:
        _cargar_vision()
        ESTADO["vision"] = True
        _cargar_texto()
        ESTADO["texto"] = True
        obs.registrar(evento="arranque", estado="listo", latencia_s=time.perf_counter() - t0)
    except Exception as e:   # noqa: BLE001
        ESTADO["error"] = type(e).__name__
        obs.registrar(evento="arranque", estado="error", error=type(e).__name__)


# ----------------------------------------------------------------- manejadores
def responder_texto(pregunta):
    rid, t0 = obs.nuevo_id(), time.perf_counter()
    pregunta = (pregunta or "").strip()
    if not pregunta:
        return "Escribe una pregunta sobre un destino colombiano (clima, vuelos, hoteles, restaurantes)."
    if not ESTADO["texto"]:
        return "El sistema todavía está cargando los modelos. Intenta de nuevo en un minuto."
    truncada = len(pregunta) > MAX_CHARS_PREGUNTA
    try:
        texto, meta = _motores["texto"](pregunta[:MAX_CHARS_PREGUNTA])
        obs.registrar(evento="solicitud", request_id=rid, entrada="texto", via=meta["via"],
                      n_llamadas_herramienta=meta["n_llamadas"], tope_de_pasos=meta["tope"],
                      longitud_pregunta=len(pregunta), pregunta_truncada=truncada,
                      latencia_s=time.perf_counter() - t0)
        return texto
    except Exception as e:   # noqa: BLE001
        obs.registrar(evento="solicitud", request_id=rid, entrada="texto",
                      error=type(e).__name__, latencia_s=time.perf_counter() - t0)
        return "Ocurrió un error procesando tu pregunta. Intenta de nuevo."


def responder_imagen(imagen):
    rid, t0 = obs.nuevo_id(), time.perf_counter()
    if imagen is None:
        return "Sube una foto de un paisaje."
    if not ESTADO["vision"]:
        return "El sistema todavía está cargando los modelos. Intenta de nuevo en un minuto."
    try:
        imagen = imagen.convert("RGB")
        imagen.thumbnail((1024, 1024))        # solo en memoria; la foto no se guarda
        r = _motores["vision"](imagen)
        obs.registrar(evento="solicitud", request_id=rid, entrada="imagen",
                      valvula=r["valvula"], tipo_paisaje=r["tipo"], prob=r["prob"],
                      n_destinos=len(r["destinos"]), latencia_s=time.perf_counter() - t0)
        return r["texto"]
    except Exception as e:   # noqa: BLE001
        obs.registrar(evento="solicitud", request_id=rid, entrada="imagen",
                      error=type(e).__name__, latencia_s=time.perf_counter() - t0)
        return "Ocurrió un error procesando la imagen. Prueba con otra foto (JPG o PNG)."


# ----------------------------------------------------------------- interfaz
def construir_interfaz():
    import gradio as gr
    with gr.Blocks(title="TravelGenie") as demo:
        gr.Markdown("# TravelGenie\nRecomendador de viajes por Colombia: pregunta por un destino "
                    "o sube la foto de un paisaje. Cubre 14 destinos con datos de clima, vuelos, "
                    "hoteles, restaurantes y atractivos. Si no tiene datos, te lo dice.")
        with gr.Tab("Pregunta"):
            q = gr.Textbox(label="Tu pregunta", lines=2, max_length=MAX_CHARS_PREGUNTA,
                           placeholder="¿Cuál es el mejor mes para viajar a Cartagena?")
            gr.Examples(["¿Qué restaurantes hay en Medellín?", "¿Cuándo es mejor viajar a Santa Marta?",
                         "¿Qué hoteles recomiendas en Quibdó?"], inputs=q)
            btn = gr.Button("Preguntar", variant="primary")
            salida = gr.Markdown()
            btn.click(responder_texto, q, salida, api_name="texto")
        with gr.Tab("Foto de un paisaje"):
            img = gr.Image(type="pil", label="Foto (playa, montaña, selva, cafetal, pueblo colonial, llanura)")
            btn2 = gr.Button("Buscar destinos", variant="primary")
            salida2 = gr.Markdown()
            btn2.click(responder_imagen, img, salida2, api_name="imagen")
        gr.Markdown("*No se guardan ni tus preguntas ni tus fotos. Los registros solo llevan "
                    "tipo de solicitud, latencia y resultado.*")
    demo.queue(default_concurrency_limit=1)
    return demo


def crear_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    import gradio as gr
    app = FastAPI()

    @app.get("/health")
    def health():
        listo = ESTADO["texto"] and ESTADO["vision"]
        cuerpo = {"status": "ok" if listo else ("error" if ESTADO["error"] else "loading"),
                  "texto": ESTADO["texto"], "vision": ESTADO["vision"],
                  "modo_texto": MODO_TEXTO, "version": VERSION,
                  "umbral_vision": _motores.get("umbral")}
        return JSONResponse(cuerpo, status_code=200 if listo else 503)

    return gr.mount_gradio_app(app, construir_interfaz(), path="/")


if __name__ == "__main__":
    import uvicorn
    threading.Thread(target=calentar, daemon=True).start()
    uvicorn.run(crear_app(), host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), log_level="warning")
