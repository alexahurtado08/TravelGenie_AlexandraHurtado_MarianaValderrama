"""
Observabilidad de TravelGenie: registros JSON y métricas en W&B (opcional).

Reglas de privacidad (se cumplen por construcción):
- Solo se escriben los campos de CAMPOS_PERMITIDOS; cualquier otro se descarta.
- NUNCA se registra el texto de la pregunta, ni la imagen, ni la respuesta, ni la
  IP del usuario. De la pregunta solo se guarda su longitud; de la foto, solo el
  tipo de paisaje y la confianza.
- De un error solo se guarda el nombre de la excepción (su mensaje podría traer
  texto del usuario).

Un registro por solicitud, en formato JSON Lines: una línea por evento, a la salida
estándar (los "Logs" de Hugging Face Spaces) y al archivo logs/travelgenie.jsonl.

W&B se activa solo si existe la variable de entorno WANDB_API_KEY. Si falla,
la app sigue funcionando: la observabilidad nunca debe tumbar el servicio.
"""

import json
import os
import sys
import threading
import time
import uuid

RUTA_LOG = os.environ.get("TRAVELGENIE_LOG", "logs/travelgenie.jsonl")
PROYECTO_WANDB = os.environ.get("WANDB_PROJECT", "travelgenie-final")

CAMPOS_PERMITIDOS = {
    "ts", "request_id", "evento", "entrada",       # entrada: "texto" | "imagen"
    "via",                                         # agente: rag_avanzado | sin_cobertura | ...
    "n_llamadas_herramienta", "tope_de_pasos",
    "longitud_pregunta", "pregunta_truncada",
    "componentes", "valvula", "tipo_paisaje", "prob", "n_destinos",
    "latencia_s", "error", "estado", "detalle",
}

_lock = threading.Lock()
_wandb = {"run": None, "intentado": False}
_contadores = {"solicitudes": 0, "errores": 0, "valvula": 0}


def _limpiar(d):
    out = {}
    for k, v in d.items():
        if k not in CAMPOS_PERMITIDOS:
            continue
        if isinstance(v, float):
            v = round(v, 4)
        elif isinstance(v, str):
            v = v[:60]
        elif not isinstance(v, (int, bool, type(None))):
            continue
        out[k] = v
    return out


def nuevo_id():
    return uuid.uuid4().hex[:12]


def registrar(**campos):
    """Escribe un evento JSON (solo campos permitidos) y, si hay W&B, sus métricas."""
    campos.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    ev = _limpiar(campos)
    linea = json.dumps(ev, ensure_ascii=False)
    with _lock:
        print(linea, file=sys.stdout, flush=True)
        try:
            os.makedirs(os.path.dirname(RUTA_LOG) or ".", exist_ok=True)
            with open(RUTA_LOG, "a", encoding="utf-8") as f:
                f.write(linea + "\n")
        except OSError:
            pass   # en un disco de solo lectura se sigue con la salida estándar
    if ev.get("evento") == "solicitud":
        _wandb_log(ev)
    return ev


def _wandb_run():
    if _wandb["intentado"]:
        return _wandb["run"]
    _wandb["intentado"] = True
    if not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        import wandb
        _wandb["run"] = wandb.init(project=PROYECTO_WANDB, job_type="serving",
                                   config={"componente": "app"}, reinit=True)
    except Exception as e:   # noqa: BLE001  (nunca tumbar la app)
        registrar(evento="wandb", estado="desactivado", detalle=type(e).__name__)
    return _wandb["run"]


def _wandb_log(ev):
    _contadores["solicitudes"] += 1
    _contadores["errores"] += 1 if ev.get("error") else 0
    _contadores["valvula"] += 1 if ev.get("valvula") else 0
    run = _wandb_run()
    if run is None:
        return
    try:
        m = {"latencia_s": ev.get("latencia_s"),
             "error": int(bool(ev.get("error"))),
             "valvula": int(bool(ev.get("valvula"))),
             "entrada_imagen": int(ev.get("entrada") == "imagen"),
             "solicitudes_total": _contadores["solicitudes"],
             "tasa_errores": _contadores["errores"] / _contadores["solicitudes"],
             "tasa_valvula": _contadores["valvula"] / _contadores["solicitudes"]}
        if ev.get("prob") is not None:
            m["prob_vision"] = ev["prob"]
        run.log({k: v for k, v in m.items() if v is not None})
    except Exception:   # noqa: BLE001
        pass
