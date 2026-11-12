"""
Técnica de eficiencia: reducir la precisión del agente (Qwen2.5-1.5B) y medir qué se gana
y qué se pierde. Componente: SOLO el agente (la decisión de llamar a verificar_cobertura).

TOLERANCIA (fijada ANTES de correr; no se cambia después de ver resultados):
  Calidad (obligatoria, contra fp32, en final/eval_agente_decisiones.py):
    1. acierto de la vía no baja más de 1 caso (≈2.5 pts) frente a fp32;
    2. la válvula no pierde nada: acierto en 'sin_cobertura' y en 'fuera_de_tema' >= fp32;
    3. (informativo) acuerdo con fp32: % de casos con la misma decisión exacta.
  Ganancia (al menos una):
    A. p95 de latencia del agente >= 25% menor que fp32, o
    B. memoria del modelo >= 40% menor con p95 no peor en más de 10%.
  Se adopta la variante que cumpla calidad y ganancia; si varias, la de menor p95.
  Si ninguna, se queda fp32 y se reporta como resultado negativo.

Uso (desde la raíz del repo, en Colab con GPU):
    pip install -q bitsandbytes
    python final/eficiencia.py --decisiones            # fp32, bf16, fp16, int8, nf4
    python final/eficiencia.py --decisiones fp32 fp16  # solo algunas variantes
    python final/eficiencia.py --e2e fp32 fp16 --repeticiones 3
        # preguntas completas del eval M3 (con RAG real): latencia p50/p95 y % de
        # cumplimiento de las reglas de verificacion.py (sin juez)
Escribe final/resultados_eficiencia.csv/json y final/resultados_eficiencia_e2e.csv/json.
"""

import csv
import gc
import json
import os
import statistics
import sys
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
os.chdir(RAIZ)
sys.path[:0] = [RAIZ, AQUI]

import agente
from eval_agente_decisiones import CASOS

VARIANTES = ["fp32", "bf16", "fp16", "int8", "nf4"]   # bf16 = lo que cargaba la app por defecto
TOL_CASOS = 1            # casos de diferencia aceptados en acierto de vía
GANANCIA_P95 = 0.25      # reducción mínima de p95
GANANCIA_MEM = 0.40      # reducción mínima de memoria
P95_NO_PEOR = 0.10       # p95 no peor en más de 10% (opción B)


# ------------------------------------------------------------------ carga de variantes
def cargar_variante(nombre):
    """Carga Qwen en la precisión pedida y lo deja en agente._estado. Devuelve el footprint en GB."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    liberar()
    tok = AutoTokenizer.from_pretrained(agente.MODELO_AGENTE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {}
    # OJO: con transformers 5.x, sin `dtype` el modelo carga en la precisión del checkpoint
    # (bf16 para Qwen2.5), no en fp32. Por eso fp32 y bf16 se piden de forma explícita.
    if nombre == "fp32":
        kw["dtype"] = torch.float32
    elif nombre == "bf16":
        kw["dtype"] = torch.bfloat16
    elif nombre == "fp16":
        kw["dtype"] = torch.float16
    elif nombre == "int8":
        from transformers import BitsAndBytesConfig
        kw.update(quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map={"": 0})
    elif nombre == "nf4":
        from transformers import BitsAndBytesConfig
        kw.update(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16),
            device_map={"": 0})
    else:
        raise ValueError(nombre)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        llm = AutoModelForCausalLM.from_pretrained(agente.MODELO_AGENTE, **kw)
    except TypeError:                       # transformers 4.x usa torch_dtype
        if "dtype" in kw:
            kw["torch_dtype"] = kw.pop("dtype")
        llm = AutoModelForCausalLM.from_pretrained(agente.MODELO_AGENTE, **kw)
    if nombre in ("fp32", "fp16"):
        llm = llm.to(device)
    llm.eval()
    agente._estado.update(tok=tok, llm=llm, device=device)
    return llm.get_memory_footprint() / 1e9


def liberar():
    import torch
    for k in ("llm", "tok", "device"):
        agente._estado.pop(k, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ evaluación de decisiones
def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def p(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def evaluar_decisiones(llm=None, n_calentamiento=3):
    """Corre los casos con el RAG anulado (solo se mide el agente). `llm` permite inyectar un
    sustituto para pruebas sin modelo. Devuelve (filas_detalle, resumen)."""
    original = agente.rag_avanzado.sistema_rag_avanzado
    agente.rag_avanzado.sistema_rag_avanzado = lambda q: ""
    try:
        for q, *_ in CASOS[:n_calentamiento]:
            agente.sistema_agente(q, llm=llm)
        filas, tiempos = [], []
        for q, grupo, via_esp, llamada_esp in CASOS:
            _sync(); t0 = time.perf_counter()
            agente.sistema_agente(q, llm=llm)
            _sync(); dt = time.perf_counter() - t0
            u = agente.ULTIMO
            llamadas = [c["argumentos"].get("destino") for c in u["llamadas"]]
            tiempos.append(dt)
            filas.append({"pregunta": q, "grupo": grupo, "via_esperada": via_esp, "via": u["via"],
                          "via_ok": u["via"] == via_esp,
                          "llamo": bool(llamadas), "destino_arg": llamadas[0] if llamadas else None,
                          "llamada_ok": None if llamada_esp is None else (bool(llamadas) == llamada_esp),
                          "tope_de_pasos": u["tope_de_pasos"], "latencia_s": round(dt, 4)})
    finally:
        agente.rag_avanzado.sistema_rag_avanzado = original

    def acierto(grupo=None, campo="via_ok"):
        xs = [f[campo] for f in filas if (grupo is None or f["grupo"] == grupo) and f[campo] is not None]
        return sum(xs) / len(xs) if xs else None
    resumen = {"n": len(filas), "via_acc": acierto(), "llamada_acc": acierto(campo="llamada_ok"),
               "acc_cubierta": acierto("cubierta"), "acc_sin_cobertura": acierto("sin_cobertura"),
               "acc_sin_ciudad": acierto("sin_ciudad"), "acc_fuera_de_tema": acierto("fuera_de_tema"),
               "p50_s": round(statistics.median(tiempos), 4), "p95_s": round(p(tiempos, 0.95), 4)}
    return filas, resumen


def veredicto(res, base):
    """Aplica la tolerancia fijada arriba. `res` y `base` son resúmenes (base = fp32)."""
    n = res["n"]
    c1 = res["via_acc"] * n >= base["via_acc"] * n - TOL_CASOS
    c2 = (res["acc_sin_cobertura"] >= base["acc_sin_cobertura"]
          and res["acc_fuera_de_tema"] >= base["acc_fuera_de_tema"])
    g_a = res["p95_s"] <= base["p95_s"] * (1 - GANANCIA_P95)
    g_b = (res["memoria_gb"] <= base["memoria_gb"] * (1 - GANANCIA_MEM)
           and res["p95_s"] <= base["p95_s"] * (1 + P95_NO_PEOR))
    return {"calidad_ok": bool(c1 and c2), "ganancia_ok": bool(g_a or g_b),
            "adoptar": bool(c1 and c2 and (g_a or g_b)),
            "detalle": {"c1_via": bool(c1), "c2_valvula": bool(c2), "g_p95": bool(g_a), "g_mem": bool(g_b)}}


def modo_decisiones(variantes):
    import torch
    resumenes, detalles = {}, {}
    for v in variantes:
        try:
            mem = cargar_variante(v)
        except Exception as e:   # noqa: BLE001  (p. ej. falta bitsandbytes)
            print(f"[{v}] no se pudo cargar: {type(e).__name__}: {str(e)[:120]}")
            continue
        filas, r = evaluar_decisiones()
        r["memoria_gb"] = round(mem, 3)
        r["pico_gpu_gb"] = (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                            if torch.cuda.is_available() else None)
        resumenes[v], detalles[v] = r, filas
        print(f"[{v}] vía {r['via_acc']:.3f}  cubierta {r['acc_cubierta']:.2f}  "
              f"sin_cob {r['acc_sin_cobertura']:.2f}  fuera {r['acc_fuera_de_tema']:.2f}  "
              f"p50 {r['p50_s']}s  p95 {r['p95_s']}s  mem {r['memoria_gb']} GB")
    if "fp32" not in resumenes:
        print("Sin fp32 no hay base de comparación.")
        return
    base = resumenes["fp32"]
    for v, r in resumenes.items():
        r["acuerdo_con_fp32"] = round(sum(
            (a["via"], a["destino_arg"]) == (b["via"], b["destino_arg"])
            for a, b in zip(detalles[v], detalles["fp32"])) / len(CASOS), 3)
        r["veredicto"] = veredicto(r, base) if v != "fp32" else {"adoptar": False, "detalle": "base"}
    campos = ["variante", "n", "via_acc", "llamada_acc", "acc_cubierta", "acc_sin_cobertura",
              "acc_sin_ciudad", "acc_fuera_de_tema", "acuerdo_con_fp32", "p50_s", "p95_s",
              "memoria_gb", "pico_gpu_gb", "calidad_ok", "ganancia_ok", "adoptar"]
    with open(os.path.join(AQUI, "resultados_eficiencia.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(campos)
        for v, r in resumenes.items():
            ve = r["veredicto"]
            w.writerow([v] + [r.get(c) for c in campos[1:13]] +
                       [ve.get("calidad_ok", ""), ve.get("ganancia_ok", ""), ve.get("adoptar", "")])
    with open(os.path.join(AQUI, "resultados_eficiencia.json"), "w", encoding="utf-8") as f:
        json.dump({"resumen": resumenes, "detalle": detalles}, f, indent=2, ensure_ascii=False)
    print("\nVeredicto contra la tolerancia (fp32 es la base):")
    for v, r in resumenes.items():
        if v == "fp32":
            continue
        ve = r["veredicto"]
        print(f"  {v}: calidad {'OK' if ve['calidad_ok'] else 'NO'} | ganancia {'OK' if ve['ganancia_ok'] else 'NO'}"
              f" | adoptar={ve['adoptar']}  {ve['detalle']}  acuerdo={r['acuerdo_con_fp32']}")
    dif = [(v, a["pregunta"], a["via"]) for v in detalles if v != "fp32"
           for a, b in zip(detalles[v], detalles["fp32"]) if a["via"] != b["via"]]
    if dif:
        print("\nCasos donde la vía cambió frente a fp32 (para el análisis de errores):")
        for d in dif:
            print("  ", d)


# ------------------------------------------------------------------ extremo a extremo
def modo_e2e(variantes, repeticiones):
    from eval_set_m3 import eval_set_m3
    from verificacion import verificar_criterio
    import rag_avanzado
    filas = []
    for v in variantes:
        cargar_variante(v)
        rag_avanzado._cargar()
        for q in [c["input"] for c in eval_set_m3[:2]]:     # calentamiento
            agente.sistema_agente(q)
        tiempos, cumple = [], []
        for _ in range(repeticiones):
            for caso in eval_set_m3:
                _sync(); t0 = time.perf_counter()
                r = agente.sistema_agente(caso["input"])
                _sync(); tiempos.append(time.perf_counter() - t0)
                cumple.append(bool(verificar_criterio(r, caso)[0]))
        fila = {"variante": v, "n_mediciones": len(tiempos),
                "pct_cumple_verificacion": round(100 * sum(cumple) / len(cumple), 1),
                "p50_s": round(statistics.median(tiempos), 3), "p95_s": round(p(tiempos, 0.95), 3)}
        filas.append(fila)
        print(fila)
    with open(os.path.join(AQUI, "resultados_eficiencia_e2e.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0])); w.writeheader(); w.writerows(filas)
    with open(os.path.join(AQUI, "resultados_eficiencia_e2e.json"), "w", encoding="utf-8") as f:
        json.dump(filas, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in ("--decisiones", "--e2e"):
        print(__doc__)
    else:
        rep = 3
        if "--repeticiones" in a:
            i = a.index("--repeticiones"); rep = int(a[i + 1]); del a[i:i + 2]
        vs = a[1:] or VARIANTES
        modo_decisiones(vs) if a[0] == "--decisiones" else modo_e2e(vs, rep)
