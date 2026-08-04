# ## Paso 3 — Unir municipios + clima, y calcular el baseline sin IA
# 
# Cruza los 1.122 municipios con el clima de su departamento, y calcula `meses_recomendados`: una heurística simple (temperatura entre 18-28°C + baja precipitación) que puntúa cada mes. Este es el **baseline no-ML** de referencia del proyecto — antes de meter cualquier modelo, esto ya da una primera respuesta razonable a "¿cuándo viajar?".

"""
Paso 3 — Unir municipios (Paso 1, DIVIPOLA) + clima mensual por
departamento (Paso 2) en un dataset estructurado, y calcular una primera
predicción heurística de "meses recomendados para viajar" (el baseline
sin ML que pide la sección 5 de la plantilla del proyecto).
"""

import json
import os

MESES_NOMBRE = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril", 5: "mayo", 6: "junio",
    7: "julio", 8: "agosto", 9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre",
}

# Rango de temperatura considerado "agradable" para turismo — ajusta a tu criterio
TEMP_MIN_AGRADABLE = 18
TEMP_MAX_AGRADABLE = 28


def cargar_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def calcular_meses_recomendados(clima_mensual: dict, top_n: int = 3) -> list:
    """Heurística baseline: puntúa cada mes por temperatura dentro de rango
    agradable y baja precipitación, y devuelve los top_n meses."""
    scored = []
    for mes_str, valores in clima_mensual.items():
        mes = int(mes_str)
        temp = valores.get("temp_prom_c")
        precip = valores.get("precipitacion_prom_mm")
        if temp is None or precip is None:
            continue

        if TEMP_MIN_AGRADABLE <= temp <= TEMP_MAX_AGRADABLE:
            score_temp = 1.0
        else:
            dist = min(abs(temp - TEMP_MIN_AGRADABLE), abs(temp - TEMP_MAX_AGRADABLE))
            score_temp = max(0.0, 1.0 - dist / 10)

        score_precip = max(0.0, 1.0 - precip / 300)

        score = 0.6 * score_temp + 0.4 * score_precip
        scored.append((mes, round(score, 3)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [{"mes": MESES_NOMBRE[m], "score": s} for m, s in scored[:top_n]]


def main():
    municipios = cargar_json("data/00_divipola_plano.json")
    clima = cargar_json("data/02_clima.json")

    dataset_unido = []
    deptos_sin_clima = set()

    for m in municipios:
        depto = (m.get("departamento") or "").strip()
        clima_depto = clima.get(depto)

        registro = dict(m)
        if clima_depto:
            registro["clima_mensual"] = clima_depto["clima_mensual"]
            registro["meses_recomendados"] = calcular_meses_recomendados(
                clima_depto["clima_mensual"]
            )
        else:
            deptos_sin_clima.add(depto)
            registro["clima_mensual"] = None
            registro["meses_recomendados"] = None

        dataset_unido.append(registro)

    os.makedirs("data", exist_ok=True)
    out_path = "data/03_dataset_unido.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset_unido, f, ensure_ascii=False, indent=2)

    print(f"Listo: {len(dataset_unido)} municipios -> {out_path}")
    if deptos_sin_clima:
        print(f"\nDepartamentos sin clima cargado:")
        for d in sorted(deptos_sin_clima):
            print(f"  - {d}")

    if dataset_unido:
        print("\nEjemplo de registro:")
        print(json.dumps(dataset_unido[0], ensure_ascii=False, indent=2))


main()
