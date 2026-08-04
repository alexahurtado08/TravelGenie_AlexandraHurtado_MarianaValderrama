# ## Paso 1 — Departamentos y municipios de Colombia (DIVIPOLA)
# 
# Descarga el listado oficial de los 1.122 municipios de Colombia con su departamento y coordenadas, desde el dataset `gdxc-w37w` de `datos.gov.co` (licencia CC BY-SA 4.0).
# 
# Esta es la tabla base de todo el proyecto — el "esqueleto" geográfico sobre el que después se cruzan clima, vuelos y lugares turísticos. **No** incluye información de turismo por sí sola (se descartó una primera versión basada en el Registro Nacional de Turismo por no tener coordenadas).

"""
Paso 1 (versión simple) — Listado oficial de departamentos y municipios de
Colombia (DIVIPOLA), dataset_id "xdk5-pm3f" en datos.gov.co.

No trae hoteles ni establecimientos — es solo la división político-
administrativa: cada departamento con sus municipios. Úsenlo como el
"esqueleto" sobre el cual luego cruzan clima, turismo, vuelos, etc.

Uso:
    pip install requests --break-system-packages
    python3 paso1_departamentos_municipios.py
"""

import json
import os
import requests

DATASET_ID = "gdxc-w37w"  # DIVIPOLA - Códigos municipios (confirmado activo)
SODA_URL = f"https://www.datos.gov.co/resource/{DATASET_ID}.json"
LIMIT = 5000


def fetch_all() -> list:
    all_rows = []
    offset = 0
    while True:
        params = {"$limit": LIMIT, "$offset": offset}
        resp = requests.get(SODA_URL, params=params, timeout=30)
        if resp.status_code == 404:
            raise RuntimeError(
                f"El dataset '{DATASET_ID}' ya no existe o cambió de ID "
                f"(404). Corre explorar_datasets.py con el query "
                f"'divipola departamentos municipios' para encontrar el "
                f"ID vigente y actualiza DATASET_ID en este script."
            )
        resp.raise_for_status()
        page = resp.json()
        if not page:
            break
        if offset == 0:
            print(f"Columnas reales del dataset: {list(page[0].keys())}")
            print("(si 'dpto'/'nom_mpio' no aparecen tal cual, ajusta las")
            print(" llaves usadas más abajo en normalize_row)\n")
        all_rows.extend(page)
        if len(page) < LIMIT:
            break
        offset += LIMIT
    return all_rows


def normalize_row(row: dict) -> dict:
    """Columnas confirmadas de gdxc-w37w:
    cod_dpto, dpto, cod_mpio, nom_mpio, tipo_municipio, longitud, latitud"""
    lower_map = {k.lower(): k for k in row.keys()}

    def get(*candidates):
        for c in candidates:
            if c in lower_map:
                return row[lower_map[c]]
        return None

    lat = get("latitud")
    lon = get("longitud")

    def to_float(value):
        if value in (None, ""):
            return None
        return float(str(value).replace(",", "."))

    return {
        "departamento": get("dpto"),
        "cod_departamento": get("cod_dpto"),
        "municipio": get("nom_mpio"),
        "cod_municipio": get("cod_mpio"),
        "tipo_municipio": get("tipo_municipio"),  # ej. "Municipio" / "Capital"
        "latitud": to_float(lat),
        "longitud": to_float(lon),
    }


def main():
    print(f"Descargando {DATASET_ID} (DIVIPOLA)...")
    try:
        rows = fetch_all()
    except RuntimeError as e:
        print(f"\n!! {e}")
        return
    print(f"{len(rows)} filas recibidas\n")

    normalizados = [normalize_row(r) for r in rows]

    # agrupa por departamento para que quede fácil de recorrer/validar
    por_departamento = {}
    for r in normalizados:
        depto = r["departamento"] or "SIN_DEPARTAMENTO"
        por_departamento.setdefault(depto, []).append({
            "municipio": r["municipio"],
            "latitud": r["latitud"],
            "longitud": r["longitud"],
        })

    os.makedirs("data", exist_ok=True)

    with open("data/00_municipios_por_departamento.json", "w", encoding="utf-8") as f:
        json.dump(por_departamento, f, ensure_ascii=False, indent=2)

    with open("data/00_divipola_plano.json", "w", encoding="utf-8") as f:
        json.dump(normalizados, f, ensure_ascii=False, indent=2)

    print(f"Listo: {len(por_departamento)} departamentos, {len(normalizados)} municipios")
    print("-> data/00_municipios_por_departamento.json (agrupado)")
    print("-> data/00_divipola_plano.json (lista plana)")
    if normalizados:
        print("\nEjemplo:")
        print(json.dumps(normalizados[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
