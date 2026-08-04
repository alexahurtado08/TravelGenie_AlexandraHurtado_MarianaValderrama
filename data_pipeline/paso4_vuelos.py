# ## Paso 4 — Precios de vuelo por mes (Travelpayouts)
# 
# Consulta la **Travelpayouts Data API** (`month-matrix`) mes por mes, para rutas desde 4 ciudades hub (Bogotá, Medellín, Cali, Barranquilla) hacia 19 aeropuertos comerciales de Colombia, y guarda el precio más barato encontrado por mes.
# 
# *Nota histórica: se evaluó inicialmente usar la API de Amadeus, pero cerró su portal self-service el 17 de julio de 2026, por lo cual se migró a Travelpayouts.* El token se lee desde el gestor de secretos de Colab (`userdata`), nunca queda escrito en el código — importante porque este repositorio es público.

"""
Paso 4 — Mejor mes para viajar por ruta, vía Travelpayouts Data API.

Genera rutas automáticamente desde un hub (Bogotá) hacia un catálogo de
aeropuertos comerciales de Colombia, en vez de escribirlas a mano una por
una. Cada resultado incluye el municipio/departamento del destino para
poder unirlo con data/03_dataset_unido.json en el Paso 5.

Importante: sin el parámetro 'month', el endpoint solo devuelve precios
cacheados recientes (todos caen en el mismo mes cercano), no un
comparativo real del año. Por eso aquí se consulta mes por mes de forma
explícita para los próximos 12 meses.
"""

import json
import os
import time
from datetime import date
import requests

from google.colab import userdata

TRAVELPAYOUTS_TOKEN = userdata.get("TRAVELPAYOUTS_TOKEN")

BASE_URL = "https://api.travelpayouts.com/v2/prices/month-matrix"

# -----------------------------------------------------------------------
# Catálogo de aeropuertos comerciales de Colombia (IATA -> municipio/depto).
# municipio/departamento deben coincidir (sin importar mayúsculas/tildes)
# con los valores de data/00_divipola_plano.json para que el Paso 5 los
# pueda unir. Agrega/quita destinos según lo que tu proyecto recomiende.
# -----------------------------------------------------------------------
AEROPUERTOS = {
    "BOG": {"municipio": "BOGOTA D.C.", "departamento": "BOGOTA"},          # hub
    "CTG": {"municipio": "CARTAGENA DE INDIAS", "departamento": "BOLIVAR"},
    "SMR": {"municipio": "SANTA MARTA", "departamento": "MAGDALENA"},
    "ADZ": {"municipio": "SAN ANDRES",  "departamento": "SAN ANDRES, PROVIDENCIA Y SANTA CATALINA"},
    "MDE": {"municipio": "MEDELLIN",    "departamento": "ANTIOQUIA"},
    "CLO": {"municipio": "SANTIAGO DE CALI", "departamento": "VALLE DEL CAUCA"},
    "BAQ": {"municipio": "BARRANQUILLA","departamento": "ATLANTICO"},
    "PEI": {"municipio": "PEREIRA",     "departamento": "RISARALDA"},
    "ARM": {"municipio": "ARMENIA",     "departamento": "QUINDIO"},
    "MZL": {"municipio": "MANIZALES",   "departamento": "CALDAS"},
    "RCH": {"municipio": "RIOHACHA",    "departamento": "LA GUAJIRA"},
    "LET": {"municipio": "LETICIA",     "departamento": "AMAZONAS"},
    "CUC": {"municipio": "SAN JOSE DE CUCUTA", "departamento": "NORTE DE SANTANDER"},
    "IBE": {"municipio": "IBAGUE",      "departamento": "TOLIMA"},
    "PSO": {"municipio": "PASTO",       "departamento": "NARIÑO"},
    "MTR": {"municipio": "MONTERIA",    "departamento": "CORDOBA"},
    "NVA": {"municipio": "NEIVA",       "departamento": "HUILA"},
    "VVC": {"municipio": "VILLAVICENCIO","departamento": "META"},
    "PPN": {"municipio": "POPAYAN",     "departamento": "CAUCA"},
    "TCO": {"municipio": "TUNJA",       "departamento": "BOYACA"},  # aprox. (Tunja no tiene aeropuerto comercial grande)
}

HUBS = ["BOG", "MDE", "CLO", "BAQ"]  # ciudades de origen más comunes en Colombia
RUTAS = [
    {"origen": hub, "destino": iata}
    for hub in HUBS
    for iata in AEROPUERTOS
    if iata != hub
]

MESES_A_CONSULTAR = 12  # próximos 12 meses desde hoy

print(f"Se van a consultar {len(RUTAS)} rutas x {MESES_A_CONSULTAR} meses "
      f"= {len(RUTAS) * MESES_A_CONSULTAR} llamadas. Puede tardar bastante "
      f"(reduce HUBS o MESES_A_CONSULTAR si quieres una corrida más rápida).")


def proximos_meses(n: int) -> list:
    """YYYY-MM-01 para los próximos n meses, empezando por el mes actual."""
    hoy = date.today()
    meses = []
    y, m = hoy.year, hoy.month
    for _ in range(n):
        meses.append(f"{y:04d}-{m:02d}-01")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return meses


def precio_para_mes(origen: str, destino: str, mes: str):
    headers = {"x-access-token": TRAVELPAYOUTS_TOKEN}
    params = {
        "currency": "cop",
        "origin": origen,
        "destination": destino,
        "month": mes,
        "show_to_affiliates": "true",
    }
    resp = requests.get(BASE_URL, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    if not body.get("success"):
        raise RuntimeError(body.get("error", "error desconocido"))
    datos = body.get("data", [])
    if not datos:
        return None
    return min(d["value"] for d in datos if "value" in d)


def main():
    resultados = []

    for ruta in RUTAS:
        print(f"-> {ruta['origen']} -> {ruta['destino']}")
        precios_por_mes = []

        for mes in proximos_meses(MESES_A_CONSULTAR):
            try:
                precio = precio_para_mes(ruta["origen"], ruta["destino"], mes)
            except (requests.RequestException, RuntimeError) as e:
                print(f"   !! Error en {mes[:7]}: {e}")
                continue

            if precio is not None:
                print(f"   {mes[:7]}: {precio:,.0f} COP")
                precios_por_mes.append({"mes": mes[:7], "precio": precio})
            else:
                print(f"   {mes[:7]}: sin datos")

            time.sleep(0.3)  # evita saturar el rate limit

        if not precios_por_mes:
            print("   (sin datos para ningún mes en esta ruta)")
            continue

        mas_barato = min(precios_por_mes, key=lambda x: x["precio"])
        destino_info = AEROPUERTOS.get(ruta["destino"], {})
        resultados.append({
            "origen": ruta["origen"],
            "destino": ruta["destino"],
            "destino_municipio": destino_info.get("municipio"),
            "destino_departamento": destino_info.get("departamento"),
            "precios_por_mes": precios_por_mes,
            "mes_mas_barato": mas_barato["mes"],
            "precio_mas_barato": mas_barato["precio"],
        })

    os.makedirs("data", exist_ok=True)
    out_path = "data/04_vuelos.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    print(f"\nListo -> {out_path}")
    if resultados:
        print("\nEjemplo de registro:")
        print(json.dumps(resultados[0], ensure_ascii=False, indent=2))


main()
