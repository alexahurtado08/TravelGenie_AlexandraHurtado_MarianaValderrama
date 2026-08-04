# ## Paso 2 — Clima histórico mensual por departamento
# 
# Usa la API de **Open-Meteo** (gratuita, sin API key) para traer promedios mensuales de temperatura y precipitación de los últimos 5 años, por cada una de las 32 capitales de departamento (detectadas automáticamente desde el Paso 1).
# 
# Se usa clima **histórico promedio por mes**, no clima actual puntual — es lo que realmente se necesita para responder "¿cuál es el mejor mes para viajar?".

"""
Paso 2 — Clima histórico mensual por departamento de Colombia.

Toma las capitales de departamento automáticamente desde el archivo que
generó el Paso 1 (data/00_divipola_plano.json), así que cubre los 32
departamentos sin necesidad de escribirlos a mano.

Usa Open-Meteo Archive API (gratis, sin API key) para promedios mensuales
de temperatura y precipitación de los últimos años, que es lo que
necesitas para decidir "la mejor época para viajar" (no un solo dato
puntual de clima actual).

Uso:
    pip install requests --break-system-packages
    python3 paso2_clima.py
"""

import json
import os
import statistics
from datetime import date
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
YEARS_BACK = 5  # años de histórico a promediar


def cargar_capitales() -> dict:
    """Lee data/00_divipola_plano.json y elige, por departamento, el
    municipio marcado como capital (tipo_municipio contiene 'Capital').
    Si un departamento no tiene ninguno marcado así, usa el primer
    municipio de la lista como respaldo."""
    with open("data/00_divipola_plano.json", encoding="utf-8") as f:
        municipios = json.load(f)

    capitales = {}
    for m in municipios:
        depto = m["departamento"]
        es_capital = (m.get("tipo_municipio") or "").lower().find("capital") != -1
        if depto not in capitales or es_capital:
            capitales[depto] = m
            if es_capital:
                capitales[depto]["_confirmada"] = True

    faltan_confirmar = [d for d, m in capitales.items() if not m.get("_confirmada")]
    if faltan_confirmar:
        print("Aviso: estos departamentos no tenían 'Capital' explícita en "
              "tipo_municipio, se usó el primer municipio encontrado como "
              "aproximación (revisa si aplica):")
        for d in faltan_confirmar:
            print(f"  - {d} -> {capitales[d]['municipio']}")
        print()

    return capitales


def fetch_monthly_climate(lat: float, lon: float) -> dict:
    """Trae temperatura y precipitación diaria de los últimos N años y las
    agrupa por mes (1-12) para sacar promedios."""
    end = date.today()
    start = date(end.year - YEARS_BACK, 1, 1)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "temperature_2m_mean,precipitation_sum",
        "timezone": "America/Bogota",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    by_month_temp = {m: [] for m in range(1, 13)}
    by_month_precip = {m: [] for m in range(1, 13)}

    dates = data["daily"]["time"]
    temps = data["daily"]["temperature_2m_mean"]
    precs = data["daily"]["precipitation_sum"]

    for d, t, p in zip(dates, temps, precs):
        month = int(d.split("-")[1])
        if t is not None:
            by_month_temp[month].append(t)
        if p is not None:
            by_month_precip[month].append(p)

    monthly = {}
    for m in range(1, 13):
        monthly[m] = {
            "temp_prom_c": round(statistics.mean(by_month_temp[m]), 1) if by_month_temp[m] else None,
            "precipitacion_prom_mm": round(statistics.mean(by_month_precip[m]), 1) if by_month_precip[m] else None,
        }
    return monthly


def main():
    capitales = cargar_capitales()
    print(f"{len(capitales)} departamentos detectados\n")

    result = {}
    for depto, info in capitales.items():
        municipio = info["municipio"]
        lat, lon = info["latitud"], info["longitud"]
        if lat is None or lon is None:
            print(f"-> {depto} ({municipio}): sin coordenadas, se omite")
            continue

        print(f"-> Clima histórico para {depto} ({municipio})...")
        try:
            monthly = fetch_monthly_climate(lat, lon)
        except requests.RequestException as e:
            print(f"   !! Error: {e}")
            continue

        result[depto] = {
            "municipio_referencia": municipio,
            "lat": lat,
            "lon": lon,
            "clima_mensual": monthly,
        }

    os.makedirs("data", exist_ok=True)
    out_path = "data/02_clima.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nListo -> {out_path}")


# -----------------------------------------------------------------------
# BLOQUE OPCIONAL — OpenWeather (clima actual / forecast 5 días)
# Descomenta y usa si tu materia pide específicamente esta API.
# Regístrate gratis en https://openweathermap.org/api
# -----------------------------------------------------------------------
# OPENWEATHER_API_KEY = "TU_API_KEY_AQUI"
#
# def fetch_current_weather(lat, lon):
#     url = "https://api.openweathermap.org/data/2.5/weather"
#     params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric", "lang": "es"}
#     resp = requests.get(url, params=params, timeout=15)
#     resp.raise_for_status()
#     data = resp.json()
#     return {
#         "temp_actual_c": data["main"]["temp"],
#         "descripcion": data["weather"][0]["description"],
#         "timestamp": data["dt"],  # unix time
#     }


if __name__ == "__main__":
    main()
