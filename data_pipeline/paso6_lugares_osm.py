# ## Paso 6 — Hoteles, restaurantes y atractivos (OpenStreetMap)
# 
# Consulta la **Overpass API** de OpenStreetMap (datos bajo licencia ODbL, atribución obligatoria — ver README) para traer nombres de hoteles, restaurantes y atractivos turísticos alrededor de cada uno de los destinos que sí tienen datos de vuelo.
# 
# El servidor público de Overpass es compartido y se satura fácilmente, así que esta celda reintenta con espera ante error 429, alterna entre 3 servidores espejo, y guarda progreso después de cada destino — se puede interrumpir y retomar sin perder lo ya conseguido.

"""
Paso 6 — Hoteles, restaurantes y atractivos por destino, vía OpenStreetMap
Overpass API. Reintenta con espera en 429, cambia de espejo en otros
errores, y guarda progreso incremental (puedes parar y retomar).
"""

import json
import os
import time
import requests

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/cgi/interpreter",
]

HEADERS = {
    "User-Agent": "TravelGenie-EAFIT-ProyectoAcademico/1.0 (uso educativo)",
    "Accept": "application/json",
}

RADIO_METROS = 15000

CATEGORIAS = {
    "tourism": ["hotel", "guest_house", "attraction", "museum", "viewpoint"],
    "amenity": ["restaurant"],
}


def construir_query(lat, lon, radio=RADIO_METROS):
    partes = []
    for key, valores in CATEGORIAS.items():
        patron = "|".join(valores)
        partes.append(f'node["{key}"~"^({patron})$"](around:{radio},{lat},{lon});')
        partes.append(f'way["{key}"~"^({patron})$"](around:{radio},{lat},{lon});')
    cuerpo = "\n  ".join(partes)
    return f"""
[out:json][timeout:60];
(
  {cuerpo}
);
out center tags;
"""


def consultar_overpass(lat, lon):
    query = construir_query(lat, lon)
    ultimo_error = None
    for url in OVERPASS_URLS:
        for intento in range(2):
            try:
                resp = requests.post(url, data={"data": query}, headers=HEADERS, timeout=90)
                if resp.status_code == 429:
                    espera = 30
                    print(f"   (429 en {url}, esperando {espera}s antes de reintentar...)")
                    time.sleep(espera)
                    continue
                resp.raise_for_status()
                return resp.json().get("elements", [])
            except requests.RequestException as e:
                ultimo_error = e
                print(f"   (falló {url}: {e})")
                break
        continue
    raise ultimo_error if ultimo_error else RuntimeError("todos los servidores fallaron")


def normalizar_elemento(el):
    tags = el.get("tags", {})
    if el["type"] == "node":
        lat, lon = el.get("lat"), el.get("lon")
    else:
        centro = el.get("center", {})
        lat, lon = centro.get("lat"), centro.get("lon")

    return {
        "nombre": tags.get("name"),
        "categoria": tags.get("tourism") or tags.get("amenity"),
        "lat": lat,
        "lon": lon,
        "direccion": tags.get("addr:street"),
    }


def main():
    with open("data/05_dataset_final.json", encoding="utf-8") as f:
        municipios = json.load(f)

    destinos = [m for m in municipios if m.get("vuelos_por_origen")]
    print(f"Consultando lugares para {len(destinos)} destinos "
          f"(radio {RADIO_METROS/1000:.0f} km)...")

    os.makedirs("data", exist_ok=True)
    out_path = "data/06_lugares.json"

    try:
        with open(out_path, encoding="utf-8") as f:
            resultado = json.load(f)
        print(f"Retomando: {len(resultado)} destinos ya guardados de una corrida previa.")
    except FileNotFoundError:
        resultado = {}

    for m in destinos:
        nombre = m["municipio"]
        if nombre in resultado:
            print(f"-> {nombre} (ya estaba guardado, se omite)")
            continue

        print(f"-> {nombre}")
        try:
            elementos = consultar_overpass(m["latitud"], m["longitud"])
        except requests.RequestException as e:
            print(f"   !! Error final: {e}")
            continue

        lugares = [normalizar_elemento(el) for el in elementos]
        lugares = [l for l in lugares if l["nombre"]]

        hoteles = [l for l in lugares if l["categoria"] in ("hotel", "guest_house")]
        restaurantes = [l for l in lugares if l["categoria"] == "restaurant"]
        atractivos = [l for l in lugares if l["categoria"] in ("attraction", "museum", "viewpoint")]

        resultado[nombre] = {
            "hoteles": hoteles,
            "restaurantes": restaurantes,
            "atractivos": atractivos,
        }
        print(f"   {len(hoteles)} hoteles, {len(restaurantes)} restaurantes, "
              f"{len(atractivos)} atractivos")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(resultado, f, ensure_ascii=False, indent=2)

        time.sleep(5)

    print(f"\nListo: {len(resultado)}/{len(destinos)} destinos -> {out_path}")


main()
