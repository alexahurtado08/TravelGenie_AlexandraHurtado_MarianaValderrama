# ## Paso 5 — Unificar municipios + clima + vuelos
# 
# Combina la salida del Paso 3 (municipios + clima) con la del Paso 4 (vuelos), y calcula `mejor_epoca_combinada`: compara el mes más barato para volar contra los mejores meses por clima, señalando si coinciden (combinación ideal) o si hay que elegir entre precio y clima.
# 
# El cruce se hace por la pareja **(municipio, departamento)**, no solo por nombre de municipio — Colombia tiene municipios homónimos en distintos departamentos (ej. "San Andrés" existe en el archipiélago y también en Santander); cruzar solo por nombre mezclaba los datos de ambos.

"""
Paso 5 — Unificar data/03_dataset_unido.json (municipios + clima) con
data/04_vuelos.json (precios por mes) en un solo dataset final por
destino, listo para alimentar el modelo del M1.

Solo los municipios que SÍ tienen vuelo consultado en el Paso 4 quedan
con info de precios; el resto queda solo con clima (útil para otras
partes del proyecto, pero sin señal de precio).
"""

import json
import os


def normalizar(texto):
    """Para comparar nombres sin pelearse con mayúsculas/tildes/puntuación."""
    if not texto:
        return ""
    reemplazos = str.maketrans("ÁÉÍÓÚÑ", "AEIOUN")
    limpio = texto.upper().translate(reemplazos)
    limpio = limpio.replace(",", "").replace(".", "")
    return " ".join(limpio.split())  # colapsa espacios múltiples


def calcular_mejor_epoca_combinada(meses_recomendados_clima, vuelos):
    """Cruza los mejores meses por clima con el mes más barato encontrado
    entre TODOS los orígenes disponibles para ese destino. Heurística
    simple: si el mes más barato también aparece en los meses
    recomendados por clima, es la mejor opción posible (barato + buen
    clima). Si no, se reportan ambos por separado para que el usuario
    decida el trade-off."""
    if not meses_recomendados_clima or not vuelos:
        return None

    meses_clima = {m["mes"] for m in meses_recomendados_clima}
    mas_barato_global = min(vuelos, key=lambda v: v["precio_mas_barato"])
    mes_barato = mas_barato_global["mes_mas_barato"]

    return {
        "coincide_barato_y_buen_clima": mes_barato in meses_clima,
        "mejor_origen": mas_barato_global["origen"],
        "mes_mas_barato": mes_barato,
        "precio_mas_barato": mas_barato_global["precio_mas_barato"],
        "mejores_meses_clima": sorted(meses_clima),
    }


def main():
    with open("data/03_dataset_unido.json", encoding="utf-8") as f:
        municipios = json.load(f)

    with open("data/04_vuelos.json", encoding="utf-8") as f:
        vuelos = json.load(f)

    # ahora puede haber varias rutas (distintos orígenes) hacia el mismo
    # destino, así que se agrupan en una lista en vez de sobreescribir.
    # La clave es (municipio, departamento) y NO solo municipio, porque
    # Colombia tiene municipios homónimos en departamentos distintos
    # (ej. "San Andrés" existe en el archipiélago Y en Santander).
    vuelos_por_municipio = {}
    for v in vuelos:
        key = (normalizar(v.get("destino_municipio")), normalizar(v.get("destino_departamento")))
        if key[0]:
            vuelos_por_municipio.setdefault(key, []).append(v)

    final = []
    con_vuelo = 0
    for m in municipios:
        key = (normalizar(m.get("municipio")), normalizar(m.get("departamento")))
        vuelos_del_destino = vuelos_por_municipio.get(key)

        registro = dict(m)
        registro["vuelos_por_origen"] = vuelos_del_destino  # None si no hay ruta consultada
        registro["mejor_epoca_combinada"] = calcular_mejor_epoca_combinada(
            m.get("meses_recomendados"), vuelos_del_destino
        )
        if vuelos_del_destino:
            con_vuelo += 1
        final.append(registro)

    os.makedirs("data", exist_ok=True)
    out_path = "data/05_dataset_final.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"Listo: {len(final)} municipios totales, {con_vuelo} con datos "
          f"de vuelo -> {out_path}")

    # muestra un ejemplo de un municipio que SÍ tenga vuelo, para verificar
    ejemplo = next((r for r in final if r["vuelos_por_origen"]), None)
    if ejemplo:
        print("\nEjemplo de registro con vuelo:")
        print(json.dumps(ejemplo, ensure_ascii=False, indent=2))
    else:
        print("\nAviso: ningún municipio hizo match con los vuelos del "
              "Paso 4. Revisa que 'destino_municipio' en data/04_vuelos.json "
              "coincida con los nombres de municipio del Paso 1 (imprime "
              "ambos y compara si hace falta ajustar AEROPUERTOS).")


main()
