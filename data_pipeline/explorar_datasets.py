# ## Paso 0 — Explorar datasets vigentes en datos.gov.co
# 
# Herramienta auxiliar, no genera datos del proyecto en sí. `datos.gov.co` es un portal donde los `dataset_id` a veces cambian o se dan de baja sin aviso (nos pasó más de una vez). Esta celda busca, por palabra clave, los datasets que están **activos ahora mismo** y muestra sus columnas reales — así se evita adivinar un ID viejo y encontrarse con un error 404.

"""
Explorar datasets de datos.gov.co antes de meterlos a paso1_datos_gov.py

Usa la Socrata Discovery API para buscar datasets reales (evita el error
403 que da cuando se copia un ID de una URL tipo /w/xxxx-yyyy/ que es una
"vista/story" y no un dataset consultable).

Uso:
    python3 explorar_datasets.py
"""

import requests

DISCOVERY_URL = "https://api.us.socrata.com/api/catalog/v1"


def buscar_datasets(query: str, limit: int = 20) -> list:
    params = {
        "domains": "www.datos.gov.co",
        "q": query,
        "only": "datasets",
        "limit": limit,
    }
    resp = requests.get(DISCOVERY_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("results", [])


def inspeccionar_columnas(dataset_id: str):
    """Trae 1 fila real del dataset para ver los nombres de columna exactos."""
    url = f"https://www.datos.gov.co/resource/{dataset_id}.json"
    resp = requests.get(url, params={"$limit": 1}, timeout=30)
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        return []
    return list(rows[0].keys())


def main():
    resultados = buscar_datasets("divipola departamentos municipios", limit=20)

    print(f"Encontrados {len(resultados)} datasets para 'turismo':\n")
    for r in resultados:
        resource = r.get("resource", {})
        dataset_id = resource.get("id")
        nombre = resource.get("name")
        print(f"- {dataset_id}  |  {nombre}")

        try:
            columnas = inspeccionar_columnas(dataset_id)
            print(f"    columnas: {columnas}\n")
        except requests.RequestException as e:
            print(f"    !! no se pudo leer: {e}\n")


if __name__ == "__main__":
    main()
