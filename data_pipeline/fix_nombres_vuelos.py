# ## Corrección puntual — nombres oficiales de municipio
# 
# Ajuste posterior sobre `data/04_vuelos.json` ya generado (sin repetir las ~900 llamadas a la API): tres ciudades tienen nombre oficial largo en el DIVIPOLA distinto al nombre coloquial usado al armar el catálogo de aeropuertos del Paso 4 (Cartagena → "Cartagena de Indias", Cali → "Santiago de Cali", Cúcuta → "San José de Cúcuta"). Sin este ajuste, esos 3 destinos no cruzaban correctamente en el Paso 5.

import json

AEROPUERTOS_CORREGIDO = {
    "CTG": {"municipio": "CARTAGENA DE INDIAS", "departamento": "BOLIVAR"},
    "CLO": {"municipio": "SANTIAGO DE CALI", "departamento": "VALLE DEL CAUCA"},
    "CUC": {"municipio": "SAN JOSE DE CUCUTA", "departamento": "NORTE DE SANTANDER"},
}

with open("data/04_vuelos.json", encoding="utf-8") as f:
    vuelos = json.load(f)

corregidos = 0
for v in vuelos:
    fix = AEROPUERTOS_CORREGIDO.get(v["destino"])
    if fix:
        v["destino_municipio"] = fix["municipio"]
        v["destino_departamento"] = fix["departamento"]
        corregidos += 1

with open("data/04_vuelos.json", "w", encoding="utf-8") as f:
    json.dump(vuelos, f, ensure_ascii=False, indent=2)

print(f"Corregidos {corregidos} registros")
