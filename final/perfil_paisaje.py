"""
Descripción de paisaje de cada destino, qué tipo de lugar es, para poder emparejar
una foto con los destinos de la base.
"""

TIPOS = ["playa", "selva", "montaña", "cafetero", "ciudad_colonial", "llanura"]

PERFIL = {
    "SANTA MARTA":   ["playa"],
    "RIOHACHA":      ["playa", "llanura"],      
    "BARRANQUILLA":  ["playa"],
    "MONTERÍA":      ["llanura"],
    "NEIVA":         ["llanura"],               
    "VILLAVICENCIO": ["llanura"],               
    "LETICIA":       ["selva"],
    "TUNJA":         ["montaña", "ciudad_colonial"],
    "PASTO":         ["montaña"],
    "MANIZALES":     ["montaña", "cafetero"],
    "PEREIRA":       ["cafetero"],
    "MEDELLÍN":      ["montaña"],
    "IBAGUÉ":        ["montaña"],
    "POPAYÁN":       ["ciudad_colonial"],
}

assert all(t in TIPOS for ts in PERFIL.values() for t in ts)
assert all(any(t in ts for ts in PERFIL.values()) for t in TIPOS), "tipo sin destinos"


def destinos_de(tipo):
    """Destinos cuyo perfil incluye ese tipo de paisaje."""
    return [d for d, ts in PERFIL.items() if tipo in ts]
