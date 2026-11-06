"""
Perfil de paisaje de cada destino: qué tipo(s) de lugar es, para poder ligar
una foto con los destinos de la base.

Esta tabla la escribió el equipo con conocimiento general de Colombia (no sale
de los datos descargados): es una decisión de diseño y puede estar equivocada.
Se pueden cambiar los tipos de un destino sin tocar nada más.

Tipos: playa, selva, montaña, cafetero, ciudad_colonial, llanura
(llanura incluye sabana y desierto).
"""

TIPOS = ["playa", "selva", "montaña", "cafetero", "ciudad_colonial", "llanura"]

PERFIL = {
    "SANTA MARTA":   ["playa"],
    "RIOHACHA":      ["playa", "llanura"],      # costa y desierto de La Guajira
    "BARRANQUILLA":  ["playa"],
    "MONTERÍA":      ["llanura"],
    "NEIVA":         ["llanura"],               # desierto de la Tatacoa cerca
    "VILLAVICENCIO": ["llanura"],               # llanos orientales
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
