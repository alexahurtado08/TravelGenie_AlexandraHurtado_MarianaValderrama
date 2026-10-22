"""
Eval set de M3 — TravelGenie: los 10 casos de eval_set.py (ya corregidos tras la
retroalimentación de M2) más 2 preguntas con respuesta verificable en el corpus
(Santa Marta y Pereira), porque 10 casos daban pocos con datos exactos
(n pequeño = números ruidosos). Son 12 casos.

Fuera del corpus: Cartagena, Quibdó, San Andrés (x2). Fuera de dominio: Francia
y tiquete falso.
"""

import contextlib
import io

with contextlib.redirect_stdout(io.StringIO()):   # silencia los print() de eval_set.py
    from eval_set import eval_set as _eval_set_m2

_NUEVAS = [
    {'input': '¿Cuál es el mejor mes para viajar a Santa Marta?',
     'esperado': 'Los meses con mejor clima en Santa Marta son enero, marzo y abril. El vuelo más barato que encontré es en noviembre, desde MDE, por $41,704 COP, pero noviembre no coincide con los meses de mejor clima: hay que elegir entre pagar menos o viajar con mejor clima.',
     'criterio': 'usa los datos reales de Santa Marta (meses de clima, mes, origen y precio de vuelo), sin inventar cifras ni mezclar datos de otras ciudades',
     'verificar': {'contiene_todos': ['enero', 'marzo', 'abril', 'noviembre', 'MDE', '41,704']}},

    {'input': '¿Cuándo es más barato volar a Pereira y desde dónde?',
     'esperado': 'El vuelo más barato a Pereira que encontré es en agosto, desde BOG, por $70,175 COP.',
     'criterio': 'mes, origen y precio coinciden con los datos de Pereira; falla si cita cifras de otra ciudad',
     'verificar': {'contiene_todos': ['agosto', 'BOG', '70,175']}},
]

eval_set_m3 = list(_eval_set_m2) + _NUEVAS

assert len(eval_set_m3) == 12
assert all('input' in e and 'esperado' in e and 'criterio' in e and 'verificar' in e for e in eval_set_m3), \
    'Cada ejemplo necesita input, esperado, criterio y verificar.'
