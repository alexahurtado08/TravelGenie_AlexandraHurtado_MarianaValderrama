"""
Eval set de M3 

Parte del eval set de M2 solo con dos cambios, ambos
justificados por el corpus real de 14 destinos 

  1) La pregunta de Cartagena esperaba el mejor mes (enero) y un precio de
     vuelo, datos que el corpus NO tiene: ningún sistema con RAG puede
     cumplirla. Pasa a ser una pregunta FUERA DEL CORPUS: lo correcto es
     admitir que no hay datos.
  2) Se añaden 2 preguntas con respuesta verificable en el corpus (Santa Marta
     y Pereira), porque el set original tenía muy pocos casos respondibles

Casos fuera del corpus: Cartagena, Quibdó, San André y  Fuera de dominio:
Francia y tiquete falso.
"""

import contextlib
import io

with contextlib.redirect_stdout(io.StringIO()):   # silencia los print() de eval_set.py
    from eval_set import eval_set as _eval_set_m2

_CARTAGENA = {
    'input': '¿Cuál es el mejor mes para viajar a Cartagena de Indias?',
    'esperado': 'Reconoce que no tiene datos de Cartagena (no está entre los destinos cubiertos) en vez de inventar un mes de clima o un precio de vuelo.',
    'criterio': 'admite la falta de datos explícitamente; falla si responde con datos de otra ciudad o inventa mes/precio',
}

_NUEVAS = [
    {'input': '¿Cuál es el mejor mes para viajar a Santa Marta?',
     'esperado': 'Indica que los meses con mejor clima en Santa Marta son enero, marzo y abril, y que el vuelo más económico encontrado es en noviembre desde MDE por $41,704 COP, explicando que ese mes no coincide con los de mejor clima (trade-off entre precio y clima).',
     'criterio': 'usa los datos reales de Santa Marta (meses de clima, mes, origen y precio de vuelo), sin inventar cifras ni mezclar datos de otras ciudades'},

    {'input': '¿Cuándo es más barato volar a Pereira y desde dónde?',
     'esperado': 'Indica que el vuelo más económico encontrado a Pereira es en agosto, desde BOG, por $70,175 COP.',
     'criterio': 'mes, origen y precio coinciden con los datos de Pereira; falla si cita cifras de otra ciudad'},
]

eval_set_m3 = [_CARTAGENA if 'Cartagena' in e['input'] else e for e in _eval_set_m2] + _NUEVAS

assert len(eval_set_m3) == 12
assert sum('Cartagena' in e['input'] for e in eval_set_m3) == 1
assert all('input' in e and 'esperado' in e and 'criterio' in e for e in eval_set_m3), \
    'Cada ejemplo necesita input, esperado y criterio.'
