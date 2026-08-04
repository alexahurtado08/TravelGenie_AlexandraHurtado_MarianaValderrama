# Pipeline de recolección y preparación de datos

Estos scripts se corren en orden (cada uno consume la salida del anterior).
Todos se ejecutaron originalmente en Google Colab.

```
1. paso1_departamentos_municipios.py  -> data/00_divipola_plano.json (1122 municipios, DIVIPOLA)
                                       -> data/00_municipios_por_departamento.json
2. paso2_clima.py                     -> data/02_clima.json (clima histórico mensual, Open-Meteo)
3. paso3_unir_datos.py                -> data/03_dataset_unido.json (municipios + clima)
4. paso4_vuelos.py                    -> data/04_vuelos.json (precios por mes, Travelpayouts)
   [fix_nombres_vuelos.py]            -> corrección puntual de nombres de municipio, sin repetir llamadas a la API
5. paso5_dataset_final.py             -> data/05_dataset_final.json (todo unido)
6. paso6_lugares_osm.py               -> data/06_lugares.json (hoteles/restaurantes/atractivos, OpenStreetMap)
7. paso7_generar_pares_entrenamiento.py -> data/07_train.jsonl, data/07_val.jsonl (dataset de fine-tuning)

explorar_datasets.py — herramienta auxiliar para encontrar datasets vigentes
en datos.gov.co cuando un dataset_id deja de funcionar (pasó más de una vez).
```

## Cómo correr estos scripts

Los scripts usan rutas relativas (`data/archivo.json`), y la carpeta
`data/` vive en la **raíz del repositorio**, no dentro de esta carpeta.
Corre los scripts desde la raíz del repo, no desde dentro de
`data_pipeline/`:

```bash
# desde la raíz del repo:
python3 data_pipeline/paso1_departamentos_municipios.py
python3 data_pipeline/paso2_clima.py
# ...etc
```

O, en Colab, copia el contenido de cada script a una celda tal como se
hizo originalmente (ver el historial de este proyecto).

## Fuentes externas usadas

- **datos.gov.co** (Socrata SODA API) — DIVIPOLA, dataset `gdxc-w37w`.
- **Open-Meteo Archive API** — clima histórico, gratuita, sin API key.
- **Travelpayouts Data API** — precios de vuelo. Requiere token de cuenta
  gratuita (ver `paso4_vuelos.py`, usa el gestor de secretos de Colab, no
  queda escrito en el código).
- **OpenStreetMap** (vía Overpass API) — hoteles, restaurantes, atractivos.
  Datos bajo licencia ODbL — ver atribución en el README principal.

