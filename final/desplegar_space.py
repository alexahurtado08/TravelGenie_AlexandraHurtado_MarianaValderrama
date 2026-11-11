"""
Sube TravelGenie a Hugging Face Spaces (solo los archivos que la app necesita).

Uso, desde la raíz del repo (en Colab o en tu computador):
    pip install -q huggingface_hub
    export HF_TOKEN=hf_xxx          # token con permiso de ESCRITURA (huggingface.co/settings/tokens)
    python final/desplegar_space.py USUARIO/travelgenie

En Colab:  import os; os.environ["HF_TOKEN"] = "hf_xxx"
Qué hace: arma una carpeta temporal con el código, los datos, el adaptador LoRA y el
README con la configuración del Space; crea el Space si no existe y la sube.
No sube chroma_db/ (la app lo reconstruye al arrancar), ni las fotos de prueba, ni
los resultados de evaluación.
Los secretos (WANDB_API_KEY) se configuran en Settings > Variables and secrets del Space,
nunca en el código.
"""

import os
import shutil
import sys
import tempfile

ARCHIVOS = ["agente.py", "rag_naive.py", "rag_avanzado.py",
            "data/08_corpus.json", "data/09_chunks.json",
            "data_pipeline/paso9_indexar_chroma.py",
            "final/app.py", "final/observabilidad.py", "final/vision.py",
            "final/perfil_paisaje.py", "final/umbral_vision.json"]
CARPETAS = ["travelgenie-lora-adapter"]

README = """---
title: TravelGenie
emoji: 🧭
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: final/app.py
pinned: false
---

TravelGenie: recomendador de viajes por Colombia (texto o foto de un paisaje).
Proyecto de STAI, Universidad EAFIT. Código completo en GitHub:
https://github.com/alexahurtado08/TravelGenie_AlexandraHurtado_MarianaValderrama
"""


def main():
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit(__doc__)
    repo_id = sys.argv[1]
    falta = [a for a in ARCHIVOS if not os.path.isfile(a)]
    if falta:
        sys.exit(f"Faltan archivos en el repo: {falta}\n"
                 "(umbral_vision.json sale de: python final/eval_vision.py --tune)")
    from huggingface_hub import HfApi
    tmp = tempfile.mkdtemp()
    for a in ARCHIVOS:
        os.makedirs(os.path.join(tmp, os.path.dirname(a)), exist_ok=True)
        shutil.copy(a, os.path.join(tmp, a))
    for c in CARPETAS:
        shutil.copytree(c, os.path.join(tmp, c))
    shutil.copy("final/requirements_space.txt", os.path.join(tmp, "requirements.txt"))
    with open(os.path.join(tmp, "README.md"), "w", encoding="utf-8") as f:
        f.write(README)
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(folder_path=tmp, repo_id=repo_id, repo_type="space",
                      commit_message="Despliegue de TravelGenie")
    print(f"Listo. Construcción en: https://huggingface.co/spaces/{repo_id}")
    print(f"Health check:  https://{repo_id.replace('/', '-').lower()}.hf.space/health")


if __name__ == "__main__":
    main()
