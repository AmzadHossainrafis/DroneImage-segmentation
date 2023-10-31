import os 
from pathlib import Path
import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')



project_name = "Camvid_segmentation"
list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_loader.py",
    f"src/{project_name}/components/models.py",
    f"src/{project_name}/components/trainer.py",
    f"src/{project_name}/components/data_injection.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/martices.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/train_pipeline.py",
    "artifacts/model_ckpt/.gitkeep",
    "fig/.gitkeep",
    "dataset/.gitkeep",
    "tests/.gitkeep",
    "config/config.yaml",
    "requirements.txt",
    "LICENSE"
    "setup.py",
    ".gitignore",
    "scripts.py",
    "README.md",
    "app.py",
    "runtime.txt",
    "Procfile",
    "Makefile",
    "notebook/trials.ipynb",
    "notebook/EDA.ipynb",
    "templates/index.html",
    "static/style.css"


]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")

