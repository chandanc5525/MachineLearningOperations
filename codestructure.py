# ----------------------------------
# Project Structure Creation Script
# ----------------------------------



import os

# Define folders and files
structure = {
    "config": ["__init__.py", "config.yaml"],
    "data/raw": [],
    "data/processed": [],
    "data/external": [],
    "notebooks": ["01_data_exploration.ipynb", "02_feature_engineering.ipynb"],
    "src/data": ["__init__.py", "data_ingestion.py"],
    "src/features": ["__init__.py", "feature_engineering.py"],
    "src/models": ["__init__.py", "model_train.py"],
    "src/evaluation": ["__init__.py", "evaluate_model.py"],
    "src/pipeline": ["__init__.py", "pipeline.py"],
    "artifacts": ["__init__.py"],
    "artifacts/model": ["__init__.py"],
    "artifacts/logs": ["__init__.py"]
}

# Create folders and files
for folder, files in structure.items():
    os.makedirs(folder, exist_ok=True)
    for file in files:
        path = os.path.join(folder, file)
        if not os.path.exists(path):
            with open(path, "w") as f:
                if file.endswith(".py"):
                    f.write("# This is a Python file\n")
                else:
                    f.write("")  # empty file for yaml or notebook

print("Project structure created successfully!")
