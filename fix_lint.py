import os

replacements = {
    "main.py": [("== True", "is True")],
    "scratch/verify_inference.py": [("== True", "is True")],
    "src/data_preparation/dataset.py": [("== True", "is True")],
    "src/model_training/train.py": [('f"\\n--- Epoch', '"\\n--- Epoch'), ('f"[*] Successfully', '"[*] Successfully')]
}

for filepath, reps in replacements.items():
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        for old, new in reps:
            content = content.replace(old, new)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Fixed {filepath}")
