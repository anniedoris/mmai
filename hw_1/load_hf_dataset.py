import os
from datasets import load_dataset
from tqdm import tqdm

OUT_DIR = "/orcd/data/faez/001/annie/mmai/data"

IMG_DIR = os.path.join(OUT_DIR, "images")
CODE_DIR = os.path.join(OUT_DIR, "code")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(CODE_DIR, exist_ok=True)

dataset = load_dataset(
    "CADCODER/DeepCAD-CQ-Vision-Paired",
    split=f"train"
)

# Save images and code from existing dataset locally
for i, ex in enumerate(tqdm(dataset, total=len(dataset), desc="Saving dataset")):
    img = ex["image"]   # usually PIL.Image
    img_path = os.path.join(IMG_DIR, f"{i}.png")
    img.save(img_path)

    code = ex["code"]   # adjust key if needed
    code_path = os.path.join(CODE_DIR, f"{i}.py")

    with open(code_path, "w") as f:
        f.write(code)
        
    if i % 100 == 0:
        print(f"Saved {i}")
