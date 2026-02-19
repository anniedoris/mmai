from modality_helpers import *
import os

RANGE = 2000
image_inputs = [f"/orcd/data/faez/001/annie/mmai/data/images/{i}.png" for i in range(RANGE)]
cad_code_inputs = [f"/orcd/data/faez/001/annie/mmai/data/code/{i}.py" for i in range(RANGE)]

def main():
    
    # Generate synthetic text descriptions using Qwen VL 8B
    generate_object_descriptions("/orcd/data/faez/001/annie/mmai/data/images", "/orcd/data/faez/001/annie/mmai/data/code", num_generate=RANGE)
    
    # Generate synthetic image inputs
    generate_synthetic_image(image_inputs, model_type="closed_source")

    # Generate point cloud inputs
    os.makedirs("/orcd/data/faez/001/annie/mmai/data/point_clouds", exist_ok=True)
    for cad_code_input in tqdm(cad_code_inputs, desc="Processing CAD codes"):
        generate_point_cloud(cad_code_input)

if __name__ == "__main__":
    main()