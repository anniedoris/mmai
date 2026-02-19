from google import genai
from google.genai import types
from PIL import Image
import random
import os
import torch
from tqdm import tqdm
from diffusers import QwenImageEditPipeline
import trimesh
import cadquery as cq
import numpy as np
from io import BytesIO
import base64
from vllm import LLM, SamplingParams

##################################################
#### Generating synthetic photographs of CAD models ####
##################################################
MATERIALS = [
    "brushed stainless steel",
    "anodized aluminum",
    "machined aluminum with visible tool marks",
    "polished chrome steel",
    "matte black powder-coated steel",
    "cast iron with rough surface texture",
    "carbon fiber composite",
    "translucent industrial plastic",
    "white injection-molded plastic",
]

SURFACES = [
    "a dark walnut wooden workbench",
    "a brushed metal laboratory table",
    "a concrete factory floor",
    "a clean white inspection table",
    "a black matte photography surface",
    "a stainless steel workshop counter",
    "a light oak desktop",
]

LIGHTING = [
    "soft diffused studio lighting",
    "bright overhead industrial lighting",
    "dramatic side lighting with strong shadows",
    "cool fluorescent workshop lighting",
    "warm ambient lighting",
    "high-contrast inspection lighting",
    "cinematic spotlight lighting",
]

PERSPECTIVE = [
    "Slightly change the camera angle.",
    "Slightly change the camera zoom.",
    "Slightly change the camera angle and zoom.",
    ""
]

def make_prompt():
    material = random.choice(MATERIALS)
    surface = random.choice(SURFACES)
    lighting = random.choice(LIGHTING)
    perspective = random.choice(PERSPECTIVE)

    return (
        f"Render this CAD model as a {material} component "
        f"resting on {surface}. Use {lighting}. "
        "Photorealistic, high detail, sharp focus. Maintain geometric fidelity to the original CAD model."
    )

def generate_synthetic_image(seed_images, save_dir="/orcd/data/faez/001/annie/mmai/data/syn_real", model_type="closed_source"):
    os.makedirs(save_dir, exist_ok=True)
    if model_type == "closed_source":
        client = genai.Client(api_key="AIzaSyDST66RVhA5-sRs0iQb7FZOxq2LF5cDQ_A")
        for seed_image in tqdm(seed_images, desc="Generating synthetic images"):
            input_image = Image.open(seed_image)
            prompt = make_prompt()
            print(f"Using prompt: {prompt}")
            response = client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[
                    prompt,
                    input_image
                ],
                config=types.GenerateContentConfig()
            )
            for part in response.parts:
                if part.inline_data:
                    image = part.as_image()
                    file_name = os.path.basename(seed_image).split(".")[0]
                    image.save(f"{save_dir}/{file_name}_gemini.png")
                    
    elif model_type == "open_source":
        pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
        print("pipeline loaded")
        pipeline.to(torch.bfloat16)
        pipeline.to("cuda")
        pipeline.set_progress_bar_config(disable=None)
        for seed_image in tqdm(seed_images, desc="Generating synthetic images"):
            image = Image.open(seed_image).convert("RGB")
            prompt = make_prompt()
            print(f"Using prompt: {prompt}")
            prompt = prompt
            inputs = {
                "image": image,
                "prompt": prompt,
                "generator": torch.manual_seed(0),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": 50,
            }

            with torch.inference_mode():
                output = pipeline(**inputs)
                output_image = output.images[0]
                file_name = os.path.basename(seed_image).split(".")[0]
                output_image.save(f"{save_dir}/{file_name}_qwen.png")
    else:
        raise ValueError("No valid model type specified")
    return

##################################################
#### Generating point clouds from the CAD ####
##################################################
# # Note: below point cloud code is adapted from Cadrille paper
def generate_point_cloud(cad_code_path, save_dir="/orcd/data/faez/001/annie/mmai/data/point_clouds", n_points=2048):
    """Load a CAD file and sample N surface points via tessellation. Same method as Cadrille."""
    code = read_py_as_string(cad_code_path)
    local_vars = {}
    exec(code, {"cq": cq, "__builtins__": __builtins__}, local_vars)
    result = local_vars['solid']
    compound = result.val()
    mesh = compound_to_mesh(compound)
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    mesh.apply_translation(-center)
    extent = np.max(mesh.extents)
    if extent > 1e-7:
        mesh.apply_scale(1.0 / extent)
    mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    pred_points, _ = trimesh.sample.sample_surface(mesh, n_points)
    pc = np.asarray(pred_points, dtype=np.float32)
    file_name = os.path.basename(cad_code_path).split(".")[0]
    np.save(f"{save_dir}/{file_name}.npy", pc)
    return cad_code_path, pred_points

def read_py_as_string(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    return content

def compound_to_mesh(compound):
    vertices, faces = compound.tessellate(0.001, 0.1)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)


##################################################
#### Generate descriptions of the CAD objects ####
##################################################
PROMPT = ("Describe the CAD object provided based on the image and the following CAD code: {} \n Keep your description concise (under 100 words) and focused on the object's geometry and function (if discernible).")

def encode_image(image: Image.Image, im_type='jpeg') -> str:
        buffered = BytesIO()
        image.save(buffered, format=im_type)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_prompt(image_path, prompt):
    # Read in the image
    image = Image.open(image_path).convert("RGB")
    # Resize image to 448x448
    image = image.resize((448, 448), Image.LANCZOS)
    prompt = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": f"data:image/{'png'};base64,{encode_image(image, im_type='png')}"}},
                                {"type": "text", "text": prompt}]
                }
    ]
    return prompt

def generate_object_descriptions(image_dir, cad_code_dir, num_generate, save_dir="/orcd/data/faez/001/annie/mmai/data/syn_descriptions"):

    os.makedirs(save_dir, exist_ok=True)
    
    chunks = []
    inds = list(range(num_generate))
    inds = [str(ind) for ind in inds]
    
    if num_generate > 500:
        # Break into groups of 500
        for i in range(0, num_generate, 500):
            chunk = inds[i:i+500]
            chunks.append(chunk)
    else:
        chunks.append(inds)

    prompts = []
    for chunk in tqdm(chunks, desc="Processing chunks"):
        mini_batch_prompts = []
        for ind in tqdm(chunk, desc="Generating prompts", leave=False):
            image_path = f"{image_dir}/{ind}.png"
            code_path = f"{cad_code_dir}/{ind}.py"
            code = read_py_as_string(code_path)
            finished_prompt = generate_prompt(image_path, PROMPT.format(code))
            mini_batch_prompts.append(finished_prompt)
        prompts.append(mini_batch_prompts)

    print(f"{torch.cuda.device_count()} GPUs detected for realism evaluation.")
    model = LLM(model="Qwen/Qwen3-VL-8B-Instruct",
                tensor_parallel_size=torch.cuda.device_count(),
                max_model_len=4096*2)

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1024)

    full_responses = []
    for chunk in prompts:
        results = model.chat(chunk, sampling_params=sampling_params)
        chunk_responses = [result.outputs[0].text for result in results]
        full_responses.append(chunk_responses)

    flattened_responses = [response for chunk in full_responses for response in chunk]
    
    for i, response in tqdm(enumerate(flattened_responses), total=len(flattened_responses), desc="Saving descriptions"):
        with open(f"{save_dir}/{i}.txt", "w") as f:
            f.write(response)

    return flattened_responses