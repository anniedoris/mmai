import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go

def get_clip_embeddings(list_of_image_paths, model_name = "openai/clip-vit-large-patch14-336", batch_size = 32):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    
    all_embeddings = []
    
    with torch.inference_mode():
        for i in tqdm(range(0, len(list_of_image_paths), batch_size)):
            
            batch_paths = list_of_image_paths[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            features = model.get_image_features(**inputs)
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8) # normalize embeddings and each feature has dimension 768
            
            print("Features shape:")
            print(features.shape)

            all_embeddings.append(features.cpu())

    embs = torch.cat(all_embeddings, dim=0).numpy()  # (N=2000, D=768)
    return embs

def visualize_data_distribution(data, labels = None, x_feature = "t-SNE 1", y_feature = "t-SNE 2", num_components = 2,
    perplexity = 1, num_iterations = 250, savename = "visualizations/data_distribution.html", hover_text = None, color_by_label = False, highlight_indices = None):
    """
    Visualizes the distribution of a specified feature in a DataFrame.

    Args:
        data (np.array): The dataset that you plan on using as a 2D representation.
    """
    tsne = TSNE(n_components = num_components, perplexity = perplexity, max_iter = num_iterations)
    tsne_data = tsne.fit_transform(data)


    if hover_text is None:
        hover_text = labels if labels is not None else None

    if color_by_label and labels is not None:
        fig = px.scatter(
            x=tsne_data[:, 0],
            y=tsne_data[:, 1],
            color=labels.astype(str),
            hover_name=hover_text,
            title="t-SNE Visualization"
        )
    else:
        fig = px.scatter(
            x=tsne_data[:, 0],
            y=tsne_data[:, 1],
            hover_name=hover_text,
            title="t-SNE Visualization"
        )

        # Force single color
        fig.update_traces(marker=dict(color="blue"))
        
    # Add highlighted points
    if highlight_indices is not None:
        highlight_indices = np.atleast_1d(highlight_indices)

        fig.add_trace(
            go.Scatter(
                x=tsne_data[highlight_indices, 0],
                y=tsne_data[highlight_indices, 1],
                mode="markers",
                text= [highlight_indices.astype(str)],
                marker=dict(
                    color='red',
                    size=16,
                    line=dict(width=1, color="black")
                ),
                name="Highlighted",
                showlegend=False
            )
        )
        
    fig.write_html(savename)
    return

def visualize_grayscale_image(sample_num, dir="/orcd/data/faez/001/annie/mmai/data/images"):
    img_path = f"{dir}/{sample_num}.png"
    img = Image.open(img_path)
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray')
    plt.axis('off') # remove axes
    plt.tight_layout()
    plt.savefig('visualizations/grayscale_image.png', dpi=600)
    plt.close()
    return
    
def visualize_photorealistic_image(sample_num, dir="/orcd/data/faez/001/annie/mmai/data/syn_real"):
    img_path = f"{dir}/{sample_num}_gemini.png"
    img = Image.open(img_path)
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray')
    plt.axis('off') # remove axes
    plt.tight_layout()
    plt.savefig('visualizations/photorealistic_image.png', dpi=600)
    plt.close()
    return
    
def visualize_point_cloud(sample_num, dir="/orcd/data/faez/001/annie/mmai/data/point_clouds"):
    points = np.load(f"{dir}/{sample_num}.npy")
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:,0],
        y=points[:,1],
        z=points[:,2],
        mode='markers',
        marker=dict(size=2)
    )])

    fig.update_layout(
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    fig.write_html(f"visualizations/point_cloud.html")
    return

def visualize_code(sample_num, dir="/orcd/data/faez/001/annie/mmai/data/code"):
    with open(f"{dir}/{sample_num}.py", "r") as f:
        code = f.read()
    print("-" * 50)
    print(f"Code for sample number {sample_num}:")
    print("-" * 50)
    print(code)
    return

def visualize_description(sample_num, dir="/orcd/data/faez/001/annie/mmai/data/syn_descriptions"):
    with open(f"{dir}/{sample_num}.txt", "r") as f:
        description = f.read()
    
    print("-" * 50)
    print(f"Description for sample number {sample_num}:")
    print("-" * 50)
    print(description)
    return

def get_command_counts(num_samples, dir="/orcd/data/faez/001/annie/mmai/data/code"):
    extrude_counts = []
    arc_counts = []
    circle_counts = []
    line_counts = []
    
    for ind in range(num_samples):
        with open(f"{dir}/{ind}.py", "r") as f:
            code = f.read()
        extrude_count = code.count("extrude")
        extrude_counts.append(extrude_count)
        
        circle_count = 0
        if "circle" in code:
            circle_count = code.count("circle")
            
        line_count = 0
        if "line" in code:
            line_count = code.count("lineTo")
            
        circle_counts.append(circle_count)
        line_counts.append(line_count)
    
    plt.figure(figsize=(8, 6))
    sns.histplot(extrude_counts, bins=20, kde=False, discrete=True)
    plt.title("Distribution of Extrude Commands per Sample")
    plt.xlabel("Number of Extrude per Sample")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig('visualizations/extrude_count_distribution.png', dpi=600)
    plt.close()
    
    plt.figure(figsize=(8, 6))
    sns.histplot(circle_counts, bins=20, kde=False, discrete=True)
    total_circles = sum(circle_counts)
    plt.title(f"Distribution of Circle Commands per Sample\nTotal Circle Commands: {total_circles}")
    plt.xlabel("Number of Circles per Sample")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig('visualizations/circle_count_distribution.png', dpi=600)
    plt.close()
    
    plt.figure(figsize=(8, 6))
    sns.histplot(line_counts, bins=20, kde=False, discrete=True)
    total_lines = sum(line_counts)
    plt.title(f"Distribution of Line Commands per Sample\nTotal Line Commands: {total_lines}")
    plt.xlabel("Number of Lines per Sample")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig('visualizations/line_count_distribution.png', dpi=600)
    plt.close()
    return