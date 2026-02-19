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
    perplexity = 1, num_iterations = 250, savename = "data_distribution.html", hover_text = None, color_by_label = False):
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
        
    fig.write_html(savename)
    return
