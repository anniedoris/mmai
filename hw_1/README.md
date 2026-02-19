


### Extract Modalities (20 pts)
My modality extraction pipeline first locally downloads the CAD-Coder dataset (from huggingface) I developed in my prior work (mentioned in my writeup). This dataset consists of ~147k samples of images and corresponding CAD code. To download the dataset, run:

```
python load_hf_dataset.py
```

My focus in this homework is to generate other additional modalities from this dataset. I generate synthetic photographs of the objects (using Gemini), synthetic CAD model descriptions (using Qwen), and point clouds. To generate these modalities, run:

```
python generate_modalities.py
```

Please note: I am running this code on the ORCD computing cluster to use H100s and to accommodate storage of this large dataset. Please update save paths in the code to your local/preferred directories.

### Data Visualizations (15 pts)
For the t-SNE dataset distribution plots discussed in the writeup, run:

```
python visualize_data.py --tsne
```