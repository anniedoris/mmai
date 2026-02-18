from datasets import load_dataset
import Pillow

NUM_ROWS = 1000

dataset = load_dataset("CADCODER/DeepCAD-CQ-Vision-Paired", split=f"train[:{NUM_ROWS}]")

for data_point in dataset:
    print(data_point)