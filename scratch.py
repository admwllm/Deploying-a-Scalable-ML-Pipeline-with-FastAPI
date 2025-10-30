import numpy as np
import pandas as pd
import os

project_path = os.getcwd()
precision_path = os.path.join(project_path, "data", "model_precision.npy")

p = np.load(precision_path)
print(f"Model precision: {precision_path}: {p:.4f}")