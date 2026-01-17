import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# 1. Load your improved data
# Columns: Procs, CPU, Wait, RAM, Frag, Mouse, Latency
df = pd.read_csv('training_data_augmented.csv', header=None)

# 2. Use K-Means to "Discover" the 4 Kernel Mindsets
# This groups similar system behaviors together
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(df)

# Let's see what the AI "found" (Printing cluster centers)
print("Clusters Discovered (Centers):")
centers = kmeans.cluster_centers_
for i, center in enumerate(centers):
    print(f"Mindset {i}: {center}")

# 3. Train the Brain (The Classifier)
# It learns to look at a state and predict the Cluster ID
clf = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=42)
clf.fit(df, clusters)

# 4. Export to ONNX for Rust
initial_type = [('float_input', FloatTensorType([None, 7]))]
onx = convert_sklearn(clf, initial_types=initial_type)

with open("kernel_brain.onnx", "wb") as f:
    f.write(onx.SerializeToString())

print("\nSuccess! kernel_brain.onnx has been generated.")
