import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# 1. LOAD DATA
df = pd.read_csv('training_data.csv', header=None)
X = df.values
print(f"Loaded {len(X)} samples.")

# 2. AUTOMATIC LABELING (K-Means)
# This groups your data into 4 patterns
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# 3. TRAIN THE BRAIN (MLP)
# Learning to map State -> Label
brain = MLPClassifier(
    hidden_layer_sizes=(16, 8), 
    activation='relu', 
    max_iter=1000, 
    random_state=42
)
brain.fit(X, labels)
print("Brain training complete.")

# 4. PREDICTION TEST (The verification step you requested)
# We ask the brain to predict the mindset for every row in your training data
predictions = brain.predict(X)

# Create a new dataframe: Original 7 columns + 1 Prediction column
results_df = pd.DataFrame(X)
results_df['Predicted_Mindset'] = predictions

# Save to CSV
results_df.to_csv('prediction_results.csv', index=False, header=False)
print("Verification file 'prediction_results.csv' created.")

# 5. EXPORT TO ONNX
initial_type = [('float_input', FloatTensorType([None, 7]))]
onx = convert_sklearn(brain, initial_types=initial_type)

with open("kernel_brain.onnx", "wb") as f:
    f.write(onx.SerializeToString())

print("Model 'kernel_brain.onnx' exported successfully.")

# --- ANALYTICS: What do the numbers mean? ---
print("\n--- MINDSET ANALYSIS ---")
for i in range(4):
    center = kmeans.cluster_centers_[i]
    # We identify which cluster is which by looking at the averages
    print(f"Mindset {i}:")
    print(f"  Avg CPU: {center[1]:.4f}")
    print(f"  Avg RAM: {center[3]:.4f}")
    print(f"  Avg Mouse Activity: {center[5]:.4f}")
    print(f"  Avg Input Delay: {center[6]:.4f}")
