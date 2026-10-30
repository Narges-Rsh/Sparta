import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
results_dir = Path("/home/nrashvan/sparta_org/Sparta/results_ShanghaiTech/results_binarized_smoothed/")
# Change this to the specific CSV file name you want to see
video_file = "10_0075.csv" 
eer_threshold = 0.4118760839178665

# 1. Load the data
csv_path = results_dir / video_file
if not csv_path.exists():
    print(f"File {video_file} not found. Please check the path.")
else:
    # Col 0: GT, Col 1: Raw Score, Col 2: Binarized, Col 3: Smoothed
    df = pd.read_csv(csv_path, header=None)
    
    # 2. Preprocess: Handle infinities as requested
    # -inf becomes 0, +inf becomes 1
    scores = df.iloc[:, 1].replace({float('-inf'): 0, float('inf'): 1}).values
    gt = df.iloc[:, 0].values
    frames = np.arange(len(scores))

    # 3. Create the Plot
    plt.figure(figsize=(15, 6))
    
    # Plot the Anomaly Score
    plt.plot(frames, scores, label='Anomaly Score (Model)', color='#1f77b4', linewidth=1.5)
    
    # Plot the EER Threshold line
    plt.axhline(y=eer_threshold, color='green', linestyle='--', alpha=0.7, label=f'EER Threshold ({eer_threshold:.4f})')
    
    # Shade the Ground Truth Anomaly regions (where GT == 1)
    # This highlights the "Actual" events in red
    plt.fill_between(frames, 0, max(scores.max(), 1), where=(gt == 1), 
                     color='red', alpha=0.2, label='Actual Anomaly (GT)')
    
    # Optional: Highlight where the model actually predicted an anomaly (Smoothed Binarized)
    pred_smoothed = df.iloc[:, 3].values
    plt.fill_between(frames, 0, 0.05, where=(pred_smoothed == 1), 
                     color='blue', alpha=0.5, label='Model Prediction (Smoothed)')

    # Formatting the Plot
    plt.title(f'Anomaly Detection Analysis: {video_file}', fontsize=14)
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, max(scores.max(), 1.0) * 1.1)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'plot_{video_file.split(".")[0]}.png')
    print(f"Success! Plot saved as plot_{video_file.split('.')[0]}.png")