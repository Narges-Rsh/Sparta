import pandas as pd
import numpy as np
import os
from pathlib import Path

# --- Configuration ---
# Source directory with raw SPARTA scores
input_dir = Path("/home/nrashvan/sparta_org/Sparta/results_ShanghaiTech/csv_files/")
# Final output directory for the 4-column CSVs
output_dir = Path("/home/nrashvan/sparta_org/Sparta/results_ShanghaiTech/results_binarized_smoothed/")
# SPARTA EER Threshold
eer_threshold = 0.4118760839178665
window_size = 7

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def apply_sparta_tumbling_smoothing(series):
    """
    Applies a dynamic majority-vote filter on non-overlapping blocks.
    Convention: 0 = Anomaly, 1 = Normal.
    Logic:
    - If 1s > 0s: Block becomes [1, 1, ..., 1] (Normal)
    - If 0s > 1s: Block becomes [0, 0, ..., 0] (Anomaly)
    - If Tie: Block remains unchanged.
    """
    arr = series.values.astype(int).copy()
    n = len(arr)
    
    
    for i in range(0, n, window_size):
        block = arr[i:i+window_size]
        num_zeros = np.sum(block == 0)
        num_ones = np.sum(block == 1)
        
        if num_ones > num_zeros:
            # Clear majority Normal
            arr[i:i+window_size] = 1
        elif num_zeros > num_ones:
            # Clear majority Anomaly
            arr[i:i+window_size] = 0
        else:
            # Tie (e.g., 3 zeros and 3 ones in a block of 6 at the end)
            # Do nothing, keep original values
            pass
            
    return arr

print(f"Starting SPARTA Integrated Post-Processing...")
print(f"EER Threshold: {eer_threshold} | Window Size: 7")

for csv_file in sorted(input_dir.glob("*.csv")):
    # 1. Load the CSV
    df = pd.read_csv(csv_file, header=None, names=['GT', 'Score'])
    
    # 2. Handle Infinities and Binarization
    # We replace inf values and then apply thresholding
    cleaned_score = df['Score'].replace({float('-inf'): 0, float('inf'): 1})
    df['Binarized'] = np.where(cleaned_score <= eer_threshold, 0, 1)
    
    # 3. Apply the updated smoothing logic (same as STG-NF)
    df['Smoothed_Binarized'] = apply_sparta_tumbling_smoothing(df['Binarized'])
    
    # 4. Save to final location with 4 columns: GT, Score, Binarized, Smoothed
    output_path = output_dir / csv_file.name
    df.to_csv(output_path, index=False, header=False)

print(f"Done! SPARTA results saved to: {output_dir}")