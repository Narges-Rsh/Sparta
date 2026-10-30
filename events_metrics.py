import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration ---
results_dir = Path("/home/nrashvan/sparta_org/Sparta/results_ShanghaiTech/results_binarized_smoothed/")
tiou_threshold = 0.5

def get_intervals(array):
    """Identifies consecutive sequences of 1s and returns (start, end) tuples."""
    padded = np.concatenate([[0], array, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts, ends))

def calculate_tiou(int_a, int_b):
    """Calculates Temporal Intersection over Union (tIoU) between two intervals."""
    s1, e1 = int_a
    s2, e2 = int_b
    intersection = max(0, min(e1, e2) - max(s1, s2) + 1)
    union = (e1 - s1 + 1) + (e2 - s2 + 1) - intersection
    return intersection / union if union > 0 else 0

# --- Global Counters ---
total_gt_events = 0
total_pred_events = 0
total_tp, total_fp, total_fn = 0, 0, 0
pred_durations_list = []

# --- Header for the Master Table ---
print(f"{'File Name':<18} | {'GT':<3} | {'Pred':<4} | {'Diff':<4} | {'TP':<3} | {'FP':<3} | {'FN':<3}")
print("-" * 65)

for csv_file in sorted(results_dir.glob('*.csv')):
    df = pd.read_csv(csv_file, header=None)
    
    # Extract ground truth (Col 0) and smoothed predictions (Col 3)
    # Assumes 0=Normal, 1=Anomaly
    gt_intervals = get_intervals(df.iloc[:, 0].values)
    pred_intervals = get_intervals(df.iloc[:, 3].values)
    
    file_tp = 0
    matched_pred_indices = set()
    matched_gt_indices = set()

    # Step 1: Matching logic (Best Match per GT event)
    for g_idx, g_int in enumerate(gt_intervals):
        best_iou = 0
        best_p_idx = -1
        
        for p_idx, p_int in enumerate(pred_intervals):
            if p_idx in matched_pred_indices:
                continue
            
            iou = calculate_tiou(g_int, p_int)
            if iou > best_iou:
                best_iou = iou
                best_p_idx = p_idx
        
        if best_iou >= tiou_threshold:
            file_tp += 1
            matched_gt_indices.add(g_idx)
            matched_pred_indices.add(best_p_idx)

    # Step 2: File-level metrics
    num_gt = len(gt_intervals)
    num_pred = len(pred_intervals)
    diff_val = num_pred - num_gt
    file_fp = num_pred - len(matched_pred_indices)
    file_fn = num_gt - len(matched_gt_indices)
    
    # Update durations for predicted events
    if num_pred > 0:
        for p_int in pred_intervals:
            pred_durations_list.append(p_int[1] - p_int[0] + 1)

    # Update Global Totals
    total_gt_events += num_gt
    total_pred_events += num_pred
    total_tp += file_tp
    total_fp += file_fp
    total_fn += file_fn

    # Print Row
    print(f"{csv_file.name:<18} | {num_gt:<3} | {num_pred:<4} | {diff_val:+4d} | {file_tp:<3} | {file_fp:<3} | {file_fn:<3}")

# --- Final Metrics Calculation ---
precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
avg_dur = np.mean(pred_durations_list) if pred_durations_list else 0

# --- Print Final Summary ---
print("\n" + "="*50)
print(f"FINAL INTEGRATED SUMMARY (tIoU Threshold: {tiou_threshold})")
print("="*50)
print(f"Total Actual (GT) Events:      {total_gt_events}")
print(f"Total Predicted Events:        {total_pred_events}")
print(f"Total True Positives (TP):     {total_tp}")
print(f"Total False Positives (FP):    {total_fp}")
print(f"Total False Negatives (FN):    {total_fn}")
print("-" * 50)
print(f"Precision:                     {precision:.4f}")
print(f"Recall:                        {recall:.4f}")
print(f"F1-Score:                      {f1:.4f}")
print(f"Avg Predicted Duration:        {avg_dur:.2f} frames")
print("="*50)