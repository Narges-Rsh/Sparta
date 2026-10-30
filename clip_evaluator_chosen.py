import os
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve,auc
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import rc
import cv2
import csv

seg_len = 30
sigma = 7
plot_eer = False 
target_fnr = 0.1



# specify the directory path
path1 = "/home/galinezh/TCVAE_modified/results_recon_CPCC/CPCC_2/train4/scores"
video_path = '/home/galinezh/TCVAE_modified/data/CPCC_2/test_clips'
save_path = '/home/galinezh/TCVAE_modified/results_recon_CPCC/chosen_clips/c2/'

# save_dir = '/home/galinezh/anomaly_baseline/combined_results_final'
# order_path = "/home/galinezh/TCVAE_modified/results/shang_score_pose_no_vis/order.txt"
# initialize an empty dataframe



# with open(order_path, "r") as file:
#     # Read the lines of the file into a list
#     order = file.readlines()
clip_list = os.listdir(path1)
clip_list = sorted(fn for fn in clip_list if fn.endswith('.csv'))   
pred_list = []
recon_list = []
result = pd.DataFrame()
    
# loop through all files in the directory
for file in clip_list:
    file = file.split('.')[0] + '.csv'
    name = file.split('.')[0]
    id = (int(name.split('_')[0]), int(name.split('_')[1]))
    # read the csv file
    df1 = pd.read_csv(os.path.join(path1, file), header=None)

    df1 = df1.rename(columns={0: 'gt', 1: 'score'})

    pred_list.append(np.array(df1['score'].tolist()))
    result = result.append(df1, ignore_index=True)



pred_np = np.concatenate(pred_list, axis=0)
pred_np[pred_np == np.inf] = pred_np[pred_np != np.inf].max()
pred_np[pred_np == -1 * np.inf] = pred_np[pred_np != -1 * np.inf].min()
index = 0
for score in range(len(pred_list)):
        for t in range(pred_list[score].shape[0]):
            pred_list[score][t] = pred_np[index]
            index += 1
for s in range(len(pred_list)):
        for sig in range(1, sigma):
            pred_list[s] = gaussian_filter1d(pred_list[s], sigma=sig)
pred_scores_np = np.concatenate(pred_list)


count_list = []
counter = 0 
for clip in clip_list:
    
    clip_name = clip.split('.')[0] + '.mp4'
    clip_path = os.path.join(video_path, clip_name)
    cap = cv2.VideoCapture(clip_path)
    
    # Get the frames per second (fps) and frame count
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if clip != '2_259.csv':
        counter = counter + frame_count
        continue
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # You can choose the codec based on your needs
    out = cv2.VideoWriter(os.path.join(save_path, clip.split('.')[0] + '.mp4'), fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    count_list.append(frame_count)
    
    # Process each frame
    for i in range(frame_count):
        ret, frame = cap.read()

        # Check if the frame is successfully read
        if not ret:
            break
        
        if pred_scores_np[counter] >= 0.9:
        # Write the value on each frame
            cv2.putText(frame, "Anomaly Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        else:
            cv2.putText(frame, "Normal Frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        counter = counter + 1
            # Write the frame to the output video
        out.write(frame)
            

    # Release the video capture and writer objects
    cap.release()
    out.release()

exit()

n = 0 
roc = []
for idx, clip_name in zip(count_list, clip_list):
    if 1 in (result['gt'][n:n+idx]).values and 0 in (result['gt'][n:n+idx]).values:
        auc_roc_pred  = roc_auc_score(result['gt'][n:n+idx],pred_scores_np[n:n+idx])
        roc.append([auc_roc_pred, clip_name])
        with open(os.path.join(save_path, 'clip_roc.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(roc)
    n = n + idx
    




