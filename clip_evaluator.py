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
target_fnr = 0.1
pred_auc = 66.61
recon_auc = 66.12
save_clips = True
plot_fig = False


# specify the directory path
path_recon = "/home/galinezh/galinezh/TCVAE_modified/results_recon_tokenizations/no_kalman/PST/CHAD/scores"
path_pred = "/home/galinezh/galinezh/TCVAE_modified/results_pred_tokenizations_biga/results_pred_tokenizations/Frozen_Recon_Encoder/PST/CHAD/scores_batch1"
video_path = '/home/galinezh/galinezh/TCVAE_modified/data/combined/test_videos'
save_path = '/home/galinezh/galinezh/TCVAE_modified/results_recon_tokenizations/no_kalman/PST/CHAD/clips'
figure_path = '/home/galinezh/galinezh/TCVAE_modified/results_recon_tokenizations/no_kalman/PST/CHAD/plots'

# save_dir = '/home/galinezh/anomaly_baseline/combined_results_final'
# order_path = "/home/galinezh/TCVAE_modified/results/shang_score_pose_no_vis/order.txt"
# initialize an empty dataframe



# with open(order_path, "r") as file:
#     # Read the lines of the file into a list
#     order = file.readlines()
clip_list = os.listdir(path_recon)
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
    df1 = pd.read_csv(os.path.join(path_recon, file), header=None)
    df2 = pd.read_csv(os.path.join(path_pred, file), header=None)
    
    df1 = df1.rename(columns={0: 'gt', 1: 'recon_score'})
    df2 = df2.rename(columns={0: 'gt', 1: 'pred_score'})
    
    recon_list.append(np.array(df1['recon_score'].tolist()))
    pred_list.append(np.array(df2['pred_score'].tolist()))
    result = result.append(pd.concat([df1, df2['pred_score']], axis=1), ignore_index=True)


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


recon_np = np.concatenate(recon_list, axis=0)
recon_np[recon_np == np.inf] = recon_np[recon_np != np.inf].max()
recon_np[recon_np == -1 * np.inf] = recon_np[recon_np != -1 * np.inf].min()
index = 0
for score in range(len(recon_list)):
        for t in range(recon_list[score].shape[0]):
            recon_list[score][t] = recon_np[index]
            index += 1
for s in range(len(recon_list)):
        for sig in range(1, sigma):
            recon_list[s] = gaussian_filter1d(recon_list[s], sigma=sig)
recon_scores_np = np.concatenate(recon_list)
scaler1 = StandardScaler()
scaler2 = StandardScaler()

pred_scaled = scaler1.fit_transform(pred_scores_np.reshape(-1, 1))
recon_scaled = scaler2.fit_transform(recon_scores_np.reshape(-1, 1))


count_list = []
counter = 0 


for clip in clip_list:
    clip_name = clip.split('.')[0] + '.mp4'
    clip_path = os.path.join(video_path, clip_name)
    cap = cv2.VideoCapture(clip_path)
    
    # Get the frames per second (fps) and frame count
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # You can choose the codec based on your needs
    out = cv2.VideoWriter(os.path.join(save_path, clip.split('.')[0] + '.mp4'), fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    count_list.append(frame_count)
    
    # Process each frame
    for i in range(frame_count):
        ret, frame = cap.read()

        # Check if the frame is successfully read
        if not ret:
            break
        
        combined_score = (pred_auc/(pred_auc+recon_auc))*pred_scaled[counter] + (recon_auc/(pred_auc+recon_auc))*recon_scaled[counter]
        # Write the value on each frame
        cv2.putText(frame, f"Score: {combined_score[0]:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        counter = counter + 1
        # Write the frame to the output video
        if save_clips:
            out.write(frame)

    # Release the video capture and writer objects
    cap.release()
    out.release()

n = 0 
roc = []
for idx, clip_name in zip(count_list, clip_list):
    if 1 in (result['gt'][n:n+idx]).values and 0 in (result['gt'][n:n+idx]).values:
        combined_score_np = (pred_auc/(pred_auc+recon_auc))*pred_scaled + (recon_auc/(pred_auc+recon_auc))*recon_scaled
        auc_roc_pred  = roc_auc_score(result['gt'][n:n+idx],combined_score_np[n:n+idx])
        roc.append([auc_roc_pred, clip_name])
        if plot_fig:
            figure_name = clip_name.replace('.csv', '.pdf')
            figure_file = f"{figure_path}/{figure_name}"
            frame_index = np.arange(0, idx)
            anomaly_scores = combined_score_np[n:n+idx]
            ground_truth =  np.array(result['gt'][n:n+idx])
            plt.figure(figsize=(12, 4))
            plt.tight_layout()
            plt.plot(frame_index, anomaly_scores, label='Anomaly Score')
            plt.xticks(np.arange(0, idx, step=50))
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.xlabel('Frame Index')
            plt.ylabel('Anomaly Score')
            plt.legend()
            start_index = 0
            end = 0
            i = 0
            while i<len(ground_truth):
                if ground_truth[i] == 1:
                    start_index = i
                    for j in range (i+1, len(ground_truth)):
                        if ground_truth[j] == 0 or j == len(ground_truth)-1:
                            if j == len(ground_truth)-1:
                                end = j
                            else:
                                end = j-1
                            i = j+1
                            break
                    plt.axvspan(start_index, end, color='pink', alpha=0.5)
                else:
                    i = i+1
                
                # Save the figure as a PDF
            plt.savefig(figure_file, format='pdf')

            # Close the plot to free memory
            plt.close()
                            
            
        with open(os.path.join(save_path, 'clip_roc_combined.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(roc)
    n = n + idx
    