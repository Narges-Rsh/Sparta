import os
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve,auc
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import rc

seg_len = 24
sigma = 7
plot_eer = False 
target_fnr = 0.1
pred_auc = 50
recon_auc = 50
SHANG_HR = False

SHANGHAITECH_HR_SKIP = [(1, 130), (1, 135), (1, 136), (6, 144), (6, 145), (12, 152)]


# specify the directory path
path1 = "/home/galinezh/galinezh/TCVAE_modified/PoseWatch_results_NWPUC/pred/best_scores2"
path2 = "/home/galinezh/galinezh/TCVAE_modified/PoseWatch_results_NWPUC/recon/best_scores"
# save_dir = '/home/galinezh/anomaly_baseline/combined_results_final'
# order_path = "/home/galinezh/TCVAE_modified/results/shang_score_pose_no_vis/order.txt"
# initialize an empty dataframe
result = pd.DataFrame()


# with open(order_path, "r") as file:
#     # Read the lines of the file into a list
#     order = file.readlines()
clip_list = os.listdir(path1)
clip_list = sorted(fn for fn in clip_list if fn.endswith('.csv'))   
pred_list = []
recon_list = []
    
# loop through all files in the directory
for file in clip_list:
    file = file.split('.')[0] + '.csv'
    name = file.split('.')[0]
    id = (int(name.split('_')[0]), int(name.split('_')[1]))
    if SHANG_HR:
        if id in SHANGHAITECH_HR_SKIP:
            continue
    # read the csv file
    df1 = pd.read_csv(os.path.join(path1, file), header=None)
    df2 = pd.read_csv(os.path.join(path2, file), header=None)
    df1 = df1.rename(columns={0: 'gt', 1: 'pred_score'})
    df2 = df2.rename(columns={0: 'gt', 1: 'recon_score'})
    pred_list.append(np.array(df1['pred_score'].tolist()))
    recon_list.append(np.array(df2['recon_score'].tolist()))
    result = result.append(pd.concat([df1, df2['recon_score']], axis=1), ignore_index=True)

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


print ("*********************Combined************************")


combined_score = (pred_auc/(pred_auc+recon_auc))*pred_scaled + (recon_auc/(pred_auc+recon_auc))*recon_scaled
# combined_score = (pred_auc/(pred_auc+recon_auc))*pred_scaled + (recon_auc/(pred_auc+recon_auc))*recon_scaled

auc_roc = roc_auc_score(result['gt'],combined_score)

precision, recall, thresholds = precision_recall_curve(result['gt'],combined_score)

auc_pr = auc(recall, precision)

fpr, tpr, threshold = roc_curve(result['gt'],combined_score, pos_label=1)
fnr = 1 - tpr
eer_th = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print('AUC ROC: {}'.format(auc_roc))
print('AUC PR: {}'.format(auc_pr))
print('EER: {}'.format(eer))
print('EER TH: {}'.format(eer_th))

idx_closest_fnr = np.argmin(np.abs(fnr - target_fnr))

# Get the corresponding threshold and FPR values
threshold_at_target_fnr = threshold[idx_closest_fnr]
fpr_at_target_fnr = fpr [idx_closest_fnr]
    
print('10ER: {}'.format(fpr_at_target_fnr))
print('10ER TH: {}'.format(threshold_at_target_fnr))

from matplotlib.font_manager import FontProperties

if plot_eer:
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman'] + rcParams['font.serif']
    csfont = {'fontname':'Times New Roman'}
    
    custom_font = FontProperties(family='serif', style='normal', size=12, weight='bold')
    # plt.plot(threshold, fpr, label='FPR', color='#3BBA88')
    # plt.plot(threshold, fnr, label='FNR', color='#408DB8')
    # plt.plot([eer_th, eer_th], [0, 1], 'k--', label=f'EER ({eer:.3f})', linewidth=2, color='#FF9053')
    # plt.plot([threshold_at_target_fnr, threshold_at_target_fnr], [0, 1], 'k--', label=f'10ER ({fpr_at_target_fnr:.3f})', linewidth=2, color='#D64973')
    # plt.xlabel('Threshold' ,  fontproperties=custom_font)
    # plt.ylabel('Error Rate',  fontproperties=custom_font)
    # plt.title('PoseWatch-H Error Rate on CHAD',  fontproperties=custom_font)
    plt.rcParams['pdf.fonttype'] = 42
    # # plt.xticks(fontname='Times New Roman', fontsize=10)
    # # plt.yticks(fontname='Times New Roman', fontsize=10)
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('CHAD_transformer2.png')
    # plt.savefig('CHAD_EER.pdf', format='pdf')
    
    fig, ax = plt.subplots(figsize=(6, 2.5))  # Width: 8 inches, Height: 6 inches

    # Plot data
    ax.plot(threshold[threshold<4], fpr[threshold<4], label='FPR', color='#3BBA88')
    ax.plot(threshold[threshold<4], fnr[threshold<4], label='FNR', color='#408DB8')
    ax.plot([eer_th, eer_th], [0, 1], 'k--', label=f'EER ({eer:.3f})', linewidth=2, color='#FF9053')
    # ax.plot([threshold_at_target_fnr, threshold_at_target_fnr], [0, 1], 'k--', label=f'10ER ({fpr_at_target_fnr:.3f})', linewidth=2, color='#D64973')

    # Customize axis labels and title
    ax.set_xlabel('Threshold', fontproperties=custom_font)
    ax.set_ylabel('Error Rate', fontproperties=custom_font)
    ax.set_title('PoseWatch-H Error Rate on HR-SHT', fontproperties=FontProperties(family='serif', style='normal', size=14, weight='bold'))
    
    ax.tick_params(axis='both', which='minor', labelsize=12, width=1, length=5)  # Adjust labelsize, width, and length as needed
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(FontProperties(family='serif', style='normal', size=10, weight='bold'))
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(FontProperties(family='serif', style='normal', size=10, weight='bold'))
    legend = ax.legend()
    for label in legend.get_texts():
        label.set_fontproperties(FontProperties(family='serif', style='normal', size=10, weight='bold'))
    # Add grids
    ax.grid(True)

    # Save the plot as a PDF
    # plt.tight_layout()  # Ensures proper spacing
    plt.savefig('error_rate_plot_shanghr_PST.png')
    plt.savefig('error_rate_plot_shanghr_PST.pdf')
        
    

print ("*********************Reconstruction************************")
auc_roc_recon  = roc_auc_score(result['gt'], recon_scores_np)

precision, recall, thresholds = precision_recall_curve(result['gt'], recon_scores_np)

auc_pr_recon = auc(recall, precision)

fpr, tpr, threshold = roc_curve(result['gt'], recon_scores_np, pos_label=1)
fnr = 1 - tpr
eer_th_recon = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
eer_recon = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print('AUC ROC: {}'.format(auc_roc_recon))
print('AUC PR: {}'.format(auc_pr_recon))
print('EER: {}'.format(eer_recon))
print('EER TH: {}'.format(eer_th_recon))

idx_closest_fnr = np.argmin(np.abs(fnr - target_fnr))

# Get the corresponding threshold and FPR values
threshold_at_target_fnr = threshold[idx_closest_fnr]
fpr_at_target_fnr = fpr [idx_closest_fnr]
print('10ER: {}'.format(fpr_at_target_fnr))
print('10ER TH: {}'.format(threshold_at_target_fnr))

print ("*********************Prediction************************")
auc_roc_pred  = roc_auc_score(result['gt'],pred_scores_np)

precision, recall, thresholds = precision_recall_curve(result['gt'], pred_scores_np)

auc_pr_pred = auc(recall, precision)

fpr, tpr, threshold = roc_curve(result['gt'],pred_scores_np, pos_label=1)
fnr = 1 - tpr
eer_th_pred = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
eer_pred = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print('AUC ROC: {}'.format(auc_roc_pred))
print('AUC PR: {}'.format(auc_pr_pred))
print('EER: {}'.format(eer_pred))
print('EER TH: {}'.format(eer_th_pred))

idx_closest_fnr = np.argmin(np.abs(fnr - target_fnr))

# Get the corresponding threshold and FPR values
threshold_at_target_fnr = threshold[idx_closest_fnr]
fpr_at_target_fnr = fpr [idx_closest_fnr]
print('10ER: {}'.format(fpr_at_target_fnr))
print('10ER TH: {}'.format(threshold_at_target_fnr))
print("Done!")