import numpy as np 
import os
import json


directory_path = '/home/galinezh/TCVAE_modified/data/corridor/track_numpy_iitb_train'
save_dir = '/home/galinezh/TCVAE_modified/data/corridor/pose/train/'
# Get a list of all files in the directory
file_list = os.listdir(directory_path)

# Loop through the files and load .npy files
npy_data = []  # List to store loaded data
data_dict = {}
for file_name in file_list:
    if file_name.endswith('.npy'):
        file_path = os.path.join(directory_path, file_name)
        data = np.load(file_path)
        camera_id = data[:, 0][0]
        video_id = data[:, 1][0]
        video_name = str(int(camera_id))+'_'+str(int(video_id))
        if video_name in data_dict:
        # Access the list using the key and append the new element
            data_dict[video_name].append(data[:, 2:])
        else:
            # If the key doesn't exist, create a new entry with the list containing the element
            data_dict[video_name] = [data[:, 2:]]
            
for video in data_dict:
    video_data = data_dict[video]
    video_dict = {}
    for segment in video_data:
        person_dict = {}
        person_id = segment[0][76]
        for frame_data in segment:
            pose_x = frame_data[1:26]
            pose_y = frame_data[26:51]
            vis = frame_data[51:76]
            frame_num = frame_data[0]
            combined = [elem for triplet in zip(pose_x, pose_y, vis) for elem in triplet]
            d = {'keypoints':combined, 'scores': 0}
            person_dict[frame_num] = d
     
        video_dict[person_id] = person_dict
        
    with open(save_dir + video + '.json', 'w') as json_file:
        json.dump(video_dict, json_file)
        print('Fuck you!')