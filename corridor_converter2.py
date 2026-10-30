import numpy as np 
import os
import json


directory_path = '/home/galinezh/TCVAE_modified/data/avenue_raw/Avenue_extracted/avenue_testing_videos/'
save_dir = '/home/galinezh/TCVAE_modified/data/Avenue/pose/test/'
# Get a list of all files in the directory
file_list = os.listdir(directory_path)

# Loop through the files and load .npy files
npy_data = []  # List to store loaded data
data_dict = {}
for file_name in file_list:
    if file_name.endswith('.json'):
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        video_dict = {}
        ids = sorted(set([d['idx'] for d in data ]))
        for id in ids:
            person_dict = {}
            for frame_data in data:
                if frame_data ['idx'] == id:
                    frame_dict = {}
                    frame = int(frame_data['image_id'].split('.')[0])
                    frame_dict['keypoints'] = frame_data['keypoints']
                    frame_dict['scores'] = frame_data['score']
                    person_dict[frame] = frame_dict 
            video_dict[id] = person_dict
    
    
    with open(save_dir + file_name , 'w') as json_file:
        json.dump(video_dict, json_file)
        print('Fuck you!')