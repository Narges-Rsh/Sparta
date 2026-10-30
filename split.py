import os 
import shutil

path = "/home/galinezh/TCVAE_modified/test_split_chad.txt"
anno_path = "/home/galinezh/TCVAE_modified/data/combined_Alphapose/"
move_dir_train = "/home/galinezh/TCVAE_modified/data/combined/pose/train/"
move_dir_test = "/home/galinezh/TCVAE_modified/data/combined/pose/test/"

with open(path, "r") as file:
    split_list = [line.strip() for line in file.readlines()]

for filename in os.listdir(anno_path):
    if filename.split('.')[0] in split_list:
        shutil.copy2(anno_path+filename, move_dir_test)
    else:
       shutil.copy2(anno_path+filename, move_dir_train) 




    