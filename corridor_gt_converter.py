import os
import shutil

# Replace 'source_directory' with the path to the folder containing several subfolders with .npy files
source_directory = '/home/galinezh/TCVAE_modified/data/corridor/Annotation'

# Replace 'destination_directory' with the path to the new directory where you want to move the .npy files
destination_directory = '/home/galinezh/TCVAE_modified/data/corridor/gt/test_frame_mask'

# Loop through the subdirectories and move the .npy files
for root, dirs, files in os.walk(source_directory):
    for file in files:
        if file.endswith('.npy'):
            source_file_path = os.path.join(root, file)
            destination_file_path = os.path.join(destination_directory, '1_'+file[3:])
            # Rename and move the file to the destination directory
            shutil.move(source_file_path, destination_file_path)

print("All .npy files have been moved and renamed to the destination directory.")


