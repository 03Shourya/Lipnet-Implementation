import os
import random
import shutil

# Define source, target, and additional folders
source_folder = '../lipnet_flask/data/s1'
target_folder = 'test_data/s1'
another_folder = '../lipnet_flask/data/alignments/s1'  # Folder where the corresponding .mpg files are found
yet_another_folder = 'test_data/alignments/s1'  # Folder to store files with corresponding extensions like .mp4

# Ensure the target folders exist, if not create them
if not os.path.exists(target_folder):
    os.makedirs(target_folder)
if not os.path.exists(yet_another_folder):
    os.makedirs(yet_another_folder)

# List all files in the source folder
files = os.listdir(source_folder)

# Filter out directories (if any)
files = [f for f in files if os.path.isfile(os.path.join(source_folder, f))]

# Select 30% of the files randomly
num_files_to_copy = int(0.30 * len(files))
selected_files = random.sample(files, num_files_to_copy)

# Copy selected files to the target folder and handle corresponding files in yet_another_folder
for file in selected_files:
    # Remove the '.mpg' extension if present
    base_name, ext = os.path.splitext(file)
    base_name = base_name.split('/')[-1]
    if ext.lower() == '.mpg':
        # Construct the file names
        source_path = os.path.join(source_folder, file)
        target_path = os.path.join(target_folder, file)
        
        # Copy the .mpg file to the target folder
        shutil.copy(source_path, target_path)
        
        # Now, look for the same base name in another folder with a different extension (e.g., .mp4)
        matching_file = base_name + '.align'  # Example: change to the appropriate extension
        another_file_path = os.path.join(another_folder, matching_file)
        
        # If the corresponding file exists in the another folder, copy it to the yet_another_folder
        if os.path.exists(another_file_path):
            shutil.copy(another_file_path, yet_another_folder)
            print(f"Copied corresponding file {matching_file} from {another_folder} to {yet_another_folder}")
        else:
            print(f"Corresponding file {matching_file} not found in {another_folder}")
    
print(f"Copied {num_files_to_copy} files to {target_folder}")
