import os
from shutil import copyfile

file_name_txt = 'file_name.txt'
target_folder = 'dataset/'
source_folder = 'target/' # source images store in folder name target :)

with open(file_name_txt, mode='r') as f:
    data = f.readlines()

"""
replace png = JPG
"""

for line in data:
    line = line.rstrip("\n") # move newlines in image name
    source_path = os.path.join(source_folder, line.replace("png", "JPG"))
    target_path = os.path.join(target_folder, line.replace("png", "JPG"))
    copyfile(source_path, target_path)
    print("copy file from {} to {}".format(source_path, target_path))