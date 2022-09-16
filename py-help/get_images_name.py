import os

path_fol_label = 'coffe_leaves_labels/'
store_img_name = 'file_name.txt'
list_img_name = []


"""
img: IMG_3036.png
label: IMG_3036_gt.png
"""

for name in os.listdir(path_fol_label):
    name = name.replace("_gt", "")
    list_img_name.append(name)

with open(store_img_name, mode='w') as f:
    for name in list_img_name:
        f.writelines(name + '\n')

f.close()
# print(len(list_img_name))
# print(list_img_name)
