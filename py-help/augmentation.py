import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import glob2
import cv2
from pathlib import Path
import numpy as np



imgs_path = glob2.glob('images/*.jpg')
labels_path = glob2.glob('seg_labels/*.png')

des_aug_img = 'aug_imgs/'
des_aug_label = 'aug_labels/'

nums_aug = len(imgs_path) # number images

imgs_path.sort()
labels_path.sort()

# Define augmentation pipline

sometimes = lambda aug: iaa.Sometimes(0.5, aug)


seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(
            rotate=(-90, 90),
            shear=(-8, 8),
            scale={"x": (0.8, 1.2), "y":(0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}        
        ),
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        sometimes(iaa.Crop(percent=(0, 0.1)))
    ], random_order=True
)

for i in range(nums_aug):

    img = cv2.imread(imgs_path[i])
    label = cv2.imread(labels_path[i], 0)
    h, w = img.shape[:2]
    img_name = imgs_path[i].split('/')[1][:-4]
    seg_map = SegmentationMapsOnImage(label, shape=(h, w))

    for iter in range(5):

        img_aug_iter, segmap_aug_iter = seq(image=img, segmentation_maps=seg_map)
        mask = segmap_aug_iter.draw(size=(h, w))[0]

        new_img_aug_name = img_name + "_aug_" + str(iter) + ".png"
        new_label_aug_name = img_name + "_aug_label_" + str(iter) + ".png"
 
        new_path_img_aug = Path(des_aug_img, new_img_aug_name)
        new_path_label_aug = Path(des_aug_label, new_label_aug_name)

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        mask = np.select([mask==63, mask==134, mask == 135], [1, 2, 2], mask)

        cv2.imwrite(str(new_path_img_aug), img_aug_iter)
        cv2.imwrite(str(new_path_label_aug), mask)


