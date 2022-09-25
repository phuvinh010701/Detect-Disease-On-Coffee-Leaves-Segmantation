import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img, to_categorical
import numpy as np
import glob2
from tensorflow.keras.utils import Sequence

def parse():
    parser = argparse.ArgumentParser(description="Unet semantic segmantation detectection")
    parser.add_argument("--img_path", type=str, help="path to images")
    parser.add_argument("--label_path", type=str, help="path to labels")
    parser.add_argument("--model", type=str, help="path to h5 model")
    parser.add_argument("--batch_size", type=int, help="num image evalute in one batch")
    parser.add_argument("--shape", type=int, help="input shape")
    return parser.parse_args()

class DataGenerator(Sequence):
    def __init__(self, all_filenames, labels, batch_size, input_dim, n_channels, shuffle=True):
        self.all_filenames = all_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.all_filenames) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        all_filenames_temp = [self.all_filenames[k] for k in indexes]
        X, Y = self.__data_generation(all_filenames_temp)

        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.all_filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, all_filenames_temp):
      
        X = np.empty((self.batch_size, *self.input_dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.input_dim, self.n_channels))
        
        for i, (fn, label_fn) in enumerate(all_filenames_temp):
            img = load_img(fn, target_size=self.input_dim)
            label = load_img(label_fn, target_size=self.input_dim, color_mode='grayscale')

            img = img_to_array(img)
            label = img_to_array(label)

            img /= 255.0
            label = to_categorical(label, num_classes=3)

            X[i, ]= img
            Y[i, ] = label
        return X, Y


def main():
    args = parse()
    u = load_model(args.model)
    img_path = glob2.glob(args.img_path + '*')
    label_path = glob2.glob(args.label_path + '*')
    val_generator = DataGenerator(
        all_filenames = list(zip(img_path, label_path)),
        labels = label_path,
        batch_size = args.batch_size,
        input_dim = (args.shape, args.shape),
        n_channels = 3,
        shuffle = True,
    ) 
    u.evaluate(val_generator)

if __name__ == "__main__":
	main()