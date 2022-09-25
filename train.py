from model import *
import argparse
import glob2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import img_to_array, load_img, to_categorical
from sklearn.model_selection import train_test_split

def parser():
    parser = argparse.ArgumentParser(description="Unet semantic segmantation")
    parser.add_argument("--img_path", type=str, 
                        help="path to folder contain images")
    parser.add_argument("--label_path", type=str, 
                    help="path to folder contain label")                    
    parser.add_argument("--shape", type=int, default=256)
    parser.add_argument("--batch_size", default=16, type=int,
                        help="number of images to be processed at the same time")
    return parser.parse_args()

def check_arguments_errors(args):
    if not os.path.exists(args.img_path):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.label_path):
        raise(ValueError("Invalid label path {}".format(os.path.abspath(args.weights))))

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
    args = parser()
    check_arguments_errors(args)
    img_path = glob2.glob(args.img_path + '*')
    label_path = glob2.glob(args.label_path + '*')

    img_path.sort()
    label_path.sort()
    train_img_paths, val_img_paths, train_label_paths, val_label_paths = train_test_split(img_path, label_path, test_size = 0.2)

    train_generator = DataGenerator(
    all_filenames = list(zip(train_img_paths, train_label_paths)),
    labels = train_label_paths,
    batch_size = args.batch_size,
    input_dim = (args.shape, args.shape),
    n_channels = 3,
    shuffle = True,
    )
    
    val_generator = DataGenerator(
    all_filenames = list(zip(val_img_paths, val_label_paths)),
    labels = val_label_paths,
    batch_size = args.batch_size,
    input_dim = (args.shape, args.shape),
    n_channels = 3,
    shuffle = True,
    ) 
    
    u = unet()
    history = u.fit(train_generator,
          steps_per_epoch=len(train_generator),
          validation_data=val_generator,
          validation_steps=5,
          epochs=100,
          callbacks = (EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True))
    )
    
    u.save('segmantation.h5')

if __name__ == "__main__":
    main()
