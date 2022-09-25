import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img
import numpy as np

def parse():
	parser = argparse.ArgumentParser(description="Unet semantic segmantation detectection")
	parser.add_argument("--img_path", type=str, help="path to image")
	parser.add_argument("--model", type=str, help="path to h5 model")
	return parser.parse_args()

def main():
	args = parse()
	u = load_model(args.model)
	img_path = args.img_path

	img_test = load_img(img_path, target_size=(256, 256))
	img_test = img_to_array(img_test)
	img_test /= 255.0
	test_img_input=np.expand_dims(img_test, 0)
	prediction = u(test_img_input)
	predicted_img= np.argmax(prediction, axis=3)[0,:,:]
	plt.figure(figsize=(12, 8))
	plt.subplot(121)
	plt.axis('off')
	plt.title('Testing Image')
	plt.imshow(img_test[:,:,:])

	plt.subplot(122)
	plt.axis('off')
	plt.title('Prediction on test image')
	plt.imshow(predicted_img)
	plt.show()

if __name__ == "__main__":
	main()