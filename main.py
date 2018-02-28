
import window
import patch
import os
import cv2
import model
import train
import numpy as np
from keras.models import model_from_json
# import test

opt = {
	'seed': 91,
	'threads': 2,
	'learningRate': 1e-3,
	'weightDecay': 0,
	'momentum': 0,
	'epochs': 20,
	'batch_size': 20,
	'learningRateDecay': 1e-7,
	'clipnorm': 15
}

def load_imgs(folderpath):
	imgs = []
	files = os.listdir(folderpath)
	for filename in files:
		img = cv2.imread(folderpath+filename, 0)
		if img is None:
			continue
		imgs.append(img)
	return imgs

train_images = load_imgs("dataset/train/")
train_cleaned_images = load_imgs("dataset/train_cleaned/")
# test_images = load_imgs("dataset/test/")

print("num_train_images: " + str(len(train_images)))
# print("num_test_images: " + str(len(test_images)))

train_data, train_patch_num = patch.img2patch(train_images)
train_data = np.array(train_data)
train_data = train_data.reshape(train_data.shape[0], 50, 50, 1)
# retrain_images = patch.patch2img(train_data, train_images)

print("Train data")
print(len(train_data))

train_cleaned_data, train_cleaned_patch_num = patch.img2patch(train_cleaned_images)
train_cleaned_data = np.array(train_cleaned_data)
train_cleaned_data = train_cleaned_data.reshape(train_cleaned_data.shape[0], 50, 50, 1)
# retrain_cleaned_images = patch.patch2img(train_cleaned_data, train_cleaned_images)

# test_data, test_patch_num = patch.img2patch(test_images)

# execute
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
train_model = model_from_json(loaded_model_json)
train_model.load_weights("model.h5")
print("==> training")
print("==> training")
train.train(train_data, train_cleaned_data, train_model, opt)


