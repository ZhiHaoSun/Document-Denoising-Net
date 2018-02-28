
import window
import patch
import os
import cv2
import model
import train
import numpy as np
import time
from keras.models import model_from_json
# import test

def load_imgs(folderpath):
	count = 2
	imgs = []
	names = []
	files = os.listdir(folderpath)
	for filename in files:
		img = cv2.imread(folderpath+filename, 0)
		if img is None:
			print(filename)
			continue
		imgs.append(img)
		names.append(filename)
		count -= 1
		if count <= 0:
			break
	return imgs, names

def gen_submission(cleaned_images, img_names):

	print("=> Save Submission File")
	fp = open("submission.csv", "w")

	fp.write("id,value\n")

	for i in range(0, len(cleaned_images)):

		img_id = img_names[i]
		img_id = img_id[0:len(img_id)-4]
		cleaned_img = cleaned_images[i]
		w = len(cleaned_img[0])
		h = len(cleaned_img)

		for k in range(0, w):
			for j in range(1, h):
				fp.write(img_id + "_" + str(j) + "_" +  str(k) + "," + str(cleaned_img[j][k]) + "\n")
	fp.close()

test_images, names = load_imgs("dataset/receipts/")
print("num_test_images: " + str(len(test_images)))

start = time.time()
test_data, test_patch_num = patch.img2patch(test_images)
print('Img2Patch time: ' +  str(time.time() - start))
test_data = np.array(test_data)
test_data = test_data.reshape(test_data.shape[0], 50, 50, 1)
print(test_data.shape[0])

# execute
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
train_model = model_from_json(loaded_model_json)
train_model.load_weights("model.h5")
print("==> training")

start = time.time()
results = train_model.predict(test_data)
print('Predict time: ' + str(time.time() - start))

print(results)
images = patch.patch2img(results, test_images)

count = 0
for image in images:
	cv2.imwrite('dataset/test_cleaned/' + names[count], image)
	count += 1

# gen_submission(images, names)

