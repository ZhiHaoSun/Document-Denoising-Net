import math
import numpy as np

patch_size = 50
overlap = 30

def hanning(size):
	v = np.divide(np.arange(size), size)
	v2 = np.ones(size)

	v2[0: size] = v
	hanv = - np.cos(v2 * 2 * math.pi) * 0.5 + 0.5

	ret = np.outer(hanv, hanv) + 0.01
	ret = np.divide(ret, ret.sum())

	return ret

# img2patch
def img2patch(images, patch_size=patch_size):

	patch_num = 0
	for image in images:
		num_y = math.ceil((image.shape[0] - patch_size) / (patch_size - overlap )) + 1
		num_x = math.ceil((image.shape[1] - patch_size) / (patch_size - overlap )) + 1
		patch_num = patch_num + num_x * num_y

	data = []

	for i in range(0,len(images)):
		image = images[i]
		size_x = image.shape[1]
		size_y = image.shape[0]
		num_y = math.ceil((size_y - patch_size) / (patch_size - overlap )) + 1
		num_x = math.ceil((size_x - patch_size) / (patch_size - overlap )) + 1

		for sx in range(0, num_x):
			for sy in range(0, num_y):
				x = (patch_size - overlap) * sx
				y = (patch_size - overlap) * sy
				if x + patch_size >= size_x:
					x = size_x - patch_size
				if y + patch_size >= size_y:
					y = size_y - patch_size

				data.append(image[y:y+patch_size, x:x+patch_size])

	return data, patch_num

# patch2img
def patch2img(data, original_images):

	count = 0
	images = []
	window = hanning(patch_size)

	for i in range(0, len(original_images)):
		size_y = original_images[i].shape[0]
		size_x = original_images[i].shape[1]
		
		images.append(np.zeros((size_y, size_x), np.float64))
		weight = np.zeros((size_y, size_x), np.float64)
		num_y = math.ceil((size_y - patch_size) / (patch_size - overlap )) + 1
		num_x = math.ceil((size_x - patch_size) / (patch_size - overlap )) + 1

		for sx in range(0, num_x):
			for sy in range(0, num_y):
				x = (patch_size - overlap) * sx
				y = (patch_size - overlap) * sy
				if x + patch_size >= size_x:
					x = size_x - patch_size
				if y + patch_size >= size_y:
					y = size_y - patch_size
				img_copy = images[i]

				mul = np.multiply(data[count].reshape(patch_size, patch_size), window)

				img_copy[y:y+patch_size, x:x+patch_size] += mul
				weight[y: y+patch_size, x:x+patch_size] += window
				images[i] = img_copy
				count = count + 1

		images[i] = np.divide(images[i], weight)
	return images
