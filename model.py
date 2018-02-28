
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import LeakyReLU
from keras.layers import Input, Dense
from keras.layers import Lambda, Activation
from keras import backend as K
from keras import losses



def newmodel(patch_size = 50):

	# input dimensions
	nfeats = 1

	# filter size
	filtsize = 3

	# hidden units
	nstates = [96,96,96,96,96]

	# model:
	model = Sequential()

	# stage 1 : Convolution
	model.add(Conv2D(nstates[0], (3, 3), strides=(1,1), padding='same', input_shape=(patch_size, patch_size, 1)))
	model.add(LeakyReLU(alpha=0.1))

	# stage 2 : Convolution
	model.add(Conv2D(nstates[1], (1, 1)))
	model.add(LeakyReLU(alpha=0.1))

	# stage 3 : Convolution
	model.add(Conv2D(nstates[2], (3, 3), strides=(1,1), padding='same'))
	model.add(LeakyReLU(alpha=0.1))

	# stage 4 : Convolution
	model.add(Conv2D(nstates[3], (1, 1)))
	model.add(LeakyReLU(alpha=0.1))

	# stage 5 : Convolution
	model.add(Conv2D(nstates[4], (3, 3), strides=(1,1), padding='same'))
	model.add(LeakyReLU(alpha=0.1))

	# stage 6 : Convolution
	model.add(Conv2D(1, (3, 3), strides=(1,1), padding='same'))

	model.add(LeakyReLU(alpha=0.001))
	model.add(Lambda(lambda x: -x + 1))
	model.add(LeakyReLU(alpha=0.001))
	model.add(Lambda(lambda x: -x + 1))
	
	return model

def newthreshold(patch_size=50):
	model = Sequential()
	model.add(Activation('relu', input_shape=(patch_size,patch_size,1)))
	model.add(Lambda(lambda x: -x + 1))
	model.add(Activation('relu'))
	model.add(Lambda(lambda x: -x + 1))

	return model