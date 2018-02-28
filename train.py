import numpy as np
from keras import optimizers
from keras import backend as K

def custom_loss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


# functions

def train(train_data, cleaned_data, model, opt):

	rms = optimizers.Adam(lr=opt['learningRate'], beta_1=0.9, beta_2=0.999, decay=opt['learningRateDecay'], clipnorm=opt['clipnorm'])
	model.compile(loss=custom_loss, optimizer=rms)
	print(model.summary())
	model.fit(train_data, cleaned_data, epochs=opt['epochs'], batch_size=opt['batch_size'], verbose=2, validation_split=0.1)

	model_json = model.to_json()
	with open("model.json", "w") as json_file:
	    json_file.write(model_json)

	# save/log current net
	model.save_weights("model.h5")
	print("Saved model to disk")
 

