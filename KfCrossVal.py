import os
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

kf = sklearn.model_selection.KFold(n_splits=5, shuffle=False, random_state=None)

train_data = 0 #Se debe definir
Y = 0 #Esto tambien
image_dir = '0' #Y esto



idg = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.3,
                         fill_mode='nearest',
                         horizontal_flip = True,
                         rescale=1./255)


def get_model_name(k):
    return 'model_'+str(k)+'.h5'



VALIDATION_ACCURACY = []
VALIDATION_LOSS = []



save_dir = '/saved_models/'
fold_var = 1



for train_index, val_index in kf.split(np.zeros(n),Y):
	training_data = train_data.iloc[train_index]
	validation_data = train_data.iloc[val_index]
	
	train_data_generator = idg.flow_from_dataframe(training_data, directory = image_dir,
						       x_col = "filename", y_col = "label",
						       class_mode = "categorical", shuffle = True)
	valid_data_generator  = idg.flow_from_dataframe(validation_data, directory = image_dir,
							x_col = "filename", y_col = "label",
							class_mode = "categorical", shuffle = True)
	
	# CREATE NEW MODEL
	model = create_new_model()
	# COMPILE NEW MODEL
	model.compile(loss='categorical_crossentropy',
		      optimizer=opt,
		      metrics=['accuracy'])
	
	# CREATE CALLBACKS
	checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+get_model_name(fold_var), 
							monitor='val_accuracy', verbose=1, 
							save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	# There can be other callbacks, but just showing one because it involves the model name
	# This saves the best model
	# FIT THE MODEL
	history = model.fit(train_data_generator,
			    epochs=num_epochs,
			    callbacks=callbacks_list,
			    validation_data=valid_data_generator)
	#PLOT HISTORY
	#		:
	#		:
	
	# LOAD BEST MODEL to evaluate the performance of the model
	model.load_weights("/saved_models/model_"+str(fold_var)+".h5")
	
	results = model.evaluate(valid_data_generator)
	results = dict(zip(model.metrics_names,results))
	
	VALIDATION_ACCURACY.append(results['accuracy'])
	VALIDATION_LOSS.append(results['loss'])
	
	tf.keras.backend.clear_session()
	
	fold_var += 1