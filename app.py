#!/usr/bin/env python
import os
from flask import Flask, request, jsonify
import keras
from keras.optimizers import Adam
import numpy as np
from keras.preprocessing import image
import json
import keras.backend as K
from utils import decode_img, classes
from resnet import ResNet50
import tensorflow as tf


# Model
K.clear_session()
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
model = ResNet50(input_shape = (64, 64, 3), classes = 47)
print(model.summary())
model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.002), 
              loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('model.hdf5')
graph = tf.get_default_graph()
   


# initialization
app = Flask(__name__)


@app.route('/api/predict', methods=['POST'])
def predict():
	data = request.data
	dataDict = json.loads(data)
	img_base64 = dataDict['image']

	
	img = np.expand_dims(decode_img(img_base64), axis=0)
	img = np.resize(img, (1,64,64,3))

	global graph
	with graph.as_default():
		result = model.predict(img)
		predicted_class = list(classes.keys())[list(classes.values()).index(result.argmax())]
    
	print("The predicted class is:", predicted_class)

	return jsonify({'prediction': predicted_class})


if __name__ == '__main__': 
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)