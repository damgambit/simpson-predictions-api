#!/usr/bin/env python
import os
from flask import Flask, request
import keras
from keras.optimizers import Adam
import numpy as np
from keras.preprocessing import image

from resnet import ResNet50, classes

model = ResNet50(input_shape = (64, 64, 3), classes = 47)
print(model.summary())
model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.002), 
              loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

model.load_weights('model.h5')


def decode_img(img_base64):
  decode_str = img_base64.decode("base64")
  file_like = cStringIO.StringIO(decode_str)
  img = PIL.Image.open(file_like)
  # rgb_img[c, r] is the pixel values.
  rgb_img = img.convert("RGB")
  return rgb_img


def make_prediction(file):
    test_image = image.load_img(file, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    classes = training_set.class_indices
    predicted_class = list(classes.keys())[list(classes.values()).index(result.argmax())]
    
    print("The predicted class is:", predicted_class)
    display(Image(filename=file, width=256, height=256))



# initialization
app = Flask(__name__)


@app.route('/api/predict', methods=['POST'])
def predict():
	print('in')
	print(request.data)
	return jsonify({'prediction': 'Hello'})


if __name__ == '__main__': 
    app.run(host='0.0.0.0', debug=True)