#!/usr/bin/env python
import os
from flask import Flask
import keras

from resnet import ResNet50

model = ResNet50(input_shape = (64, 64, 3), classes = 47)
print(model.summary())
model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.002), 
              loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())





# initialization
app = Flask(__name__)


@app.route('/api/predict')
def predict():
    return jsonify({'data': 'Hello'})


if __name__ == '__main__': 
    app.run(host='0.0.0.0')