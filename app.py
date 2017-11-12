#!/usr/bin/env python
import os
from flask import Flask



# initialization
app = Flask(__name__)


@app.route('/api/precict')
def predict():
    return jsonify({'data': 'Hello'})


if __name__ == '__main__': 
    app.run(host='0.0.0.0')