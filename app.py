# Dependent libraries
# For API
from flask import Flask, request, jsonify
# For model
import pickle
# For Mathametical calculations
import numpy as np

model = pickle.load( open( "model.pkl", "rb" ) ) 
# API definition
app = Flask(__name__)

@app.route('/')
def home():
    return "hello"

@app.route('/predict', methods=['POST'])
# Function to predict the survival
def predict():
    int_feature=[float(x) for x in request.form.values()]
    feature=[np.array(int_feature)]
    prediction=model.predict(feature)
    print(prediction)

if __name__ == '__main__':
    app.run()