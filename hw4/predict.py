import pickle
from flask import Flask, request, jsonify
import numpy as np

with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)

def prepare_features(ride):
    features ={}
    features['PU_DO']='{}_{}'.format(ride['PULocationID'],ride['DOLocationID'])
    features['trip_distance']=ride['trip_distance']
    return features

def predict_mean(features):
    X_val = dv.transform(features)
    y_pred = lr.predict(X_val)
    return np.mean(y_pred)


app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)

    pred = predict_mean(features)

    result = {'duration': pred}
    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
    