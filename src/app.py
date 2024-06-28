from flask import Flask, render_template,request
import pickle
import numpy as np
import pandas as pd
import sys
import os
from utils.model_utils import *

app = Flask(__name__)
secret_key = '7829'
app.config['SECRET_KEY'] = secret_key

@app.route("/")
def Dash():
    return render_template('Dash.html')

@app.route("/form", methods=['POST', 'GET'])
def form():
    correlacao_positiva1 = request.form.get('correlacao_positiva1')
    correlacao_negativa1 = request.form.get('correlacao_negativa1')
    correlacao_positiva2 = request.form.get('correlacao_positiva2')
    correlacao_negativa2 = request.form.get('correlacao_negativa2')
    predict = np.array([0])
    if correlacao_positiva1 is not None or correlacao_negativa1 is not None or correlacao_positiva2 is not None or correlacao_negativa2 is not None:
        X = pd.DataFrame({'Corr1': float(correlacao_positiva1), 'Corr2': float(correlacao_positiva2), 
                        'Corr3': float(correlacao_negativa1), 'Corr4': float(correlacao_negativa2)}, index=[0])

        X = preprocessor(X=X)
        with open('notebooks/model/mini_fraud_detection_clf.sav', 'rb') as file:
            model = pickle.load(file)
        predict = model.predict(X)
    return render_template('form.html', result=predict)

@app.route("/model", methods=['POST', 'GET'])
def model():
    return render_template('model.html')

if __name__ in '__main__':
    app.run(debug=True)