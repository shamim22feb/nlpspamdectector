# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 23:23:55 2020

@author: SAM
"""

import nltk
import pickle
from flask import render_template,Flask,request,url_for

model = pickle.load(open('model_nlp.pkl','rb'))
cv = pickle.load(open('transform.pkl','rb'))
app= Flask(__name__)

@app.route('/')
def home():
    render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST' :
        message = request.form['message']
        data=[message]
        vect = cv.transform(data).toarray()
        predict = model.predict(vect)
    return render_template('result.html',prediction = predict)

if __name__=='__main__':
    app.run()
    
