from fastapi import FastAPI
import uvicorn
from ad_click import ad
import pandas as pd
import numpy as np
import pickle

app = FastAPI()
# pickle_in = open("model.pkl","rb")
# classifier = pickle.load(pickle_in)

with open(r'.\model\model.pkl', 'rb') as f:
    clf2 = pickle.load(f)

@app.get('/')
def index():
    return { 'message': 'Hello, user'}

@app.get('/{name}')
def get_name(name: str):
    return { 'message': f'Hello, {name}'}

@app.post('/predict')
def predict_ad(data: ad):
    data = data.dict()
    age=data['age']
    sex=data['sex']
    estimated_sal= data['EstimatedSalary']
    print(age,sex,estimated_sal)

    prediction = clf2.predict([[age,sex,estimated_sal]])
    print('********************************************')
    print(prediction)
    print('********************************************')
    if(prediction[0] == 0):
        prediction="not click"
    else:
        prediction="click"
    return {
        'prediction': prediction
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)