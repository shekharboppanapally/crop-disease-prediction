from flask import Flask,redirect,url_for,render_template,request
import pickle
import numpy as np
import requests
import os
from skimage.io import imread
from skimage.transform import resize

apple_disease_model_path = 'models/apple.pkl'
apple_disease_model = pickle.load(open(apple_disease_model_path, 'rb'))

cherry_disease_model_path = 'models/cherry.pkl'
cherry_disease_model = pickle.load(open(cherry_disease_model_path, 'rb'))

grape_disease_model_path = 'models/grape.pkl'
grape_disease_model = pickle.load(open(grape_disease_model_path, 'rb'))

maize_disease_model_path = 'models/maize.pkl'
maize_disease_model = pickle.load(open(maize_disease_model_path, 'rb'))

tomato_disease_model_path = 'models/tomato.pkl'
tomato_disease_model = pickle.load(open(tomato_disease_model_path, 'rb'))

potato_disease_model_path = 'models/potato.pkl'
potato_disease_model = pickle.load(open(potato_disease_model_path, 'rb'))

app=Flask(__name__)

@app.route('/')
def welcome():
     return render_template('homepage.html')

@app.route('/submit',methods=['POST','GET'])
def submit():
    if request.method=='POST':
        crop=(request.form['Crop'])
        file=(request.files['file'])
        if crop == 'Apple' :
            CATAGORIES=['Apple_scab','Black_rot','Cedar_apple_rust']
            flat_data=[]
            url=(file)
            img=imread(url,pilmode="RGB")
            img_resized=resize(img,(150,150,3))
            flat_data.append(img_resized.flatten())
            flat_data=np.array(flat_data)
            y_out=apple_disease_model.predict(flat_data)
            y_out=CATAGORIES[y_out[0]]
            return render_template('result.html', result=y_out)
        if crop == 'Cherry' :
            CATAGORIES=['cherry_healthy','cherry_powdery_mildew']
            flat_data=[]
            url=(file)
            img=imread(url)
            img_resized=resize(img,(150,150,3))
            flat_data.append(img_resized.flatten())
            flat_data=np.array(flat_data)
            y_out=cherry_disease_model.predict(flat_data)
            y_out=CATAGORIES[y_out[0]]
            return render_template('result.html', result=y_out) 
        if crop == 'Grape' :
            CATAGORIES=['black_rot','esca','healthy','leaf_blight']
            flat_data=[]
            url=(file)
            img=imread(url)
            img_resized=resize(img,(150,150,3))
            flat_data.append(img_resized.flatten())
            flat_data=np.array(flat_data)
            y_out=grape_disease_model.predict(flat_data)
            y_out=CATAGORIES[y_out[0]]
            return render_template('result.html', result=y_out) 
        if crop == 'Maize' :
            CATAGORIES=['Northern_leaf_blight','cercospora_leaf_spot','common_rust','healthy']
            flat_data=[]
            url=(file)
            img=imread(url)
            img_resized=resize(img,(150,150,3))
            flat_data.append(img_resized.flatten())
            flat_data=np.array(flat_data)
            y_out=maize_disease_model.predict(flat_data)
            y_out=CATAGORIES[y_out[0]]
            return render_template('result.html', result=y_out)                 
        if crop == 'Potato' :
            CATAGORIES=['potato_early_blight','potato_healthy','potato_late_blight']
            flat_data=[]
            url=(file)
            img=imread(url)
            img_resized=resize(img,(150,150,3))
            flat_data.append(img_resized.flatten())
            flat_data=np.array(flat_data)
            y_out=potato_disease_model.predict(flat_data)
            y_out=CATAGORIES[y_out[0]]
            return render_template('result.html', result=y_out)   
        if crop == 'Tomato' :
            CATAGORIES=['tomato_early_blight','tomato_healthy','tomato_late_blight']
            flat_data=[]
            url=(file)
            img=imread(url)
            img_resized=resize(img,(150,150,3))
            flat_data.append(img_resized.flatten())
            flat_data=np.array(flat_data)
            print(img.shape) 
            y_out=potato_disease_model.predict(flat_data)
            y_out=CATAGORIES[y_out[0]]
            return render_template('result.html', result=y_out) 

if __name__=='__main__':
    app.run(debug=True)