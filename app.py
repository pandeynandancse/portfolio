from __future__ import division, print_function
from flask import Flask
from flask import Flask, render_template, request,session

from flask_session import Session
from flask_sqlalchemy import SQLAlchemy 
import re
import urllib.request
from bs4 import BeautifulSoup
import pandas as pd
import requests
app = Flask(__name__)
import numpy as np

sess = Session()
import os
import pickle





model = pickle.load(open('mlmodels/model.pkl', 'rb'))
@app.route('/mlmodel')
def mlmodel():
    return render_template('mlmodel.html')




@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('mlmodel.html', prediction_text='Employee Salary should be $ {}'.format(output))






@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)




import pyrebase
firebaseConfig = {
    "apiKey": "AIzaSyAwZKzSs_jMRfWzku2v5jgmzeuUAkVI4ds",
    "authDomain": "mlproject-88096.firebaseapp.com",
    "databaseURL": "https://mlproject-88096.firebaseio.com",
    "projectId": "mlproject-88096",
    "storageBucket": "mlproject-88096.appspot.com",
    "messagingSenderId": "115515980307",
    "appId": "1:115515980307:web:f108d243bb127632671de6",
    "measurementId": "G-VNL1548CDZ"
  } 
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

user ={}
id=0




@app.route('/login', methods=['GET', 'POST'])
def basic():
    unsuccessful = 'Please enter correct credential'
    #successful = 'Welcome '
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            # print(email,password)
            global user 
            user = dict(auth.sign_in_with_email_and_password(email,password))
            try:
                print("here")
                session['email'] = email
            except Exception as e:
                print(e)
            print(session)
            if(session['email']):
                
                return render_template('index.html',id = 1)
            else:
                return render_temmplate('index.html', id=0)
        except:
            return render_template('login.html',us = unsuccessful)


    # else:
    #     if(user['idToken']):
    #         return render_template('index.html')
    return render_template('login.html')
    









@app.route('/logout', methods=['GET','POST'])
def logout():
    # unsuccessful = 'Please enter correct credential'
    #successful = 'Welcome '
    if request.method == 'GET' or request.method=='POST': 
        
        try:
            session.pop('email',None)
            global id
            return render_template('index.html',id=0)
        except:
            return render_template('index.html')


    # else:
    #     if(user['idToken']):
    #         return render_template('index.html')
    # return render_template('.html')
    






#auth.get_account_info(user['idToken'])
#auth.send_email_verification(user['idToken])

#auth.send_password_reset_email(email)








@app.route('/contributor')
def contributor():
    return render_template('login.html')





@app.route('/forgot_password',methods=['GET','POST'])
def forgot_password():
    # unsuccessful = 'Signup failed, Please try again'
    
    if request.method == 'POST':
        email = request.form['email']
        try:
            #user = auth.get_account_info(user['idToken'])
            
            auth.send_password_reset_email(email)
            return render_template('login.html')
        except:
            return render_template('signup.html')
    # else:
    #     if(user['idToken']):
    #         return render_template('index.html')
    return render_template('forgot_password.html')
    






@app.route('/signup', methods=['GET', 'POST'])
def signup():
    unsuccessful = 'Signup failed, Please try again'
    
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            auth.create_user_with_email_and_password(email,password)
            user = auth.sign_in_with_email_and_password(email,password)
            auth.send_email_verification(user['idToken'])
            
            return render_template('login.html')
        except:
            return render_template('signup.html',us = unsuccessful)
    # else:
    #     if(user['idToken']):
    #         return render_template('index.html')
    return render_template('signup.html')







@app.route('/')
def home():
    try:
        if(session['email'] != None):
            print("fafafafa")
            print(user)
            return render_template('index.html',id=1)
    except:
        return render_template('index.html',id=0)








# #registering another python router file  
# from ml import ml_blueprint_object
# app.register_blueprint(ml_blueprint_object, url_prefix='/ml')








#registering another python router file  
from education import education_blueprint_object
app.register_blueprint(education_blueprint_object, url_prefix='/education')








#registering another python router file  
from blog import blog_blueprint_object
app.register_blueprint(blog_blueprint_object, url_prefix='/blog')






#registering another python router file  
from gallery import gallery_blueprint_object
app.register_blueprint(gallery_blueprint_object, url_prefix='/gallery')







#registering another python router file  
from android import android_blueprint_object
app.register_blueprint(android_blueprint_object, url_prefix='/android')







#registering another python router file  
from web import web_blueprint_object
app.register_blueprint(web_blueprint_object, url_prefix='/web')








#registering another python router file  
from gui import gui_blueprint_object
app.register_blueprint(gui_blueprint_object, url_prefix='/gui')













@app.route('/corona_notification')
def corona_notification():
    return render_template('corona.html',id = 1)






@app.route('/corona_notification_run')
def corona_notification_run():


    import requests
    import json
    import pyttsx3
    import speech_recognition as sr
    import re
    import threading
    import time
    import urllib.request
    import urllib.parse
    import urllib.error
    from bs4 import BeautifulSoup as bs

    a = "Jammu and Kashmir\nPunjab\nHimachal Pradesh\nHaryana\nDelhi\nRajasthan\nUttar Pradesh\nUttarakhand\nMadhya Pradesh\nChattisgarh\nGujarat\nMaharashtra\nKarnataka\nGoa\nKerala\nTamil nadu\nAndhra pradesh\nTelangana\nOrissa\nBihar\nJharkhand\nWest Bengal\nAssam\nArunach Pradesh\nSikkim\nMeghalaya\nMizoram\nNagaland\nTripura"

    states = a.split("\n")
    states_list = []
    for state in states:
        states_list.append(state.lower())


    def getData(url):
        r = urllib.request.urlopen(url)
        return r


    print("started parsing")
    myHtmlData = getData("https://www.mohfw.gov.in/")
    print("parsed")

    soup = bs(myHtmlData, 'html.parser')
    myDataStr = ""
    for tr in soup.find_all('tr'):
        myDataStr += tr.get_text()
    myDataStr = myDataStr[1:]
    itemList = myDataStr.split("\n\n")


    def speak(text):
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        for j, v in enumerate(voices):
            if voices[j].id == 'hindi':
                break

        print(voices[j].id)
        engine.setProperty('voice', voices[j].id)
        engine.setProperty('rate', 100)
        engine.setProperty('volume', 0.9)
        engine.say(text)
        engine.runAndWait()


    def get_audio():
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            print("Say something!")
            audio = r.listen(source, timeout=3)
            said = ""

            try:
                said = r.recognize_google(audio)
            except Exception as e:
                print("Exception:", str(e))

        return said.lower()


    def main():
        print("Started Program")

        END_PHRASE = "stop"

        i = 0
        # UPDATE_COMMAND = "update"

        while True:
            getAudio = []
            print("Listening...")
            text = get_audio()
            getAudio.append(text)
            print(text)
            if (getAudio[0] == 'total cases'):
                total_confirmed_in_india = itemList[34].split("\n")
                total_confirmed_cases_in_india_text = total_confirmed_in_india[0]
                total_number_of_cases_in_india = total_confirmed_in_india[1]
                total_number_of_cured_in_india = itemList[35]
                total_number_of_deaths_in_india = itemList[36]

                nTex = f'I ,am ,announcing ,about  , {total_confirmed_cases_in_india_text}  , in, which, total,' \
                        f'confirmed, cases, are, {total_number_of_cases_in_india[:-1]} , cured,and, discharge' \
                        f'd, are ,{total_number_of_cured_in_india},  ' \
                        f'and, total ,deaths, till, now, are , {total_number_of_deaths_in_india} ,'
                speak(nTex)
                continue
            for item in itemList[1:35]:
                dataList = item.split("\n")
                if dataList[1].lower() in getAudio:

                    nText = f'I ,am ,announcing ,about ,State , {dataList[1]}  in, which, total,confirmed,cases, are, {dataList[2]} , cured,and, discharged, are ,{dataList[3]},  ' \
                            f'and, total ,deaths, till, now, are , {dataList[4]} ,'

                    speak(str(nText))
                    break
                else:
                    continue

            # if text == UPDATE_COMMAND:
            #   result = "Data is being updated. This may take a moment!"
            #  data.update_data()

            # if result:
            #   speak(result)

            if text.find(END_PHRASE) != -1:  # stop loop
                print("Exit")
                break


    main()
    return render_template('corona.html',id = 1)







def load_books():    
    df = pd.read_excel("books.xlsx")
    categories = df["English Package Name"]
    return df,categories






@app.route('/springer')
def springer():
    return render_template('springer.html',id = 1)






@app.route('/springer_category_run')
def springer_category_run():
    df,categories = load_books()
    return render_template('springer_category.html',categories=set(categories),id = 1)






@app.route("/<string:category>")
def category(category):
    df,categories = load_books()
    books  = df[df["English Package Name"]==category]["Book Title"]
    return render_template('springer_category_book.html', category = category,books= set(books) ,id = 1)




def pdf(liel):
     if "/content/pdf" in liel:
         return True












@app.route("/category/<string:boook>")
def boook(boook):
    df,categories = load_books()
    url  = str(df[df["Book Title"]==boook]["OpenURL"]).split("...")
    isb =  str(df[df["Book Title"]==boook]["Print ISBN"])
    u = re.sub(" +"," ",url[0])
    uu = u.split(" ")
    lin = uu[1]
    lin = lin+"bn="
    isb = re.sub(" +"," " ,isb)
    isbb = isb.split(" ")[1]
    finall = lin + isbb
    fi = finall.split("\n")
    res = urllib.request.urlopen(fi[0])
    htmls = res.read()
    soup = BeautifulSoup(htmls,'html.parser')
    tags = soup('a')
    li=[]
    for tag in tags:
        li.append(tag.get('href',None))
    f_list = list(filter(pdf, li))

    go = f_list[0].split("/")
    link = "link.springer.com/content/pdf/"+go[3]
    
    #notifyMe("Hello","Your Downlaod will start in 3 seconds")
    # !pip install download

    co = link.split("content")

    urll = "https://" + link
    pathh = "content" 
    #path = download(urll, pathh,replace=True)
    #urllib.request.urlopen(urll)
    return render_template("springer_download_confirmation.html",ur = urll,id = 1)
    



















app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///weather.db'

db = SQLAlchemy(app)






class City(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
   

@app.route('/weather', methods=['GET', 'POST'])
def weather():
    if request.method == 'POST':
        new_city = request.form.get('city')
        
        if new_city:
            new_city_obj = City(name=new_city)

            db.session.add(new_city_obj)
            db.session.commit()

    cities = City.query.all()

    url = 'http://api.openweathermap.org/data/2.5/weather?q={}&units=imperial&appid=352318b0096c90c8b3a94a031f4ee602'

    weather_data = []

    for city in cities:

        r = requests.get(url.format(city.name)).json()

        weather = {
            'city' : city.name,
            'temperature' : r['main']['temp'],
            'description' : r['weather'][0]['description'],
            'icon' : r['weather'][0]['icon'],
        }

        weather_data.append(weather)

    return render_template('weather.html', weather_data=weather_data)








@app.route('/weather_delete_database', methods=['GET', 'POST'])
def weather_delete_database():
    if request.method == 'POST':
        db.session.query(City).delete()
        db.session.commit()
    return render_template('weather.html')













import sys

import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


import tensorflow as tf




from keras.models import load_model,Model
MODEL_PATH = 'mlmodels/model_resnet.h5'
resnet_model = load_model(MODEL_PATH)
resnet_model._make_predict_function()    # Necessary




def model_predict(img_path, ml_model):
    #print("daddadadarwgrgrggdw=================")
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224)) #load_img requires installation of pillow
    #print("daddadadarwgdawdaddjiajdi=================")
    # Preprocessing the image
    x = image.img_to_array(img)

    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x,mode='caffe')
    print(x)
    #print("dadadwdaadw=================")
    graph = tf.get_default_graph()
    with graph.as_default():
        preds = ml_model.predict(x)
    print("dadadwdaadw+++++++++++++++++++++++=================")
    return preds



@app.route('/image_classification', methods=['GET'])
def image_classification():
    
    return render_template('image_classification.html')





@app.route('/predict_image', methods=['GET', 'POST'])
def predict_image():
    try:
        if request.method == 'POST':
            #print("kokok==================")
            # Get the file from post request
            f = request.files['file']
            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)  #__file__ means current file path
            #print("pathhhh==================")
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            #print("uploaded==================")
            # Make prediction
            preds = model_predict(file_path, resnet_model)
            print("predict==================") 
            # Process your result for human
            # pred_class = preds.argmax(axis=-1)            # Simple argmax
            pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
            result = str(pred_class[0][0][1])               # Convert to string
            return result
    except:
        print("error")
        return None
    


































def get_predictions(raw_image):
    YOLO_DIR = "yolo-coco"
    CONFIDENCE = 0.5
    THRESHOLD = 0.3

    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([YOLO_DIR, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")
    
    # download model weights if not already downloaded
    model_found = 0
    files = os.listdir("yolo-coco")
    
    if "yolov3.weights" in files:
        model_found = 1

    if model_found == 0:
        download_model_weights()

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([YOLO_DIR, "yolov3.weights"])
    configPath = os.path.sep.join([YOLO_DIR, "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # load input image and grab its spatial dimensions
    nparr = np.fromstring(raw_image.data, np.uint8)
    # decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected probability is greater than the minimum probability
            if confidence > CONFIDENCE:
                # scale the bounding box coordinates back relative to the size of the image
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)
    
    predictions = []
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # append prediction box coordinates, box display colors, labels and probabilities
            predictions.append({
                "boxes": boxes[i], 
                "color": [int(c) for c in COLORS[classIDs[i]]], 
                "label": LABELS[classIDs[i]], 
                "confidence": confidences[i]
            })

    return predictions












MODEL_PATH = 'mlmodels/model_resnet.h5'
resnet_model = load_model(MODEL_PATH)
resnet_model._make_predict_function()    # Necessary






@app.route('/yolo_object_detection', methods=['GET'])
def yolo_object_detection():
    return render_template('yolo_object_detection.html')






@app.route('/detect_obj', methods=['POST'])
def detect_obj():
    predictions = get_predictions(request)



    





















if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'

    sess.init_app(app)
    app.run(host='0.0.0.0',debug=False)
