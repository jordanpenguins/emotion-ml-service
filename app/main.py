from typing import Union
import os
import cv2
import numpy as np
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


app = FastAPI()

# CLASS NAMES
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad',
               'surprise', 'neutral']

# Allow local dev origins 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    print("CWD:", os.getcwd())
    global model
    model = load_model("app/model/emotion_model.h5")


@app.get("/")
def read_root():
    return {"Hello": "World"}

def preprocess_image(file_path):
    # CascadeClassifier is a cv2 used for face detection 
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

    file = '/content/drive/MyDrive/Pictures/image_test1.jpg'
    t_image = cv2.imread(file)
    #img = image.load_img(file, grayscale=True, target_size=(48, 48))
    gray = cv2.cvtColor(t_image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.1,4)

    if(len(faces) == 0):
        exit()

    for (x,y,w,h) in faces:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
        start_row,end_row,start_col,end_col = y,y+h,x,x+h

    croppedimage = gray[start_row:end_row,start_col:end_col]
    img = cv2.resize(croppedimage,(48,48))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)

    x /= 255


    return x # returns preprocessed image

def getFrame(seconds,vidcap):
    global count, d, dcount
    vidcap.set(cv2.CAP_PROP_POS_MSEC,seconds*1000)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    hasFrames,img = vidcap.read()

    if hasFrames:
        minutes = "00"
        hours = "00"
        if seconds >= 60:
            minutes = str(seconds//60)
            seconds = seconds % 60

        if int(minutes) >= 60:
            hours = str(int(minutes)//60)
            minutes = str(int(minutes) % 60)

        min = "{:02d}".format(int(minutes))
        sec = "{:02d}".format(seconds)
        hrs = "{:02d}".format(int(hours))

        flag = 0
        frameId = vidcap.get(1)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.1,4)

        if(len(faces)==0):
            flag = 1

        if flag == 0 :
            # prepropessing each frame
            count = count + 1
            for (x,y,w,h) in faces:
                cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
                start_row,end_row,start_col,end_col = y,y+h,x,x+h

            croppedimage = gray[start_row:end_row,start_col:end_col]
            finalimg = cv2.resize(croppedimage,(48,48))

            x = image.img_to_array(finalimg)
            x = np.expand_dims(x, axis = 0)

            x /= 255

            custom = model.predict(x)
            emt = list(custom[0])
            idx = emt.index(max(emt))
            imgname = d[idx]
            dcount[imgname] = dcount[imgname] + 1

            print(str(count) + " " + imgname + " " + hrs + ":" + min + ":" + sec)
            # cv2.imwrite("/content/drive/MyDrive/detection/" + "%d_" % count +imgname+"__"+ hrs+":"+min+":"+sec+".jpg" ,img)

    return hasFrames


# process video
def process_video(video_path):
    global count, d, dcount
    count = 0
    d = {0:"angry",1:"disgust",2:"fear",3:"happiness",4:"sad",5:"surprise",6:"neutral"}
    
    # we want to get the emotion count for 
    dcount = {"angry":0,"disgust":0,"fear":0,"happiness":0,"sad":0,"surprise":0,"neutral":0}

    cap = cv2.VideoCapture(video_path)
    sec = 0
    frameRate = 2 #it will capture image in each 2 second
    success = getFrame(sec,cap)
    while success:
        sec = sec + frameRate
        success = getFrame(sec,cap)
    cap.release()

    # right now we are just returning the count of each emotion rather than displaying emotion results for every second
    return dcount 

# Video model received from the backend
def process_image(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    global count, d, dcount
    count = 0
    d = {0:"angry",1:"disgust",2:"fear",3:"happiness",4:"sad",5:"surprise",6:"neutral"}
    dcount = {"angry":0,"disgust":0,"fear":0,"happiness":0,"sad":0,"surprise":0,"neutral":0}

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return dcount

    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
        start_row, end_row, start_col, end_col = y, y + h, x, x + w

    croppedimage = gray[start_row:end_row, start_col:end_col]
    finalimg = cv2.resize(croppedimage, (48, 48))

    x = image.img_to_array(finalimg)
    x = np.expand_dims(x, axis=0)

    x /= 255

    custom = model.predict(x)
    emt = list(custom[0])
    idx = emt.index(max(emt))
    imgname = d[idx]
    dcount[imgname] = dcount[imgname] + 1
    # cv2.imwrite("/content/drive/MyDrive/detection/" + "%d_" % count + imgname + "__" + hrs + ":" + min + ":" + sec + ".jpg", img)

    return dcount

@app.post("/predict-video")
async def predict_video(video_file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await video_file.read())
        video_path = tmp.name
    predictions = process_video(video_path)
    return {"video_path": video_path, "predictions": predictions}

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    print(file)
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    predictions = process_image(tmp_path)
    os.remove(tmp_path)  # Clean up temp file
    print(predictions)
    return {"predictions": predictions}
