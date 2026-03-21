# from typing import Union
# import os
# import cv2
# import numpy as np
# import tempfile
# import requests
# from pydantic import BaseModel

# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image


# app = FastAPI()

# # CLASS NAMES
# CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad',
#                'surprise', 'neutral']

# # Allow local dev origins 
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# @app.on_event("startup")
# def startup_event():
#     global model
#     model_path = os.path.join(os.getcwd(), "app/model/emotion_model.h5")
#     print("Loading model from:", model_path)
#     model = load_model(model_path)


# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

# def preprocess_image(file_path):
#     # CascadeClassifier is a cv2 used for face detection 
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

#     file = '/content/drive/MyDrive/Pictures/image_test1.jpg'
#     t_image = cv2.imread(file)
#     # img = image.load_img(file, grayscale=True, target_size=(48, 48))
#     gray = cv2.cvtColor(t_image,cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray,1.1,4)

#     if(len(faces) == 0):
#         exit()

#     for (x,y,w,h) in faces:
#         cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
#         start_row,end_row,start_col,end_col = y,y+h,x,x+h

#     croppedimage = gray[start_row:end_row,start_col:end_col]
#     img = cv2.resize(croppedimage,(48,48))

#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis = 0)

#     x /= 255


#     return x # returns preprocessed image

# def getFrame(seconds,vidcap):
#     global count, d, dcount
#     vidcap.set(cv2.CAP_PROP_POS_MSEC,seconds*1000)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
#     hasFrames,img = vidcap.read()

#     if hasFrames:
#         minutes = "00"
#         hours = "00"
#         if seconds >= 60:
#             minutes = str(seconds//60)
#             seconds = seconds % 60

#         if int(minutes) >= 60:
#             hours = str(int(minutes)//60)
#             minutes = str(int(minutes) % 60)

#         min = "{:02d}".format(int(minutes))
#         sec = "{:02d}".format(seconds)
#         hrs = "{:02d}".format(int(hours))

#         flag = 0
#         frameId = vidcap.get(1)
#         gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray,1.1,4)

#         if(len(faces)==0):
#             flag = 1

#         if flag == 0 :
#             # prepropessing each frame
#             count = count + 1
#             for (x,y,w,h) in faces:
#                 cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
#                 start_row,end_row,start_col,end_col = y,y+h,x,x+h

#             croppedimage = gray[start_row:end_row,start_col:end_col]
#             finalimg = cv2.resize(croppedimage,(48,48))

#             x = image.img_to_array(finalimg)
#             x = np.expand_dims(x, axis = 0)

#             x /= 255

#             custom = model.predict(x)
#             emt = list(custom[0])
#             idx = emt.index(max(emt))
#             imgname = d[idx]
#             dcount[imgname] = dcount[imgname] + 1

#             print(str(count) + " " + imgname + " " + hrs + ":" + min + ":" + sec)
#             # cv2.imwrite("/content/drive/MyDrive/detection/" + "%d_" % count +imgname+"__"+ hrs+":"+min+":"+sec+".jpg" ,img)

#     return hasFrames


# # process video
# def process_video(video_path):
#     global count, d, dcount
#     count = 0
#     d = {0:"angry",1:"disgust",2:"fear",3:"happiness",4:"sad",5:"surprise",6:"neutral"}
    
#     # we want to get the emotion count for 
#     dcount = {"angry":0,"disgust":0,"fear":0,"happiness":0,"sad":0,"surprise":0,"neutral":0}

#     cap = cv2.VideoCapture(video_path)
#     sec = 0
#     frameRate = 2 #it will capture image in each 2 second
#     success = getFrame(sec,cap)
#     while success:
#         sec = sec + frameRate
#         success = getFrame(sec,cap)
#     cap.release()

#     # right now we are just returning the count of each emotion rather than displaying emotion results for every second
#     return dcount 

# def _format_seconds(seconds: int) -> str:
#     """Helper function to format seconds into HH:MM:SS string."""
#     hours = seconds // 3600
#     minutes = (seconds % 3600) // 60
#     secs = seconds % 60
#     return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# # process video
# def process_video_timestamp(video_path):
#     """
    
#     Expected Output: predictions = [
#                            {"start": "00:00:02", "end": "00:00:04", "emotion": "happy"},
#                             {"start": "00:00:04", "end": "00:00:06", "emotion": "sad"},
#                             {"start": "00:00:06", "end": "00:00:08", "emotion": "neutral"},
#                             {"start": "00:00:06", "end": "00:00:08", "emotion": "happy"},
#                           ]
#     """
#     count = 0
#     d = {0:"angry",1:"disgust",2:"fear",3:"happiness",4:"sad",5:"surprise",6:"neutral"}
#     predictions = []
    
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
#     cap = cv2.VideoCapture(video_path)
#     sec = 0
#     frameRate = 2 #it will capture image in each 2 second
#     current_emotion = None
#     current_start_time = 0

#     # Get video duration
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#     duration_seconds = int(frame_count / fps) if fps > 0 else 0
#     duration_formatted = _format_seconds(duration_seconds)

#     while True:
#         cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
#         hasFrames,img = cap.read()
#         if hasFrames:
            
#             flag = 0
#             gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(gray,1.1,4)

#             if(len(faces)==0):
#                 detected_emotion = None
#                 flag = 1

#             if flag == 0 :
#                 detected_emotion = None 
#                 # prepropessing each frame
#                 count = count + 1
#                 for (x,y,w,h) in faces:
#                     cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
#                     start_row,end_row,start_col,end_col = y,y+h,x,x+h

#                 croppedimage = gray[start_row:end_row,start_col:end_col]
#                 finalimg = cv2.resize(croppedimage,(48,48))

#                 x = image.img_to_array(finalimg)
#                 x = np.expand_dims(x, axis = 0)
#                 x /= 255
#                 custom = model.predict(x)
#                 emt = list(custom[0])
#                 idx = emt.index(max(emt))
#                 imgname = d[idx]
#                 detected_emotion = imgname
#                 if current_emotion != detected_emotion:
#                     if current_emotion is not None:
#                         predictions.append({
#                             "start": _format_seconds(current_start_time),
#                             "end": _format_seconds(sec),
#                             "emotion": current_emotion
#                         })
#                     print(f"{str(count)} {imgname} Start {_format_seconds(sec)} End {_format_seconds(sec)}")
#                     # cv2.imwrite(f"detection/frame_{count}_{current_start_time}_{sec}_{detected_emotion}.jpg", img)
#                     current_emotion = detected_emotion
#                     current_start_time = sec
#             sec = sec + frameRate
#         else:
#             break

#     if current_emotion is not None:
#         predictions.append({
#             "start": _format_seconds(current_start_time),
#             "end": _format_seconds(sec - frameRate),
#             "emotion": current_emotion
#         })
        
#     cap.release()
#     return {
#         "duration_seconds": duration_seconds,
#         "duration": duration_formatted,
#         "predictions": predictions
#     }

# # Video model received from the backend
# def process_image(image_path):
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
#     global count, d, dcount
#     count = 0
#     d = {0:"angry",1:"disgust",2:"fear",3:"happiness",4:"sad",5:"surprise",6:"neutral"}
#     dcount = {"angry":0,"disgust":0,"fear":0,"happiness":0,"sad":0,"surprise":0,"neutral":0}

#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#     if len(faces) == 0:
#         return dcount

#     for (x, y, w, h) in faces:
#         cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         start_row, end_row, start_col, end_col = y, y + h, x, x + w

#     croppedimage = gray[start_row:end_row, start_col:end_col]
#     finalimg = cv2.resize(croppedimage, (48, 48))

#     x = image.img_to_array(finalimg)
#     x = np.expand_dims(x, axis=0)

#     x /= 255

#     custom = model.predict(x)
#     emt = list(custom[0])
#     idx = emt.index(max(emt))
#     imgname = d[idx]
#     dcount[imgname] = dcount[imgname] + 1
#     # cv2.imwrite("/content/drive/MyDrive/detection/" + "%d_" % count + imgname + "__" + hrs + ":" + min + ":" + sec + ".jpg", img)

#     return dcount

# @app.post("/predict-video")
# async def predict_video(video_file: UploadFile = File(...)):
#     """
#     Predict video without timestamps
#     """
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
#         tmp.write(await video_file.read())
#         video_path = tmp.name
#     predictions = process_video(video_path)
#     os.remove(video_path) # Clean up temp file
#     return {"video_path": video_path, "predictions": predictions}

# @app.post("/predict-video-timestamp")
# async def predict_video_timestamp(video_file: UploadFile = File(...)):
#     """
#     Predict video with timestamps

#     Expected Output: {"video_path": "path/to/uploaded/video.mp4",
#                        "predictions": [
#                            {"start": "00:00:02", "end": "00:00:04", "emotion": "happy"},
#                            {"start": "00:00:04", "end": "00:00:06", "emotion": "sad"},
#                            {"start": "00:00:06", "end": "00:00:08", "emotion": "neutral"},
#                            {"start": "00:00:06", "end": "00:00:08", "emotion": "happy"},
#                        ]
#                     }
#     """
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
#         tmp.write(await video_file.read())
#         video_path = tmp.name
#     predictions = process_video_timestamp(video_path)
#     os.remove(video_path) # Clean up temp file
#     return {"video_path": video_path, "predictions": predictions}

# class VideoURLRequest(BaseModel):
#     video_url: str


# @app.post("/predict-video-timestamp-url")
# async def predict_video_timestamp_url(request: VideoURLRequest):
#     """
#     Predict video with timestamps from firebase storage URL

#     Expected Output: {"video_path": "path/to/uploaded/video.mp4",
#                        "duration": "00:00:10",
#                        "duration_seconds": 10,
#                        "predictions": [
#                            {"start": "00:00:02", "end": "00:00:04", "emotion": "happy"},
#                            {"start": "00:00:04", "end": "00:00:06", "emotion": "sad"},
#                            {"start": "00:00:06", "end": "00:00:08", "emotion": "neutral"},
#                            {"start": "00:00:06", "end": "00:00:08", "emotion": "happy"},
#                        ]
#                     }
#     """

#     try:
#         response = requests.get(request.video_url)
#         response.raise_for_status()
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
#             tmp.write(response.content)
#             video_path = tmp.name
#         predictions = process_video_timestamp(video_path)
#         os.remove(video_path)  # Clean up temp file
#         return {"video_path": video_path, "predictions": predictions["predictions"], "duration": predictions["duration"], "duration_seconds": predictions["duration_seconds"]}
#     except requests.RequestException as e:
#         return {"error": str(e)}


# @app.post("/predict-image")
# async def predict_image(file: UploadFile = File(...)):
#     print(file)
#     # Save the uploaded file to a temporary location
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
#         tmp.write(await file.read())
#         tmp_path = tmp.name

#     predictions = process_image(tmp_path)
#     os.remove(tmp_path)  # Clean up temp file
#     print(predictions)
#     return {"predictions": predictions}

# main.py - Simplified ML Service

import os
import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import tempfile
import requests
from app.video_processor import VideoEmotionProcessor
from dotenv import load_dotenv
import uvicorn
import firebase_admin
from firebase_admin import credentials, storage
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
import uuid
import subprocess
import requests
import tempfile
from typing import Optional
import tempfile

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Emotion Analysis ML Service",
    description="Video emotion analysis service - processes video and returns predictions",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = None

def get_processor():
    """Get or create video processor instance"""
    global processor
    if processor is None:
        processor = VideoEmotionProcessor()
    return processor

def cleanup_files(paths):
    """Safely delete temporary files created during processing."""
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
                logger.info(f"🧹 Deleted temporary file: {path}")
            except Exception as e:
                logger.warning(f"Failed to delete {path}: {e}")


def download_file_from_url(url: str, suffix: str = '.mp4') -> Optional[str]:
    """
    Download file from URL with detailed error logging
    """
    try:
        logger.info(f"Downloading from URL: {url}")
        logger.info("Making HTTP request...")
        
        # Add timeout and stream for large files
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # Raise exception for bad status codes
        
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Content-Type: {response.headers.get('Content-Type')}")
        logger.info(f"Content-Length: {response.headers.get('Content-Length')} bytes")
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        
        # Download with progress tracking
        total_size = int(response.headers.get('Content-Length', 0))
        downloaded = 0
        
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                temp_file.write(chunk)
                downloaded += len(chunk)
                
                # Log progress every 10%
                if total_size > 0 and downloaded % (total_size // 10) < 8192:
                    progress = (downloaded / total_size) * 100
                    logger.info(f"Download progress: {progress:.1f}%")
        
        temp_file.close()
        
        file_size = os.path.getsize(temp_file.name)
        logger.info(f"✅ Download complete: {temp_file.name} ({file_size / 1024 / 1024:.2f} MB)")
        
        return temp_file.name
        
    except requests.exceptions.Timeout:
        logger.error(f"❌ Timeout error downloading from URL: {url}")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"❌ Connection error: {e}")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"❌ HTTP error: {e}")
        logger.error(f"Response text: {response.text[:500]}")  # First 500 chars
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected error downloading file: {e}", exc_info=True)
        return None

def create_annotated_video_with_ffmpeg(input_path: str, output_path: str) -> bool:
        """
        Re-encode video with FFmpeg for better compatibility
        """
        try:
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-vcodec', 'libx264',
                '-acodec', 'aac',
                '-strict', 'experimental',
                '-b:v', '2000k',
                '-y',  # Overwrite output
                output_path
            ]
            
            logger.info(f"Re-encoding with FFmpeg: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info("✅ FFmpeg encoding successful")
                return True
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"FFmpeg encoding failed: {e}")
            return False

@app.on_event("startup")
async def startup_event():
    """Initialize processor on startup"""
    logger.info("Starting Emotion Analysis ML Service...")
    try:
        get_processor()
        logger.info("Video processor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize video processor: {e}")
        raise

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        key_path = os.path.join(base_dir, "firebase_key.json")

        if not os.path.exists(key_path):
            raise FileNotFoundError(f"Firebase key not found at {key_path}")

        if not firebase_admin._apps:
            cred = credentials.Certificate(key_path)
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'ai-emotion-tagging.firebasestorage.app'
            })
            logger.info("Firebase initialized successfully")
        else:
            logger.info("Firebase already initialized")

    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")
        raise


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Emotion Analysis ML Service",
        "version": "3.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        proc = get_processor()
        return {
            "status": "healthy",
            "model": "loaded" if proc.emotion_model else "not loaded",
            "face_detector": "ready" if proc.face_detector else "not ready"
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "error": str(e)}
        )


class VideoURLRequest(BaseModel):
    """Simple request model - only URLs needed"""
    video_url: str
    patient_photo_url: str
    frame_interval: Optional[int] = 60


@app.post("/analyze")
async def analyze_video(request: VideoURLRequest):
    """
    Analyze video emotion from URLs
    
    The ML service only processes the video and returns predictions.
    Backend handles all database operations.
    
    Input:
        {
            "video_url": "https://storage.googleapis.com/...",
            "patient_photo_url": "https://storage.googleapis.com/...",
            "frame_interval": 60  (optional, default 60)
        }
    
    Output:
        {
            "success": true,
            "duration_seconds": 120.5,
            "predictions": [
                {
                    "start": "00:00:02",
                    "end": "00:00:04",
                    "emotion": "happy",
                    "confidence": 0.87
                },
                ...
            ],
            "summary": {
                "total_predictions": 60,
                "dominant_emotion": "neutral",
                "emotion_distribution": {...}
            }
        }
    """
    video_path = None
    photo_path = None
    
    try:
        logger.info("="*60)
        logger.info("Starting video analysis")
        logger.info(f"Video URL: {request.video_url[:80]}...")
        logger.info(f"Photo URL: {request.patient_photo_url[:80]}...")
        logger.info(f"Frame interval: {request.frame_interval}")
        logger.info("="*60)
        
        # Validate frame_interval
        if request.frame_interval < 1 or request.frame_interval > 300:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="frame_interval must be between 1 and 300"
            )
        
        # Download video
        logger.info("Downloading video...")
        response = requests.get(request.video_url, stream=True, timeout=600)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            total_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                tmp_video.write(chunk)
                total_size += len(chunk)
            video_path = tmp_video.name
        
        logger.info(f"Video downloaded: {total_size / 1024 / 1024:.2f} MB")
        
        # Download patient photo
        logger.info("Downloading patient photo...")
        response = requests.get(request.patient_photo_url, timeout=60)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_photo:
            tmp_photo.write(response.content)
            photo_path = tmp_photo.name
        
        logger.info(f"Photo downloaded: {len(response.content) / 1024:.2f} KB")
        
        # Get processor
        proc = get_processor()
        
        # Set frame interval
        original_frame_interval = proc.config['frame_interval']
        proc.config['frame_interval'] = request.frame_interval
        
        # Process video - simplified version that doesn't save to Firebase
        logger.info("Processing video...")
        result = proc.process_video_simple(video_path, photo_path)
        
        # Restore frame interval
        proc.config['frame_interval'] = original_frame_interval
        
        # Clean up temp files
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(photo_path):
            os.remove(photo_path)
        
        logger.info("="*60)
        logger.info("Processing complete!")
        logger.info(f"Total predictions: {len(result.get('predictions', []))}")
        logger.info("="*60)
        
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading files: {e}")
        
        # Cleanup
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        if photo_path and os.path.exists(photo_path):
            os.remove(photo_path)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download files: {str(e)}"
        )
        
    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        
        # Cleanup
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        if photo_path and os.path.exists(photo_path):
            os.remove(photo_path)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing error: {str(e)}"
        )

def upload_to_firebase_storage(local_path: str, destination_name: str) -> str:
    """Upload file to Firebase Storage and return public URL"""
    try:
        # Verify file exists and has content
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"File not found: {local_path}")
        
        file_size = os.path.getsize(local_path)
        logger.info(f"Uploading file: {local_path} ({file_size / 1024 / 1024:.2f} MB)")
        
        if file_size < 1000:
            raise ValueError(f"File is too small ({file_size} bytes) - likely corrupted")
        
        bucket = storage.bucket()
        blob = bucket.blob(f"annotated_videos/{destination_name}")
        
        # Upload with content type
        blob.upload_from_filename(
            local_path,
            content_type='video/mp4'
        )
        
        # Make public
        blob.make_public()
        
        public_url = blob.public_url
        logger.info(f"✅ File uploaded successfully: {public_url}")
        
        return public_url
        
    except Exception as e:
        logger.error(f"Error uploading to Firebase: {e}")
        raise


@app.post("/analyze-with-video-upload")
async def analyze_and_upload_video(request: VideoURLRequest, background_tasks: BackgroundTasks):
    """
    Analyze video, create annotated version, and upload to Firebase Storage
    Returns predictions + URL to annotated video
    """
    video_path = None
    photo_path = None
    output_path = None
    
    try:
        logger.info("="*60)
        logger.info("Starting video analysis")
        logger.info(f"Video URL: {request.video_url[:80]}...")
        logger.info(f"Photo URL: {request.patient_photo_url[:80]}...")
        logger.info(f"Frame interval: {request.frame_interval}")
        logger.info("="*60)
        
        # Validate frame_interval
        if request.frame_interval < 1 or request.frame_interval > 300:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="frame_interval must be between 1 and 300"
            )
        
        # Download video
        logger.info("Downloading video...")
        try:
            logger.info("Downloading video...")
            response = requests.get(request.video_url, stream=True, timeout=600)
            response.raise_for_status()
        except Exception as e:
            import traceback
            logger.error(f"❌ Video download failed: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Video download failed: {e}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            total_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                tmp_video.write(chunk)
                total_size += len(chunk)
            video_path = tmp_video.name
        
        logger.info(f"Video downloaded: {total_size / 1024 / 1024:.2f} MB")
        
        # Download patient photo
        logger.info("Downloading patient photo...")
        response = requests.get(request.patient_photo_url, timeout=60)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_photo:
            tmp_photo.write(response.content)
            photo_path = tmp_photo.name
        
        logger.info(f"Photo downloaded: {len(response.content) / 1024:.2f} KB")
        
        output_filename = f"annotated_{uuid.uuid4()}.mp4"
        output_path = tempfile.mktemp(suffix=".mp4")

        # Get processor
        proc = get_processor()
        
        # Set frame interval
        original_frame_interval = proc.config['frame_interval']
        proc.config['frame_interval'] = request.frame_interval
        
        # Process video - simplified version that doesn't save to Firebase
        logger.info("Processing video...")
        result = proc.process_video_with_annotation(video_path, photo_path, output_path)

        upload_local_path = result.get("annotated_video_path") or output_path

        # pick a new temp path for the re-encoded file
        fixed_path = os.path.join(
            os.path.dirname(upload_local_path),
            f"fixed_{uuid.uuid4()}.mp4"
        )

        # try to re-encode
        ok = create_annotated_video_with_ffmpeg(upload_local_path, fixed_path)

        # choose which file to upload
        file_to_upload = fixed_path if ok and os.path.exists(fixed_path) else upload_local_path
        
        # Restore frame interval
        proc.config['frame_interval'] = original_frame_interval
        
        logger.info("="*60)
        logger.info("Processing complete!")
        logger.info(f"Total predictions: {len(result.get('predictions', []))}")
        logger.info("="*60)
        
        
        if result['success'] and os.path.exists(output_path):
            # Upload to Firebase Storage
            logger.info("Uploading annotated video to Firebase Storage...")
            annotated_url = upload_to_firebase_storage(file_to_upload, f"annotated_{uuid.uuid4()}.mp4")
            
            result['annotated_video_url'] = annotated_url
            logger.info(f"✅ Annotated video uploaded: {annotated_url}")
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_files, [video_path, photo_path, output_path])
        
        return result
        
    except Exception as e:
        cleanup_files([video_path, photo_path, output_path])
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        timeout_keep_alive=7200,
        log_level="info"
    )