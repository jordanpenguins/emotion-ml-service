import os
import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import tempfile
import requests
from video_processor import VideoEmotionProcessor
from dotenv import load_dotenv
import uvicorn

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