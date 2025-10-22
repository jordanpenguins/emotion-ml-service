# video_processor.py - Updated for ViT-Large model

import os
import cv2
import numpy as np
import tempfile
import logging
from typing import Dict, List, Optional, Tuple
from mtcnn import MTCNN
from deepface import DeepFace
from scipy.spatial.distance import cosine
from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTImageProcessor

logger = logging.getLogger(__name__)


class ViTEmotionModel:
    """Handler for ViT-Large emotion detection model"""
    
    def __init__(self, model_path: str):
        """
        Initialize ViT emotion model
        
        Args:
            model_path: Path to the fine-tuned ViT model directory
        """
        logger.info(f"Loading ViT model from: {model_path}")
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model and processor
        try:
            self.model = ViTForImageClassification.from_pretrained(model_path)
            self.processor = ViTImageProcessor.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Emotion labels (from RAF-DB training)
            self.emotions = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
            
            logger.info(f"✅ ViT model loaded successfully")
            logger.info(f"Model device: {next(self.model.parameters()).device}")
            logger.info(f"Emotions: {self.emotions}")
            
        except Exception as e:
            logger.error(f"Failed to load ViT model: {e}")
            raise
    
    def predict(self, face_image: np.ndarray) -> Dict:
        """
        Predict emotion from face image
        
        Args:
            face_image: Face image as numpy array (BGR format from OpenCV)
            
        Returns:
            Dictionary with emotion prediction and confidence scores
        """
        try:
            # Convert BGR to RGB
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_image
            
            # Convert to PIL Image
            pil_image = Image.fromarray(face_rgb)
            
            # Preprocess with ViT processor
            inputs = self.processor(images=pil_image, return_tensors='pt')
            pixel_values = inputs['pixel_values'].to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(pixel_values)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probs).item()
                confidence = probs[0][predicted_class].item()
            
            # Get all emotion probabilities
            all_emotions = {
                emotion: round(probs[0][i].item(), 4)
                for i, emotion in enumerate(self.emotions)
            }
            
            predicted_emotion = self.emotions[predicted_class]
            
            return {
                'emotion': predicted_emotion,
                'confidence': confidence,
                'all_emotions': all_emotions
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            # Return default response on error
            return {
                'emotion': 'Neutral',
                'confidence': 0.0,
                'all_emotions': {emotion: 0.0 for emotion in self.emotions}
            }


class VideoEmotionProcessor:
    """Main processor for video emotion analysis - Optimized for 2-hour videos"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize video processor
        
        Args:
            model_path: Path to ViT model directory (default from environment)
        """
        # Configuration
        self.config = {
            'match_threshold': 0.4,
            'min_confidence': 0.6,
            'frame_interval': 30,  # Process every 30 frames (1 per second at 30fps)
            'face_detection_confidence': 0.9,
            'facenet_model': 'Facenet512',
            'num_classes': 7,
            'max_video_duration': 7200,  # 2 hours in seconds
            
            # Temporal smoothing settings 
            'use_temporal_smoothing': True,
            'smoothing_window_size': 5,      # 5 frames 
            'smoothing_method': 'weighted',  # Use Gaussian weights
            'gaussian_sigma': 1.0,           # Controls weight distribution
        }
        
        # Initialize face detector
        self.face_detector = MTCNN()
        logger.info("MTCNN face detector initialized")
        
        # Initialize ViT emotion model
        if model_path is None:
            model_path = os.environ.get('VIT_MODEL_PATH', './app/model/vit-large-rafdb-staged')
        
        self.emotion_model = ViTEmotionModel(model_path)
        
        # Session storage for tracking progress
        self.sessions = {}
        
        logger.info("VideoEmotionProcessor initialized successfully")
        logger.info(f"Max video duration supported: {self.config['max_video_duration']/3600} hours")
        logger.info(f"Temporal smoothing: {'ENABLED (Gaussian weighted)' if self.config['use_temporal_smoothing'] else 'DISABLED'}")
        if self.config['use_temporal_smoothing']:
            logger.info(f"  Window size: {self.config['smoothing_window_size']} frames")
            logger.info(f"  Method: {self.config['smoothing_method']}")
    
    
    def compute_face_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Compute face embedding from image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Face embedding as numpy array or None if failed
        """
        try:
            logger.info(f"Computing face embedding from {image_path}")
            embedding_obj = DeepFace.represent(
                img_path=image_path,
                model_name=self.config['facenet_model'],
                enforce_detection=True,
                detector_backend='mtcnn'
            )
            
            if embedding_obj and len(embedding_obj) > 0:
                embedding = np.array(embedding_obj[0]['embedding'])
                logger.info(f"Face embedding computed successfully: shape {embedding.shape}")
                return embedding
            
            logger.warning("No face embedding found in image")
            return None
        
        except Exception as e:
            logger.error(f"Error computing embedding: {e}")
            return None
    
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in frame using MTCNN
        
        Args:
            frame: Video frame as numpy array (BGR)
            
        Returns:
            List of detected faces with location and confidence
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.face_detector.detect_faces(rgb_frame)
        
        faces = []
        for detection in detections:
            if detection['confidence'] > self.config['face_detection_confidence']:
                box = detection['box']
                x, y, w, h = box
                # Convert to (top, right, bottom, left) format
                face_location = (y, x + w, y + h, x)
                faces.append({
                    'location': face_location,
                    'confidence': detection['confidence'],
                    'keypoints': detection['keypoints']
                })
        
        return faces
    
    
    def get_face_embedding_from_frame(self, frame: np.ndarray, face_location: Tuple) -> Optional[np.ndarray]:
        """
        Extract face embedding from detected face in frame
        
        Args:
            frame: Video frame as numpy array
            face_location: Face location as (top, right, bottom, left)
            
        Returns:
            Face embedding or None if failed
        """
        try:
            top, right, bottom, left = face_location
            face_img = frame[top:bottom, left:right]
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Save to temp file for DeepFace
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            Image.fromarray(face_rgb).save(temp_file.name)
            temp_file.close()
            
            # Compute embedding
            embedding_obj = DeepFace.represent(
                img_path=temp_file.name,
                model_name=self.config['facenet_model'],
                enforce_detection=False
            )
            
            # Clean up temp file
            os.unlink(temp_file.name)
            
            if embedding_obj and len(embedding_obj) > 0:
                return np.array(embedding_obj[0]['embedding'])
            return None
        
        except Exception as e:
            logger.debug(f"Error getting face embedding from frame: {e}")
            return None
    
    
    def match_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> Dict:
        """
        Compare two face embeddings
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Dictionary with match results
        """
        distance = cosine(embedding1, embedding2)
        similarity = 1 - distance
        is_match = distance < self.config['match_threshold']
        
        return {
            'is_match': is_match,
            'distance': float(distance),
            'similarity': float(similarity),
            'confidence': float(similarity) if is_match else 0.0
        }
    
    
    def identify_patient(self, detected_faces: List[Dict], frame: np.ndarray, 
                        patient_emb: np.ndarray) -> Optional[Dict]:
        """
        Identify which face is the patient
        
        Args:
            detected_faces: List of detected faces
            frame: Video frame
            patient_emb: Patient face embedding
            
        Returns:
            Patient face information or None if not found
        """
        best_match = None
        best_confidence = 0.0
        
        for face in detected_faces:
            face_location = face['location']
            face_emb = self.get_face_embedding_from_frame(frame, face_location)
            
            if face_emb is None:
                continue
            
            match_result = self.match_faces(patient_emb, face_emb)
            
            if match_result['is_match'] and match_result['confidence'] > best_confidence:
                best_confidence = match_result['confidence']
                best_match = {
                    'face_location': face_location,
                    'match_confidence': match_result['confidence'],
                    'detection_confidence': face['confidence']
                }
        
        return best_match
    
    
    def _gaussian_weights(self, window_size: int, sigma: float = None) -> List[float]:
        """
        Generate Gaussian weights for smoothing (center has more weight)
        
        Args:
            window_size: Size of the window
            sigma: Standard deviation for Gaussian (default from config)
            
        Returns:
            List of normalized weights
        """
        if sigma is None:
            sigma = self.config['gaussian_sigma']
        
        center = window_size // 2
        weights = []
        
        for i in range(window_size):
            distance = abs(i - center)
            weight = np.exp(-(distance ** 2) / (2 * sigma ** 2))
            weights.append(weight)
        
        # Normalize weights to sum to 1
        total = sum(weights)
        weights = [w / total for w in weights]
        
        return weights
    
    
    def smooth_predictions_temporal(self, predictions: List[Dict], window_size: int = 5, 
                                   method: str = 'weighted') -> List[Dict]:
        """
        Apply temporal smoothing to emotion predictions using Gaussian weights
        
        Args:
            predictions: List of predictions with 'emotion', 'confidence', 'all_emotions'
            window_size: Number of frames to average (odd number recommended)
            method: 'simple' for uniform averaging, 'weighted' for Gaussian-weighted
            
        Returns:
            Smoothed predictions with more stable emotions
        """
        if len(predictions) < 3:
            logger.info("Too few predictions for smoothing, returning as-is")
            return predictions
        
        logger.info(f"Applying temporal smoothing (window={window_size}, method={method})...")
        
        smoothed = []
        
        for i in range(len(predictions)):
            # Get window indices
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(predictions), i + window_size // 2 + 1)
            window = predictions[start_idx:end_idx]
            
            # Calculate weights
            if method == 'weighted':
                weights = self._gaussian_weights(len(window))
                if i == 0:
                    logger.info(f"📊 Gaussian weights for window size {len(window)}: {[round(w, 3) for w in weights]}")
            else:
                weights = [1.0 / len(window)] * len(window)
            
            # Aggregate emotion probabilities with weights
            emotion_sums = {}
            
            for pred, weight in zip(window, weights):
                for emotion, prob in pred['all_emotions'].items():
                    emotion_sums[emotion] = emotion_sums.get(emotion, 0) + (prob * weight)
            
            # Get dominant emotion after smoothing
            dominant_emotion = max(emotion_sums.items(), key=lambda x: x[1])
            
            # Keep original timestamps
            smoothed.append({
                'start': predictions[i]['start'],
                'end': predictions[i]['end'],
                'emotion': dominant_emotion[0],
                'confidence': round(dominant_emotion[1], 3),
                'all_emotions': {k: round(v, 3) for k, v in emotion_sums.items()}
            })
        
        # Log smoothing statistics
        original_emotions = [p['emotion'] for p in predictions]
        smoothed_emotions = [p['emotion'] for p in smoothed]
        changes = sum(1 for o, s in zip(original_emotions, smoothed_emotions) if o != s)
        
        logger.info(f"Smoothing complete: {changes}/{len(predictions)} predictions changed ({changes/len(predictions)*100:.1f}%)")
        
        return smoothed
    
    
    def _count_emotions(self, predictions: List[Dict]) -> Dict[str, int]:
        """Count emotion distribution"""
        counts = {}
        for pred in predictions:
            emotion = pred['emotion']
            counts[emotion] = counts.get(emotion, 0) + 1
        return counts
    
    
    def _log_emotion_distribution(self, predictions: List[Dict], label: str = ""):
        """Log emotion distribution for debugging"""
        counts = self._count_emotions(predictions)
        total = len(predictions)
        
        logger.info(f"📊 {label} Emotion Distribution:")
        for emotion, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            logger.info(f"   {emotion}: {count} ({percentage:.1f}%)")
    
    
    def get_session_status(self, session_id: str) -> Dict:
        """
        Get processing status for a session
        
        Args:
            session_id: Session UUID
            
        Returns:
            Dictionary with status information
        """
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        return {
            'status': 'not_found',
            'progress': 0,
            'message': 'Session not found'
        }
    
    
    def process_video_simple(self, video_path: str, photo_path: str) -> Dict:
        """
        Simplified video processing with ViT-Large and Gaussian temporal smoothing
        
        Args:
            video_path: Local path to video file
            photo_path: Local path to patient photo
            
        Returns:
            Dictionary with predictions and metadata
        """
        try:
            logger.info("="*60)
            logger.info("STARTING VIDEO PROCESSING WITH VIT-LARGE")
            logger.info("="*60)
            
            # Validate files
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            if not os.path.exists(photo_path):
                raise FileNotFoundError(f"Photo file not found: {photo_path}")
            
            # Compute patient embedding
            logger.info("Computing patient face embedding...")
            patient_embedding = self.compute_face_embedding(photo_path)
            
            if patient_embedding is None:
                raise ValueError("Failed to compute patient embedding")
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Video: {total_frames} frames, {duration:.1f}s, {fps:.2f} FPS")
            logger.info(f"Processing every {self.config['frame_interval']} frames")
            
            # Store raw predictions with ALL emotions
            raw_predictions = []
            frame_count = 0
            processed_count = 0
            patient_detected_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every N frames
                if frame_count % self.config['frame_interval'] == 0:
                    timestamp = frame_count / fps if fps > 0 else 0
                    processed_count += 1
                    
                    # Detect faces
                    detected_faces = self.detect_faces(frame)
                    
                    if detected_faces:
                        # Identify patient
                        patient_face = self.identify_patient(detected_faces, frame, patient_embedding)
                        
                        if patient_face:
                            patient_detected_count += 1
                            
                            # Extract face with margin
                            top, right, bottom, left = patient_face['face_location']
                            margin = 20
                            h, w = frame.shape[:2]
                            top = max(0, top - margin)
                            left = max(0, left - margin)
                            bottom = min(h, bottom + margin)
                            right = min(w, right + margin)
                            
                            face_img = frame[top:bottom, left:right]
                            
                            # Skip small faces
                            if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                                frame_count += 1
                                continue
                            
                            # Predict emotion using ViT model
                            emotion_result = self.emotion_model.predict(face_img)
                            
                            # Format timestamp
                            start_time = self.format_timestamp(timestamp)
                            end_time = self.format_timestamp(timestamp + (self.config['frame_interval'] / fps))
                            
                            # Store raw prediction with ALL emotions
                            raw_predictions.append({
                                "start": start_time,
                                "end": end_time,
                                "emotion": emotion_result['emotion'],
                                "confidence": round(emotion_result['confidence'], 3),
                                "all_emotions": emotion_result['all_emotions']
                            })
                            
                            # Log progress
                            if processed_count % 100 == 0:
                                logger.info(f"Processed {processed_count} frames | "
                                          f"Patient detected: {patient_detected_count} | "
                                          f"Progress: {frame_count/total_frames*100:.1f}%")
                
                frame_count += 1
            
            cap.release()
            
            logger.info(f"Raw processing complete: {len(raw_predictions)} predictions")
            
            if len(raw_predictions) == 0:
                raise ValueError("No predictions generated - patient not detected in video")
            
            # Log raw emotion distribution (before smoothing)
            logger.info("")
            self._log_emotion_distribution(raw_predictions, "BEFORE SMOOTHING")
            
            # Apply Gaussian temporal smoothing
            if self.config['use_temporal_smoothing'] and len(raw_predictions) >= 3:
                logger.info("")
                predictions = self.smooth_predictions_temporal(
                    raw_predictions,
                    window_size=self.config['smoothing_window_size'],
                    method=self.config['smoothing_method']
                )
                
                # Log smoothed emotion distribution
                logger.info("")
                self._log_emotion_distribution(predictions, "AFTER SMOOTHING")
            else:
                predictions = raw_predictions
                logger.info("Smoothing skipped (disabled or too few predictions)")
            
            # Calculate summary from smoothed predictions
            emotion_counts = self._count_emotions(predictions)
            
            # Calculate emotion percentages
            total = len(predictions)
            emotion_percentages = {
                emotion: round((count / total) * 100, 1)
                for emotion, count in emotion_counts.items()
            }
            
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else None
            
            # Calculate average confidence
            avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions) if predictions else 0
            
            logger.info("")
            logger.info("="*60)
            logger.info("PROCESSING COMPLETE")
            logger.info("="*60)
            logger.info(f"Model: ViT-Large (RAF-DB fine-tuned)")
            logger.info(f"Total predictions: {len(predictions)}")
            logger.info(f"Dominant emotion: {dominant_emotion}")
            logger.info(f"Average confidence: {avg_confidence:.3f}")
            logger.info(f"Detection rate: {patient_detected_count / processed_count * 100:.1f}%")
            logger.info(f"Smoothing: {self.config['smoothing_method']} (window={self.config['smoothing_window_size']})")
            logger.info("="*60)
            
            return {
                "success": True,
                "duration_seconds": round(duration, 2),
                "predictions": predictions,
                "summary": {
                    "total_predictions": len(predictions),
                    "dominant_emotion": dominant_emotion,
                    "emotion_distribution": emotion_counts,
                    "emotion_percentages": emotion_percentages,
                    "average_confidence": round(avg_confidence, 3),
                    "detection_rate": round(patient_detected_count / processed_count * 100, 1) if processed_count > 0 else 0,
                    "smoothing_applied": self.config['use_temporal_smoothing'],
                    "smoothing_method": self.config['smoothing_method'],
                    "smoothing_window": self.config['smoothing_window_size'],
                    "model": "ViT-Large (RAF-DB)"
                }
            }
            
        except Exception as e:
            logger.error(f"Error in process_video_simple: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    
    def format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"