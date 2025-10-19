import os
import cv2
import numpy as np
import tempfile
import requests
from typing import Dict, List, Tuple, Optional
from PIL import Image
from mtcnn import MTCNN
from deepface import DeepFace
from scipy.spatial.distance import cosine
import logging
from model_handler import EmotionModelHandler
import time
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore, storage

logger = logging.getLogger(__name__)



class VideoEmotionProcessor:
    """Main processor for video emotion analysis - Optimized for 2-hour videos"""
    
    def __init__(self):
        # Configuration
        self.config = {
            'match_threshold': 0.4,
            'min_confidence': 0.6,
            'frame_interval': 30,  # Process every 30 frames (1 per second at 30fps)
            'face_detection_confidence': 0.9,
            'facenet_model': 'Facenet512',
            'num_classes': 7,
            'batch_save_size': 50,  # Save to Firebase every 50 records for long videos
            'max_video_duration': 7200,  # 2 hours in seconds
        }
        
        # # Initialize Firebase with service account
        # if not firebase_admin._apps:
        #     # Path to your service account key
        #     cred_path = os.path.join(
        #         os.path.dirname(os.path.dirname(__file__)),
        #         'credentials',
        #         'fyp-mcs17-firebase-adminsdk-fbsvc-78498f8376.json'
        #     )
            
        #     # Check if file exists
        #     if not os.path.exists(cred_path):
        #         raise FileNotFoundError(
        #             f"Firebase credentials not found at: {cred_path}\n"
        #             "Please download your service account key from Firebase Console"
        #         )
            
        #     # Initialize with credentials
        #     cred = credentials.Certificate(cred_path)
        #     firebase_admin.initialize_app(cred, {
        #         'storageBucket': 'fyp-mcs17.firebasestorage.app'  
        #     })
        #     logger.info(f"Firebase initialized with credentials from {cred_path}")
        
        # # Initialize Firestore and Storage
        # self.db = firestore.client()
        # self.bucket = storage.bucket()

        # logger.info("Firebase services initialized successfully")
        
        # Initialize face detector
        self.face_detector = MTCNN()
        logger.info("MTCNN face detector initialized")
        
        # Initialize emotion model
        model_path = os.path.join("app", "model", "emotion_model.pth")
        self.emotion_model = EmotionModelHandler(model_path, self.config['num_classes'])
        
        # Session storage for tracking progress
        self.sessions = {}
        
        logger.info("VideoEmotionProcessor initialized successfully")
        logger.info(f"Max video duration supported: {self.config['max_video_duration']/3600} hours")
    
    def download_file(self, url: str, filename: str, source: str = 'auto') -> str:
        """
        Download file from URL (supports Firebase, HTTP)
        Optimized for large files with progress tracking
        
        Args:
            url: File URL
            filename: Local filename
            source: 'firebase', 'http', or 'auto' to detect
            
        Returns:
            Path to downloaded file
        """
        try:
            # Auto-detect source
            if source == 'auto':
                if 'firebase' in url or 'firebasestorage' in url or url.startswith('gs://'):
                    source = 'firebase'
                else:
                    source = 'http'
            
            logger.info(f"Downloading {filename} from {source}: {url[:100]}...")
            
            # Download using HTTP (works for Firebase signed URLs)
            return self._download_http(url, filename)
            
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            raise
    
    def _download_http(self, url: str, filename: str) -> str:
        """
        Download file via HTTP with progress tracking
        Optimized for large video files
        """
        start_time = time.time()
        response = requests.get(url, stream=True, timeout=600)  # 10 min timeout for initial connection
        response.raise_for_status()
        
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=os.path.splitext(filename)[1]
        )
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks for better performance
        last_log_time = time.time()
        
        for chunk in response.iter_content(chunk_size=chunk_size):
            temp_file.write(chunk)
            downloaded += len(chunk)
            
            # Log progress every 5 seconds or every 50MB
            current_time = time.time()
            if (current_time - last_log_time > 5) or (downloaded % (50 * 1024 * 1024) == 0):
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    speed = downloaded / (current_time - start_time) / 1024 / 1024  # MB/s
                    logger.info(f"Download progress: {progress:.1f}% ({downloaded / 1024 / 1024:.1f} MB) - Speed: {speed:.2f} MB/s")
                else:
                    logger.info(f"Downloaded: {downloaded / 1024 / 1024:.1f} MB")
                last_log_time = current_time
        
        temp_file.close()
        elapsed_time = time.time() - start_time
        logger.info(f"Download complete: {temp_file.name} ({downloaded / 1024 / 1024:.2f} MB in {elapsed_time:.1f}s)")
        return temp_file.name
    
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
    
    def save_to_firebase_batch(self, session_id: str, emotion_data: List[Dict]) -> bool:
        """
        Save emotion analysis results to Firebase Firestore in batches
        Optimized for large datasets from long videos
        
        Args:
            session_id: Session UUID
            emotion_data: List of emotion records
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not emotion_data:
                logger.warning("No emotion data to save")
                return True
            
            logger.info(f"Saving {len(emotion_data)} records to Firebase Firestore in batches")
            
            # Firestore batch write (max 500 operations per batch)
            batch_size = 500
            total_batches = (len(emotion_data) - 1) // batch_size + 1
            
            for i in range(0, len(emotion_data), batch_size):
                batch_data = emotion_data[i:i + batch_size]
                batch = self.db.batch()
                
                for data in batch_data:
                    # Create document reference with auto-generated ID
                    doc_ref = self.db.collection('emotion_analysis').document()
                    
                    # Prepare record
                    record = {
                        'session_id': session_id,
                        'frame': data['frame'],
                        'timestamp': data['timestamp'],
                        'emotion': data['emotion'],
                        'confidence': data['confidence'],
                        'match_confidence': data['match_confidence'],
                        'num_faces': data['num_faces'],
                        'emotion_probabilities': data['all_emotions'],
                        'created_at': firestore.SERVER_TIMESTAMP
                    }
                    
                    batch.set(doc_ref, record)
                
                # Commit batch
                batch.commit()
                logger.info(f"Saved batch {i//batch_size + 1}/{total_batches} ({len(batch_data)} records)")
            
            logger.info(f"All {len(emotion_data)} records saved successfully to Firebase")
            return True
            
        except Exception as e:
            logger.error(f"Error saving to Firebase: {e}", exc_info=True)
            return False
    
    def save_to_firebase(self, session_id: str, emotion_data: List[Dict]) -> bool:
        """
        Wrapper for batch save (backward compatibility)
        """
        return self.save_to_firebase_batch(session_id, emotion_data)
    
    def _get_emotion_summary(self, emotion_data: List[Dict]) -> Dict:
        """
        Generate summary statistics from emotion data
        
        Args:
            emotion_data: List of emotion records
            
        Returns:
            Dictionary with summary statistics
        """
        if not emotion_data:
            return {
                'total_records': 0,
                'emotion_distribution': {},
                'average_confidence': 0.0,
                'average_match_confidence': 0.0,
                'dominant_emotion': None,
                'emotion_percentages': {}
            }
        
        # Count emotions
        emotion_counts = {}
        for data in emotion_data:
            emotion = data['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Calculate percentages
        total_records = len(emotion_data)
        emotion_percentages = {
            emotion: round((count / total_records) * 100, 2)
            for emotion, count in emotion_counts.items()
        }
        
        # Find dominant emotion
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else None
        
        # Calculate averages
        avg_confidence = sum(d['confidence'] for d in emotion_data) / len(emotion_data)
        avg_match_confidence = sum(d['match_confidence'] for d in emotion_data) / len(emotion_data)
        
        return {
            'total_records': total_records,
            'emotion_distribution': emotion_counts,
            'emotion_percentages': emotion_percentages,
            'dominant_emotion': dominant_emotion,
            'average_confidence': round(avg_confidence, 3),
            'average_match_confidence': round(avg_match_confidence, 3)
        }
    
    def process_video_files(self, patient_id: str, session_id: str, video_id: str,
                           video_path: str, photo_path: str) -> Dict:
        """
        Process video using local file paths (for Celery task with file uploads)
        Optimized for 2-hour videos with incremental saving and progress tracking
        
        Args:
            patient_id: Patient UUID
            session_id: Session UUID
            video_id: Video UUID
            video_path: Local path to video file
            photo_path: Local path to patient photo
            
        Returns:
            Dictionary with processing results
        """
        
        start_time = time.time()
        
        # Update session status
        self.sessions[session_id] = {
            'status': 'processing',
            'progress': 0,
            'start_time': datetime.now().isoformat()
        }
        
        # Save session to Firebase for persistence
        try:
            self.db.collection('processing_sessions').document(session_id).set({
                'status': 'processing',
                'progress': 0,
                'patient_id': patient_id,
                'video_id': video_id,
                'start_time': firestore.SERVER_TIMESTAMP
            })
        except Exception as e:
            logger.warning(f"Failed to save session to Firebase: {e}")
        
        try:
            # Validate files exist
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            if not os.path.exists(photo_path):
                raise FileNotFoundError(f"Photo file not found: {photo_path}")
            
            # Compute patient embedding
            self.sessions[session_id] = {'status': 'computing_embedding', 'progress': 10}
            logger.info("Computing patient face embedding...")
            patient_embedding = self.compute_face_embedding(photo_path)
            
            if patient_embedding is None:
                raise ValueError("Failed to compute patient embedding. Ensure face is visible in photo.")
            
            logger.info(f"Patient embedding computed: {patient_embedding.shape}")
            
            # Process video
            self.sessions[session_id] = {'status': 'processing_video', 'progress': 20}
            logger.info(f"Processing video from {video_path}...")
            
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Video info: {total_frames} frames, {duration:.1f}s ({duration/60:.1f} min), {fps:.2f} FPS, {width}x{height}")
            
            # Check if video is too long
            if duration > self.config['max_video_duration']:
                logger.warning(f"Video duration ({duration:.1f}s) exceeds maximum ({self.config['max_video_duration']}s)")
                logger.warning(f"Consider increasing frame_interval for better performance")
            
            emotion_data = []
            emotion_data_buffer = []  # Buffer for incremental saving
            frame_count = 0
            processed_count = 0
            patient_detected_count = 0
            last_save_time = time.time()
            
            logger.info(f"Processing every {self.config['frame_interval']} frames")
            logger.info(f"Expected samples: ~{total_frames // self.config['frame_interval']}")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every N frames
                if frame_count % self.config['frame_interval'] == 0:
                    timestamp = frame_count / fps if fps > 0 else 0
                    processed_count += 1
                    
                    # Update progress
                    if total_frames > 0:
                        progress = 20 + int((frame_count / total_frames) * 70)
                        elapsed = time.time() - start_time
                        estimated_total = elapsed / (frame_count / total_frames) if frame_count > 0 else 0
                        remaining = estimated_total - elapsed
                        
                        self.sessions[session_id] = {
                            'status': 'processing_video',
                            'progress': progress,
                            'processed_frames': processed_count,
                            'total_expected': total_frames // self.config['frame_interval'],
                            'elapsed_time': f"{elapsed/60:.1f} min",
                            'estimated_remaining': f"{remaining/60:.1f} min" if remaining > 0 else "calculating..."
                        }
                        
                        # Update Firebase every 5%
                        if progress % 5 == 0:
                            try:
                                self.db.collection('processing_sessions').document(session_id).update({
                                    'progress': progress,
                                    'status': 'processing_video',
                                    'processed_frames': processed_count
                                })
                            except:
                                pass
                    
                    # Detect faces
                    detected_faces = self.detect_faces(frame)
                    
                    if detected_faces:
                        # Identify patient
                        patient_face = self.identify_patient(detected_faces, frame, patient_embedding)
                        
                        if patient_face:
                            patient_detected_count += 1
                            
                            # Extract face for emotion analysis
                            top, right, bottom, left = patient_face['face_location']
                            margin = 20
                            h, w = frame.shape[:2]
                            top = max(0, top - margin)
                            left = max(0, left - margin)
                            bottom = min(h, bottom + margin)
                            right = min(w, right + margin)
                            
                            face_img = frame[top:bottom, left:right]
                            
                            # Skip if face is too small
                            if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                                logger.debug(f"Face too small at frame {frame_count}, skipping")
                                continue
                            
                            # Predict emotion using EVA-02
                            emotion_result = self.emotion_model.predict(face_img)
                            
                            # Store result
                            record = {
                                'frame': frame_count,
                                'timestamp': round(timestamp, 2),
                                'emotion': emotion_result['emotion'],
                                'confidence': emotion_result['confidence'],
                                'match_confidence': patient_face['match_confidence'],
                                'num_faces': len(detected_faces),
                                'all_emotions': emotion_result['all_emotions']
                            }
                            
                            emotion_data.append(record)
                            emotion_data_buffer.append(record)
                            
                            # Incremental save for long videos (every 50 records or every 5 minutes)
                            current_time = time.time()
                            if len(emotion_data_buffer) >= self.config['batch_save_size'] or \
                               (current_time - last_save_time) > 300:  # 5 minutes
                                logger.info(f"Incremental save: {len(emotion_data_buffer)} records")
                                if self.save_to_firebase_batch(session_id, emotion_data_buffer):
                                    emotion_data_buffer = []  # Clear buffer after successful save
                                    last_save_time = current_time
                                else:
                                    logger.warning("Incremental save failed, will retry later")
                            
                            # Log progress
                            if processed_count % 100 == 0:
                                logger.info(f"Processed {processed_count} frames | "
                                          f"Patient detected: {patient_detected_count} ({patient_detected_count/processed_count*100:.1f}%) | "
                                          f"Latest: {emotion_result['emotion']} ({emotion_result['confidence']:.2f}) | "
                                          f"Progress: {frame_count/total_frames*100:.1f}%")
                
                frame_count += 1
            
            cap.release()
            
            processing_time = time.time() - start_time
            logger.info(f"Video processing complete in {processing_time/60:.1f} minutes")
            logger.info(f"Processed {processed_count} frames, detected patient in {patient_detected_count} frames")
            
            # Check if we detected the patient at all
            if patient_detected_count == 0:
                logger.warning("Patient was not detected in any frame!")
            
            # Save remaining buffer data
            if emotion_data_buffer:
                logger.info(f"Saving final batch: {len(emotion_data_buffer)} records")
                self.sessions[session_id] = {'status': 'saving_results', 'progress': 90}
                success = self.save_to_firebase_batch(session_id, emotion_data_buffer)
                
                if not success:
                    raise Exception("Failed to save final batch to Firebase")
            
            # Clean up files
            try:
                if os.path.exists(video_path):
                    os.unlink(video_path)
                    logger.info(f"Cleaned up video file: {video_path}")
                if os.path.exists(photo_path):
                    os.unlink(photo_path)
                    logger.info(f"Cleaned up photo file: {photo_path}")
            except Exception as e:
                logger.warning(f"Error cleaning up files: {e}")
            
            # Update final status
            self.sessions[session_id] = {'status': 'completed', 'progress': 100}
            
            # Update Firebase session status
            try:
                self.db.collection('processing_sessions').document(session_id).update({
                    'status': 'completed',
                    'progress': 100,
                    'completed_at': firestore.SERVER_TIMESTAMP,
                    'total_records': len(emotion_data),
                    'processing_time_minutes': round(processing_time / 60, 2)
                })
            except Exception as e:
                logger.warning(f"Failed to update session status: {e}")
            
            # Prepare response
            result = {
                'success': True,
                'session_id': session_id,
                'patient_id': patient_id,
                'video_id': video_id,
                'total_records': len(emotion_data),
                'processing_time': round(processing_time, 2),
                'processing_time_minutes': round(processing_time / 60, 2),
                'video_info': {
                    'total_frames': total_frames,
                    'duration': round(duration, 2),
                    'duration_minutes': round(duration / 60, 2),
                    'fps': round(fps, 2),
                    'resolution': f"{width}x{height}",
                    'processed_frames': processed_count,
                    'patient_detected': patient_detected_count,
                    'detection_rate': round(patient_detected_count / processed_count * 100, 1) if processed_count > 0 else 0,
                    'frame_interval': self.config['frame_interval']
                },
                'emotion_summary': self._get_emotion_summary(emotion_data)
            }
            
            logger.info(f"Processing complete: {len(emotion_data)} emotion records in {processing_time/60:.1f} minutes")
            logger.info(f"Average processing speed: {processed_count / processing_time:.2f} frames/sec")
            return result
            
        except Exception as e:
            logger.error(f"Error processing video: {e}", exc_info=True)
            self.sessions[session_id] = {'status': 'error', 'progress': 0, 'error': str(e)}
            
            # Update Firebase with error
            try:
                self.db.collection('processing_sessions').document(session_id).update({
                    'status': 'error',
                    'error': str(e),
                    'failed_at': firestore.SERVER_TIMESTAMP
                })
            except:
                pass
            
            # Clean up files on error
            try:
                if video_path and os.path.exists(video_path):
                    os.unlink(video_path)
                if photo_path and os.path.exists(photo_path):
                    os.unlink(photo_path)
            except:
                pass
            
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id,
                'patient_id': patient_id,
                'video_id': video_id
            }
    
    def process_video(self, patient_id: str, session_id: str, 
                     patient_photo_url: str, video_url: str,
                     patient_photo_source: str = 'auto',
                     video_source: str = 'auto') -> Dict:
        """
        Process video using URLs (downloads files first)
        Optimized for 2-hour videos
        
        Args:
            patient_id: Patient UUID
            session_id: Session UUID
            patient_photo_url: URL to patient photo
            video_url: URL to video
            patient_photo_source: Source type for photo ('auto', 'firebase', 'http')
            video_source: Source type for video ('auto', 'firebase', 'http')
            
        Returns:
            Dictionary with processing results
        """
        
        # Update session status
        self.sessions[session_id] = {'status': 'downloading', 'progress': 0}
        
        patient_photo_path = None
        video_path = None
        
        try:
            # Download patient photo
            logger.info("Downloading patient photo...")
            patient_photo_path = self.download_file(
                patient_photo_url, 
                'patient.jpg',
                source=patient_photo_source
            )
            
            # Download video
            logger.info("Downloading video...")
            video_path = self.download_file(
                video_url, 
                'video.mp4',
                source=video_source
            )
            
            # Process using file paths
            result = self.process_video_files(
                patient_id=patient_id,
                session_id=session_id,
                video_id=session_id,  # Use session_id as video_id if not provided
                video_path=video_path,
                photo_path=patient_photo_path
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in process_video: {e}", exc_info=True)
            self.sessions[session_id] = {'status': 'error', 'progress': 0, 'error': str(e)}
            
            # Clean up downloaded files
            try:
                if patient_photo_path and os.path.exists(patient_photo_path):
                    os.unlink(patient_photo_path)
                if video_path and os.path.exists(video_path):
                    os.unlink(video_path)
            except:
                pass
            
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id
            }
    
    def get_session_status(self, session_id: str) -> Dict:
        """
        Get processing status for a session
        
        Args:
            session_id: Session UUID
            
        Returns:
            Dictionary with status information
        """
        # Check in-memory cache first
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        # Check Firebase Firestore for session
        try:
            doc = self.db.collection('processing_sessions').document(session_id).get()
            if doc.exists:
                data = doc.to_dict()
                return {
                    'status': data.get('status', 'unknown'),
                    'progress': data.get('progress', 0),
                    'message': 'Session found in Firebase'
                }
        except Exception as e:
            logger.error(f"Error checking session in Firebase: {e}")
        
        # Check if emotion analysis results exist
        try:
            query = self.db.collection('emotion_analysis').where('session_id', '==', session_id).limit(1)
            results = query.stream()
            
            if any(results):
                return {
                    'status': 'completed',
                    'progress': 100,
                    'message': 'Session found in database'
                }
        except Exception as e:
            logger.error(f"Error checking emotion records: {e}")
        
        return {
            'status': 'not_found',
            'progress': 0,
            'message': 'Session not found'
        }

    # video_processor.py - Add this method to VideoEmotionProcessor class

    def process_video_simple(self, video_path: str, photo_path: str) -> Dict:
        """
        Simplified video processing - just analyze and return predictions
        No Firebase/Firestore saving - backend handles that
        
        Args:
            video_path: Local path to video file
            photo_path: Local path to patient photo
            
        Returns:
            Dictionary with predictions and metadata
        """
        import cv2
        
        try:
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
            
            predictions = []
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
                            
                            # Extract face
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
                            
                            # Predict emotion
                            emotion_result = self.emotion_model.predict(face_img)
                            
                            # Format timestamp
                            start_time = self.format_timestamp(timestamp)
                            end_time = self.format_timestamp(timestamp + (self.config['frame_interval'] / fps))
                            
                            # Add prediction
                            predictions.append({
                                "start": start_time,
                                "end": end_time,
                                "emotion": emotion_result['emotion'],
                                "confidence": round(emotion_result['confidence'], 3)
                            })
                            
                            # Log progress
                            if processed_count % 100 == 0:
                                logger.info(f"Processed {processed_count} frames | "
                                        f"Patient detected: {patient_detected_count} | "
                                        f"Progress: {frame_count/total_frames*100:.1f}%")
                
                frame_count += 1
            
            cap.release()
            
            logger.info(f"Processing complete: {len(predictions)} predictions")
            
            # Calculate summary
            emotion_counts = {}
            for pred in predictions:
                emotion = pred['emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else None
            
            return {
                "success": True,
                "duration_seconds": round(duration, 2),
                "predictions": predictions,
                "summary": {
                    "total_predictions": len(predictions),
                    "dominant_emotion": dominant_emotion,
                    "emotion_distribution": emotion_counts,
                    "detection_rate": round(patient_detected_count / processed_count * 100, 1) if processed_count > 0 else 0
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