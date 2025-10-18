import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import logging
import timm

logger = logging.getLogger(__name__)


class EVAForEmotionRecognition(nn.Module):
    """
    EVA-02 Large based classifier for emotion recognition
    Exactly matching your Colab EVA02Classifier architecture
    """
    
    def __init__(self, num_classes=8, pretrained=True, dropout_rate=0.5):
        super(EVAForEmotionRecognition, self).__init__()
        
        # Load EVA-02 Large from timm
        logger.info("Loading EVA-02 Large model...")
        self.backbone = timm.create_model(
            'eva02_large_patch14_224',  # Using Large model for better accuracy
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'  # Global average pooling
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        logger.info(f"EVA-02 feature dimension: {self.feature_dim}")
        
        # Freeze backbone initially for stable training
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Advanced classification head (same as your EVA02Classifier)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 1024),
            nn.GELU(),  # EVA-02 uses GELU activation
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Extract features from EVA-02 backbone
        features = self.backbone(x)
        
        # Classification
        logits = self.classifier(features)
        return logits
    
    def unfreeze_backbone(self, num_layers_to_unfreeze=-1):
        """
        Unfreeze backbone for fine-tuning
        
        Args:
            num_layers_to_unfreeze: Number of layers to unfreeze (-1 for all)
        """
        if num_layers_to_unfreeze == -1:
            # Unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("Unfroze entire EVA-02 backbone")
        else:
            # Unfreeze last N layers
            if hasattr(self.backbone, 'blocks'):
                total_blocks = len(self.backbone.blocks)
                for i in range(max(0, total_blocks - num_layers_to_unfreeze), total_blocks):
                    for param in self.backbone.blocks[i].parameters():
                        param.requires_grad = True
                logger.info(f"Unfroze last {num_layers_to_unfreeze} blocks of EVA-02")


class EmotionModelHandler:
    """Handler for EVA-02 emotion recognition model"""
    
    def __init__(self, model_path: str = '/Users/leezhiwin/emotion-ml-service/app/model/emotion_model.pth', num_classes: int = 7):
        """
        Initialize the emotion model
        
        Args:
            model_path: Path to the trained model weights (.pth file)
            num_classes: Number of emotion classes (default: 7 for RAF-DB)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.num_classes = num_classes
        
        # RAF-DB emotion labels (7 classes)
        if num_classes == 7:
            self.emotion_labels = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']
        # FER2013 labels (7 classes, different order)
        elif num_classes == 7:
            self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        # FER+ or custom (8 classes)
        elif num_classes == 8:
            self.emotion_labels = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
        else:
            # Generic labels
            self.emotion_labels = [f'emotion_{i}' for i in range(num_classes)]
        
        logger.info(f"Emotion labels: {self.emotion_labels}")
        
        # Image preprocessing transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load model
        self.model = self._load_model(model_path)
        logger.info("Emotion model loaded successfully")
    
    def _load_model(self, model_path: str):
        """
        Load the EVA-02 model with trained weights
        
        Args:
            model_path: Path to model weights
            
        Returns:
            Loaded model
        """
        try:
            logger.info(f"Loading model from {model_path}")
            
            # Create model instance (pretrained=False since we're loading trained weights)
            model = EVAForEmotionRecognition(
                num_classes=self.num_classes,
                pretrained=False,  # We'll load our trained weights
                dropout_rate=0.5
            )
            
            # Load trained weights
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    logger.info(f"Loaded checkpoint from epoch: {checkpoint.get('epoch', 'unknown')}")
                    if 'best_val_acc' in checkpoint:
                        logger.info(f"Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load state dict
            model.load_state_dict(state_dict, strict=True)
            
            # Move to device and set to eval mode
            model = model.to(self.device)
            model.eval()
            
            logger.info("Model loaded and ready for inference")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = image[:, :, ::-1]  # BGR to RGB
            else:
                # Handle grayscale
                image_rgb = np.stack([image] * 3, axis=-1) if len(image.shape) == 2 else image
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb.astype('uint8'))
            
            # Apply transforms
            tensor = self.transform(pil_image)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            return tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def predict(self, image: np.ndarray) -> dict:
        """
        Predict emotion from face image
        
        Args:
            image: Face image as numpy array (BGR format from OpenCV)
            
        Returns:
            Dictionary with:
                - emotion: Predicted emotion label
                - confidence: Confidence score (0-1)
                - all_emotions: Dict of all emotion probabilities
        """
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            # Get predicted emotion
            predicted_emotion = self.emotion_labels[predicted_idx.item()]
            confidence_score = confidence.item()
            
            # Get all emotion probabilities
            all_probs = probabilities[0].cpu().numpy()
            all_emotions = {
                label: float(prob) 
                for label, prob in zip(self.emotion_labels, all_probs)
            }
            
            result = {
                'emotion': predicted_emotion,
                'confidence': round(confidence_score, 4),
                'all_emotions': all_emotions
            }
            
            logger.debug(f"Prediction: {predicted_emotion} (confidence: {confidence_score:.4f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def predict_batch(self, images: list) -> list:
        """
        Predict emotions for a batch of images
        
        Args:
            images: List of face images as numpy arrays
            
        Returns:
            List of prediction dictionaries
        """
        try:
            # Preprocess all images
            tensors = [self.preprocess_image(img) for img in images]
            batch_tensor = torch.cat(tensors, dim=0)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
            
            # Process results
            results = []
            for i in range(len(images)):
                probs = probabilities[i]
                confidence, predicted_idx = torch.max(probs, 0)
                
                predicted_emotion = self.emotion_labels[predicted_idx.item()]
                confidence_score = confidence.item()
                
                all_probs = probs.cpu().numpy()
                all_emotions = {
                    label: float(prob) 
                    for label, prob in zip(self.emotion_labels, all_probs)
                }
                
                results.append({
                    'emotion': predicted_emotion,
                    'confidence': round(confidence_score, 4),
                    'all_emotions': all_emotions
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error during batch prediction: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_architecture': 'EVA-02 Large',
            'num_classes': self.num_classes,
            'emotion_labels': self.emotion_labels,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'input_size': '224x224',
            'backbone_frozen': trainable_params == 0
        }


# Test function
if __name__ == "__main__":
    import cv2
    import os
    
    # Test model loading
    model_path = os.environ.get('EVA_MODEL_PATH', '/Users/leezhiwin/emotion-ml-service/app/model/emotion_model.pth')
    
    print("Testing EmotionModelHandler...")
    print(f"Model path: {model_path}")
    
    try:
        model_handler = EmotionModelHandler(model_path, num_classes=7)
        
        # Print model info
        info = model_handler.get_model_info()
        print("\n📊 Model Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test with dummy image
        print("\n🧪 Testing prediction with dummy image...")
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = model_handler.predict(dummy_image)
        
        print("\n✅ Test prediction result:")
        print(f"  Emotion: {result['emotion']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  All emotions:")
        for emotion, prob in result['all_emotions'].items():
            print(f"    {emotion}: {prob:.4f}")
        
        print("\n✅ Model handler test passed!")
        
    except Exception as e:
        print(f"\n❌ Error testing model handler: {e}")
        import traceback
        traceback.print_exc()