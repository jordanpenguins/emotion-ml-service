import cv2
import numpy as np
from model_handler import EmotionModelHandler

# Test model
model = EmotionModelHandler('model/emotion_model.pth')

# Create test image
test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

# Predict
result = model.predict(test_image)

print("✅ Model loaded successfully!")
print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"All emotions: {result['all_emotions']}")