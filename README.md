# Emotion tagging ML service using FastAPI

## Docker workflow for the ML-Service
Edit code → create new image → tag the image → push the image into Artifact Registry in Google Cloud Platform


```bash
# Authenticate with Google Cloud
gcloud auth login
gcloud config set project wired-name-467603-a7

# Configure Docker to use Google Artifact Registry
gcloud auth configure-docker asia-southeast1-docker.pkg.dev

# Build your Docker image
docker build -t emotion-ml-service:latest .

# Tag your image for Artifact Registry
docker tag emotion-ml-service:latest asia-southeast1-docker.pkg.dev/wired-name-467603-a7/ml-service/emotion-ml-service:latest

# Push your image to Artifact Registry
docker push asia-southeast1-docker.pkg.dev/wired-name-467603-a7/ml-service/emotion-ml-service:latest

# Deploy to Cloud Run
gcloud run deploy emotion-ml-service \
  --image asia-southeast1-docker.pkg.dev/wired-name-467603-a7/ml-service/emotion-ml-service:latest \
  --region asia-southeast1 \
  --platform managed \
  --allow-unauthenticated
```