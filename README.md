# Emotion tagging ML service using fast api 

## Docker workflow for the ML-Service
Edit code → create new image → tag the image → push the image into artistry registry in google cloud platform

## Personal Notes
* Google Cloud Run supports only up to 32 MB size for uploads via HTTP1. If you want to upload videos larger than the limit, then we have to look into Google Cloud Storage or Firebase storage for the Fast API to process directly

```

gcloud login
glocud config set project wired-name-467603-a7

gcloud auth configure-docker asia-southeast1-docker.pkg.dev


 docker tag myimage:latest asia-southeast1-docker.pkg.dev/wired-name-467603-a7/ml-service/myimage:latest


docker push asia-southeast1-docker.pkg.dev/wired-name-467603-a7/ml-service/myimage:latest





```

