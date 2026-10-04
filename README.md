# WasteVision

## End-to-End Computer Vision MLOps Platform

WasteVision is a production-oriented computer vision and MLOps platform for classifying waste images into six categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

The project demonstrates the complete lifecycle of a deployable machine learning system — from reproducible dataset management and model training to experiment tracking, model inference, authentication, database persistence, containerization, testing, and ARM64 deployment preparation.

---

## Project Overview

The goal of WasteVision is not only to train an image classification model, but to build a complete ML system around the model.

The system provides:

- Six-class waste image classification
- Reproducible data management
- PyTorch transfer learning
- MLflow experiment tracking
- DVC-based data/pipeline versioning
- FastAPI inference API
- JWT authentication
- Email verification
- PostgreSQL prediction storage
- User prediction history
- Dockerized production application
- ARM64-compatible container
- Automated testing
- Cloud deployment preparation

---

## Architecture

```text
                    ┌──────────────────┐
                    │    TrashNet      │
                    │     Dataset      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │       DVC        │
                    │ Data Versioning  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Data Preparation │
                    │ & Dataset Split  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    PyTorch       │
                    │ MobileNetV3-Small│
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     MLflow       │
                    │ Experiment Track │
                    │ Model Artifacts  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  FastAPI Backend │
                    │                  │
                    │ Authentication   │
                    │ Inference        │
                    │ Prediction API   │
                    └────────┬─────────┘
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
        ┌─────────────────┐     ┌─────────────────┐
        │   PostgreSQL    │     │   Jinja2 Web UI │
        │                 │     │                 │
        │ Users           │     │ Login           │
        │ Predictions     │     │ Dashboard       │
        │ History         │     │ Image Upload    │
        └─────────────────┘     └─────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │      Docker      │
                    │ Production Image │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ ARM64 Validation │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Oracle Cloud A1  │
                    │   Deployment     │
                    │     Pending      │
                    └──────────────────┘
```

---

# Key Results

| Metric | Result |
|---|---:|
| Classification Classes | 6 |
| Model | MobileNetV3-Small |
| Test Accuracy | **85.00%** |
| Macro Precision | **86.78%** |
| Macro Recall | **83.34%** |
| Macro F1 | **84.04%** |
| Weighted F1 | **85.12%** |
| Model Size | ~6 MB |
| Best Epoch | 6 |
| Inference | CPU compatible |
| ARM64 Image | Validated |

---

# Tech Stack

## Machine Learning

- Python
- PyTorch
- TorchVision
- MobileNetV3-Small
- Transfer Learning
- Pillow
- NumPy
- Pandas
- Scikit-learn

## MLOps

- DVC
- MLflow
- Git
- GitHub
- Reproducible ML pipelines
- Model artifact/version management

## Backend

- FastAPI
- Uvicorn
- Pydantic
- SQLAlchemy
- Alembic

## Authentication & Security

- JWT
- PyJWT
- Argon2
- pwdlib
- HTTP-only authentication cookies
- Email verification

## Database

- PostgreSQL

## Frontend

- HTML
- CSS
- Jinja2
- FastAPI/Jinja2 templates

## DevOps

- Docker
- Linux
- ARM64
- GitHub Actions
- Pytest

## Deployment

- Oracle Cloud Ampere A1
- Nginx
- HTTPS

> Oracle Cloud deployment is currently pending due to regional A1 capacity availability.

---

# Machine Learning Model

WasteVision uses **MobileNetV3-Small** with PyTorch transfer learning.

MobileNetV3-Small was selected because it provides a good balance between:

- Classification performance
- Model size
- CPU inference performance
- Memory requirements
- Deployment cost

This makes it suitable for lightweight cloud and edge-oriented deployment.

---

# Dataset

The project uses the **TrashNet** dataset.

The model performs six-class classification:

```text
cardboard
glass
metal
paper
plastic
trash
```

The original dataset is treated as immutable.

Data preparation and dataset splitting are designed to be reproducible, while hidden files, unsupported files, and non-image files are ignored.

Large datasets are not committed directly to Git.

---

# MLOps Pipeline

WasteVision follows an ML lifecycle rather than treating model training as a standalone script.

```text
Raw Dataset
     │
     ▼
    DVC
     │
     ▼
Data Preparation
     │
     ▼
Train / Validation / Test
     │
     ▼
Model Training
     │
     ▼
    MLflow
     │
     ├── Parameters
     ├── Training Metrics
     ├── Validation Metrics
     ├── Model Artifacts
     └── Experiment History
     │
     ▼
Model Artifact
     │
     ▼
FastAPI Inference
     │
     ▼
Docker
     │
     ▼
Deployment
```

---

# DVC

DVC is used for reproducible dataset and ML pipeline management.

The project separates:

- Raw data
- Processed data
- Dataset splits
- Training
- Evaluation

The objective is to make model development reproducible rather than manually modifying datasets between experiments.

---

# MLflow

MLflow is used for experiment tracking and model management.

Tracked information includes:

- Training parameters
- Training loss
- Validation loss
- Accuracy
- Precision
- Recall
- F1 score
- Model artifacts
- Model versions
- Experiment information

The trained baseline model is stored as:

```text
artifacts/mlflow_baseline/best_model.pt
```

---

# Backend API

WasteVision provides a FastAPI REST API.

## Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "service": "WasteVision"
}
```

## Readiness Check

```http
GET /ready
```

Example response:

```json
{
  "status": "ready",
  "service": "WasteVision"
}
```

## Version

```http
GET /version
```

Example:

```json
{
  "service": "WasteVision",
  "version": "0.1.0"
}
```

## Prediction

```http
POST /api/v1/predict
```

The endpoint accepts an image and returns the predicted waste category.

Example response:

```json
{
  "prediction_id": 8,
  "prediction": "cardboard",
  "confidence": 0.9999585,
  "model": "wastevision",
  "version": "1",
  "inference_time_ms": 1524.27
}
```

---

# Authentication

The application includes user authentication.

Authentication flow:

```text
Register
   │
   ▼
Email Verification
   │
   ▼
Login
   │
   ▼
JWT Token
   │
   ▼
Authenticated API Requests
```

Implemented features:

- User registration
- Password hashing with Argon2
- Email verification
- JWT authentication
- Authenticated prediction requests
- User-specific prediction history

---

# Prediction History

Predictions are persisted in PostgreSQL.

Stored information includes:

- Prediction ID
- User ID
- Predicted class
- Confidence
- Model name
- Model version
- Inference time
- Timestamp

Images themselves are not persisted.

Users can only access their own prediction history.

---

# Web Application

WasteVision includes a lightweight server-rendered web interface using Jinja2.

The interface provides:

- Landing page
- User registration
- Login
- Email verification
- Dashboard
- Image upload
- Prediction results
- Prediction history
- Logout

The frontend intentionally avoids a large JavaScript framework because the primary goal of this project is demonstrating the ML/MLOps backend and deployment workflow.

---

# Project Structure

```text
WasteVision/
│
├── data/
│   ├── dataset-original/
│   ├── processed/
│   └── splits/
│
├── src/
│   └── wastevision/
│       ├── data/
│       ├── models/
│       ├── training/
│       └── inference/
│
├── api/
│   ├── main.py
│   ├── dependencies.py
│   └── routes/
│
├── auth/
│   ├── jwt.py
│   ├── hashing.py
│   ├── email_verification.py
│   └── routes.py
│
├── database/
│   ├── models.py
│   └── session.py
│
├── services/
│   └── email.py
│
├── web/
│   └── routes.py
│
├── templates/
│
├── static/
│
├── tests/
│
├── configs/
│
├── config/
│   └── settings.py
│
├── artifacts/
│   └── mlflow_baseline/
│
├── metrics/
│
├── docker/
│   └── production/
│       └── Dockerfile
│
├── k8s/
│
├── .github/
│   └── workflows/
│
├── dvc.yaml
├── params.yaml
├── requirements.txt
├── requirements-prod.txt
├── alembic.ini
├── .gitignore
├── .dockerignore
├── .dvcignore
└── README.md
```

---

# Docker

The application is packaged as a production-oriented Docker image.

The image includes:

- FastAPI application
- PyTorch CPU runtime
- MobileNetV3-Small model
- Database integration
- Web interface
- Non-root application user
- Container health check

Build the image:

```bash
docker build \
  -f docker/production/Dockerfile \
  -t wastevision:local .
```

Run the container:

```bash
docker run \
  --env-file .env \
  -p 8000:8000 \
  wastevision:local
```

---

# ARM64 Support

The production image was validated for ARM64 compatibility.

```text
Architecture: arm64
Operating System: linux
```

This is important for deployment on ARM-based cloud infrastructure such as Oracle Cloud Ampere A1.

The model was also validated inside the ARM64 container with:

```text
Input:  [1, 3, 224, 224]
Output: [1, 6]
```

---

# Testing

The project includes automated tests covering important application components.

Run the test suite:

```bash
PYTHONPATH=src pytest -v
```

Current result:

```text
47 passed
1 warning
```

Tests cover areas including:

- API endpoints
- Authentication
- User registration
- Login
- Email verification
- Prediction API
- Prediction history
- Database behavior
- Model/inference integration
- Health and readiness checks

---

# CI/CD

GitHub Actions is used to automate project validation.

The CI pipeline is designed to:

```text
Git Push
   │
   ▼
Install Dependencies
   │
   ▼
Run Tests
   │
   ▼
Run Linting
   │
   ▼
Build Docker Image
   │
   ▼
Validation
```

Model retraining is intentionally not triggered on every commit.

---

# Deployment

## Target Architecture

```text
GitHub
   │
   ▼
Oracle Cloud Always Free
Ampere A1 ARM64 VM
   │
   ├── Nginx
   │
   ├── WasteVision FastAPI
   │
   └── PostgreSQL
```

Production PostgreSQL will remain on a private Docker network and will not be publicly exposed.

Public traffic will be served through:

```text
HTTPS
  │
  ▼
Nginx
  │
  ▼
FastAPI :8000
  │
  ▼
PostgreSQL
```

### Deployment Status

- [x] Docker image
- [x] ARM64 image validation
- [x] FastAPI production container
- [x] PostgreSQL integration
- [x] Database migrations
- [x] Authentication flow
- [x] Model inference
- [x] Prediction persistence
- [ ] Oracle A1 public deployment
- [ ] Nginx
- [ ] HTTPS
- [ ] Public API URL

> Oracle Cloud Ampere A1 deployment is currently pending due to capacity availability.

---

# Local Development

## 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd WasteVision
```

## 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure environment variables

Create:

```text
.env
```

Example:

```dotenv
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/wastevision
JWT_SECRET_KEY=<LONG_RANDOM_SECRET>
EMAIL_BACKEND=development
EMAIL_VERIFICATION_BASE_URL=http://127.0.0.1:8000
WEB_COOKIE_SECURE=false
```

Never commit `.env` to Git.

## 5. Run database migrations

```bash
alembic upgrade head
```

## 6. Start the API

```bash
uvicorn api.main:app --reload
```

Application:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

---

# Example Workflow

```text
1. Register user
        ↓
2. Verify email
        ↓
3. Login
        ↓
4. Upload waste image
        ↓
5. MobileNetV3-Small inference
        ↓
6. Return prediction + confidence
        ↓
7. Store prediction in PostgreSQL
        ↓
8. View prediction history
```

---

# Engineering Decisions

### Why MobileNetV3-Small?

The model was selected because the project targets lightweight inference and inexpensive deployment.

### Why FastAPI?

FastAPI provides:

- Type-safe request validation
- Automatic API documentation
- Good Python ML integration
- Lightweight deployment

### Why PostgreSQL?

PostgreSQL provides reliable persistence for:

- Users
- Authentication state
- Prediction history

### Why Docker?

Docker provides a reproducible runtime environment and simplifies deployment across development and cloud environments.

### Why DVC?

DVC provides reproducible dataset and pipeline management without committing large datasets directly to Git.

### Why MLflow?

MLflow provides experiment tracking and model artifact/version management.

### Why ARM64?

ARM64 validation prepares the application for cost-efficient ARM cloud infrastructure such as Oracle Ampere A1.

---

# Security Considerations

The application implements:

- Password hashing with Argon2
- JWT-based authentication
- Email verification
- HTTP-only authentication cookies
- Environment-based secrets
- User-level prediction isolation
- Non-root Docker container
- Private PostgreSQL networking in production

Secrets are intentionally excluded from Git.

---

# Limitations

Current limitations include:

- TrashNet is relatively small compared with production-scale datasets.
- The model is optimized for demonstration and lightweight deployment rather than maximum classification accuracy.
- CPU inference latency depends heavily on the deployment environment.
- Oracle Cloud public deployment is pending infrastructure capacity.
- The current email system supports development/testing and requires a production SMTP/email provider for public deployment.
- The application is not intended to replace industrial waste-sorting systems without additional validation.

---

# Future Improvements

Potential future improvements include:

- Production email provider integration
- Public Oracle Cloud deployment
- HTTPS with Nginx
- GitHub Actions deployment
- Model monitoring
- Prediction monitoring dashboard
- Confidence thresholding
- Model performance monitoring
- Additional waste categories
- Larger production-oriented dataset
- Model optimization / quantization
- Improved inference latency
- Automated model promotion through MLflow

---

# What This Project Demonstrates

WasteVision demonstrates practical experience across the complete ML engineering lifecycle:

```text
Computer Vision
      +
Deep Learning
      +
Data Versioning
      +
Experiment Tracking
      +
Model Management
      +
REST APIs
      +
Authentication
      +
Database Engineering
      +
Testing
      +
Docker
      +
ARM64 Deployment
      +
Cloud Infrastructure
```

The project demonstrates how a trained computer vision model can be transformed into a maintainable, testable, and deployable machine learning application.

---

# License

This project is intended for educational, portfolio, and demonstration purposes.

See the dataset's original license and attribution requirements before redistributing the dataset.

