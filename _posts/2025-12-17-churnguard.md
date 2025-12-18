---
layout: post
title: "ChurnGuard: End-to-End MLOps Pipeline for Customer Churn Prediction"
excerpt: "ChurnGuard is a production-ready machine learning system that predicts customer churn for telecom companies. Beyond the model itself, this project demonstrates the complete MLOps lifecycle: experiment tracking, containerization, multi-service orchestration, and cloud deployment. The architecture mirrors how ML systems are built at Telecom companies, where the ability to deploy and maintain models in production is as critical as model accuracy."
date: 2025-12-18
---

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat&logo=xgboost&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-232F3E?style=flat&logo=amazon-aws&logoColor=white)

ChurnGuard is a production-ready machine learning system that predicts customer churn for telecom companies. Beyond the model itself, this project demonstrates the complete MLOps lifecycle: experiment tracking, containerization, multi-service orchestration, and cloud deployment. The architecture mirrors how ML systems are built at companies like T-Mobile, Verizon, and AT&T, where the ability to deploy and maintain models in production is as critical as model accuracy.

**Live Demo:** `http://54.158.47.223:8000/docs` *(EC2 instance may be stopped to save costs)*

![autodoc ai](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/UI.png?raw=true)

---

## The Problem

Customer churn represents one of the largest controllable costs in the telecom industry. The numbers tell the story:

**Business Impact:**
- Average telecom churn rate: 15-25% annually
- Cost to acquire new customer: $300-400
- Cost to retain existing customer: $50-75
- A 1% reduction in churn can represent $10M+ in annual savings for mid-size carriers

**The Real Challenge:**
Building an accurate churn model is only 20% of the problem. The other 80% is:
- Getting the model into production where it can influence decisions
- Tracking experiments to know which model version performs best
- Ensuring the model runs consistently across environments
- Deploying updates without breaking existing systems
- Monitoring performance over time

Most data scientists can build a model in a Jupyter notebook. Fewer can deploy it as a production service. ChurnGuard demonstrates the complete journey from notebook to cloud endpoint.

---

## The Solution

ChurnGuard implements a full MLOps pipeline with four key components:

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MLOps Pipeline                                 │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │   Training  │    │  Experiment │    │   Docker    │    │    Cloud    │   │
│  │   Pipeline  │───►│  Tracking   │───►│  Container  │───►│  Deployment │   │
│  │  (XGBoost)  │    │  (MLflow)   │    │  (FastAPI)  │    │  (AWS EC2)  │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│                                                                             │
│  Local Development ─────────────────────────────────► Production            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1. ML Model (XGBoost Classifier)

**Why XGBoost?**
- Handles mixed feature types (categorical + numerical) natively
- Built-in regularization prevents overfitting
- Fast training and inference
- Industry standard for tabular data problems

**Training Pipeline:**
```python
# Simplified training flow
def train_model():
    # 1. Load and clean data
    df = load_data("telco_churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # 2. Encode categorical features
    for col in categorical_columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    # 3. Train with logged hyperparameters
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1
    )
    model.fit(X_train, y_train)
    
    # 4. Log everything to MLflow
    mlflow.log_params({"n_estimators": 100, "max_depth": 5})
    mlflow.log_metrics({"roc_auc": 0.84, "accuracy": 0.80})
    mlflow.xgboost.log_model(model, "model")
```

### 2. Experiment Tracking (MLflow)

**The Problem It Solves:**
Without experiment tracking, model development looks like this:
- `model_v1.pkl`, `model_v2_final.pkl`, `model_v2_final_ACTUAL.pkl`
- "Which hyperparameters did I use for that model with 0.84 AUC?"
- "Was that result with the old data or the new data?"

**MLflow Solution:**
Every training run automatically logs:
- **Parameters**: n_estimators, max_depth, learning_rate
- **Metrics**: accuracy, precision, recall, F1, ROC AUC
- **Artifacts**: trained model, encoders, feature names
- **Metadata**: git commit, timestamp, run duration

![autodoc ai](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/MLflow_exoeriment.png?raw=true)

**Key Insight from Experiments:**

I ran two configurations to compare:

| Configuration | n_estimators | max_depth | learning_rate | Accuracy |
|---------------|--------------|-----------|---------------|----------|
| Simple        | 100          | 5         | 0.10          | 79.9%    |
| Complex       | 200          | 7         | 0.05          | 78.9%    |

The simpler model won. More trees and deeper depth led to overfitting, not better performance. This is exactly the kind of insight that's impossible without proper experiment tracking.

![autodoc ai](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/MLflow.png?raw=true)

The parallel coordinates plot visualizes how hyperparameters affect accuracy. Red line (higher accuracy) shows the simpler configuration outperforming the complex one.

### 3. Containerization (Docker + docker-compose)

**The Problem It Solves:**
"It works on my machine" is the most expensive phrase in software development. A model that runs in your Jupyter notebook but fails in production is worthless.

**Docker Solution:**
Package the entire environment, including Python version, dependencies, and model artifacts, into a container that runs identically everywhere.

```dockerfile
FROM python:3.11-slim
WORKDIR /app

# Install dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose for Multi-Service Orchestration:**

Production ML systems rarely run alone. ChurnGuard uses docker-compose to run the API and MLflow tracking server together:

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - mlflow
      
  mlflow:
    image: python:3.11-slim
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0
```

One command (`docker-compose up`) starts everything.

### 4. Cloud Deployment (AWS)

**The Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                        AWS Cloud                                │
│                                                                 │
│   ┌─────────┐         ┌─────────────────────────────────────┐   │
│   │   ECR   │         │         EC2 Instance                │   │
│   │ (image  │◄────────│  ┌───────────────────────────────┐  │   │
│   │ storage)│  pull   │  │    Docker Container           │  │   │
│   └─────────┘         │  │  ┌─────────────────────────┐  │  │   │
│        ▲              │  │  │  FastAPI + XGBoost      │  │  │   │
│        │              │  │  └─────────────────────────┘  │  │   │
│   docker push         │  └───────────────────────────────┘  │   │
│        │              │              │                      │   │
└────────┼──────────────┴──────────────┼──────────────────────────┘   
         │                             │                           
    ┌────┴─────┐                  port 8000                        
    │ Local PC │                       │                           
    │  (CLI)   │                       ▼                           
    └──────────┘               ┌───────────────┐                   
                               │   Internet    │                   
                               │   (users)     │                   
                               └───────────────┘                   
```

**Services Used:**

| Service | Purpose | Why |
|---------|---------|-----|
| **ECR** | Container registry | Private storage for Docker images, integrated with AWS IAM |
| **EC2** | Compute | Runs the container, scales as needed |
| **IAM** | Security | Manages permissions between services |
| **Security Groups** | Firewall | Controls which ports are accessible |

![MLflow Comparison](Screenshots/ec2.png)

**Deployment Flow:**
1. Build Docker image locally
2. Push to ECR (AWS container registry)
3. Launch EC2 instance with IAM role for ECR access
4. User data script automatically pulls image and starts container
5. Security group opens port 8000 to internet

---

## API Design

The API follows REST conventions with three endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Service information |
| `/health` | GET | Health check with model status |
| `/predict` | POST | Churn prediction |

### Prediction Request/Response

**Request:**
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 70.35
}
```

**Response:**
```json
{
  "churn_probability": 0.752,
  "churn_prediction": true,
  "confidence": 0.752,
  "risk_level": "high"
}
```

![MLflow Comparison](Screenshots/prediction.png)

**Risk Level Logic:**
- `high`: probability >= 0.7
- `medium`: probability >= 0.4
- `low`: probability < 0.4

This classification enables business rules like "auto-route high-risk customers to retention team."

---

## Model Performance

**Dataset:** [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- 7,043 customers
- 19 features
- 26.5% churn rate (imbalanced)

**Performance Metrics:**

| Metric | Score | Interpretation |
|--------|-------|----------------|
| ROC AUC | 0.840 | Strong discriminative power |
| Accuracy | 79.9% | Overall correctness |
| Precision | 65.3% | When we predict churn, we're right 65% of the time |
| Recall | 51.9% | We catch 52% of actual churners |
| F1 Score | 57.8% | Harmonic mean of precision/recall |

**Business Interpretation:**

The precision/recall tradeoff matters for business decisions:
- **High precision, lower recall**: Fewer false alarms, but miss some churners
- **High recall, lower precision**: Catch more churners, but waste resources on false positives

For retention campaigns with limited budget, precision matters more. For "save every customer" strategies, recall matters more. The current model balances both.

---

## Key Technical Decisions

### Why FastAPI over Flask?
- **Automatic documentation**: Swagger UI generated from type hints
- **Async support**: Better performance under load
- **Pydantic validation**: Request/response validation built-in
- **Modern Python**: Type hints, async/await

### Why Docker over direct deployment?
- **Reproducibility**: Same environment everywhere
- **Isolation**: Dependencies don't conflict with system
- **Portability**: Works on any machine with Docker
- **Scalability**: Easy to run multiple instances

### Why EC2 over Lambda/SageMaker?
- **Learning value**: Understanding the fundamentals before abstractions
- **Cost control**: ~$7/month vs $50+ for managed services
- **Flexibility**: Full control over the environment
- **Transferable skills**: Docker/EC2 knowledge applies everywhere

### Why MLflow over W&B/Neptune?
- **Open source**: No vendor lock-in
- **Local first**: Works without cloud account
- **Simple**: Does one thing well
- **Industry adoption**: Used at many companies

---

## Quick Start

### Prerequisites
- Python 3.11+
- Docker Desktop

### Local Development

```bash
# Clone repository
git clone https://github.com/pmcavallo/churnguard.git
cd churnguard

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Download data and train model
python data/download_data.py
python -m model.train

# Run API
uvicorn api.main:app --reload

# Access at http://localhost:8000/docs
```

### Docker

```bash
# Build and run single container
docker build -t churnguard:latest .
docker run -p 8000:8000 churnguard:latest

# Or run full stack with docker-compose
docker-compose up --build

# Access:
# - API: http://localhost:8000/docs
# - MLflow: http://localhost:5000
```

### MLflow Experiment Tracking

```bash
# View past experiments
mlflow ui

# Access at http://localhost:5000
```

---

## Project Structure

```
churnguard/
├── api/
│   ├── main.py              # FastAPI application with endpoints
│   └── schemas.py           # Pydantic request/response models
├── model/
│   ├── train.py             # Training script with MLflow integration
│   └── predict.py           # Inference logic with model loading
├── data/
│   └── download_data.py     # Dataset downloader
├── notebooks/
│   └── 01_exploration.ipynb # Initial data exploration
├── artifacts/               # Saved models (gitignored)
├── mlruns/                  # MLflow experiment data (gitignored)
├── Screenshots/             # Documentation images
├── Dockerfile               # Container definition
├── docker-compose.yml       # Multi-service orchestration
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Cost Analysis

**AWS Costs (Monthly):**

| Service | Usage | Cost |
|---------|-------|------|
| EC2 (t3.micro) | Always on | ~$7.50 |
| ECR | Image storage | ~$0.10 |
| Data transfer | Minimal | ~$0.50 |
| **Total** | | **~$8/month** |

**Cost Optimization Tips:**
- Stop EC2 when not demoing (cost drops to ~$1/month for storage)
- Use spot instances for 70% savings
- Set billing alerts to avoid surprises

---

## Lessons Learned

**1. Simpler models often win**

The experiment comparison showed that adding complexity (more trees, deeper depth) hurt performance. This counterintuitive result is only visible with proper experiment tracking.

**2. Docker layer caching matters**

Ordering Dockerfile commands correctly (dependencies before code) reduced rebuild time from 5 minutes to 30 seconds during development.

**3. Security groups are the first debugging step**

When the API didn't respond externally, the issue was security group configuration, not the application. Cloud networking is different from local development.

**4. User data scripts need logging**

EC2 user data runs silently. Adding logging to the startup script would have made debugging faster.

---

## Future Enhancements

| Enhancement | Complexity | Value |
|-------------|------------|-------|
| CI/CD with GitHub Actions | Medium | Auto-deploy on git push |
| Model monitoring (data drift) | Medium | Detect performance degradation |
| A/B testing infrastructure | High | Compare model versions in production |
| Kubernetes deployment | High | Auto-scaling for production load |
| Feature store integration | High | Consistent features across training/serving |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| ML Framework | XGBoost, scikit-learn |
| API | FastAPI, Uvicorn, Pydantic |
| Containerization | Docker, docker-compose |
| Experiment Tracking | MLflow |
| Cloud | AWS (ECR, EC2, IAM) |
| Language | Python 3.11 |

---

*"The gap between a Jupyter notebook and a production ML system is where most data science projects die. ChurnGuard bridges that gap, demonstrating every step from model training to cloud deployment. The model itself is straightforward. The value is in the infrastructure that makes it production-ready."*
