# Credit Lending Risk Analysis

## 📌 Project Overview

Credit Lending Risk Analysis is an end-to-end MLOps project designed to predict customer creditworthiness and classify applicants into four risk categories (`P1`, `P2`, `P3`, `P4`). The project combines machine learning, experiment tracking, model versioning, containerization, CI/CD, and cloud deployment to deliver a production-ready credit risk assessment system.

---

### Application Dashboard
![Credit Risk Dashboard](assets/dashboard.png)

---

## 🎯 Business Problem

Financial institutions must accurately assess the credit risk of loan applicants to reduce defaults and optimize lending decisions. This project leverages machine learning models and MLOps practices to automate risk classification and provide real-time predictions through a web interface.

---

## 🏗️ Project Architecture

```text
Data Ingestion
      ↓
Data Validation
      ↓
Data Transformation
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Model Evaluation
      ↓
MLflow Model Registry
      ↓
FastAPI Inference Service
      ↓
Streamlit Frontend
      ↓
Docker & Docker Compose
      ↓
GitHub Actions CI/CD
      ↓
AWS EC2 Deployment
```

---

## 📂 Project Structure

```text
Credit_Lending_Risk_Analysis/

├── app/                        # FastAPI application
├── src/                        # MLOps pipeline components
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── data_transformation.py
│   ├── feature_engineering.py
│   ├── model_trainer.py
│   ├── model_evaluation.py
│   └── model_registry.py
│
├── config/                     # Configuration files
├── data/                       # Raw and processed datasets
├── notebooks/                  # EDA and experimentation notebooks
├── reports/                    # Evaluation reports
├── tests/                      # Unit tests
├── streamlit_app.py            # Streamlit frontend
├── dvc.yaml                    # DVC pipeline
├── params.yaml                 # Pipeline parameters
├── Dockerfile.fastapi          # Backend container
├── Dockerfile.streamlit        # Frontend container
├── docker-compose.yml          # Multi-container orchestration
├── requirements.txt
└── README.md
```

---

## ⚙️ Technologies Used

### Machine Learning

* Scikit-learn
* XGBoost
* Random Forest
* Decision Tree

### MLOps

* MLflow
* DVC
* GitHub Actions
* Docker
* Docker Compose

### Backend

* FastAPI
* Pydantic
* Uvicorn

### Frontend

* Streamlit

### Cloud & Deployment

* AWS EC2
* Docker Hub

### Data Processing

* Pandas
* NumPy

---

## 🔬 Feature Engineering

* Missing value treatment
* Chi-Square Test for categorical variables
* ANOVA Test for numerical variables
* Variance Inflation Factor (VIF) analysis
* Feature selection
* Label Encoding
* Ordinal Encoding

---

## 🤖 Models Trained

* Decision Tree Classifier
* Random Forest Classifier
* XGBoost Classifier

Hyperparameter optimization was performed using RandomizedSearchCV and tracked through MLflow experiments.

---

## 📊 Model Performance

| Metric            | Value  |
| ----------------- | ------ |
| Accuracy          | 75.98% |
| Weighted F1 Score | 77.08% |
| Macro F1 Score    | 70.58% |

### Class-wise Performance

| Class | Precision | Recall | F1 Score |
| ----- | --------- | ------ | -------- |
| P1    | 0.736     | 0.813  | 0.773    |
| P2    | 0.889     | 0.808  | 0.847    |
| P3    | 0.403     | 0.570  | 0.472    |
| P4    | 0.787     | 0.684  | 0.732    |

---

## 🚀 MLOps Features

### DVC Pipeline

* Data versioning
* Pipeline orchestration
* Reproducible experiments

### MLflow

* Experiment tracking
* Model registry
* Model versioning
* Champion model deployment

### Docker

* Separate frontend and backend containers
* Portable deployment environment

### CI/CD

* Automated Docker image build
* Automated image push to Docker Hub
* Automated deployment to AWS EC2 via GitHub Actions

---

## 🌐 Application Components

### FastAPI Backend

* Health endpoint
* Prediction endpoint
* Model loading from MLflow Registry
* Real-time inference

### Streamlit Frontend

* User-friendly interface
* Input validation
* Feature descriptions
* Real-time prediction dashboard

---

## 🐳 Running with Docker Compose

```bash
docker compose up -d
```

Backend:

```text
http://localhost:8000
```

Frontend:

```text
http://localhost:8501
```

---

## 📈 Key Achievements

* Built a complete end-to-end MLOps pipeline.
* Automated training, evaluation, tracking, and deployment workflows.
* Integrated MLflow Model Registry with production inference.
* Implemented CI/CD using GitHub Actions.
* Deployed containerized application on AWS EC2.
* Delivered real-time multiclass credit risk prediction through FastAPI and Streamlit.

---

## 👨‍💻 Author

Sudhanshu Nandan

Email: [sdntheone@gmail.com](mailto:sdntheone@gmail.com)

GitHub: https://github.com/sdntheone

LinkedIn: https://www.linkedin.com/in/sudhanshu-nandan
