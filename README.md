# Telco Customer Churn Prediction & LTV Analytics System


## 1. Executive Summary

### Project Overview
The Telco Customer Churn Prediction & LTV Analytics System is a comprehensive predictive analytics solution designed for telecommunications and subscription-based businesses. This 4-week project aims to leverage machine learning to predict customer churn probability and calculate Customer Lifetime Value (LTV), enabling data-driven retention strategies and optimized marketing investments.

### Objectives
- **Primary Goal**: Build an end-to-end ML pipeline that predicts customer churn with >85% accuracy
- **Secondary Goal**: Calculate accurate Customer Lifetime Value for strategic customer segmentation
- **Tertiary Goal**: Deploy production-ready APIs and interactive dashboards for business stakeholders

### Expected Business Impact
- **Retention Improvement**: 15-25% reduction in churn through proactive intervention
- **Marketing ROI**: 30-40% improvement through targeted campaigns to high-LTV customers
- **Revenue Protection**: Early identification of at-risk customers worth $500K+ in annual revenue
- **Strategic Planning**: Data-driven insights for product development and pricing strategies

### Key Deliverables
1. PostgreSQL database with cleaned, feature-engineered telco customer data
2. Trained ML models (Churn Classifier + LTV Regressor) with SHAP explainability
3. RESTful FastAPI service with single/batch prediction endpoints
4. Interactive Apache Superset/Metabase dashboards
5. Docker-containerized deployment package
6. Comprehensive technical documentation

---

## 2. Technical Architecture

### System Architecture (Text Diagram)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                   │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐         ┌─────────────────────────────────┐  │
│  │  Raw CSV Data    │────────>│   PostgreSQL Database           │  │
│  │  (Telco Churn)   │         │   - customer_data table         │  │
│  └──────────────────┘         │   - predictions_log table       │  │
│                                │   - model_metrics table         │  │
│                                └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ML PROCESSING LAYER                             │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Feature Engineering Pipeline (Python/Pandas)                 │  │
│  │  - Data cleaning & validation                                 │  │
│  │  - Feature encoding (OneHot, Label, Ordinal)                  │  │
│  │  - Feature scaling (StandardScaler)                           │  │
│  │  - Feature selection (correlation analysis, RFE)              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                │                                     │
│                                ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  ML Models (Scikit-Learn, XGBoost)                           │  │
│  │  ┌────────────────────┐    ┌────────────────────────────┐   │  │
│  │  │ Churn Classifier   │    │  LTV Regressor             │   │  │
│  │  │ - Random Forest    │    │  - Linear Regression       │   │  │
│  │  │ - XGBoost          │    │  - Random Forest           │   │  │
│  │  │ - Logistic Reg     │    │  - Gradient Boosting       │   │  │
│  │  └────────────────────┘    └────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                │                                     │
│                                ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Model Explainability (SHAP)                                  │  │
│  │  - Feature importance analysis                                │  │
│  │  - Individual prediction explanations                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       API LAYER (FastAPI)                            │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  RESTful Endpoints                                            │  │
│  │  - POST /predict/churn (single customer)                      │  │
│  │  - POST /predict/churn/batch (multiple customers)             │  │
│  │  - POST /predict/ltv (single customer)                        │  │
│  │  - POST /predict/ltv/batch (multiple customers)               │  │
│  │  - GET /model/metrics (performance stats)                     │  │
│  │  - GET /explain/{customer_id} (SHAP values)                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   VISUALIZATION LAYER                                │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Apache Superset / Metabase Dashboards                        │  │
│  │  - Churn Risk Heatmap                                         │  │
│  │  - LTV Distribution by Segment                                │  │
│  │  - Feature Importance Visualization                           │  │
│  │  - Model Performance Metrics                                  │  │
│  │  - Customer Segmentation Analysis                             │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow Description

1. **Ingestion Phase**: Raw CSV data is loaded into PostgreSQL using SQLAlchemy ORM
2. **Preprocessing Phase**: Python scripts extract data, perform EDA, and engineer features
3. **Training Phase**: Multiple ML models are trained and evaluated; best performers are serialized
4. **Inference Phase**: FastAPI receives requests, loads models, generates predictions with SHAP explanations
5. **Storage Phase**: Predictions are logged back to PostgreSQL for audit and analysis
6. **Visualization Phase**: BI tools connect to PostgreSQL to render real-time dashboards

### Component Interactions

- **Database ↔ ML Models**: SQLAlchemy provides ORM layer for seamless data extraction and model training
- **ML Models ↔ API**: Joblib/Pickle serialization enables model loading in FastAPI endpoints
- **API ↔ Visualization**: REST endpoints provide real-time predictions; dashboards query historical data from PostgreSQL
- **SHAP ↔ API**: Explainability layer generates feature importance for each prediction request

---

## 3. Tech Stack Details

### PostgreSQL Setup

**Version**: PostgreSQL 14+

**Database Schema**:

```sql
-- Main customer data table
CREATE TABLE customer_data (
    customer_id VARCHAR(50) PRIMARY KEY,
    gender VARCHAR(10),
    senior_citizen INTEGER,
    partner VARCHAR(5),
    dependents VARCHAR(5),
    tenure INTEGER,
    phone_service VARCHAR(5),
    multiple_lines VARCHAR(20),
    internet_service VARCHAR(20),
    online_security VARCHAR(20),
    online_backup VARCHAR(20),
    device_protection VARCHAR(20),
    tech_support VARCHAR(20),
    streaming_tv VARCHAR(20),
    streaming_movies VARCHAR(20),
    contract VARCHAR(20),
    paperless_billing VARCHAR(5),
    payment_method VARCHAR(50),
    monthly_charges DECIMAL(10,2),
    total_charges DECIMAL(10,2),
    churn VARCHAR(5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Predictions log table
CREATE TABLE predictions_log (
    prediction_id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customer_data(customer_id),
    churn_probability DECIMAL(5,4),
    churn_prediction VARCHAR(5),
    ltv_prediction DECIMAL(10,2),
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(20)
);

-- Model metrics table
CREATE TABLE model_metrics (
    metric_id SERIAL PRIMARY KEY,
    model_name VARCHAR(50),
    model_version VARCHAR(20),
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    roc_auc DECIMAL(5,4),
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_customer_churn ON customer_data(churn);
CREATE INDEX idx_customer_tenure ON customer_data(tenure);
CREATE INDEX idx_predictions_timestamp ON predictions_log(prediction_timestamp);
```

### Python Libraries

**Core Data Processing**:
- **Pandas (v2.0+)**: DataFrame operations, data cleaning, feature engineering
- **NumPy (v1.24+)**: Numerical computations, array operations
- **SQLAlchemy (v2.0+)**: ORM for PostgreSQL interactions, connection pooling

**Machine Learning**:
- **Scikit-Learn (v1.3+)**: Preprocessing, models, metrics, model selection
- **XGBoost (v2.0+)**: Gradient boosting for churn classification and LTV regression
- **Imbalanced-Learn (v0.11+)**: SMOTE for handling class imbalance

**Model Explainability**:
- **SHAP (v0.42+)**: TreeExplainer, LinearExplainer, visualization plots

**API Development**:
- **FastAPI (v0.104+)**: Async REST API framework
- **Pydantic (v2.0+)**: Request/response validation
- **Uvicorn (v0.24+)**: ASGI server

**Visualization**:
- **Matplotlib (v3.7+)**: Static plots
- **Seaborn (v0.12+)**: Statistical visualizations
- **Plotly (v5.17+)**: Interactive charts

**Utilities**:
- **Joblib (v1.3+)**: Model serialization
- **Python-dotenv (v1.0+)**: Environment variables
- **Psycopg2 (v2.9+)**: PostgreSQL adapter

---

## 4. Detailed 4-Week Timeline

### Week 1: Data Ingestion & Exploratory Data Analysis

**Day 1**: Environment Setup - Install PostgreSQL, Python environment, configure SQLAlchemy, initialize Git

**Day 2**: Data Ingestion - Download dataset, create schema, load CSV to PostgreSQL, validate integrity

**Day 3**: EDA Part 1 - Descriptive statistics, churn distribution, univariate visualizations, identify missing values

**Day 4**: EDA Part 2 - Bivariate analysis, correlation matrix, chi-square tests, document insights

**Day 5**: Data Quality - Handle missing values, detect outliers, validate consistency, create quality report

**Day 6**: Initial Feature Engineering - Create derived features, encode categoricals, prepare scaling, save preprocessed data

**Day 7**: Week 1 Review - Compile EDA findings, create data dictionary, review code, prepare stakeholder presentation

### Week 2: Feature Engineering & Model Development

**Day 8**: Advanced Feature Engineering - Interaction features, aggregate services, time-based features, correlation analysis

**Day 9**: Feature Selection - RFE with Random Forest, mutual information scores, variance threshold, finalize feature set

**Day 10**: Churn Model Part 1 - Train/test split, handle imbalance with SMOTE, train baseline models, evaluate metrics

**Day 11**: Churn Model Part 2 - Train XGBoost and Gradient Boosting, hyperparameter tuning, cross-validation, select best model

**Day 12**: LTV Model - Calculate LTV target, train regression models, evaluate RMSE/MAE/R², select best model

**Day 13**: SHAP Explainability - Configure SHAP, generate values, create visualizations, integrate into pipeline, document top features

**Day 14**: Model Serialization - Serialize models with Joblib, save preprocessing pipeline, validate on test set, log metrics to database

### Week 3: LTV Calculation & API Development

**Day 15**: LTV Refinement - Cohort-based analysis, confidence intervals, customer segmentation, create visualizations

**Day 16**: FastAPI Setup - Initialize project structure, create Pydantic schemas, setup ORM models, configure environment

**Day 17**: Prediction Endpoints - Implement churn and LTV endpoints (single and batch), add validation and error handling

**Day 18**: SHAP Integration - Create explain endpoint, generate force plots, implement feature importance ranking, add caching

**Day 19**: Monitoring Endpoints - Implement metrics endpoint, health checks, request logging, prediction logging to database

**Day 20**: API Testing - Write unit tests with pytest, test all endpoints, generate OpenAPI docs, create Postman collection

**Day 21**: API Optimization - Async processing for batch, rate limiting, model loading optimization, Docker containerization

### Week 4: Visualization, Deployment & Documentation

**Day 22**: Superset Setup - Install via Docker, configure PostgreSQL connection, create datasets, setup authentication

**Day 23**: Dashboard Part 1 - Churn Risk Dashboard with contract analysis, high-risk customers, trends, tenure distribution

**Day 24**: Dashboard Part 2 - LTV Analytics Dashboard with segmentation, top customers, revenue forecast, risk scatter plot

**Day 25**: Dashboard Part 3 - Model Performance Dashboard with confusion matrix, ROC curve, precision-recall, feature importance

**Day 26**: Docker Containerization - Create Dockerfile, docker-compose.yml for multi-container setup, test locally, create env templates

**Day 27**: Documentation - Write comprehensive README, technical documentation, API usage examples, record demo video

**Day 28**: Final Testing - End-to-end integration testing, load test API, validate dashboards, create deployment checklist, stakeholder handover

---

## 5. Data Source Information

### Telco Customer Churn Dataset

**Source**: Kaggle / IBM Sample Data Sets  
**URL**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

**Characteristics**:
- **Rows**: 7,043 customer records
- **Columns**: 21 features (19 predictors + 1 target + 1 ID)
- **Target**: Churn (Yes/No) - Binary classification
- **Distribution**: ~73% No Churn, ~27% Churn (imbalanced)

### Feature Schema

**Demographics** (4): customerID, gender, SeniorCitizen, Partner, Dependents

**Services** (9): tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies

**Account** (5): Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges

**Target**: Churn (Yes/No)

### Data Quality Considerations

- **Missing Values**: TotalCharges has 11 missing (0.16%) - customers with 0 tenure
- **Inconsistencies**: TotalCharges stored as object type (contains spaces)
- **Outliers**: MonthlyCharges >$100 are legitimate premium customers
- **Class Imbalance**: 26.5% churn rate - use SMOTE or class weights

---

## 6. Machine Learning Strategy

### Churn Prediction

**Type**: Binary Classification

**Models**: Logistic Regression (baseline), Random Forest, XGBoost

**Pipeline**:
1. Train/test split (80/20, stratified)
2. Handle imbalance with SMOTE
3. Train multiple models
4. Hyperparameter tuning with GridSearchCV
5. Evaluate with accuracy, precision, recall, F1, ROC-AUC

**Targets**: Accuracy >85%, Precision >80%, Recall >75%, F1 >77%, ROC-AUC >0.85

### LTV Prediction

**Type**: Regression

**Formula**: `LTV = MonthlyCharges × PredictedTenure × (1 - ChurnProbability)`

**Models**: Linear Regression, Random Forest Regressor, XGBoost Regressor

**Metrics**: RMSE <$500, MAE <$400, R² >0.75, MAPE <15%

### SHAP Explainability

**Purpose**: Transparent, interpretable predictions

**Implementation**: TreeExplainer for tree models, generate waterfall/force/summary plots

**Value**: Identify churn drivers, personalize retention strategies, regulatory compliance

---

## 7. API Design

### Endpoints

**POST /predict/churn** - Single customer churn prediction with probability and risk factors

**POST /predict/churn/batch** - Batch churn predictions for multiple customers

**POST /predict/ltv** - Customer lifetime value prediction with confidence intervals

**GET /explain/{customer_id}** - SHAP explanation with feature contributions

**GET /model/metrics** - Model performance statistics (accuracy, precision, recall, etc.)

**GET /health** - API health check

### Request/Response Format

- **Content-Type**: application/json
- **Authentication**: Bearer token (JWT) for production
- **Rate Limiting**: 100 requests/minute per API key
- **Error Handling**: Structured error responses with status codes

---

## 8. Deployment Strategy

### Docker Architecture

**Multi-container setup** with docker-compose:
- **postgres**: PostgreSQL 14 database
- **api**: FastAPI application
- **superset**: Apache Superset dashboards

### Environment Configuration

**.env file** with database credentials, API keys, model paths, logging configuration

### Deployment Options

**Local**: Virtual environment + uvicorn  
**Docker**: docker-compose up for full stack  
**Production**: Managed PostgreSQL (RDS/Cloud SQL) + Container orchestration (ECS/GKE/AKS) + Load balancer + CI/CD pipeline

### Documentation

- README with installation and usage
- API docs (Swagger UI, ReDoc)
- Model documentation
- Deployment guide

---

## 9. Success Metrics

### Model Performance

**Churn**: Accuracy ≥85%, Precision ≥80%, Recall ≥75%, F1 ≥77%, ROC-AUC ≥0.85

**LTV**: RMSE ≤$500, R² ≥0.75, MAPE ≤15%

### Business KPIs

**Retention**: Reduce churn from 26.5% to 20-22% (4+ percentage points)

**Marketing ROI**: Improve from 2:1 to 3:1 through targeted campaigns

**Revenue Protection**: Retain 40% of high-risk customers = $400K saved

**Operational**: API response <200ms, 99.5% uptime, monthly model retraining

### Measurement

- A/B testing (control vs. ML-driven interventions)
- Real-time monitoring dashboards
- Alert system for model degradation

---

## 10. Risk Mitigation

### Challenge 1: Data Quality Issues

**Risk**: Missing values, inconsistencies, outliers

**Mitigation**: Automated validation pipeline, data cleaning scripts, quality alerts, historical snapshots for rollback

### Challenge 2: Model Performance Degradation

**Risk**: Accuracy decreases due to data drift

**Mitigation**: Automated monthly retraining, performance monitoring with alerts, feature distribution tracking, version rollback capability

### Challenge 3: Class Imbalance

**Risk**: Model biased toward majority class

**Mitigation**: SMOTE oversampling, adjust classification threshold, optimize F1-score, use ensemble methods

### Challenge 4: API Performance Under Load

**Risk**: Slow response times or crashes

**Mitigation**: Caching, async processing, request queuing, performance monitoring, rate limiting, load balancing

### Challenge 5: Stakeholder Adoption

**Risk**: Business users don't trust predictions

**Mitigation**: SHAP explanations for transparency, training sessions, user-friendly dashboards, pilot program with success stories

### Challenge 6: Model Interpretability Requirements

**Risk**: Regulatory or business need for explainable decisions

**Mitigation**: SHAP integration from day one, feature importance documentation, decision audit trail, transparent model selection

### Fallback Strategies

- Maintain previous model versions for quick rollback
- Rule-based fallback system if ML fails
- Manual override capability for critical decisions
- Comprehensive logging for debugging
- Regular backup of models and data


