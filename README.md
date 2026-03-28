Lunar Seismic Event Classification — Autonomous Structural Health Monitoring for Lunar Habitats
A supervised machine learning pipeline for classifying NASA Apollo lunar seismic events to enable autonomous structural safety monitoring for future Moon bases.

Overview
This project builds an end-to-end ML classification framework using the NASA Apollo Passive Seismic Experiment (PSE) dataset — 13,057 seismic events recorded on the Moon between 1969 and 1977 by four seismometers deployed during the Apollo missions. The goal is to automatically identify the type of each seismic event (deep moonquake, shallow moonquake, meteoroid impact, etc.) to enable future autonomous lunar habitat safety systems to distinguish structurally hazardous events from benign ones — without requiring human supervision.

Problem Statement
Future long-duration human habitats on the Moon face a unique seismic threat. The Moon's highly fractured crust causes seismic energy to reverberate for up to an hour per event — far longer than on Earth. With communication delays making real-time human oversight impractical, there is a critical need for an intelligent onboard classifier that can:

Detect and classify incoming seismic events in real time
Distinguish structurally hazardous events (shallow moonquakes, deep moonquakes) from benign ones (thermal cracking, background noise)
Trigger automated safety alerts and maintenance planning


Dataset
PropertyValueSourceNASA Apollo PSE — Planetary Data System (PDS)Events13,057Recording period1969–1977Seismic stationsS12, S14, S15, S16Features (engineered)108 → 50 after VIF pruningTarget classes8 (Deep MQ classified/unclassified, Meteoroid, Shallow MQ, Short-Period, LM Impact, S-IVB, Special)

Models Trained
ModelAccuracyF1 (Weighted)Naive Baseline58.3%0.577Ridge (L2) Logistic Regression39.3%0.471Lasso (L1) Logistic Regression35.8%0.444Random Forest87.4%0.867Gradient Boosting ⭐87.8%0.871

Key Features of the Pipeline

Inf-safe feature engineering — log amplitude features computed with .clip(lower=0) before log1p() to prevent infinity values; ratio features removed entirely
VIF multicollinearity diagnostics — Variance Inflation Factor analysis to detect and remove redundant features before training linear models
Chronological train/test split — no shuffling, 80/20 time-ordered split to prevent temporal data leakage
TimeSeriesSplit cross-validation — 5-fold CV that preserves event chronology throughout GridSearchCV hyperparameter tuning
Full evaluation suite — Accuracy, Precision, Recall, F1 (weighted + macro), ROC-AUC, confusion matrices, cross-fold stability plots, and bias-variance learning curves
Dual classification tasks — 8-class event type identification + binary hazardous vs non-hazardous detection


Tech Stack
Python 3.10 | pandas | numpy | scikit-learn | matplotlib | seaborn | Jupyter Notebook

Project Structure
├── lunar_seismic_ml_ready.csv       # Feature-engineered dataset (108 cols)
├── LUNAR_SEISMIC_ML.ipynb           # Main notebook — full pipeline
└── ML_Report                        # Full project report

Data Sources

NASA PDS — Apollo Seismic Data
UT Austin — Nakamura Event Catalog
