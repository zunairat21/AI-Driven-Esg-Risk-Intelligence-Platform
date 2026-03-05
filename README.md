# AI-Driven ESG Risk Intelligence Platform

## Business Problem
Manual assessment of Environmental, Social, and Governance (ESG) risks from large volumes of unstructured news and company data is time-consuming, subjective, and difficult to scale. Investors, organizations, and regulators require transparent and data-driven ESG risk evaluation to support responsible and informed decision-making.

## Proposed AI Solution
This project aims to develop an AI-driven ESG risk intelligence platform that applies machine learning and natural language processing to ESG-related data. The system is designed to:
- Predict ESG risk scores (regression)
- Classify ESG risk levels (low, medium, high)
- Explain model predictions using explainable AI techniques (SHAP)
- Enable natural-language ESG question answering through a Retrieval-Augmented Generation (RAG) approach

## Project Roadmap
This repository is intentionally developed **phase by phase** to reflect real-world AI system development:

- Phase 0: Project foundation and business problem definition  
- Phase 1: ESG risk score prediction (regression)  
- Phase 2: ESG risk level classification  
- Phase 3: Explainable AI using SHAP  
- Phase 4: ESG news sentiment analysis using FinBERT  
- Phase 5: RAG-based ESG question answering  
- Phase 6: Streamlit application integration
## Phase 1 – ESG Risk Score Regression (Baseline Completed)

Phase 1 focuses on building a clean and reliable baseline model to predict ESG risk scores from structured ESG and financial data.

**What was implemented:**
- Modular data preprocessing using `Pipeline` and `ColumnTransformer`
- Proper train-test split with prevention of data leakage
- Baseline Linear Regression model
- Model evaluation using MAE, RMSE, and R² metrics
- Trained model and preprocessing pipeline saved for reuse

This phase establishes a strong foundation for further model improvements and downstream ESG risk analysis.


## Phase 1.1 – Regression Model Improvement (Random Forest)

To improve upon the baseline Linear Regression model, a Random Forest Regressor was introduced to capture non-linear relationships and feature interactions within ESG and financial data.

**Enhancements introduced:**
- Random Forest Regressor trained using the same preprocessing and train-test split
- Fair model comparison using MAE, RMSE, and R² metrics
- Automated selection of the best-performing model based on test RMSE
- Best model persisted for downstream usage

This enhancement demonstrates a systematic approach to model improvement while maintaining evaluation fairness and reproducibility.

## Phase 2 – ESG Risk Level Classification

In Phase 2, the project extends beyond predicting a numerical ESG risk score by introducing a classification system that categorizes companies into interpretable ESG risk levels.

To achieve this, the continuous ESG risk score was converted into categorical labels using defined thresholds:

Low Risk

Medium Risk

High Risk

This transformation enables easier interpretation for business stakeholders such as investors, analysts, and compliance teams.

The classification pipeline reuses the existing preprocessing workflow developed in Phase 1 to maintain consistency and prevent duplicated data handling logic.

A baseline multiclass classification model using Logistic Regression was trained and evaluated using accuracy and detailed classification metrics including precision, recall, and F1-score.

This phase establishes a structured classification framework for ESG risk analysis.

## Phase 2.1 – Model Improvement with Random Forest Classifier

After establishing Logistic Regression as a baseline classifier, the model performance was further improved by introducing a Random Forest Classifier.

Random Forest was selected because it can capture non-linear relationships and complex feature interactions that are common in ESG and financial datasets.

Both models were trained using the same preprocessing pipeline and evaluated using consistent classification metrics to ensure a fair comparison.

The system automatically selects and saves the best-performing model based on evaluation results, ensuring that the most effective classifier is used for downstream prediction tasks.


## Tech Stack (Planned)
- Python  
- Pandas, NumPy, Scikit-learn  
- PyTorch & Hugging Face Transformers  
- SHAP (Explainable AI)  
- Sentence Transformers, FAISS  
- Streamlit  

---
