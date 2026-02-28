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


## Tech Stack (Planned)
- Python  
- Pandas, NumPy, Scikit-learn  
- PyTorch & Hugging Face Transformers  
- SHAP (Explainable AI)  
- Sentence Transformers, FAISS  
- Streamlit  

---
