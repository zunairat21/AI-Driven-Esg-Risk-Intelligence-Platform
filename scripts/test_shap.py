import sys
import os 
#Add project root to path 
sys.path.append(os.path.abspath("."))
import pandas as pd 

from src.explainability.shap_explainer import ShapExplainer

##Load dataset 
df = pd .read_csv("data/company_esg_financial_dataset.csv")


##Define target column 
target_column = "ESG_Overall"

##Separate Features
X = df.drop(columns=[target_column])

#Initialize ShapExplainer
explainer = ShapExplainer(
    pipeline_path = "models/rf_regression_pipeline.joblib"
)

#Preprocess data
X_transformed = explainer.preprocess_data(X)

#Build ShapExplainer
explainer.build_explainer(X_transformed)

#Generate shap values
shap_values = explainer.compute_shap_values(X_transformed)

#Create summary plot 
explainer.summary_plot(
    shap_values, X_transformed
)
