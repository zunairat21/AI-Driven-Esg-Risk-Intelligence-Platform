import pandas as pd 
import joblib

from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import classification_report ,accuracy_score

from data_preprocessing import load_data, preprocess_data

##Create ESG Risk Level Labels

def create_risk_level(df, score_column="ESG_Risk_Score":
  def map_score(score):
    if score <= 30:
      return "Low"
    elif score<=60:
      return "Medium:
    else:
      return "High"


df["ESG_Risk_Level"] = df[score_column].apply(map_score)

return df

##Encode Risk Levels

def encode_risk_level(df):
  label_mapping = {
    "Low":0,
    "Medium":1,
    "High":2
  }

df["Esg_Risk_Level_Label"] =df[ESG_Risk_Level].map(label_mapping)

return df


##Training Function 

def train_risk_level_model(data_path):
  
  ##Load Data
  df = load_data(data_path)

  ##Create risk_level_labels

  df = create_risk_level(df)

  ##Encode risk level
  df = encode_risk_level(df)

  ##Prepare features and target variable 
  target = "ESG_Risk_Level_Label"

  X_train,X_test,y_train, y_test,preprocessor = preprocess_data(df, target)

  ##Fit preprocessing

  preprocessor.fit(X_train)
  X_train_processed = preprocessor.transform(X_train)
  X_test_processed = preprocessor.transform(X_test)

  ##Train Logistic Regression(multiclass )
  model = LogisticRegression(max_iter=1000)

  model.fit(X_train_processed, y_train)

  ##Predictions
  y_pred= model.predict(X_test_processed)

  ##Evaluation

  acc = accuracy_score(y_test, y_pred)

  print("Classification Accuracy:",acc)
  print("\nClassification Report:")
  print(classification_report(y_test, y_pred)

  ##Save model 
  joblib.dump(model,"esg_risk_level_model.pkl")
  joblib.dump(preprocessor,"risk_level_preprocessor.pkl")
  print("\nModel saved successfully.")

  if __name__ = "__main__":
    train_risk_level_model(
      data_path="data/company_esg_financial_dataset.csv"
    )

        
  
  

  
  
                      
