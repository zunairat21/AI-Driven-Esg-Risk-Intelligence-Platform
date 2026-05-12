import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklear.compose import columnTransformer 

def load_data(file_path: str) -> pd.DataFrame:
  """
  Load ESG Financial Dataset from a CSV file
  """

def preprocess_data(df: pd.DataFrame, target_column:str):
  """
  Prepare feature and target , and build preprocessing pipeline

  """
#Separate features and target 
X = df.drop(columns = [target_column])
y = df[target_column]

#Select numeric columns
numeric_features = X.select_dtypes(include = ["int64", "float64"]).columns

#Select numeric preprocessing pipeline
numeric_pipeline = PipeLine(
  steps = [
    ("scale", StandardScaler())
  ]
)
#Apply preprocessing only to numeric columns
preprocessor = ColumnTransformer(
  transformers=[
    ("num", numeric_pipeline, numeric_features)
  ]
)

#Train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

return X_train, X_test, y_train, y_test, preprocessor
    
