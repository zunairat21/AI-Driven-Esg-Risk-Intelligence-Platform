# # ESG Risk Score regression training pipeline

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import load_data, preprocess_data


def train_model(data_path: str, target_column: str):
    """
    Train and evaluate an ESG risk score regression model.
    """

    # 1. Load data
    df = load_data(data_path)

    # 2. Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
        df, target_column
    )

    # 3. Fit preprocessor ONLY on training data
    preprocessor.fit(X_train)

    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 4. Train regression model
    model = LinearRegression()
    model.fit(X_train_processed, y_train)

    # 5. Evaluate model
    y_pred = model.predict(X_test_processed)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print("Model Evaluation Metrics:")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # 6. Save model and preprocessor
    joblib.dump(model, "esg_risk_score_model.pkl")
    joblib.dump(preprocessor, "preprocessor.pkl")

    print("Model and preprocessor saved successfully.")


if __name__ == "__main__":
    # Update the path locally when running
    train_model(
        data_path="data/company_esg_financial_dataset.csv",
        target_column="ESG_Risk_Score"
    )

##-------Random Forest Regression------
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_processed, y_train)
rf_pred=rf_model.predict(X_test_processed)

rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = mean_squared_error(y_test, rf_pred, squared=False)
    rf_r2 = r2_score(y_test, rf_pred)

    print("\nRandom Forest Evaluation Metrics:")
    print(f"MAE  : {rf_mae:.4f}")
    print(f"RMSE : {rf_rmse:.4f}")
    print(f"R²   : {rf_r2:.4f}")

#---Model selection-----
if rf_rmse < rmse:
    best_model = rf_model
    best_model_name = "RandomForestRegressor"
else:
    best_model = model
    best_model_name = "LinearRegression"

joblib.dump(best_model,"best_esg_risk_score_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")

print(f"\nBest Model selected :{best_model_name}")

