import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load Dataset
df = pd.read_excel(r"C:\Users\abc\Documents\Customer Churn Prediction & Lifetime Value (1) cleaned dataset.xlsx")
print("Dataset Loaded Successfully")
print(df.head())

# Dataset Information
print("\nDataset Information:")
print(df.info())

print("\nDataset Shape:", df.shape)

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove Customer ID
df.drop("customerID", axis=1, inplace=True)
print("\nCustomer ID Removed Successfully")

# Calculate Customer Lifetime Value (LTV)
df["LTV"] = df["MonthlyCharges"] * df["tenure"]
print("\nLTV Column Created Successfully")
print(df[["MonthlyCharges", "tenure", "LTV"]].head())

# Encode Categorical Variables
le = LabelEncoder()
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = le.fit_transform(df[col].astype(str))
print("\nCategorical Variables Encoded Successfully")
print(df.head())

# Define Features and Target Variable
X = df.drop(["LTV", "MonthlyCharges", "tenure"], axis=1)
y = df["LTV"]
print("\nFeatures and Target Variable Created Successfully")
print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nDataset Split Successfully")
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Display Sample Data
print("\nTraining Data:")
print(X_train.head())
print("\nTarget Values:")
print(y_train.head())

#Linear Regression
from sklearn.linear_model import LinearRegression
# Create Linear Regression Model
lr = LinearRegression()
# Train the Model
lr.fit(X_train, y_train)
# Make Predictions
lr_prediction = lr.predict(X_test)
print("\nLinear Regression Model Trained Successfully")

# Train the Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
# Create Random Forest Regressor
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
# Train the Model
rf.fit(X_train, y_train)
print("\nRandom Forest Regressor Model Trained Successfully")
# Make Predictions
rf_prediction = rf.predict(X_test)
print("\nPredictions Generated Successfully")
# Display First 10 Predictions
print("\nFirst 10 Predicted LTV Values:")
print(rf_prediction[:10])
# Display Total Number of Predictions
print("\nTotal Predictions:", len(rf_prediction))

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
#Evaluate Linear Regression
print("\n========== Linear Regression ==========")
print("Mean Absolute Error (MAE):",
      mean_absolute_error(y_test, lr_prediction))
print("Root Mean Squared Error (RMSE):",
      np.sqrt(mean_squared_error(y_test, lr_prediction)))
print("R² Score:",
      r2_score(y_test, lr_prediction))

#Evaluate Random Forest
print("\n========== Random Forest ==========")
print("Mean Absolute Error (MAE):",
      mean_absolute_error(y_test, rf_prediction))
print("Root Mean Squared Error (RMSE):",
      np.sqrt(mean_squared_error(y_test, rf_prediction)))
print("R² Score:",
      r2_score(y_test, rf_prediction))

#Evaluate XGBoost
from xgboost import XGBRegressor
xgb = XGBRegressor(
    n_estimators=200,
    random_state=42
)
xgb.fit(X_train, y_train)
xgb_prediction = xgb.predict(X_test)

print("XGBoost Regressor Model Trained Successfully")
print("\nFirst 5 Predictions:")
print(xgb_prediction[:5])

print("\n========== XGBoost ==========")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, xgb_prediction))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, xgb_prediction)))
print("R² Score:", r2_score(y_test, xgb_prediction))

import joblib
joblib.dump(lr, "ltv_model.pkl")
print("\nBest Model Saved Successfully")


print("\n==========================================")
print("Best Model Selection")
print("==========================================")

print("\nLinear Regression")
print("MAE :", mean_absolute_error(y_test, lr_prediction))
print("RMSE:", np.sqrt(mean_squared_error(y_test, lr_prediction)))
print("R² Score:", r2_score(y_test, lr_prediction))

print("\nRandom Forest Regressor")
print("MAE :", mean_absolute_error(y_test, rf_prediction))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_prediction)))
print("R² Score:", r2_score(y_test, rf_prediction))

print("\nXGBoost Regressor")
print("MAE :", mean_absolute_error(y_test, xgb_prediction))
print("RMSE:", np.sqrt(mean_squared_error(y_test, xgb_prediction)))
print("R² Score:", r2_score(y_test, xgb_prediction))

print("\nBest Model: Linear Regression")
print("Reason:")
print("- Lowest MAE")
print("- Lowest RMSE")
print("- Highest R² Score")





