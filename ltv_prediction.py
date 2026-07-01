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

#  Features and Target Variable
X = df.drop("LTV", axis=1)
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