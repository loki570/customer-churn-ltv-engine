import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
file_path = r"C:\Users\PC\Desktop\Customer_Churn_LTV\data\Customer Churn Prediction & Lifetime Value.xlsx"

df = pd.read_excel(file_path)

# Display missing values
print("Missing Values Before Cleaning")
print(df.isnull().sum())

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df.fillna(0, inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Encode categorical columns
encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col].astype(str))

# Save cleaned dataset
output_path = r"C:\Users\PC\Desktop\Customer_Churn_LTV\data\cleaned_data.csv"

df.to_csv(output_path, index=False)

print("\n✅ Data Preprocessing Completed")
print("Cleaned dataset saved successfully.")