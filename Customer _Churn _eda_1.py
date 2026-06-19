#Analyze Numerical Variables
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_excel(r"C:\Users\abc\Documents\Customer Churn Prediction & Lifetime Value (1) cleaned dataset.xlsx")
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
print("Summary Statistics:")
print(df[numerical_cols].describe())
plt.figure(figsize=(18, 10))

for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram of {col}')
for i, col in enumerate(numerical_cols, 4):
    plt.subplot(2, 3, i)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()



