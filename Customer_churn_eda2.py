#Bivariate Analysis (Feature vs Churn Analysis)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_excel(r"C:\Users\abc\Documents\Customer Churn Prediction & Lifetime Value (1) cleaned dataset.xlsx")
plt.figure(figsize=(20, 12))

# 1. Contract Type vs Churn
plt.subplot(2, 3, 1)
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Contract Type vs Churn')
plt.xticks(rotation=20)

# 2. Tenure vs Churn
plt.subplot(2, 3, 2)
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title('Tenure vs Churn')

# 3. Monthly Charges vs Churn
plt.subplot(2, 3, 3)
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges vs Churn')

# 4. Payment Method vs Churn
plt.subplot(2, 3, 4)
sns.countplot(x='PaymentMethod', hue='Churn', data=df)
plt.title('Payment Method vs Churn')
plt.xticks(rotation=45)

# 5. Internet Service vs Churn
plt.subplot(2, 3, 5)
sns.countplot(x='InternetService', hue='Churn', data=df)
plt.title('Internet Service vs Churn')

# 6. Tech Support vs Churn
plt.subplot(2, 3, 6)
sns.countplot(x='TechSupport', hue='Churn', data=df)
plt.title('Tech Support vs Churn')

plt.tight_layout()
plt.show()



