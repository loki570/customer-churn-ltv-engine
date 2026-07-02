# Load Dataset new data
import pandas as pd
df=pd.read_excel(r"C:\Users\abc\Documents\Customer Churn Prediction & Lifetime Value (1) cleaned dataset.xlsx")
print(df)


import pandas as pd
df=pd.read_excel(r"C:\Users\abc\Documents\Customer Churn Prediction & Lifetime Value (1) cleaned dataset.xlsx")
print(df.head())
print(df.shape)

#Dataset Information
import pandas as pd
df=pd.read_excel(r"C:\Users\abc\Documents\Customer Churn Prediction & Lifetime Value (1) cleaned dataset.xlsx")
print(df.info())

# Summary Statistics
import pandas as pd
df=pd.read_excel(r"C:\Users\abc\Documents\Customer Churn Prediction & Lifetime Value (1) cleaned dataset.xlsx")
print(df.describe())

#Check missing values
import pandas as pd
df=pd.read_excel(r"C:\Users\abc\Documents\Customer Churn Prediction & Lifetime Value (1) cleaned dataset.xlsx")
print(df.isnull().sum())

#Check Duplicate Records
import pandas as pd
df=pd.read_excel(r"C:\Users\abc\Documents\Customer Churn Prediction & Lifetime Value (1) cleaned dataset.xlsx")
print(df.duplicated().sum())

#Analyze Churn Distribution
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_excel(r"C:\Users\abc\Documents\Customer Churn Prediction & Lifetime Value (1) cleaned dataset.xlsx")
churn_count = df['Churn'].value_counts()
churn_percentage = df['Churn'].value_counts(normalize=True) * 100
print("Customer Count:")
print(churn_count)
print("\nCustomer Percentage:")
print(churn_percentage.round(2))
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

