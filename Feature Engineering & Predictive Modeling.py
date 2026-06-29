#Tenure Group and customer count in each group
import pandas as pd
df = pd.read_excel(r"C:\Users\abc\Documents\Customer Churn Prediction & Lifetime Value (1) cleaned dataset.xlsx")
df['TenureGroup'] = pd.cut(
    df['tenure'],
    bins=[-1, 12, 24, 48, 72],
    labels=['0-12', '13-24', '25-48', '49-72']
)
print(df[['tenure', 'TenureGroup']])
print("\nTenure Group Distribution:")
print(df['TenureGroup'].value_counts())

#Average Revenue Per Month
import pandas as pd
df = pd.read_excel(r"C:\Users\abc\Documents\Customer Churn Prediction & Lifetime Value (1) cleaned dataset.xlsx")
df['AvgRevenue'] = df['TotalCharges'] / (df['tenure'] + 1)
print(df[['tenure', 'TotalCharges', 'AvgRevenue']].head(50))

#Long-Term Customer
import pandas as pd
df = pd.read_excel(r"C:\Users\abc\Documents\Customer Churn Prediction & Lifetime Value (1) cleaned dataset.xlsx")
df['LongTermCustomer'] = df['tenure'].apply(
    lambda x: 1 if x > 24 else 0
)
print(df[['tenure', 'LongTermCustomer']])
print("\nLong-Term Customer Distribution:")
print(df['LongTermCustomer'].value_counts())

#Remove Customer ID
import pandas as pd
df = pd.read_excel(r"C:\Users\abc\Documents\Customer Churn Prediction & Lifetime Value (1) cleaned dataset.xlsx")
df.drop('customerID', axis=1, inplace=True)






