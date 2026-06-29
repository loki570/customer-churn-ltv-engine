## Encode categorical columns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
df = pd.read_excel(r"C:\Users\abc\Documents\Customer Churn Prediction & Lifetime Value (1) cleaned dataset.xlsx")
print("===== Categorical Columns =====")
print(df.select_dtypes(include=['object', 'string']).columns)
le = LabelEncoder()
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = le.fit_transform(df[col].astype(str))
print("\n===== Dataset After Encoding =====")
print(df.head(10))
print("\n===== Data Types After Encoding =====")
print(df.dtypes)
df.to_excel(r"C:\Users\abc\Documents\Customer_Churn_Encoded.xlsx", index=False)




#Separate Features and Target
import pandas as pd
df = pd.read_excel(r"C:\Users\abc\Documents\Customer Churn Prediction & Lifetime Value (1) cleaned dataset.xlsx")
# Separate Features (Independent Variables)
X = df.drop('Churn', axis=1)
# Separate Target Variable (Dependent Variable)
y = df['Churn']
print("===== FEATURES (X) =====")
print(X.head())
print("\n===== TARGET VARIABLE (y) =====")
print(y.head())
print("\n===== SHAPE OF FEATURES AND TARGET =====")
print("Features (X):", X.shape)
print("Target (y):", y.shape)


#Train-Test Split
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_excel(r"C:\Users\abc\Documents\Customer Churn Prediction & Lifetime Value (1) cleaned dataset.xlsx")
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
print("Training Features (X_train):", X_train.shape)
print("Testing Features (X_test):", X_test.shape)
print("Training Target (y_train):", y_train.shape)
print("Testing Target (y_test):", y_test.shape)


#Logistic Regression
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
df = pd.read_excel(r"C:\Users\abc\Documents\Customer Churn Prediction & Lifetime Value (1) cleaned dataset.xlsx")
le = LabelEncoder()
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = le.fit_transform(df[col].astype(str))
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Train Logistic Regression Model
model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Model Evaluation
print("===== Logistic Regression Results =====")
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall   :", round(recall_score(y_test, y_pred), 4))
print("F1 Score :", round(f1_score(y_test, y_pred), 4))

print("\n===== Classification Report =====")
print(classification_report(y_test, y_pred))


#Random Forest
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_excel(r"C:\Users\abc\Documents\Customer Churn Prediction & Lifetime Value (1) cleaned dataset.xlsx")
df.drop('customerID', axis=1, inplace=True)
le = LabelEncoder()

for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Features and Target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("===== RANDOM FOREST RESULTS =====\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


#XGBOOST Model
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


df = pd.read_excel(r"C:\Users\abc\Documents\Customer Churn Prediction & Lifetime Value (1) cleaned dataset.xlsx")
df.drop('customerID', axis=1, inplace=True)
le = LabelEncoder()

for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# XGBoost Model
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
print("===== XGBOOST RESULTS =====\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


#Model Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
print("\n===== MODEL EVALUATION =====\n")

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Precision
precision = precision_score(y_test, y_pred)
print("\nPrecision:", precision)

# Recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# F1 Score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

# Full Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))