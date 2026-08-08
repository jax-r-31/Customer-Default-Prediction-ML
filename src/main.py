# Importing libraries.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Loading the Dataset.
df = pd.read_csv("../dataset/dataset.csv")
df.head()


# Handling missing values.
numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# Cleaning numeric columns with comma values.
columns_to_clean = ["customer_income", "loan_amnt"]
for col in columns_to_clean:
    df[col] = df[col].replace({",": ""}, regex=True)
    df[col] = pd.to_numeric(df[col], errors="coerce")


# Re-checking for any new missing numeric values.
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())


# Encoding Categorical columns to Numeric Labels.
label_encoder = LabelEncoder()

df["home_ownership"] = label_encoder.fit_transform(df["home_ownership"])
df["loan_intent"] = label_encoder.fit_transform(df["loan_intent"])
df["loan_grade"] = label_encoder.fit_transform(df["loan_grade"])
df["historical_default"] = label_encoder.fit_transform(df["historical_default"])
df["Current_loan_status"] = label_encoder.fit_transform(df["Current_loan_status"])


# Feature Selection.
X = df.drop(["customer_id", "Current_loan_status"], axis=1)
y = df["Current_loan_status"]


# Splitting and Imputing the data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# Training the Model with AdaBoost.
model = AdaBoostClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)


# Evaluating the Model.
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Default", "Default"], yticklabels=["No Default", "Default"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()
print(f"Accuracy: {accuracy*100:.4f}%")

