from snowflake.snowpark import Session
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Use Snowflake session
session = Session.builder.getOrCreate()

# Load engineered features from Feature Store
df = session.table("FEATURE_STORE.TITANIC_FEATURES_FINAL").to_pandas()
print("Data loaded from Feature Store")
print(df.head())

# Select features and label
feature_cols = ["SEX_ENCODED", "PCLASS", "AGE", "FAMILY_SIZE", "FARE_PER_PERSON", "HAS_CABIN"]
label_col = "SURVIVED"
X = df[feature_cols].fillna(0)
y = df[label_col]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# Train Logistic Regression
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
print("Model training complete")

# Evaluate accuracy
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Model Accuracy: {acc:.4f}")

# Show sample predictions
results = pd.DataFrame({
    "PassengerID": df.iloc[y_test.index].PASSENGERID.values,
    "Actual": y_test.values,
    "Predicted": preds
})
print("Sample Predictions:")
print(results.head(10))

# Confusion matrix
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Did not survive", "Survived"],
            yticklabels=["Did not survive", "Survived"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
