
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score
)

# Project paths
DATA_PATH = "data/transactions.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading transaction dataset...")

# Read the complete dataset in chunks to manage memory
chunks = []
for chunk in pd.read_csv(DATA_PATH, chunksize=100_000):
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
del chunks

print("Dataset shape:", df.shape)

# Select prediction target and input features
target = "isFraud"

drop_columns = [
    "isFraud",
    "isFlaggedFraud",
    "nameOrig",
    "nameDest"
]

X = df.drop(columns=drop_columns, errors="ignore")
y = df[target].astype(int)

# One-hot encode transaction type; handle numeric and categorical features
categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ))
])

print("Splitting data into training and testing sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training model...")

model.fit(X_train, y_train)

print("\nEvaluating model...")
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

if y_test.nunique() == 2:
    print(
        "Average precision:",
        round(average_precision_score(y_test, probabilities), 4)
    )

joblib.dump(model, MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)
print("Training and evaluation completed.")
