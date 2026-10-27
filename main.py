
import pandas as pd

file_path = "data/transactions.csv"

print("Loading dataset...")

df = pd.read_csv(file_path, nrows=1000)

print("\nDataset loaded successfully!")
print("\nFirst 5 transactions:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nFraud transaction counts:")
print(df["isFraud"].value_counts())
print("\nDuplicate rows:")
print(df.duplicated().sum())
print("\nTransaction types:")
print(df["type"].value_counts())
print("\nFraud percentage:")
print(df["isFraud"].value_counts(normalize=True) * 100)
print("\nTransaction amount statistics:")
print(df["amount"].describe())
print("\nAverage amount by transaction type:")
print(df.groupby("type")["amount"].mean())
print("\nFraud counts by transaction type:")
print(pd.crosstab(df["type"], df["isFraud"]))

print("\nFraud percentage by transaction type:")
print(df.groupby("type")["isFraud"].mean() * 100)
import matplotlib.pyplot as plt

fraud_counts = df["isFraud"].value_counts()

fraud_counts.plot(kind="bar")

plt.title("Legitimate vs Fraudulent Transactions")
plt.xlabel("Transaction Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")
for index, value in enumerate(fraud_counts):
    plt.text(index, value, str(value), ha="center", va="bottom")

plt.show()
type_counts = df["type"].value_counts()

type_counts.plot(kind="bar")

plt.title("Transaction Count by Type")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")

plt.show()
print("\nFeature and target columns:")

features = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

target = "isFraud"

print("\nFeatures:")
print(features)

print("\nTarget:")
print(target)

print("\nTarget label counts:")
print(df[target].value_counts())
print("\nFeature data types:")
print(df[features].dtypes)
print("\nPreparing features for machine learning:")

X = df[features].copy()
y = df[target].copy()

X = pd.get_dummies(X, columns=["type"], dtype=int)

print("\nPrepared feature columns:")
print(X.columns.tolist())

print("\nPrepared feature shape:", X.shape)
print("\nTarget shape:", y.shape)
print("\nTarget class percentages:")
print(y.value_counts(normalize=True) * 100)