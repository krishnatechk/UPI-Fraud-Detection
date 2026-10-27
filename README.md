# UPI Fraud Detection & Risk Analysis System

## 1. Project Overview
This project uses machine learning to estimate whether a financial transaction may be fraudulent. It provides an interactive web application built with Streamlit.

**Dataset note:** This project uses the PaySim synthetic financial transaction dataset for educational purposes. It is not trained on verified real-world UPI transaction data.

## 2. Features
- Fraud-risk prediction using a Random Forest classifier
- Interactive Streamlit web interface
- Transaction type, amount, and balance inputs
- Estimated fraud probability
- Model evaluation using a confusion matrix, classification report, and average precision

## 3. Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Random Forest Classifier
- Streamlit
- Joblib

## 4. Dataset
The dataset contains 6,362,620 transactions and 11 columns, including transaction type, amount, sender and receiver balances, and fraud labels.

## 5. Model Performance
Results from the current test run:

| Metric | Fraud class (1) |
|---|---:|
| Precision | 0.98 |
| Recall | 0.78 |
| F1-score | 0.87 |
| Average precision | 0.949 |

Recall of 0.78 means the model detected approximately 78% of fraudulent transactions in the test set. Results may differ on other data.

## 6. Project Structure
```text
UPI_Fraud_Detection/
├── data/
│   └── transactions.csv       # Dataset (not tracked by Git)
├── models/
│   └── fraud_model.pkl        # Generated after model training
├── app.py
├── main.py
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 7. How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Start the web application:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

Open the local URL displayed in the terminal.

## 8. Disclaimer
This is an educational project using synthetic data. Its predictions are estimates and should not be used to make real financial decisions.
