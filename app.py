
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="UPI Fraud Detection",
    page_icon="🛡️",
    layout="centered"
)


@st.cache_resource
def load_model():
    try:
        return joblib.load("models/fraud_model.pkl")
    except FileNotFoundError:
        st.error(
            "Trained model not found. "
            "Please run: python train_model.py"
        )
        st.stop()
    except Exception as e:
        st.error(f"Could not load the trained model: {e}")
        st.stop()

model = load_model()

st.title("🛡️ UPI Fraud Detection System")
st.write(
    "Enter transaction details to estimate whether "
    "a transaction may be fraudulent."
)

transaction_type = st.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
)

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=1000.0,
    step=100.0
)

oldbalance_org = st.number_input(
    "Sender's Balance Before Transaction",
    min_value=0.0,
    value=5000.0,
    step=100.0
)

newbalance_org = st.number_input(
    "Sender's Balance After Transaction",
    min_value=0.0,
    value=4000.0,
    step=100.0
)

oldbalance_dest = st.number_input(
    "Receiver's Balance Before Transaction",
    min_value=0.0,
    value=2000.0,
    step=100.0
)

newbalance_dest = st.number_input(
    "Receiver's Balance After Transaction",
    min_value=0.0,
    value=3000.0,
    step=100.0
)

step = st.number_input(
    "Time Step",
    min_value=1,
    value=1,
    step=1
)
threshold = st.slider(
    "Fraud Detection Threshold",
    min_value=0.05,
    max_value=0.9,
    value=0.5,
    step=0.05
)
if st.button("Check Transaction", type="primary"):
    transaction = pd.DataFrame([{
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalance_org,
        "newbalanceOrig": newbalance_org,
        "oldbalanceDest": oldbalance_dest,
        "newbalanceDest": newbalance_dest
    }])
    probability = model.predict_proba(transaction)[0][1]

    st.subheader("Prediction Result")
    st.metric("Estimated Fraud Probability", f"{probability:.2%}")
    st.caption(f"Detection threshold: {threshold:.0%}")

    if probability >= threshold:
        st.error("Potential Fraud Detected")
        st.write("This transaction should be reviewed.")
    else:
        st.success("Likely Legitimate Transaction")
        st.write("The model did not flag this transaction at the current threshold.")
   

st.caption(
    "Educational demonstration using a synthetic transaction dataset. "
    "Predictions are estimates, not proof of fraud."
)
