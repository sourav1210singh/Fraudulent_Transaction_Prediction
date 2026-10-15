# Fraudulent Transaction Prediction

A credit card fraud detection model, served through a Streamlit app that scores a single
transaction from user input.

## The problem

Fraud is rare. In a realistic transaction dataset the fraudulent class is a fraction of a
percent, so a model that predicts "not fraud" every time still scores high accuracy and is
completely useless. The work here is about catching the rare class without drowning the
business in false alarms.

## Approach

- **Model:** XGBoost classifier
- **Class imbalance:** SMOTE oversampling on the training split
- **Evaluation:** precision and recall rather than accuracy, since accuracy is misleading here

## Feature engineering

The model uses 30+ features, and most of them are derived rather than raw:

| Feature group | Examples |
|---|---|
| Scaled amounts | `log_amt`, `log_city_pop` |
| Time decomposition | year, month, day, hour |
| Behavioural flags | `Is_Late_Night`, `Is_Weekend`, `Is_Outside_Merchant_Hours` |
| Geography | `Distance_km` between cardholder and merchant |
| Encoded categoricals | state, merchant, job |
| One-hot categories | 14 spend categories, gender, age band |

## What the data showed

- Transactions between midnight and 6am carry a higher fraud rate, and the effect strengthens
  as the amount rises
- A large distance between the cardholder's address and the merchant's location is a strong
  signal
- Spend category matters: online categories behave differently from in-person ones

## Running it

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
