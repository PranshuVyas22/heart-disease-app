# Heart Disease Prediction System

## Overview

This project implements a machine learning–based system for predicting the likelihood of heart disease using clinical attributes. The application is deployed as an interactive web interface, enabling real-time inference based on patient input parameters.

The objective is to support early risk identification by leveraging supervised learning techniques and interpretable feature analysis.

---

## Live Application

https://heart-disease-app-7fkabuvcjgmssw5ccxdtxs.streamlit.app

---

## Methodology

### Data Processing

* Categorical feature encoding using one-hot encoding
* Feature scaling using StandardScaler
* Consistent preprocessing pipeline applied during both training and inference

### Model Development

* Baseline model: Logistic Regression
* Final model: Random Forest Classifier
* Evaluation metrics: Accuracy, Precision, Recall, F1-score
* Performance: ~88% accuracy on test data

### Optimization

* Decision threshold adjusted from 0.5 to 0.4
* Objective: reduce false negatives in a healthcare context

---

## Features

* Real-time prediction based on user input
* Probability-based risk estimation
* Feature importance visualization
* Structured and interpretable input interface
* Deployment via Streamlit

---

## Key Predictive Features

* ST_Slope
* Maximum Heart Rate (MaxHR)
* Oldpeak (ST depression)
* Exercise-Induced Angina
* Cholesterol

---

## Technology Stack

* Python
* scikit-learn
* pandas, numpy
* matplotlib
* Streamlit

---

## Project Structure

```
app.py               # Streamlit application
model.pkl            # Trained Random Forest model
scaler.pkl           # Feature scaler
columns.pkl          # Encoded feature schema
requirements.txt     # Dependencies
HeartFailure.ipynb   # Model development notebook
```

---

## Reproducibility

To run locally:

```bash
git clone https://github.com/PranshuVyas22/heart-disease-app.git
cd heart-disease-app
pip install -r requirements.txt
streamlit run app.py
```

---

## Use Cases

* Clinical risk screening (educational or experimental use)
* Demonstration of end-to-end ML deployment
* Feature importance analysis in tabular healthcare data

---

## Limitations

* Model trained on a specific dataset; may not generalize universally
* Not intended for clinical decision-making without validation
* Performance dependent on input data distribution

---

## Author

Pranshu Vyas
