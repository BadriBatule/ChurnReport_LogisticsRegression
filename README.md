# 📉 Customer Churn Forecasting Using Logistic Regression in R

This project uses logistic regression to predict customer churn for a telecom company. The model identifies high-risk customers, helping the business reduce churn through targeted interventions.

---

## 🔍 Project Objective

To build a predictive model that:
- Classifies whether a customer will churn (`Yes` or `No`)
- Identifies key features driving churn
- Enables data-driven decisions for retention strategy

---

## 🗂 Dataset

- **Source:** [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Rows:** 7,043 customers
- **Features:** 20 columns including demographics, contract details, billing, and services

---

## 📦 Technologies Used

- **Language:** R
- **Libraries:**
  - `tidyverse` for data cleaning and visualization
  - `caret` for data partitioning and model evaluation
  - `pROC` for ROC and AUC analysis
  - `e1071` for extended classification metrics

---

## 🧠 Key Steps

1. **Data Cleaning & Preprocessing**
   - Removed `customerID`
   - Converted character variables to factors
   - Handled missing values

2. **Model Development**
   - Trained a logistic regression model (`glm`)
   - Predicted churn probabilities
   - Evaluated performance with confusion matrix and ROC curve

3. **Evaluation Metrics**
   - **Accuracy:** 81.2%
   - **Recall (Sensitivity):** 58.8%
   - **AUC:** 0.85 (Excellent)
   - **Confusion Matrix:** Generated using test dataset

4. **Business-Driven Visualizations**
   - Churn by contract type
   - Churn by tenure group
   - Churn by monthly charges
   - Churn by billing method
   - Churn by paperless billing

---

## 📈 Insights & Recommendations

| Insight                         | Action Recommendation                                      |
|----------------------------------|------------------------------------------------------------|
| Month-to-month contracts churn more | Offer discounts for switching to long-term contracts       |
| New customers churn more        | Focus retention efforts in first 12 months                 |
| Electronic check users churn most | Promote credit card or bank auto-pay billing              |
| Paperless billing increases churn | Improve engagement for paperless users                    |
| High bills linked to churn      | Provide loyalty rewards or flexible billing plans          |

---

## 📂 Repository Structure

