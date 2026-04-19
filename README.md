# 📊 Customer Churn Analysis Project

> **Predicting customer churn using Exploratory Data Analysis and Machine Learning**

---

## 🧠 Problem Statement

Customer churn is a major problem for businesses — customers stop using services or cancel subscriptions, leading to significant revenue loss. This project analyzes real telecom customer data to **identify the key factors causing churn** and builds a machine learning model to predict which customers are likely to leave.

---

## 🎯 Objectives

- ✅ Identify customers who are likely to churn
- ✅ Analyze factors influencing churn behavior
- ✅ Build a machine learning model to predict churn
- ✅ Provide actionable business solutions to reduce churn

---

## 📂 Dataset Description

**Dataset:** [Telco Customer Churn — IBM Sample Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)

| Feature | Description |
|---|---|
| `customerID` | Unique customer identifier |
| `tenure` | Number of months the customer has stayed |
| `Contract` | Contract type (Month-to-month, One year, Two year) |
| `MonthlyCharges` | Monthly billing amount |
| `TotalCharges` | Total amount charged |
| `Churn` | **Target variable** — Yes / No |

Each row represents **one unique customer**.

---

## 🧹 Data Preprocessing

```python
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
```

**Steps:**
- Converted `TotalCharges` from string to numeric type
- Removed rows with missing values
- Ensured clean, analysis-ready data

---

## 📊 Exploratory Data Analysis (EDA)

### 🔹 1. Churn Distribution

```python
sns.countplot(x='Churn', data=df)
```

![Churn Distribution](Screenshot 2026-04-19 113408)

> 💡 **Insight:** Around **26–27% of customers are churning** — a significant business concern.

---

### 🔹 2. Tenure vs Churn

```python
sns.boxplot(x='Churn', y='tenure', data=df)
```

![Tenure vs Churn](Screenshot 2026-04-19 113422)

> 💡 **Insight:** Churned customers have a **median tenure of ~10 months**, while retained customers average ~38 months. **New users are at much higher risk.**

---

### 🔹 3. Monthly Charges vs Churn

```python
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
```

![Monthly Charges vs Churn](Screenshot 2026-04-19 113439)

> 💡 **Insight:** Customers paying **higher monthly charges tend to churn more** — high expectations may not be met.

---

### 🔹 4. Contract Type vs Churn

```python
sns.countplot(x='Contract', hue='Churn', data=df)
```

![Contract Type vs Churn](Screenshot 2026-04-19 113449)

> 💡 **Insight:** **Month-to-month customers have the highest churn rate.** Two-year contract customers almost never churn.

---

## 🤖 Machine Learning Model

### Encoding Categorical Variables

```python
df_ml = df.copy()
df_ml.drop('customerID', axis=1, inplace=True)
df_ml = pd.get_dummies(df_ml, drop_first=True)
```

### Training the Model

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = df_ml.drop('Churn_Yes', axis=1)
y = df_ml['Churn_Yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print(model.score(X_test, y_test))  # ~79-81% accuracy
```

> 🎯 **Model Accuracy: ~79–81%** using Logistic Regression

---

## 🔍 Feature Importance

```python
importance = pd.Series(model.coef_[0], index=X.columns)
importance.sort_values().tail(10).plot(kind='barh')
plt.title("Top Factors Influencing Churn")
```

![Feature Importance](Screenshot 2026-04-19 113500)

> 💡 **Top churn drivers:**
> - 🥇 `InternetService_Fiber optic` — Highest impact
> - 🥈 `PaperlessBilling_Yes`
> - 🥉 `PaymentMethod_Electronic check`
> - `SeniorCitizen`, `MultipleLines`, `StreamingMovies`

---

## 🧪 Predicting New Customer Churn

```python
new_customer = pd.DataFrame({
    'tenure': [5],
    'MonthlyCharges': [90],
    # ... other features
})

new_customer_encoded = pd.get_dummies(new_customer)
new_customer_encoded = new_customer_encoded.reindex(columns=X.columns, fill_value=0)

prediction = model.predict(new_customer_encoded)
probability = model.predict_proba(new_customer_encoded)

print(f"Will Churn: {prediction[0]}")
print(f"Churn Probability: {probability[0][1]:.2%}")
```

---

## 📊 Key Insights Summary

| Factor | Observation |
|---|---|
| **Tenure** | New customers (< 10 months) churn the most |
| **Monthly Charges** | Higher charges correlate with higher churn |
| **Contract Type** | Month-to-month = highest churn; Two-year = lowest |
| **Internet Service** | Fiber optic users churn more than DSL |
| **Payment Method** | Electronic check users have higher churn |

---

## 💡 Business Recommendations

| Action | Target Group |
|---|---|
| 🎁 Improve onboarding experience | New customers (< 6 months) |
| 💰 Offer discounts for annual/bi-annual plans | Month-to-month customers |
| 🌐 Improve fiber optic service quality | Fiber optic subscribers |
| 📲 Send proactive check-ins and retention offers | High monthly charge customers |
| 🏅 Loyalty rewards program | Long-tenure customers |

---

## 🚀 Business Impact

- 🔍 Identifies high-risk customers **before** they leave
- 📉 Enables proactive retention campaigns
- 💸 Reduces revenue loss from churned accounts
- 😊 Improves overall customer satisfaction

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-green?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-lightblue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plots-red)

| Library | Purpose |
|---|---|
| `pandas` | Data loading and preprocessing |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Data visualization |
| `scikit-learn` | Machine learning model |

---

## 📁 Project Structure

```
customer-churn-analysis/
│
├── FINAL_PROJECT_Customer_Churn_Analysis.ipynb   # Main Colab notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv          # Dataset
├── README.md                                      # Project documentation
└── plots/
    ├── plot1_churn_dist.png
    ├── plot2_tenure.png
    ├── plot3_monthly_charges.png
    ├── plot4_contract.png
    └── plot5_feature_importance.png
```

---

## 🏁 Conclusion

This project successfully identifies the key drivers of customer churn in a telecom company and builds a Logistic Regression model with ~80% accuracy. By combining **exploratory data analysis** with **machine learning**, businesses can proactively identify at-risk customers and take targeted retention actions — reducing churn and improving profitability.

---

*Made with ❤️ using Python, Pandas, and Scikit-Learn*
