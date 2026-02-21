# Employee Retention Prediction (Logistic Regression) 📊

A Python implementation of a **Logistic Regression** classification model to predict employee attrition.

This project analyzes an HR analytics dataset to identify the key factors that cause employees to leave and builds a model to classify future retention probabilities.

It demonstrates the transition from predicting continuous values (Linear Regression) to predicting binary categorical outcomes (Yes/No, Leave/Stay) using probability mapping.

---

## 🧠 The Mathematics

Unlike Linear Regression, which outputs any real value, Logistic Regression outputs a probability between **0 and 1**.

We achieve this by wrapping a linear equation inside a **Sigmoid (Logistic) function**.

---

### 1️⃣ The Linear Combination

First, we compute the weighted sum of the input features (just like Linear Regression).

Where:
- $x_i$ = feature values (e.g., satisfaction, hours, salary)
- $w_i$ = learned weights (coefficients)
- $b$ = bias (intercept)

$$
z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b
$$

Or in compact vector form:

$$
z = \mathbf{w}^T \mathbf{x} + b
$$

---

### 2️⃣ The Sigmoid (Logistic) Function

To convert $z$ into a probability, we apply the Sigmoid function:

$$
\hat{y} = \frac{1}{1 + e^{-z}}
$$

This function maps any real number into the interval:

$$
0 < \hat{y} < 1
$$

Where:
- $\hat{y}$ = predicted probability that the employee leaves

---

### 3️⃣ The Decision Boundary

The predicted probability $\hat{y}$ is converted into a class label using a threshold (typically 0.5):

- If $\hat{y} \ge 0.5$ → Predict **1** (Employee Leaves)
- If $\hat{y} < 0.5$ → Predict **0** (Employee Stays)

---

## 🔍 Data Exploration & Engineering

Before modeling, Exploratory Data Analysis (EDA) was performed on **14,999 employee records** to identify the variables most strongly associated with attrition.

---

### 📌 Feature Selection Insights

By grouping the dataset by the target variable (`left`), clear patterns emerged:

| Metric | Employees Who Left (1) | Employees Who Stayed (0) | Impact on Retention |
|--------|------------------------|--------------------------|---------------------|
| **Satisfaction Level** | 0.44 | 0.66 | High — Lower satisfaction increases churn |
| **Average Monthly Hours** | 207 | 199 | Medium — Overwork contributes to churn |
| **Promotion (Last 5 Years)** | 0.5% | 2.6% | Medium — Lack of promotion increases churn |

Cross-tabulation (`pd.crosstab`) also revealed:

- **Salary Bracket** strongly influences retention.
- **Department** was less statistically significant.

---

## ⚙️ Data Preprocessing

### ✅ Selected Features

- `satisfaction_level`
- `average_monthly_hours`
- `promotion_last_5years`
- `salary`

### 🔄 One-Hot Encoding

Machine learning models cannot process categorical text directly.

The `salary` column (`Low`, `Medium`, `High`) was converted into numerical boolean columns using:

```python
pd.get_dummies()
```

This produced:

- `salary_low`
- `salary_medium`
- `salary_high`

---

## 📊 Model Performance

The dataset was split using Scikit-Learn’s `train_test_split`:

- **Training Set:** 80%
- **Testing Set:** 20%

This ensures evaluation on unseen data.

| Metric | Value |
|--------|-------|
| **Total Dataset Size** | 14,999 records |
| **Testing Set Size** | 3,000 records |
| **Model Accuracy** | 78.4% (0.784) |

---

### 🎯 Verdict

The Logistic Regression model successfully predicts employee turnover with **78.4% accuracy**, using only workplace metrics and compensation-related features.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas** — Data manipulation, EDA (`groupby`), encoding (`get_dummies`)
- **Matplotlib** — Data visualization
- **Scikit-Learn** — Data splitting and model building (`train_test_split`, `LogisticRegression`)

---

## 💻 How to Run

1. Clone the repository:

```bash
git clone https://github.com/mzu-2410z/PITB-ML-Internship/
```

2. Install dependencies:

```bash
pip install pandas matplotlib scikit-learn
```

3. Launch Jupyter Notebook:

```bash
jupyter notebook LogisticRegression.ipynb
```

---

*Built as part of the PITB Machine Learning Internship Portfolio*.
