# 📈 Linear Regression: Sales Prediction & Data Stability Analysis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange.svg" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-lightgrey.svg" />
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-green.svg" />
  <img src="https://img.shields.io/badge/Status-Completed-success.svg" />
</p>

---

## 📌 Project Overview

This project analyzes the relationship between **Marketing Ad Spend** and **Sales Revenue** using **Linear Regression**.

Beyond basic prediction, the experiment demonstrates the **Law of Large Numbers** by evaluating how different dataset sizes impact:

- Model variance  
- Coefficient stability  
- Reliability of ROI (Return on Investment) estimation  

Three datasets were tested:

- **N = 15 (Small Sample)**
- **N = 40 (Medium Sample)**
- **N = 100 (Large Sample)**

The objective is to show how increasing sample size improves model stability and financial prediction accuracy.

---

## 🎯 Business Objective

Estimate **Return on Ad Spend (ROAS)** using a regression model:

Sales = m(Ad Spend) + b

yaml
Copy code

Where:

- `m` → Estimated ROAS (Return on Ad Spend)
- `b` → Baseline sales without advertising

A stable slope (`m`) means reliable investment forecasting.

---

## 📊 Experimental Design

| Dataset Size | Variance Level | Stability | ROI Reliability |
|--------------|---------------|-----------|-----------------|
| N = 15       | High          | Low       | Weak            |
| N = 40       | Moderate      | Improved  | Acceptable      |
| N = 100      | Low           | Strong    | Reliable        |

---

## 📉 Key Insights

- Small datasets are highly sensitive to noise and outliers.
- Medium datasets improve coefficient consistency.
- Large datasets produce stable slope estimates.
- Results confirm the **Law of Large Numbers** in applied machine learning.
- Financial modeling becomes more reliable as data volume increases.

---

## 🛠 Tech Stack

- **Python 3.10+**
- **Scikit-Learn** – Model training
- **Pandas** – Data manipulation
- **Matplotlib** – Visualization

---

## 📁 Project Structure

PITB-ML-Internship/
│

├── linear_regression_analysis.py

├── requirements.txt

└── README.md

---

## 🚀 How to Run

### 1️⃣ Clone the Repository

git clone https://github.com/mzu-2410z/PITB-ML-Internship.git
2️⃣ Navigate into the Folder
cd PITB-ML-Internship
3️⃣ Install Dependencies (Optional but Recommended)
pip install -r requirements.txt
4️⃣ Run the Script
python linear_regression_analysis.py
📈 What This Project Demonstrates
✅ Linear Regression implementation
✅ Statistical intuition behind dataset size
✅ Model variance behavior
✅ Financial interpretation of regression coefficients
✅ Practical understanding of the Law of Large Numbers

🔮 Future Enhancements
Add confidence intervals for coefficient estimates

Implement Ridge & Lasso Regression

Add cross-validation comparison

Compare against Polynomial Regression

Use real-world marketing datasets

👨‍💻 Author
Muhammad Zil E Umar 
Machine Learning Intern
