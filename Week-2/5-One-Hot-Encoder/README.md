# 🚗 Handling Categorical Data in Machine Learning  
## One-Hot Encoding & Dummy Variables Explained

A hands-on Machine Learning project demonstrating how to transform text-based categorical variables into numerical representations suitable for regression models.

This project explores two industry-standard encoding methodologies while properly avoiding the **Dummy Variable Trap (Multicollinearity)** — a critical concept for building stable linear models.

---

## 📌 Project Overview

Machine Learning models operate on mathematical computations. They cannot interpret raw categorical strings such as `"BMW"` or `"Audi"` directly.

### ❌ The Incorrect Approach  
Assigning arbitrary integers:

```
Audi = 1  
BMW = 2  
Benz = 3
```

This introduces a false ordinal relationship (e.g., Benz > BMW > Audi), which distorts model learning and leads to misleading predictions.

### ✅ The Correct Approach — One-Hot Encoding  

We create binary indicator columns:

| Audi | BMW | Benz |
|------|------|------|
|  1   |  0   |  0   |
|  0   |  1   |  0   |
|  0   |  0   |  1   |

Each category becomes independent, preserving the nominal nature of the data.

---

## 🧠 Core Concepts Demonstrated

### 1️⃣ Dummy Variable Trap & Multicollinearity

When all encoded columns are included, perfect multicollinearity occurs.

Example:

```
Benz = 1 − Audi − BMW
```

Linear Regression relies on matrix inversion, and perfect multicollinearity breaks the mathematics behind it.

### ✔ Solution: Drop One Column (N − 1 Rule)

Always remove one dummy variable.  
The dropped category becomes the **baseline reference** for comparison.

---

## 🛠 Implementation Approaches

This project validates the mathematical consistency of ML by achieving identical predictions using two different preprocessing strategies.

---

### 🔹 Method A — Data Analyst Approach (Pandas)

- Used `pd.get_dummies()` for fast and readable transformation.
- Removed the baseline category using `.drop()`.
- Clean and intuitive for exploratory data analysis workflows.

---

### 🔹 Method B — ML Engineer Approach (Scikit-Learn)

- Implemented `ColumnTransformer` with `OneHotEncoder`.
- Preserved numerical columns using `remainder='passthrough'`.
- Prevented multicollinearity using NumPy slicing:

```python
X = X[:, 1:]
```

This reflects a production-ready pipeline approach commonly used in ML systems.

---

## 📊 Model Performance & Validation

The models were trained on a dataset of used car prices and evaluated using Linear Regression.

### 🎯 R² Score: **94.17%**

✔ Both Pandas and Scikit-Learn implementations produced **identical predictions down to the decimal**, proving the correctness and consistency of the preprocessing logic.

---

## 🧰 Tech Stack

- **Python 3.10+**
- **Pandas** – Data manipulation & dummy variable creation  
- **NumPy** – Numerical operations & array slicing  
- **Scikit-Learn**
  - `LinearRegression`
  - `OneHotEncoder`
  - `ColumnTransformer`

---

## ▶ How to Run

1. Clone this repository  
2. Ensure `carprices.csv` is in the root directory  
3. Open `one-hot-encoder.ipynb` using:
   - Jupyter Notebook  
   - JupyterLab  
4. Run all cells to reproduce results  

---

## 📚 What This Project Demonstrates

- Proper handling of categorical variables  
- Avoiding multicollinearity in regression  
- Clean preprocessing pipelines  
- Equivalence between analytical and production-grade ML workflows  
- Strong understanding of linear algebra fundamentals in ML  

---

*Built as part of the PITB Machine Learning Internship Portfolio.*
