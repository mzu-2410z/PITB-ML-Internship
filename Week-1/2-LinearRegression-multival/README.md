# 🚀 Startup Profit Predictor: Multiple Linear Regression

## Overview

This project applies **Multiple Linear Regression** to predict the **Profit** of a startup based on its operational expenses. 

The goal is to build a data-driven tool for **Venture Capitalists (VCs)** to estimate the potential ROI of a new company before investing, simply by analyzing their spending habits in three key areas: **R&D**, **Administration**, and **Marketing**.

---

## 📂 The Dataset

I expanded the analysis beyond standard samples by generating and testing datasets of varying sizes (**N=40** and **N=90**) to ensure model consistency.

**Features (Independent Variables):**

1. **R&D Spend:** Money spent on Research & Development
2. **Administration:** Cost of running the office/management
3. **Marketing Spend:** Money spent on advertising and growth

**Target (Dependent Variable):**

- **Profit:** The net earnings of the startup

---

## 🛠 Tech Stack

- **Python 3.10+**
- **Scikit-Learn** (Multiple Linear Regression)
- **Pandas & NumPy** (Data Generation & Processing)
- **Matplotlib** (Visualization)

---

## 📊 Key Findings & Insights

After training the model on multiple datasets, the **Coefficients** revealed the "Secret Formula" for profitability:

### 1. **R&D is King 👑**

- **Coefficient:** ~0.85 - 0.96
- **Insight:** For every $1 spent on R&D, the startup generates almost $1 in pure profit. This is the single biggest driver of success.

### 2. **Marketing is Secondary 📢**

- **Coefficient:** ~0.03 - 0.10
- **Insight:** Marketing helps, but it has a much lower ROI compared to product innovation.

### 3. **Administration is a "Silent Killer" 📉**

- **Coefficient:** Often Negative or Near Zero
- **Insight:** High administrative costs often correlate with *lower* profits. Lean operations win.

---

## 🧪 Experiments

I tested the model stability across two synthetic datasets:

- **Dataset A (40 Rows):** Showed high variance in the Admin coefficient
- **Dataset B (90 Rows):** Confirmed the dominance of R&D spend as the primary predictor, stabilizing the model for real-world use

---

## 🚀 How to Run

1. Navigate to the project folder:

```bash
cd Week-1/2-LinearRegression-multival
```

2. Run the analysis:

```bash
python startup_analysis.py
```

3. (Optional) Generate new synthetic data:

```bash
python generate_data.py
```

---

*Built as part of the PITB Machine Learning Internship Portfolio.*