# 🧪 Model Training and Testing: `train_test_split`

This repository contains my progress for **Week 2** of my AI/ML internship at **PITB**. Following the **Codebasics Machine Learning series**, I implemented and documented how to properly split datasets to evaluate model performance effectively.

---

## 📌 Overview

The core objective of this module was to move beyond simply fitting a model and focus on how well it generalizes to unseen data.

Using `sklearn.model_selection.train_test_split`, I learned how to:

- Prevent overfitting  
- Evaluate model performance on unseen data  
- Measure true predictive accuracy  

---

## 🛠️ Concepts Covered

### 🔹 Data Partitioning
Splitting the dataset into:
- **Training Set** → Used to train the model  
- **Testing Set** → Used to evaluate model performance  

### 🔹 Random State
Understanding how `random_state` ensures reproducible and consistent results across experiments.

### 🔹 Model Evaluation
Using the `.score()` method to compare predicted values against actual test values.

---

## 💻 Implementation Details

- **Library Used:** `scikit-learn`  
- **Model Applied:** Linear Regression  
- **Train/Test Ratio:**  
  - 80/20 split  
  - 70/30 split  

These ratios ensure sufficient data for both training and validation.

---

## 🚀 How to Run

### 1️⃣ Install Required Libraries

    pip install pandas matplotlib scikit-learn

### 2️⃣ Open the Jupyter Notebook

    jupyter notebook train_test_split.ipynb

---

## 📈 Learning Outcomes

By completing this exercise, I can now:

- ✅ Explain why we never test a model on the same data it was trained on  
- ✅ Efficiently split datasets using Python and Scikit-Learn  
- ✅ Evaluate and quantify model accuracy before deployment  
- ✅ Understand how proper validation improves real-world performance  

---

## 📂 Repository Structure

    ├── train_test_split.ipynb
    └── README.md

---

*Built as part of the PITB Machine Learning Internship Portfolio.*
