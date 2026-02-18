# 🤖 Machine Learning Internship - Week 1
## The Foundations: Regression, Optimization & Persistence

**"Don't just call the library. Build the engine."**

This repository documents the work and key learnings from Week 1 of my Machine Learning Internship journey. The focus was on **reverse-engineering** the core algorithms behind libraries like Scikit-Learn to understand the mathematics of how machines actually "learn."

---

## 📚 Core Concepts & Mathematics

### 1. Simple Linear Regression
We started with the fundamental algorithm for predicting a continuous value based on a single input variable.

* **Hypothesis:** Approximating the relationship with a straight line.  
    $$y = mx + b$$

* **Cost Function (Mean Squared Error):** Measuring the average squared difference between predicted and actual values.  
    $$J(m, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - (mx_i + b))^2$$

---

### 2. Multivariate Linear Regression
Expanded the logic to handle multiple input features (e.g., predicting home price based on *Area*, *Bedrooms*, and *Age*).

* **Hypothesis:**  
    $$y = m_1x_1 + m_2x_2 + ... + m_n x_n + b$$

* **Vectorized Implementation:** Instead of loops, we utilized Linear Algebra (Dot Products) for efficiency.  
    $$y = \mathbf{w} \cdot \mathbf{x} + b$$

---

### 3. Gradient Descent (The Optimization Engine) 📉
Instead of using the "black box" `.fit()` method, I implemented the optimization algorithm from scratch using Calculus.

#### The Partial Derivatives (The Gradients)
To find the direction of the steepest descent, we calculated the derivatives of the cost function with respect to the weights ($m$) and bias ($b$).

$$\frac{\partial J}{\partial m} = -\frac{2}{n} \sum_{i=1}^{n} x_i (y_i - y_{pred})$$

$$\frac{\partial J}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - y_{pred})$$

#### The Update Rule (The Learning Step)
We iteratively updated the parameters by moving against the gradient, controlled by the Learning Rate ($\alpha$).

$$m_{new} = m_{curr} - \alpha \cdot \frac{\partial J}{\partial m}$$  
$$b_{new} = b_{curr} - \alpha \cdot \frac{\partial J}{\partial b}$$

---

### Key Feature: Dynamic Convergence

Implemented a "smart stop" system using `math.isclose()` to automatically halt training when the cost function stabilizes, saving computational resources.

---

## 💾 Model Persistence

Learned how to move from "Scripts" to "Production" by saving trained models to disk. This allows for instant predictions without retraining.

* **Pickle:** Standard Python serialization (Good for general objects).  
* **Joblib:** Optimized serialization for models with large NumPy arrays (Recommended for Scikit-Learn models).

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **NumPy:** For vectorization, matrix operations, and manual gradient calculations.
* **Pandas:** For data ingestion and cleaning.
* **Scikit-Learn:** Used primarily for benchmarking our custom algorithms.
* **Joblib & Pickle:** For model deployment/saving.

---

## 📂 Repository Structure

* **Manual Implementation:** Python scripts containing the raw math and logic for Gradient Descent.
* **Scikit-Learn Implementation:** Standard benchmarks using the library.
* **Model Saving:** Demos for serializing models using Joblib/Pickle.
* **Datasets:** CSV files used for training and testing.

---

*Built as part of the PITB Machine Learning Internship Portfolio.*
