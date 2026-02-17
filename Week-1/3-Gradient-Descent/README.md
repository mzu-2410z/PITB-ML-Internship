# Gradient Descent from Scratch 📉

A Python implementation of the **Gradient Descent** optimization algorithm for Linear Regression, built entirely from scratch without using high-level ML libraries for the core logic.

This project demonstrates the mathematical "engine" behind how machine learning models learn, by iteratively minimizing a cost function to find the line of best fit.

## 🧠 The Mathematics

The goal is to find the optimal values for slope ($m$) and intercept ($b$) that minimize the error between predicted and actual values.

### 1. The Hypothesis (Prediction)
We approximate the relationship between variables $x$ and $y$ using a linear equation:

$$y_{pred} = m x + b$$

### 2. The Cost Function (Mean Squared Error)
To measure how "wrong" the line is, we calculate the average squared difference between actual ($y$) and predicted ($y_{pred}$) values:

$$J(m, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - (mx_i + b))^2$$

### 3. The Gradients (Partial Derivatives)
To minimize the cost, we calculate the slope of the cost function with respect to $m$ and $b$. This tells us which direction to step to reach the bottom of the "valley."

**Derivative with respect to $m$:**
$$\frac{\partial J}{\partial m} = -\frac{2}{n} \sum_{i=1}^{n} x_i (y_i - y_{pred})$$

**Derivative with respect to $b$:**
$$\frac{\partial J}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - y_{pred})$$

### 4. The Update Rule (Learning)
We update the weights iteratively by taking small steps ($\alpha$ or learning rate) in the opposite direction of the gradient:

$$m_{new} = m_{curr} - \alpha \cdot \frac{\partial J}{\partial m}$$

$$b_{new} = b_{curr} - \alpha \cdot \frac{\partial J}{\partial b}$$

---

## 🚀 Optimization Features

### Dynamic Convergence (`math.isclose`)
Instead of running for a fixed number of iterations, this implementation includes an intelligent "Braking System."
* It tracks the **Cost** at every step.
* It uses `math.isclose(current_cost, previous_cost, rel_tol=1e-20)` to detect when the model has stopped improving.
* **Result:** The algorithm automatically stops early when it finds the optimal solution, saving computational resources.

---

## 📊 Results: Custom Engine vs. Scikit-Learn

I benchmarked my custom implementation against the industry-standard `sklearn.linear_model.LinearRegression`.

| Metric | My Gradient Descent | Scikit-Learn (Reference) | Difference |
| :--- | :--- | :--- | :--- |
| **Slope ($m$)** | `1.017738` | `1.017736` | ~0.000002 |
| **Intercept ($b$)** | `1.915082` | `1.915219` | ~0.000137 |
| **Iterations** | `415,532` (Auto-Stop) | N/A | Saved ~60% compute |

**Verdict:** The custom algorithm achieved **identical accuracy** to the library standard while demonstrating efficient convergence.

---

## 🛠️ Tech Stack
* **Python 3.10+**
* **NumPy:** For vectorization and array operations.
* **Pandas:** For data handling.
* **Math:** For convergence logic.
* **Scikit-Learn:** Used *only* for benchmarking and verification.

## 💻 How to Run
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install numpy pandas scikit-learn
    ```
3.  Run the script:
    ```bash
    python gradient_descent.py
    ```

---
*Built as part of the PITB Machine Learning Internship Portfolio.*
