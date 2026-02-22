# Handwritten Digit Classification (Support Vector Machine) ✍️

A Python implementation of a **Support Vector Machine (SVM)** model to classify handwritten digits (0-9). This project marks a transition into handling high-dimensional data (image pixels) and utilizing advanced spatial algorithms for multi-class classification.

## 📌 Overview
Unlike algorithms that calculate simple probabilities or build decision trees, a Support Vector Machine operates geometrically. It plots data points in an N-dimensional space (where N is the number of features) and calculates the optimal "hyperplane" (a mathematical boundary) that distinctly separates different classes with the maximum possible margin. 



For this project, the model successfully classifies 10 different target categories (digits 0 through 9) using an inherently high-dimensional pixel matrix.

---

## 🔍 Data Architecture & Engineering

This model utilizes Scikit-Learn's built-in **Digits Dataset**, which consists of normalized images of hand-drawn numbers. Working with image data requires understanding how the machine interprets visual space.



### Dataset Insights & Pixel Mechanics
| Metric | Details |
| :--- | :--- |
| **Total Records** | `1,797` image samples |
| **Raw Format** | `8x8` pixel matrices (`digits.images`) |
| **Color Scale** | Grayscale integer values ranging from `0` (white) to `16` (black) |
| **Target Classes** | `10` distinct classes (Digits `0` through `9`) |

### Preprocessing: The Flattening Phase
Machine learning algorithms like SVM cannot directly process a 2D grid (an 8x8 matrix). Before training, the data was strictly transformed:
* **Flattening:** Each 8x8 matrix was intrinsically flattened into a 64-element 1D array (`digits.data`). 
* **Feature Space:** Consequently, the SVM operates in a **64-dimensional space**, where each specific dimension represents the intensity of a single pixel.

---

## 🧠 The Mathematics: How SVM Learns

### 1. The Hyperplane and Margin Maximization
For a linearly separable dataset, the model seeks a hyperplane defined by:
$$\mathbf{w}^T \mathbf{x} + b = 0$$
*(where $\mathbf{w}$ is the weight vector, $\mathbf{x}$ is the input feature vector, and $b$ is the bias).*

The primary objective of the SVM is to maximize the "margin" (the distance between the hyperplane and the closest critical data points, known as **Support Vectors**). Mathematically, this is framed as an optimization problem:
$$\min_{\mathbf{w}, b} \frac{1}{2} ||\mathbf{w}||^2$$
*Subject to the constraint that all data points are classified correctly:*
$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1 \quad \forall i$$

### 2. The Soft Margin (Regularization Parameter `C`)
Real-world data is rarely perfectly separable. The SVM introduces a penalty parameter `C` to handle misclassifications.
* A **Low `C`** creates a wider, "softer" margin that allows some errors but generalizes better.
* A **High `C`** creates a strict, "hard" margin that tries to perfectly classify all training data (risking overfitting).
*This model utilizes the highly optimized default `C=1.0`.*

### 3. The Kernel Trick (Handling Non-Linearity)
Because 64-dimensional digit structures cannot be separated by a simple flat line, the model utilizes the **Radial Basis Function (RBF) Kernel**. This mathematical trick maps the data into an even higher dimension where a linear boundary *can* be drawn, without the computational cost of actually transforming the coordinates.

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma ||\mathbf{x}_i - \mathbf{x}_j||^2)$$

The $\gamma$ (gamma) parameter defines how far the influence of a single training example reaches.

---

## 📊 Results: Model Performance

The 1,797 records were partitioned into a Training Set (80%) and a Testing Set (20%) using Scikit-Learn's `train_test_split`.

| Metric | Value |
| :--- | :--- |
| **Model Used** | `SVC(kernel='rbf')` (Support Vector Classifier) |
| **Testing Set Size** | `360` images |
| **Model Accuracy** | **~99.1%** (`0.9916`) |

**Verdict:** The Support Vector Machine demonstrated elite predictive capabilities. By leveraging the RBF kernel on flattened 64-dimensional pixel arrays, it successfully navigated the multi-class (One-vs-One) classification problem, achieving over 99% accuracy on unseen handwriting samples.

---

## 🛠️ Tech Stack
* **Python 3.10+**
* **Scikit-Learn:** For dataset extraction (`load_digits`), data partitioning (`train_test_split`), and mathematical modeling (`SVC`).
* **Pandas:** For tabular data manipulation and structural verification.

## 💻 How to Run
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install pandas scikit-learn
    ```
3.  Launch Jupyter Notebook and open the file:
    ```bash
    jupyter notebook SupportVectorMachine.ipynb
    ```

---
*Built as part of the PITB Machine Learning Internship Portfolio.*
