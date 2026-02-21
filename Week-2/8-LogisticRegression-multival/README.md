# Iris Flower Classification (Multiclass Logistic Regression) 🌸

A Python implementation of a **Multiclass Logistic Regression** model to classify iris flowers into three distinct species based on their physical measurements.

This project demonstrates the transition from binary classification (predicting between two outcomes) to multiclass classification (handling three or more distinct categories) using probability mapping.

## 🧠 The Mathematics

When dealing with more than two categories, standard Logistic Regression (which uses the Sigmoid function) is extended into Multinomial Logistic Regression using the Softmax function.

### 1. The Linear Combination
For each class $k$ (out of $K$ total classes), the model calculates a separate weighted sum of the inputs:

$$z_k = w_{k1}x_1 + w_{k2}x_2 + \dots + w_{kn}x_n + b_k$$

### 2. The Softmax Function
Instead of squashing a single value between 0 and 1, the Softmax function converts all the linear scores into a probability distribution that sums to 1. It takes the exponential of one class's score and divides it by the sum of the exponentials of all scores:

$$\hat{y}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

### 3. The Decision Boundary
The model evaluates the probabilities $\hat{y}$ for all $K$ classes (Setosa, Versicolor, Virginica) and makes its final prediction based on the class that yields the highest probability.

---

## 🔍 Data Exploration & Engineering


This model is built on the famous **Iris Dataset**, accessed directly via Scikit-Learn's internal dataset library.

### Dataset Insights
| Metric | Details |
| :--- | :--- |
| **Total Records** | `150` samples |
| **Features ($x$)** | `4` (Sepal Length, Sepal Width, Petal Length, Petal Width) |
| **Target Classes ($y$)** | `3` (0: Setosa, 1: Versicolor, 2: Virginica) |

### Data Preprocessing
* **Data Loading:** Extracted the feature matrix (`flower.data`) and the target array (`flower.target`) straight from Scikit-Learn.
* **Efficient Encoding:** Because the target variables were already mapped to numerical integers (`0, 1, 2`) within the dataset, One-Hot Encoding was not required for this pipeline.

---

## 📊 Results: Model Performance

The dataset was partitioned using an 80/20 split (`test_size=0.2`), reserving 30 records exclusively for evaluating the model on unseen data.

| Metric | Value |
| :--- | :--- |
| **Total Dataset Size** | `150` records |
| **Testing Set Size** | `30` records |
| **Model Accuracy** | `93.3%` (0.9333) |

### Error Visualization (Confusion Matrix)
To understand exactly where the model succeeded and where it made its ~6.7% error margin, a **Confusion Matrix** was plotted using Seaborn. This heatmap visually compared the `Truth` labels against the `Predicted` labels, proving that the vast majority of predictions landed perfectly on the diagonal line of accuracy.

---

## 🛠️ Tech Stack
* **Python 3.10+**
* **Scikit-Learn:** For dataset loading (`load_iris`), data partitioning (`train_test_split`), model building (`LogisticRegression`), and evaluation (`confusion_matrix`).
* **Matplotlib & Seaborn:** For rendering the high-fidelity confusion matrix heatmap.

## 💻 How to Run
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install matplotlib seaborn scikit-learn
    ```
3.  Launch Jupyter Notebook and open the file:
    ```bash
    jupyter notebook Iris.ipynb
    ```

---

*Built as part of the PITB Machine Learning Internship Portfolio.*
