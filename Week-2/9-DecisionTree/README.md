# Titanic Survivor Prediction (Decision Tree Classifier) 🚢

A Python implementation of a **Decision Tree** classification model to predict passenger survival on the Titanic. This project marks a transition into independent data preprocessing, where all feature engineering, data cleaning, and model pipeline logic were constructed entirely from scratch.

## 📌 Overview
Unlike linear models that use equations, Decision Trees learn by splitting data based on a series of questions (nodes). By analyzing historical passenger data, this model constructs a flowchart-like tree to accurately predict whether a given passenger survived the disaster based on their class, gender, and fare.



[Image of Decision Tree structure]


---

## 🔍 Data Exploration & Independent Engineering

Real-world datasets are rarely ready for machine learning out of the box. A significant portion of this project was dedicated to cleaning and structuring the data manually.

### 1. Feature Selection
The raw dataset contained several columns that were either irrelevant or too unstructured for pure numerical modeling.
* **Dropped Columns:** `PassengerId`, `Name`, `Ticket`, `Cabin`, and `Embarked` were removed to eliminate noise.
* **Handling Missing Data:** The `Age` column contained significant missing (`NaN`) values and was excluded from this iteration to maintain model integrity without data imputation.
* **Final Features ($X$):** The model was trained purely on `Pclass` (Ticket Class), `Fare` (Price paid), and Gender.

### 2. Categorical Preprocessing (Dummy Variables)
Machine learning models cannot process text like "male" or "female". 
* Used `pandas.get_dummies()` to convert the `Sex` column into boolean format.
* **Dodging the Dummy Variable Trap:** To prevent multicollinearity, the `male` column was intentionally dropped. The model relies entirely on the generated `female` column (`1` for True, `0` for False), keeping the feature matrix perfectly lean.

---

## 🧠 The Mathematics: How the Tree Learns

Decision Trees figure out the "best" questions to ask by calculating the purity of the data after a split. The model aims to create branches where the resulting groups consist mostly of one class (e.g., mostly survivors or mostly casualties). It does this using two primary mathematical concepts:

### 1. Gini Impurity (Default metric)
Gini Impurity measures the likelihood of a new, random data point being incorrectly classified if it were randomly labeled according to the distribution of data in that node. 
A pure node (all survivors) has a Gini of 0.

$$Gini = 1 - \sum_{i=1}^{C} (p_i)^2$$
*(where $p_i$ is the probability of an item belonging to class $i$)*

### 2. Entropy & Information Gain
Alternatively, trees can use Entropy, a concept from information theory that measures the level of disorder or uncertainty in a group.

$$Entropy = - \sum_{i=1}^{C} p_i \log_2(p_i)$$

The tree calculates the **Information Gain** (the reduction in Entropy or Gini Impurity) for every possible split (e.g., "Is Fare > 50?" or "Is female == 1?") and chooses the split that yields the highest Information Gain.



---

## 📊 Results: Model Performance

The perfectly structured data was partitioned into a Training Set and a Testing Set to evaluate its predictive power on unseen passengers.

| Metric | Value |
| :--- | :--- |
| **Model Used** | `DecisionTreeClassifier()` |
| **Key Features** | `Pclass`, `Fare`, `female` |
| **Model Accuracy** | **~82.6%** (`0.826`) |

**Verdict:** Relying purely on socioeconomic status (`Pclass`, `Fare`) and gender (`female`), the Decision Tree was able to predict Titanic survival outcomes with nearly 83% accuracy, confirming the historical "women and upper-class first" lifeboat policies.

---

## 🛠️ Tech Stack
* **Python 3.10+**
* **Pandas:** Used extensively for data manipulation, dropping columns, and generating dummy variables.
* **Scikit-Learn:** For data partitioning (`train_test_split`) and model building (`DecisionTreeClassifier`).

## 💻 How to Run
1.  Clone the repository and ensure you have the `titanic.csv` dataset in your directory.
2.  Install dependencies:
    ```bash
    pip install pandas scikit-learn
    ```
3.  Launch Jupyter Notebook and open the file:
    ```bash
    jupyter notebook survivors.ipynb
    ```

---
*Built as part of the PITB Machine Learning Internship Portfolio.*
