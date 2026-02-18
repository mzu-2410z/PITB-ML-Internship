# Machine Learning Model Persistence (Joblib & Pickle) 💾

A demonstration of how to **save (serialize)** and **load (deserialize)** trained Machine Learning models using **Joblib** and **Pickle**.

This project solves a critical problem in ML engineering: **avoiding redundant retraining**. By saving the model's state to a file, we can deploy it to production (e.g., a Flask/Django app) and generate predictions instantly.

---

## 🚀 The Problem & Solution

### ❌ The Problem
Training a model can take hours or days. If we keep the model in RAM, we lose it as soon as the script finishes. Retraining it every time a user wants a prediction is inefficient and impractical for large applications.

### ✅ The Solution: Serialization
We save the Python object (the trained model) into a binary file. Later, we can load this file to reconstruct the exact model state without needing the original training data.

---

# 🛠️ Method 1: Joblib (Recommended for Scikit-Learn)

`joblib` is often more efficient than `pickle` for objects that contain large NumPy arrays (which most Scikit-Learn models do).

## 1️⃣ Saving with Joblib

```python
import joblib
from sklearn.linear_model import LinearRegression

# 1. Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# 2. Save to file
joblib.dump(model, 'model_joblib')
print("✅ Model saved using Joblib!")
```

---

## 2️⃣ Loading with Joblib

```python
import joblib

# 1. Load from file
loaded_model = joblib.load('model_joblib')

# 2. Predict instantly
prediction = loaded_model.predict([[5000]])
print(f"Predicted Price: {prediction}")
```

---

# 🛠️ Method 2: Pickle (Standard Python)

`pickle` is the standard Python module for object serialization. It works well for general Python objects.

## 1️⃣ Saving with Pickle

```python
import pickle

with open('model_pickle', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved using Pickle!")
```

---

## 2️⃣ Loading with Pickle

```python
import pickle

with open('model_pickle', 'rb') as f:
    loaded_model = pickle.load(f)

prediction = loaded_model.predict([[5000]])
print(f"Predicted Price: {prediction}")
```

---

# 🆚 Joblib vs. Pickle: Which One Should You Use?

| Feature        | Joblib                                  | Pickle                     |
|---------------|-------------------------------------------|-----------------------------|
| Best For      | Large NumPy arrays (Scikit-Learn models) | General Python objects      |
| Efficiency    | Faster for large data                    | Standard speed              |
| Compatibility | Part of Scikit-Learn ecosystem            | Native to Python            |

✅ **Recommendation:** Use **Joblib** for Scikit-Learn models.  
Use **Pickle** for smaller or general Python objects.

---

# ⚠️ Security Warning

**Never load data received from an untrusted or unauthenticated source.**

Both `joblib` and `pickle` can execute arbitrary code during loading. Only load files that you have created yourself.

---

# 📦 Tech Stack

- **Python 3.10+**
- **Scikit-Learn** – Machine Learning algorithms
- **Joblib** – Optimized serialization
- **Pickle** – Standard Python serialization

---

# 💻 How to Run

## 1️⃣ Clone the repository

```bash
git clone https://github.com/mzu-2410z/PITB-ML-Internship
cd Week-1/4-save-model
```

## 2️⃣ Install dependencies

```bash
pip install numpy pandas scikit-learn joblib
```

## 3️⃣ Run the script

```bash
python load-model.py
```

---

*Built as part of the PITB Machine Learning Internship Portfolio.*
