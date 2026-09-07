# 🩺 Disease Prediction Using Machine Learning

This project predicts diseases from patient symptoms using two machine learning classification models:

- **Logistic Regression (LR)**
- **Support Vector Machine (SVM)**

The project uses a symptom-based dataset containing **4,998 patient records**, **15 symptom features**, and **3 disease classes**.

> **Note:** This project is for educational and machine-learning demonstration purposes. It should not be used as a substitute for professional medical diagnosis.

---

## 📌 Project Overview

The goal of this project is to build machine learning models that can classify a disease based on the presence or absence of common symptoms.

Each symptom is represented as:

- `1` = Symptom present
- `0` = Symptom absent

The models classify patients into one of the following disease categories:

- **Malaria**
- **Pneumonia**
- **Typhoid**

---

## 📂 Project Files

```text
Disease-Prediction/
│
├── disease_prediction.csv
├── Disease Prediction using LR .ipynb
├── Disease prediction using SVM Model.ipynb
└── README.md
```

### Files Description

| File | Description |
|---|---|
| `disease_prediction.csv` | Dataset used for training and testing |
| `Disease Prediction using LR .ipynb` | Disease prediction using Logistic Regression |
| `Disease prediction using SVM Model.ipynb` | Disease prediction using Support Vector Machine |
| `README.md` | Project documentation |

---

## 📊 Dataset

The dataset contains:

- **4,998 records**
- **15 input features**
- **1 target/label column**
- **3 disease classes**
- No missing values were found in the notebook's dataset check.

### Symptoms / Features

| # | Feature |
|---:|---|
| 1 | fever |
| 2 | cough |
| 3 | headache |
| 4 | nausea |
| 5 | vomiting |
| 6 | fatigue |
| 7 | sore_throat |
| 8 | chills |
| 9 | body_pain |
| 10 | loss_of_appetite |
| 11 | abdominal_pain |
| 12 | diarrhea |
| 13 | sweating |
| 14 | rapid_breathing |
| 15 | dizziness |

### Target

The target column is:

```text
label
```

The dataset contains three classes:

| Disease | Records |
|---|---:|
| Malaria | 1,666 |
| Pneumonia | 1,666 |
| Typhoid | 1,666 |

---

## 🛠️ Technologies and Libraries

The notebooks use Python 3.12.7 and the following libraries:

```text
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook
```

Main Scikit-learn components used include:

```python
train_test_split
LogisticRegression
SVC
accuracy_score
precision_score
recall_score
f1_score
confusion_matrix
classification_report
StandardScaler
```

---

# 🔬 Methodology

The project follows these major steps:

```text
Dataset
   ↓
Dataset Inspection
   ↓
Separate Features and Target
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Prediction
   ↓
Performance Evaluation
   ↓
Confusion Matrix
   ↓
New Patient Disease Prediction
```

The notebooks use an **80:20 train-test split** with `random_state=42` and stratification.

- Training samples: **3,998**
- Testing samples: **1,000**

---

# 🤖 Model 1: Logistic Regression

The Logistic Regression notebook trains a multiclass Logistic Regression model using:

```python
LogisticRegression(
    max_iter=2000,
    solver="lbfgs"
)
```

## Performance

The recorded results from the notebook are:

| Metric | Score |
|---|---:|
| Accuracy | **88.60%** |
| Precision | **88.58%** |
| Recall | **88.60%** |
| F1 Score | **88.57%** |

### Classification Report

| Disease | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Malaria | 0.87 | 0.83 | 0.85 | 333 |
| Pneumonia | 0.91 | 0.91 | 0.91 | 333 |
| Typhoid | 0.88 | 0.91 | 0.89 | 334 |
| **Weighted Avg.** | **0.89** | **0.89** | **0.89** | **1000** |

### Confusion Matrix

The Logistic Regression notebook produced:

```text
[[278  24  31]
 [ 18 304  11]
 [ 25   5 304]]
```

---

# 🤖 Model 2: Support Vector Machine (SVM)

The SVM notebook uses an RBF kernel:

```python
SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    random_state=42
)
```

The notebook also scales the feature values before SVM prediction.

## Performance

The recorded results from the notebook are:

| Metric | Score |
|---|---:|
| Accuracy | **87.70%** |
| Precision | **87.69%** |
| Recall | **87.70%** |
| F1 Score | **87.69%** |

---

# 📈 Model Comparison

Based on the recorded notebook results:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| **Logistic Regression** | **88.60%** | **88.58%** | **88.60%** | **88.57%** |
| **SVM** | 87.70% | 87.69% | 87.70% | 87.69% |

### Best Recorded Model

In these notebooks, **Logistic Regression** achieved the higher recorded performance:

**Accuracy: 88.60%**

compared with:

**SVM Accuracy: 87.70%**

---

# 👤 New Patient Prediction

Both notebooks include a new-patient prediction section.

For example, the Logistic Regression notebook creates a symptom list such as:

```python
new_symptoms = [
    "fever",
    "cough",
    "headache",
    "nausea"
]
```

The symptoms are converted into the same feature structure used during training, and the trained model predicts the disease.

The Logistic Regression notebook also calculates a confidence value using:

```python
probabilities = LR_model.predict_proba(new_patient)
confidence = np.max(probabilities) * 100
```

The recorded example in the Logistic Regression notebook outputs a disease prediction and confidence value.

The SVM notebook's recorded example predicts:

```text
Predicted Disease: Malaria
```

---

# 🚀 How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
```

## 2. Open the Project Folder

```bash
cd your-repository-name
```

## 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## 4. Start Jupyter Notebook

```bash
jupyter notebook
```

## 5. Run a Notebook

Open either:

```text
Disease Prediction using LR .ipynb
```

or:

```text
Disease prediction using SVM Model.ipynb
```

Make sure `disease_prediction.csv` is located in the same project directory as the notebooks.

---

# 📋 Requirements

A typical environment for this project is:

```text
Python >= 3.12
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

# 📊 Evaluation Metrics

The project evaluates the models using:

### Accuracy

Measures the overall percentage of correctly classified samples.

### Precision

Measures how many predicted samples of a class were actually that class.

### Recall

Measures how many actual samples of a class were correctly identified.

### F1 Score

Combines precision and recall into a single metric.

### Confusion Matrix

Shows the number of correct and incorrect predictions for each disease class.

---

# 🎯 Project Objectives

- Build a machine learning system for symptom-based disease classification.
- Compare Logistic Regression and SVM.
- Evaluate model performance using standard classification metrics.
- Visualize model predictions using confusion matrices.
- Demonstrate prediction for a new patient based on symptoms.

---

# ⚠️ Disclaimer

This project is developed for **academic and educational purposes**. The predictions are based on the provided dataset and machine-learning models. They are **not medical diagnoses** and should not be used for medical treatment or clinical decision-making.

---

## 👨‍💻 Author

**Gazi Md. Abu Sayeed**

M.Sc. Eng. in Information and Communication Technology (ICT)  
Khulna University of Engineering & Technology (KUET)

---

## ⭐ If You Find This Project Useful

If this project is useful for your learning or research, consider giving the repository a ⭐ on GitHub.
