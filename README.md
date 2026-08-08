# Customer Default Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This project predicts whether a customer is likely to default on a loan using supervised machine learning.

The workflow includes:

- Data Cleaning
- Missing Value Handling
- Feature Engineering
- Label Encoding
- Model Training
- Performance Evaluation

The model is built using the AdaBoost Classifier from Scikit-learn.

---

## Dataset

The dataset contains customer financial and loan-related information such as:

- Customer Income
- Loan Amount
- Loan Grade
- Home Ownership
- Loan Intent
- Historical Default
- Current Loan Status

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Machine Learning Pipeline

1. Load Dataset
2. Handle Missing Values
3. Clean Numerical Columns
4. Encode Categorical Features
5. Split Train/Test Data
6. Train AdaBoost Classifier
7. Predict Test Data
8. Evaluate Accuracy
9. Visualize Confusion Matrix

---

## Project Structure

```text
customer-default-prediction/
│
├── dataset/dataset.csv
├── notebooks/main.ipynb
├── src/main.py
├── images/confusion_matrix.png
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

Clone the repository.

```bash
git clone https://github.com/jax-r-31/customer-default-prediction.git
```

Move into the project.

```bash
cd customer-default-prediction
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Run the project.

```bash
python src/main.py
```

---

## Results

The AdaBoost Classifier achieved strong predictive performance on the customer default dataset.

Performance Metrics:

- Accuracy Score
- Confusion Matrix

---

## Output

Add your confusion matrix screenshot inside the **images** folder.

```markdown
![Confusion Matrix](../Customer-Default-Prediction-ML/images/confusion_matrix.png)
```

---

## Future Improvements

- Hyperparameter tuning
- Random Forest comparison
- XGBoost implementation
- Feature Scaling
- Cross Validation
- ROC Curve
- Precision & Recall
- Model Serialization using Joblib
- Streamlit Deployment

---

## Author

**Jay Rajput**

Computer Science Engineer

Python | Machine Learning | Data Science

GitHub:
https://github.com/jax-r-31

LinkedIn:
https://linkedin.com/in/jax-r

Portfolio:
https://jaxxxportfolio.netlify.app

---

## License

This project is licensed under the MIT License.

