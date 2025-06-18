
# Credit Lending Risk Analysis

## 📌 Project Overview
This project performs **Credit Risk Analysis** using Machine Learning models such as Random Forest, Decision Tree, and XGBoost to predict customer creditworthiness and classify prospects into risk categories (`P1`, `P2`, `P3`, `P4`). The project also includes a **Flask-based Web Application** for interactive prediction.

---

## 🔍 Problem Statement
In the banking sector, understanding a prospect's credit risk is critical. The objective is to classify loan applicants into predefined risk categories based on demographic, financial, and credit-related features using classification algorithms.

---

## 📂 Project Structure

```
credit_lending_risk_analysis/
│
├── app.py                           # Flask Web Application
├── forms.py                         # WTForms for handling form inputs
├── templates/                       # HTML templates (Flask)
│   └── index.html                   # Frontend template
├── data/                            # Raw and cleaned data files
├── Feature Engineering.ipynb        # Feature Engineering Notebook
├── Exploratory_data_analysis.ipynb  # EDA Notebook
├── data_cleaning.ipynb              # Data Cleaning Notebook
├── Model_Building.ipynb             # Model Training Notebook
├── third_model.ipynb                # Additional model experiments
├── model.joblib                     # Saved Machine Learning Model (Random Forest/XGBoost)
├── best_pipeline.joblib             # Best performing pipeline/model
├── requirements.txt                 # Required Python packages
├── README.md                        # Project documentation
├── LICENSE                          # License file
└── .gitignore                       # Git ignore file
```

---

## ⚙️ Technologies Used

- **Python (Pandas, NumPy, Scikit-learn, XGBoost)**
- **Flask** - Web framework for UI
- **HTML/CSS (Jinja2 templates)** - For frontend rendering
- **WTForms** - For form handling in Flask
- **Jupyter Notebook** - For Data Cleaning, EDA, Feature Engineering, Model Building
- **Joblib** - Model serialization
- **Excel** - Raw data storage

---

## 🧩 Key Features

1. **Data Cleaning & Preprocessing**
   - Handled missing values (`-99999`)
   - Chi-Square Tests for categorical variables
   - ANOVA & VIF for numerical variables
   - Standardization using `StandardScaler`

2. **Feature Engineering**
   - Encoding categorical variables (Label Encoding, One-Hot Encoding)
   - Removal of multicollinear features using VIF

3. **Model Building**
   - **Random Forest Classifier**
   - **XGBoost Classifier (with Hyperparameter Tuning via GridSearchCV)**
   - **Decision Tree Classifier**

4. **Model Evaluation**
   - Accuracy, Precision, Recall, F1 Score (per class)

5. **Flask Web App**
   - Predict customer risk category based on user input
   - Clean, interactive UI designed with Flask & HTML

---

## 🚀 How to Run the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sdntheone/Credit_Lending_Risk_Analysis.git
   cd Credit_Lending_Risk_Analysis
   ```

2. **Create & Activate a Virtual Environment (Optional)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   venv\Scripts\activate     # On Windows
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask App**:
   ```bash
   python app.py
   ```
   Access the app at: `http://127.0.0.1:5000/`

---

## 📊 Sample Output (XGBoost Model)
```
Accuracy: 0.82

Class P1:
Precision: 0.85
Recall: 0.80
F1 Score: 0.82
...
```

---

## ✅ Results & Conclusion

- **XGBoost performed best** among all models with the highest accuracy.
- Further model tuning or collecting more feature-rich data can improve performance.
- The **Flask UI** enables non-technical users to interact with the model and get real-time predictions.

---

## 🤝 Contact
**Sudhanshu Nandan**  
Email: *[sdntheone.com]*  
GitHub: [https://github.com/sdntheone](https://github.com/sdntheone)
