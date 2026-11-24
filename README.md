# Bank Marketing: Customer Analysis & Term Deposit Subscription Prediction

End-to-end data science project on the UCI Bank Marketing dataset: data cleaning, EDA, feature engineering, handling class imbalance, machine learning, deep learning, and business insights. Completed as part of the AI & Data Science program at Machinfy.

**Goal:** predict whether a customer will subscribe to a term deposit **before** the call is made, so the bank can prioritize who to contact.

## Dataset
- [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) (`bank-full.csv`): 45,211 customers, 16 features, binary target `y`.
- Only 11.7% of customers subscribed (strong class imbalance).
- The data file is not included in this repo. Download it from the link above.

## Workflow
1. **Data understanding & cleaning:** data types, duplicates, `unknown` categories, IQR outlier detection.
2. **EDA & visualization:** 12+ charts (bar, count, histogram, box, scatter, heatmap) with interpretations.
3. **Feature engineering:** `age_group`, `balance_group`, `campaign_category` (used for analysis; the models use the original numeric features).
4. **Preprocessing:** one-hot / label encoding, `RobustScaler` (fit on train only), 80/20 train-test split.
5. **Class imbalance:** compared **SMOTE** (train set only) vs. **class weights**.
6. **Models:** Logistic Regression, KNN, Decision Tree, Random Forest, SVC, and a Keras neural network (3 hidden layers).
7. **Evaluation:** accuracy, precision, recall, F1 and confusion matrices on an untouched test set.

## Key decision: removing `duration`
`duration` (call length) is only known **after** the call ends, so it cannot be used to decide whom to call. Keeping it inflated results, for example Logistic Regression (class weights) F1 rose from **0.37 to 0.56**. It was excluded to keep the model realistic and deployable.

## Results (test set, target class = "subscribed")

| Model | Strategy | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | SMOTE | 0.7291 | 0.25 | 0.62 | 0.36 |
| KNN | SMOTE | 0.7259 | 0.23 | 0.54 | 0.32 |
| Decision Tree | SMOTE | 0.8131 | 0.27 | 0.33 | 0.30 |
| Random Forest | SMOTE | 0.8735 | 0.46 | 0.30 | 0.36 |
| SVC | SMOTE | 0.8107 | 0.28 | 0.37 | 0.32 |
| Logistic Regression | Class weights | 0.7481 | 0.26 | 0.61 | 0.37 |
| Decision Tree | Class weights | 0.8309 | 0.28 | 0.26 | 0.27 |
| Random Forest | Class weights | 0.8897 | 0.63 | 0.21 | 0.32 |
| SVC | Class weights | 0.8188 | 0.28 | 0.32 | 0.30 |
| **Neural Network (Keras)** | SMOTE | 0.8435 | 0.38 | 0.47 | **0.42** |

- Accuracy alone is misleading here: Random Forest (class weights) has the highest accuracy (0.89) but catches only 21% of real subscribers.
- A missed subscriber (false negative) costs the bank more than an unnecessary call (false positive), so models were compared mainly on **recall and F1**.
- The neural network achieved the best F1; Logistic Regression with class weights has the best recall among classical models.

## Business insights
- A **successful previous campaign** is the strongest predictor of subscription.
- **Repeated contacts** reduce conversion; capping calls per customer makes sense.
- **Students and retirees** convert at the highest rates, while blue-collar workers convert the least.
- Customers with **housing or personal loans** are less likely to subscribe.
- Campaign **month and contact channel** matter (seasonal effect, unknown contact type performs worse).

## Notes
- The neural network's validation split is taken from the end of the SMOTE-resampled training set, so the training/validation curves are indicative only. All final metrics are computed on the untouched test set.
- Class imbalance is handled on the training data only; the test set keeps the real distribution.

## Run it
```bash
pip install -r requirements.txt
```
Download `bank-full.csv`, update the file path in the first data-loading cell, then run `bank_marketing_analysis.ipynb` (originally developed in Google Colab).

## Tech stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, imbalanced-learn, TensorFlow/Keras
