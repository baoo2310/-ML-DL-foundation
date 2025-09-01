import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

df = pd.read_csv(url)


"""
print("Dataset Info: \n", df.info())
print("Class Distribution: \n", df["Class"].value_counts())

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 284807 entries, 0 to 284806
Data columns (total 31 columns):
 #   Column  Non-Null Count   Dtype  
---  ------  --------------   -----  
 0   Time    284807 non-null  float64
 1   V1      284807 non-null  float64
 2   V2      284807 non-null  float64
 3   V3      284807 non-null  float64
 4   V4      284807 non-null  float64
 5   V5      284807 non-null  float64
 6   V6      284807 non-null  float64
 7   V7      284807 non-null  float64
 8   V8      284807 non-null  float64
 9   V9      284807 non-null  float64
 10  V10     284807 non-null  float64
 11  V11     284807 non-null  float64
 12  V12     284807 non-null  float64
 13  V13     284807 non-null  float64
 14  V14     284807 non-null  float64
 15  V15     284807 non-null  float64
 16  V16     284807 non-null  float64
 17  V17     284807 non-null  float64
 18  V18     284807 non-null  float64
 19  V19     284807 non-null  float64
 20  V20     284807 non-null  float64
 21  V21     284807 non-null  float64
 22  V22     284807 non-null  float64
 23  V23     284807 non-null  float64
 24  V24     284807 non-null  float64
 25  V25     284807 non-null  float64
 26  V26     284807 non-null  float64
 27  V27     284807 non-null  float64
 28  V28     284807 non-null  float64
 29  Amount  284807 non-null  float64
 30  Class   284807 non-null  int64
dtypes: float64(30), int64(1)
memory usage: 67.4 MB
Dataset Info: 
 None
Class Distribution:
 Class
0    284315
1       492
Name: count, dtype: int64

"""

X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

rf_model = RandomForestClassifier(random_state=42, class_weight='balanced',n_estimators=100)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
print(f"\nClassification Report: \n", classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1])


"""
print(f"ROC-AUC : {roc_auc:.5f}")

Classification Report: 
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     28435
           1       1.00      0.74      0.85        46

    accuracy                           1.00     28481
   macro avg       1.00      0.87      0.92     28481
weighted avg       1.00      1.00      1.00     28481

ROC-AUC : 0.93325

"""
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

rf_model_smote = RandomForestClassifier(random_state=42)
rf_model_smote.fit(X_resampled, y_resampled)
y_pred_smote = rf_model_smote.predict(X_test)

roc_auc_smote = roc_auc_score(y_test, rf_model_smote.predict_proba(X_test)[:,1])

"""
print("\n Class Distribution After SMOTE: \n", pd.Series(y_resampled).value_counts())

 Class Distribution After SMOTE: 
 Class
0    255880
1    255880
Name: count, dtype: int64

print(f"\nClassification Report (SMOTE): \n", classification_report(y_test, y_pred_smote))

print(f"ROC-AUC (SMOTE) : {roc_auc_smote:.5f}")

Classification Report (SMOTE):
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     28435
           1       0.88      0.80      0.84        46

    accuracy                           1.00     28481
   macro avg       0.94      0.90      0.92     28481
weighted avg       1.00      1.00      1.00     28481

ROC-AUC (SMOTE) : 0.97272
"""