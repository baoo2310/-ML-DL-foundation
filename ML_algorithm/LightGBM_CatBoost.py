import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

features = ['Pclass','Sex','Age','Fare','Embarked']
target = 'Survived'

df.fillna({'Age':df['Age'].median()}, inplace=True)
df.fillna({'Embarked':df['Embarked'].mode()[0]}, inplace=True)

label_encoder = {}

for col in ['Sex', 'Embarked']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoder[col] = le

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)

y_pred_lgb = lgb_model.predict(X_test)
"""
print(f"LightGBM Accuracy: {accuracy_score(y_test, y_pred_lgb)}")

LightGBM Accuracy: 0.8044692737430168
"""

cat_features = ['Pclass', 'Sex', 'Embarked']
cat_model = CatBoostClassifier(cat_features=cat_features, verbose=0 , iterations=100)
cat_model.fit(X_train, y_train)

cat_y_pred = cat_model.predict(X_test)

"""
print(f"Cat Boost Accuracy: {accuracy_score(y_test, cat_y_pred)}")

Cat Boost Accuracy: 0.8100558659217877
"""

xgb_model = XGBClassifier(eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
"""
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb)}")

XGBoost Accuracy: 0.770949720670391
"""

cat_model_native = CatBoostClassifier(cat_features=['Sex', 'Embarked'], verbose=0, iterations=100)
cat_model_native.fit(X_train, y_train)
cat_native_pred_y = cat_model_native.predict(X_test)

"""
print(f"Cat Model Native Accuracy: {accuracy_score(y_test, cat_native_pred_y)}")

Cat Model Native Accuracy: 0.8044692737430168
"""