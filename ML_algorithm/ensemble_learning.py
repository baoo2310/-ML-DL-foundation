from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

df = load_iris()
X, y = df.data, df.target   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

log_model = LogisticRegression()
dt_model = DecisionTreeClassifier()
knn_model = KNeighborsClassifier()

log_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

# Creating Voting Classifier

ensemble_model = VotingClassifier(
    estimators=[
        ('log_reg', log_model),
        ('decision_tree', dt_model),
        ('knn', knn_model)
    ],
    voting='hard'
)

# Train
ensemble_model.fit(X_train, y_train)
y_pred_ensemble = ensemble_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_ensemble)

y_pred_log = log_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test)

print(f"Logistic Model Accuracy: {accuracy_score(y_test, y_pred_log):.5f}")
print(f"Decision Tree Model Accuracy: {accuracy_score(y_test, y_pred_dt):.5f}")
print(f"Knn Model Accuracy: {accuracy_score(y_test, y_pred_knn):.5f}")
print(f"Ensemble Model Accuracy: {accuracy:.5f}")
