from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Dataset Info: ")
print(X.describe())
print("\n Target Classes: ", data.target_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("Accuracy without scaling: ", accuracy_score(y_test, y_pred))

# Min-Max Scaling
scaler = MinMaxScaler() 
X_scaled = scaler.fit_transform(X)

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train_scaled)
y_pred_scaled = knn_scaled.predict(X_test_scaled)
print("Accuracy with Min-Max scaling: ", accuracy_score(y_test_scaled, y_pred_scaled))

# Standard Scaling
scaler_std = StandardScaler()
X_std_scaled = scaler_std.fit_transform(X)

X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X_std_scaled, y, test_size=0.2, random_state=42)

knn_std = KNeighborsClassifier(n_neighbors=5)
knn_std.fit(X_train_std, y_train_std)
y_pred_std = knn_std.predict(X_test_std)
print("Accuracy with Standard scaling: ", accuracy_score(y_test_std, y_pred_std))