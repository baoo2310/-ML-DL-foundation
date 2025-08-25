from sklearn.datasets import load_diabetes
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

"""

print(df.head())
print(df.info())

        age       sex       bmi        bp        s1        s2        s3        s4        s5        s6  target
0  0.038076  0.050680  0.061696  0.021872 -0.044223 -0.034821 -0.043401 -0.002592  0.019907 -0.017646   151.0
1 -0.001882 -0.044642 -0.051474 -0.026328 -0.008449 -0.019163  0.074412 -0.039493 -0.068332 -0.092204    75.0
2  0.085299  0.050680  0.044451 -0.005670 -0.045599 -0.034194 -0.032356 -0.002592  0.002861 -0.025930   141.0
3 -0.089063 -0.044642 -0.011595 -0.036656  0.012191  0.024991 -0.036038  0.034309  0.022688 -0.009362   206.0
4  0.005383 -0.044642 -0.036385  0.021872  0.003935  0.015596  0.008142 -0.002592 -0.031988 -0.046641   135.0
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 442 entries, 0 to 441
Data columns (total 11 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   age     442 non-null    float64
 1   sex     442 non-null    float64
 2   bmi     442 non-null    float64
 3   bp      442 non-null    float64
 4   s1      442 non-null    float64
 5   s2      442 non-null    float64
 6   s3      442 non-null    float64
 7   s4      442 non-null    float64
 8   s5      442 non-null    float64
 9   s6      442 non-null    float64
 10  target  442 non-null    float64
dtypes: float64(11)
memory usage: 38.1 KB
None
"""

correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

correlated_features = correlation_matrix['target'].sort_values(ascending=False)

"""
print("Features Most Correlated: ", correlated_features)

Features Most Correlated:  target    1.000000
bmi       0.586450
s5        0.565883
bp        0.441482
s4        0.430453
s6        0.382483
s1        0.212022
age       0.187889
s2        0.174054
sex       0.043062
s3       -0.394789
Name: target, dtype: float64
"""

X = df.drop(columns=['target'])
y = df['target']

mutual_info = mutual_info_regression(X, y)

mi_df = pd.DataFrame({'Feature': X.columns, "Mutual Information":mutual_info})
mi_df = mi_df.sort_values(by="Mutual Information", ascending=False)

"""
print("Mutual Information Score:", mi_df)

Mutual Information Score:   Feature  Mutual Information
2     bmi            0.173084
8      s5            0.147831
7      s4            0.097852
9      s6            0.095669
6      s3            0.067233
4      s1            0.066617
3      bp            0.063938
1     sex            0.036641
5      s2            0.014339 
0     age            0.003136
"""

from sklearn.ensemble import RandomForestRegressor
import numpy as np

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

feature_importance = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, "Importance": feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

"""
print("Feature Importance From RandomForest ", importance_df)

Feature Importance:    Feature  Importance
8      s5    0.315629
2     bmi    0.276249
3      bp    0.087085
9      s6    0.070775
0     age    0.057496
5      s2    0.055368
6      s3    0.051191
4      s1    0.047251
7      s4    0.027056
1     sex    0.011901
"""

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()
plt.title("Feature importance from RandomForest")
plt.show()