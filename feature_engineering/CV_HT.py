import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
df = df[['Pclass','Sex','Age','Fare','Embarked','Survived']]

df.fillna({'Age':df['Age'].median()}, inplace=True)
df.fillna({'Embarked':df['Embarked'].mode()[0]}, inplace=True)

X = df.drop(columns=['Survived'])
y = df['Survived']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age', 'Fare']),
        ('cat', OneHotEncoder(), ['Pclass', 'Sex', 'Embarked'])
    ]
)

X_preprocess = preprocessor.fit_transform(X)

log_model = LogisticRegression()
log_score = cross_val_score(log_model, X_preprocess, y, cv=5, scoring='accuracy')
print(f"Logistic Regression Accuracy: {log_score.mean():.2f}")

rf_model = RandomForestClassifier(random_state=42)
rf_score = cross_val_score(rf_model, X_preprocess, y, cv=5, scoring='accuracy')
print(f"Random Forest Accuracy: {rf_score.mean():.2f}")

para_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=para_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_preprocess, y)

print(f"Best Hyperparameter is: {grid_search.best_params_}")
print(f"Best Score is: {grid_search.best_score_:.2f}")