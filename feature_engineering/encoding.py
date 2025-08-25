import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)


"""
print("Dataset Info: ")
print(df.info())

print("Dataset Preview: ")
print(df.head())


Dataset Info: 
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object
 11  Embarked     889 non-null    object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
None
Dataset Preview:
   PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
0            1         0       3  ...   7.2500   NaN         S
1            2         1       1  ...  71.2833   C85         C
2            3         1       3  ...   7.9250   NaN         S
3            4         1       1  ...  53.1000  C123         S
4            5         0       3  ...   8.0500   NaN         S

[5 rows x 12 columns]
"""

df_one_hot = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)


"""
print("One-hot encoded Datasets: ", df_one_hot)
One-hot encoded Datasets:       PassengerId  Survived  Pclass                                               Name   Age  SibSp  ...            Ticket     Fare  Cabin Sex_male  Embarked_Q  Embarked_S
0              1         0       3                            Braund, Mr. Owen Harris  22.0      1  ...         A/5 21171   7.2500    NaN     True       False        True
1              2         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  38.0      1  ...          PC 17599  71.2833    C85    False       False       False
2              3         1       3                             Heikkinen, Miss. Laina  26.0      0  ...  STON/O2. 3101282   7.9250    NaN    False       False        True
3              4         1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  35.0      1  ...            113803  53.1000   C123    False       False        True
4              5         0       3                           Allen, Mr. William Henry  35.0      0  ...            373450   8.0500    NaN     True       False        True
..           ...       ...     ...                                                ...   ...    ...  ...               ...      ...    ...      ...         ...         ...
886          887         0       2                              Montvila, Rev. Juozas  27.0      0  ...            211536  13.0000    NaN     True       False        True
887          888         1       1                       Graham, Miss. Margaret Edith  19.0      0  ...            112053  30.0000    B42    False       False        True
888          889         0       3           Johnston, Miss. Catherine Helen "Carrie"   NaN      1  ...        W./C. 6607  23.4500    NaN    False       False        True
889          890         1       1                              Behr, Mr. Karl Howell  26.0      0  ...            111369  30.0000   C148     True       False       False
890          891         0       3                                Dooley, Mr. Patrick  32.0      0  ...            370376   7.7500    NaN     True        True       False

[891 rows x 13 columns]
"""

label_encoder = LabelEncoder()
df['Pclass_encoded'] = label_encoder.fit_transform(df['Pclass'])
"""
print("Label Encoded: ", df[['Pclass', 'Pclass_encoded']].head())

Label Encoded:     Pclass  Pclass_encoded
0       3               2
1       1               0
2       3               2
3       1               0
4       3               2
"""

df['Ticket_frequency'] = df['Ticket'].map(df['Ticket'].value_counts())


"""
print("\n Frequency Encode Feature: ")
print(df[['Ticket', 'Ticket_frequency']].head())

Label Encoded:     Pclass  Pclass_encoded
0       3               2
1       1               0
2       3               2
3       1               0
4       3               2

 Frequency Encode Feature: 
             Ticket  Ticket_frequency
0         A/5 21171                 1
1          PC 17599                 1
2  STON/O2. 3101282                 1
3            113803                 2
4            373450                 1
"""

X = df_one_hot.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.fit_transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy with One-hot Encoder: ", accuracy_score(y_test, y_pred))
