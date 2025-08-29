import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv("bike_sharing_daily.csv")

"""
print("Dataaset Info:", df.info())
print("\nDataset preview:", df.head())

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 731 entries, 0 to 730
Data columns (total 16 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   instant     731 non-null    int64  
 1   dteday      731 non-null    object 
 2   season      731 non-null    int64  
 3   yr          731 non-null    int64  
 4   mnth        731 non-null    int64  
 5   holiday     731 non-null    int64  
 6   weekday     731 non-null    int64  
 7   workingday  731 non-null    int64  
 8   weathersit  731 non-null    int64  
 9   temp        731 non-null    float64
 10  atemp       731 non-null    float64
 11  hum         731 non-null    float64
 12  windspeed   731 non-null    float64
 13  casual      731 non-null    int64
 14  registered  731 non-null    int64
 15  cnt         731 non-null    int64
dtypes: float64(4), int64(11), object(1)
memory usage: 91.5+ KB
Dataaset Info: None

Dataset preview:    instant      dteday         season  yr  mnth  holiday  weekday  workingday  weathersit      temp        atemp       hum         windspeed  casual  registered   cnt
0                   1           2011-01-01       1      0     1        0        6           0           2       0.344167  0.363625      0.805833    0.160446     331         654   985
1                   2           2011-01-02       1      0     1        0        0           0           2       0.363478  0.353739      0.696087    0.248539     131         670   801
2                   3           2011-01-03       1      0     1        0        1           1           1       0.196364  0.189405      0.437273    0.248309     120        1229  1349
3                   4           2011-01-04       1      0     1        0        2           1           1       0.200000  0.212122      0.590435    0.160296     108        1454  1562
4                   5           2011-01-05       1      0     1        0        3           1           1       0.226957  0.229270      0.436957    0.186900      82        1518  1600
"""

# Convert dteday to datetime
df['dteday'] = pd.to_datetime(df['dteday'])

df['day_of_week'] = df['dteday'].dt.day_name()
df['month'] = df['dteday'].dt.month
df['year'] = df['dteday'].dt.year

"""
print("New Feature:", df[['dteday','day_of_week','month','year']].head())

New Feature:       dteday day_of_week  month  year
0 2011-01-01    Saturday      1  2011
1 2011-01-02      Sunday      1  2011
2 2011-01-03      Monday      1  2011
3 2011-01-04     Tuesday      1  2011
4 2011-01-05   Wednesday      1  2011
"""

#X = df[['temp']]
X = df[['temp','atemp','hum','windspeed','season','weathersit','workingday']]

y = df['cnt']

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

"""
print("\nOriginal and Polynomial Features\n", pd.DataFrame(X_poly, columns=['temp', 'temp^2']).head())

Original and Polynomial Features
        temp    temp^2
0  0.344167  0.118451
1  0.363478  0.132116
2  0.196364  0.038559
3  0.200000  0.040000
4  0.226957  0.051509
"""

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_poly_train, X_poly_test = train_test_split(X_poly, test_size=0.2, random_state=42)

model_original = LinearRegression()
model_original.fit(X_train, y_train)
y_pred_original = model_original.predict(x_test)
mse_orignal = mean_squared_error(y_test, y_pred_original)


model_poly = LinearRegression()
model_poly.fit(X_poly_train, y_train)
y_pred_poly = model_poly.predict(X_poly_test)
mse_poly = mean_squared_error(y_test, y_pred_poly)

print(f"MSE original: {mse_orignal:.2f}")
print(f"MSE poly: {mse_poly:.2f}")

"""
print(f"MSE original: {mse_orignal:.2f}")
print(f"MSE poly: {mse_poly:.2f}")

MSE original: 2391051.89
MSE poly: 2431396.49
"""