---
tag: [machine learning, scikit-learn]
toc: true
---

# 보스톤 주택 가격 예측


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

## 1. 데이터 탐색


```python
# 1.2 이하 버전
# from sklearn.datasets import load_boston
# boston = load_boston()

# df = pd.DataFrame(boston.data, columns=boston.feature_names)
# df['PRICE'] = boston.target
# df.head()
```


```python
# 1.2 이상 버전
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
df = pd.DataFrame(data, columns=feature_names)
df['PRICE'] = target
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>PRICE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>



 ```
 CRIM     per capita crime rate by town
 ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
 INDUS    proportion of non-retail business acres per town
 CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
 NOX      nitric oxides concentration (parts per 10 million)
 RM       average number of rooms per dwelling
 AGE      proportion of owner-occupied units built prior to 1940
 DIS      weighted distances to five Boston employment centres
 RAD      index of accessibility to radial highways
 TAX      full-value property-tax rate per $10,000
 PTRATIO  pupil-teacher ratio by town
 B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
 LSTAT    % lower status of the population
 MEDV     Median value of owner-occupied homes in $1000's
 ```

* CRIM: 지역별 범죄 발생률  
* ZN: 25,000평방피트를 초과하는 거주 지역의 비율
* NDUS: 비상업 지역 넓이 비율
* CHAS: 찰스강에 대한 더미 변수(강의 경계에 위치한 경우는 1, 아니면 0)
* NOX: 일산화질소 농도
* RM: 거주할 수 있는 방 개수
* AGE: 1940년 이전에 건축된 소유 주택의 비율
* DIS: 5개 주요 고용센터까지의 가중 거리
* RAD: 고속도로 접근 용이도
* TAX: 10,000달러당 재산세율
* PTRATIO: 지역의 교사와 학생 수 비율
* B: 지역의 흑인 거주 비율
* LSTAT: 하위 계층의 비율
* MEDV: 본인 소유의 주택 가격(중앙값)

- 누락 데이터 확인


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 14 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   CRIM     506 non-null    float64
     1   ZN       506 non-null    float64
     2   INDUS    506 non-null    float64
     3   CHAS     506 non-null    float64
     4   NOX      506 non-null    float64
     5   RM       506 non-null    float64
     6   AGE      506 non-null    float64
     7   DIS      506 non-null    float64
     8   RAD      506 non-null    float64
     9   TAX      506 non-null    float64
     10  PTRATIO  506 non-null    float64
     11  B        506 non-null    float64
     12  LSTAT    506 non-null    float64
     13  PRICE    506 non-null    float64
    dtypes: float64(14)
    memory usage: 55.5 KB
    

- 통계 정보


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>PRICE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.613524</td>
      <td>11.363636</td>
      <td>11.136779</td>
      <td>0.069170</td>
      <td>0.554695</td>
      <td>6.284634</td>
      <td>68.574901</td>
      <td>3.795043</td>
      <td>9.549407</td>
      <td>408.237154</td>
      <td>18.455534</td>
      <td>356.674032</td>
      <td>12.653063</td>
      <td>22.532806</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.601545</td>
      <td>23.322453</td>
      <td>6.860353</td>
      <td>0.253994</td>
      <td>0.115878</td>
      <td>0.702617</td>
      <td>28.148861</td>
      <td>2.105710</td>
      <td>8.707259</td>
      <td>168.537116</td>
      <td>2.164946</td>
      <td>91.294864</td>
      <td>7.141062</td>
      <td>9.197104</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.006320</td>
      <td>0.000000</td>
      <td>0.460000</td>
      <td>0.000000</td>
      <td>0.385000</td>
      <td>3.561000</td>
      <td>2.900000</td>
      <td>1.129600</td>
      <td>1.000000</td>
      <td>187.000000</td>
      <td>12.600000</td>
      <td>0.320000</td>
      <td>1.730000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.082045</td>
      <td>0.000000</td>
      <td>5.190000</td>
      <td>0.000000</td>
      <td>0.449000</td>
      <td>5.885500</td>
      <td>45.025000</td>
      <td>2.100175</td>
      <td>4.000000</td>
      <td>279.000000</td>
      <td>17.400000</td>
      <td>375.377500</td>
      <td>6.950000</td>
      <td>17.025000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.256510</td>
      <td>0.000000</td>
      <td>9.690000</td>
      <td>0.000000</td>
      <td>0.538000</td>
      <td>6.208500</td>
      <td>77.500000</td>
      <td>3.207450</td>
      <td>5.000000</td>
      <td>330.000000</td>
      <td>19.050000</td>
      <td>391.440000</td>
      <td>11.360000</td>
      <td>21.200000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.677083</td>
      <td>12.500000</td>
      <td>18.100000</td>
      <td>0.000000</td>
      <td>0.624000</td>
      <td>6.623500</td>
      <td>94.075000</td>
      <td>5.188425</td>
      <td>24.000000</td>
      <td>666.000000</td>
      <td>20.200000</td>
      <td>396.225000</td>
      <td>16.955000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>88.976200</td>
      <td>100.000000</td>
      <td>27.740000</td>
      <td>1.000000</td>
      <td>0.871000</td>
      <td>8.780000</td>
      <td>100.000000</td>
      <td>12.126500</td>
      <td>24.000000</td>
      <td>711.000000</td>
      <td>22.000000</td>
      <td>396.900000</td>
      <td>37.970000</td>
      <td>50.000000</td>
    </tr>
  </tbody>
</table>
</div>



- 수치형 데이터 시각화(히스토그램)


```python
df.hist(bins=50, figsize=(20, 15))
plt.show()
```


    
![png](/assets/images/2023-03-02-Machine Learning 3 (보스톤 주택 가격 예측)/output_12_0.png)
    


- 상관계수


```python
plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f')
```




    <AxesSubplot:>




    
![png](/assets/images/2023-03-02-Machine Learning 3 (보스톤 주택 가격 예측)/output_14_1.png)
    



```python
df.plot(kind='scatter', x='LSTAT', y='PRICE', alpha=0.5) # 음의 상관관계계
plt.show()
```


    
![png](/assets/images/2023-03-02-Machine Learning 3 (보스톤 주택 가격 예측)/output_15_0.png)
    



```python
df.plot(kind='scatter', x='RM', y='PRICE', alpha=0.5)
plt.show()
```


    
![png](/assets/images/2023-03-02-Machine Learning 3 (보스톤 주택 가격 예측)/output_16_0.png)
    


## 2. 데이터 준비


```python
from sklearn.model_selection import train_test_split

# 특성과 정답 분리리
X = df.drop('PRICE', axis=1)
y = df['PRICE']
print(X.shape, y.shape)
```

    (506, 13) (506,)
    


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
```

    (404, 13) (404,) (102, 13) (102,)
    

## 3. 모델 훈련

### 3.1 기본 선형 모델

- 기본 선형 모델 (정규방정식)


```python
from sklearn.linear_model import LinearRegression

# 잘못된 방법 (교차검증을 사용)
lin_reg = LinearRegression() # 모델 생성
lin_reg.fit(X_train, y_train) # 모델 훈련
pred = lin_reg.predict(X_test) # 모델 예측
```


```python
from sklearn.model_selection import cross_val_score
# 올바른 방법 (교차 검증을 사용)
lin_reg = LinearRegression() # 모델 생성

# cross_val_score(모델, 특성, 정답, 평가지표, 폴드수)
scores = cross_val_score(lin_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
lin_reg_rmse = np.sqrt(-scores.mean()) # RMSE
lin_reg_rmse
```




    4.86358080742005



- 기본 선형 모델 (경사하강법 + 특성 스케일링)


```python
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler # 평균을 0, 분산을 1로 맞춰주는 변환기

# 특성 스케일링

# 변환기 = 변환기 객체 생성()
# 변환기.fit() # 변환할 준비
# 변환기.transform() # 실제 변환

std_scaler = StandardScaler() 
X_train_scaled = std_scaler.fit_transform(X_train)
print('특성스케일링 후 평균/표준편차:', X_train_scaled.mean(), X_train_scaled.std())

sgd_reg = SGDRegressor()

# cross_val_score(모델, 특성, 정답, 평가지표, 폴드수)
scores = cross_val_score(sgd_reg, X_train_scaled, y_train, scoring='neg_mean_squared_error', cv=5)
sgd_reg_rmse = np.sqrt(-scores.mean())
sgd_reg_rmse
```

    특성스케일링 후 평균/표준편차: -9.740875280793452e-17 1.0
    




    4.892560383275695



### 3.2 다항 회귀 모델

- 다항 회귀(정규방정식)


```python
from sklearn.preprocessing import PolynomialFeatures # 원본 특성에 제곱항을 추가해주는 변환기

# 변환기 = 변환기 객체 생성()
# 변환기.fit() # 변환할 준비
# 변환기.transform() # 실제 변환

poly_feature = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_feature.fit_transform(X_train)
print(X_train.shape, X_train_poly.shape)

lin_reg = LinearRegression()

# cross_val_score(모델, 특성, 정답, 평가지표, 폴드수)
scores = cross_val_score(lin_reg, X_train_poly, y_train, scoring='neg_mean_squared_error', cv=5)
poly_lin_reg_rmse = np.sqrt(-scores.mean()) # RMSE
poly_lin_reg_rmse
```

    (404, 13) (404, 104)
    




    4.349154691468671



- 다항 회귀(경사하강법)


```python
# (1) Poly(제곱특성추가) -> (2) STD scale(표준화) -> (3) SGDRegressor (경사하강법)

# (1) Poly(제곱특성추가)
poly_feature = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_feature.fit_transform(X_train)
print(X_train.shape, X_train_poly.shape)

# (2) STD scale(표준화)
std_scaler = StandardScaler() 
X_train_poly_scaled = std_scaler.fit_transform(X_train_poly)
print('특성스케일링 후 평균/표준편차:', X_train_poly_scaled.mean(), X_train_poly_scaled.std())

# (3) SGDRegressor (경사 하강법)
sgd_reg = SGDRegressor(penalty=None, random_state=42)
scores = cross_val_score(sgd_reg, X_train_poly_scaled, y_train, scoring='neg_mean_squared_error', cv=5)
sgd_reg_rmse = np.sqrt(-scores.mean())
sgd_reg_rmse
```

    (404, 13) (404, 104)
    특성스케일링 후 평균/표준편차: 3.016965538356861e-16 0.9999999999999999
    




    3.8507394341607575



### 3.3 규제모델
- 모델 파라미터 규제 되는지 확인


```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
```

- 기본 선형회귀 모델


```python
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
```




    LinearRegression()




```python
X_train.shape
```




    (404, 13)




```python
lin_reg.intercept_, lin_reg.coef_
```




    (30.24675099392408,
     array([-1.13055924e-01,  3.01104641e-02,  4.03807204e-02,  2.78443820e+00,
            -1.72026334e+01,  4.43883520e+00, -6.29636221e-03, -1.44786537e+00,
             2.62429736e-01, -1.06467863e-02, -9.15456240e-01,  1.23513347e-02,
            -5.08571424e-01]))




```python
lin_coef = pd.Series(lin_reg.coef_, index=X_train.columns)
lin_coef_sort = lin_coef.sort_values(ascending=False)
sns.barplot(x=lin_coef_sort.values, y= lin_coef_sort.index)
plt.show()
```


    
![png](/assets/images/2023-03-02-Machine Learning 3 (보스톤 주택 가격 예측)/output_38_0.png)
    


- 릿지 회귀(Ridge)


```python
alphas = [0, 0.1, 1, 10, 100]

coef_df = pd.DataFrame()
for alpha in alphas:
  ridge_reg = Ridge(alpha=alpha, random_state=42)
  ridge_reg.fit(X_train, y_train)

  ridge_coef = pd.Series(ridge_reg.coef_, index=X_train.columns)
  ridge_coef_sort = ridge_coef.sort_values(ascending=False)
  column = 'alpha :' + str(alpha)
  coef_df[column] = ridge_coef_sort

coef_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alpha :0</th>
      <th>alpha :0.1</th>
      <th>alpha :1</th>
      <th>alpha :10</th>
      <th>alpha :100</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>RM</th>
      <td>4.438835</td>
      <td>4.445779</td>
      <td>4.464505</td>
      <td>4.195326</td>
      <td>2.438815</td>
    </tr>
    <tr>
      <th>CHAS</th>
      <td>2.784438</td>
      <td>2.750333</td>
      <td>2.545470</td>
      <td>1.813291</td>
      <td>0.550702</td>
    </tr>
    <tr>
      <th>RAD</th>
      <td>0.262430</td>
      <td>0.260043</td>
      <td>0.248882</td>
      <td>0.248031</td>
      <td>0.299014</td>
    </tr>
    <tr>
      <th>INDUS</th>
      <td>0.040381</td>
      <td>0.034896</td>
      <td>0.007498</td>
      <td>-0.026277</td>
      <td>-0.048625</td>
    </tr>
    <tr>
      <th>ZN</th>
      <td>0.030110</td>
      <td>0.030459</td>
      <td>0.032271</td>
      <td>0.035552</td>
      <td>0.039892</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.012351</td>
      <td>0.012400</td>
      <td>0.012642</td>
      <td>0.012833</td>
      <td>0.011951</td>
    </tr>
    <tr>
      <th>AGE</th>
      <td>-0.006296</td>
      <td>-0.007305</td>
      <td>-0.012191</td>
      <td>-0.015341</td>
      <td>0.000545</td>
    </tr>
    <tr>
      <th>TAX</th>
      <td>-0.010647</td>
      <td>-0.010780</td>
      <td>-0.011475</td>
      <td>-0.012744</td>
      <td>-0.014630</td>
    </tr>
    <tr>
      <th>CRIM</th>
      <td>-0.113056</td>
      <td>-0.112400</td>
      <td>-0.109234</td>
      <td>-0.107134</td>
      <td>-0.110765</td>
    </tr>
    <tr>
      <th>LSTAT</th>
      <td>-0.508571</td>
      <td>-0.510902</td>
      <td>-0.523833</td>
      <td>-0.561835</td>
      <td>-0.689539</td>
    </tr>
    <tr>
      <th>PTRATIO</th>
      <td>-0.915456</td>
      <td>-0.900771</td>
      <td>-0.828604</td>
      <td>-0.761769</td>
      <td>-0.817852</td>
    </tr>
    <tr>
      <th>DIS</th>
      <td>-1.447865</td>
      <td>-1.429608</td>
      <td>-1.338700</td>
      <td>-1.232621</td>
      <td>-1.129400</td>
    </tr>
    <tr>
      <th>NOX</th>
      <td>-17.202633</td>
      <td>-15.924459</td>
      <td>-9.537952</td>
      <td>-1.889245</td>
      <td>-0.197859</td>
    </tr>
  </tbody>
</table>
</div>



- 라쏘 회귀(Lasso)


```python
alphas = [0.05, 0.1, 0.2, 0.5, 1]

coef_df = pd.DataFrame()
for alpha in alphas:
  lasso_reg = Lasso(alpha=alpha, random_state=42)
  lasso_reg.fit(X_train, y_train)

  lasso_coef = pd.Series(lasso_reg.coef_, index=X_train.columns)
  lasso_coef_sort = lasso_coef.sort_values(ascending=False)
  column = 'alpha :' + str(alpha)
  coef_df[column] = lasso_coef_sort

coef_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alpha :0.05</th>
      <th>alpha :0.1</th>
      <th>alpha :0.2</th>
      <th>alpha :0.5</th>
      <th>alpha :1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>RM</th>
      <td>4.443676</td>
      <td>4.311687</td>
      <td>4.026917</td>
      <td>3.129886</td>
      <td>1.630489</td>
    </tr>
    <tr>
      <th>CHAS</th>
      <td>1.704029</td>
      <td>0.919952</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>RAD</th>
      <td>0.234443</td>
      <td>0.239237</td>
      <td>0.245289</td>
      <td>0.236596</td>
      <td>0.219654</td>
    </tr>
    <tr>
      <th>ZN</th>
      <td>0.034602</td>
      <td>0.034893</td>
      <td>0.034848</td>
      <td>0.032640</td>
      <td>0.028501</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.013035</td>
      <td>0.013091</td>
      <td>0.013039</td>
      <td>0.012350</td>
      <td>0.011181</td>
    </tr>
    <tr>
      <th>TAX</th>
      <td>-0.012599</td>
      <td>-0.012962</td>
      <td>-0.013317</td>
      <td>-0.013032</td>
      <td>-0.012286</td>
    </tr>
    <tr>
      <th>AGE</th>
      <td>-0.017338</td>
      <td>-0.015126</td>
      <td>-0.010294</td>
      <td>0.000000</td>
      <td>0.016395</td>
    </tr>
    <tr>
      <th>INDUS</th>
      <td>-0.023023</td>
      <td>-0.016785</td>
      <td>-0.005376</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>CRIM</th>
      <td>-0.104256</td>
      <td>-0.104157</td>
      <td>-0.103020</td>
      <td>-0.093034</td>
      <td>-0.076609</td>
    </tr>
    <tr>
      <th>NOX</th>
      <td>-0.524613</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>LSTAT</th>
      <td>-0.549276</td>
      <td>-0.564674</td>
      <td>-0.590514</td>
      <td>-0.649984</td>
      <td>-0.747107</td>
    </tr>
    <tr>
      <th>PTRATIO</th>
      <td>-0.729013</td>
      <td>-0.732247</td>
      <td>-0.741718</td>
      <td>-0.729229</td>
      <td>-0.708582</td>
    </tr>
    <tr>
      <th>DIS</th>
      <td>-1.183960</td>
      <td>-1.151487</td>
      <td>-1.094868</td>
      <td>-0.915255</td>
      <td>-0.630858</td>
    </tr>
  </tbody>
</table>
</div>



- 엘라스틱넷(ElasticNet)


```python
alphas = [0.05, 0.1, 0.2, 0.5, 1]
coef_df = pd.DataFrame()
for alpha in alphas:
  elastic_reg = ElasticNet(alpha=alpha, random_state=42)
  elastic_reg.fit(X_train, y_train)

  elastic_coef = pd.Series(elastic_reg.coef_, index=X_train.columns)
  elastic_coef_sort = elastic_coef.sort_values(ascending=False)
  column = 'alpha :' + str(alpha)
  coef_df[column] = elastic_coef_sort

coef_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alpha :0.05</th>
      <th>alpha :0.1</th>
      <th>alpha :0.2</th>
      <th>alpha :0.5</th>
      <th>alpha :1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>RM</th>
      <td>4.134773</td>
      <td>3.764341</td>
      <td>3.160552</td>
      <td>2.051658</td>
      <td>1.162996</td>
    </tr>
    <tr>
      <th>CHAS</th>
      <td>1.521003</td>
      <td>0.977221</td>
      <td>0.404020</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>RAD</th>
      <td>0.247966</td>
      <td>0.258443</td>
      <td>0.273963</td>
      <td>0.287364</td>
      <td>0.275980</td>
    </tr>
    <tr>
      <th>ZN</th>
      <td>0.035809</td>
      <td>0.037015</td>
      <td>0.038071</td>
      <td>0.037961</td>
      <td>0.035571</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.012867</td>
      <td>0.012746</td>
      <td>0.012439</td>
      <td>0.011721</td>
      <td>0.011013</td>
    </tr>
    <tr>
      <th>TAX</th>
      <td>-0.012941</td>
      <td>-0.013479</td>
      <td>-0.014028</td>
      <td>-0.014505</td>
      <td>-0.014273</td>
    </tr>
    <tr>
      <th>AGE</th>
      <td>-0.014885</td>
      <td>-0.011703</td>
      <td>-0.005211</td>
      <td>0.006508</td>
      <td>0.018591</td>
    </tr>
    <tr>
      <th>INDUS</th>
      <td>-0.027025</td>
      <td>-0.030900</td>
      <td>-0.031594</td>
      <td>-0.030560</td>
      <td>-0.020130</td>
    </tr>
    <tr>
      <th>CRIM</th>
      <td>-0.106450</td>
      <td>-0.106853</td>
      <td>-0.107092</td>
      <td>-0.103047</td>
      <td>-0.093299</td>
    </tr>
    <tr>
      <th>LSTAT</th>
      <td>-0.569848</td>
      <td>-0.600051</td>
      <td>-0.644219</td>
      <td>-0.719262</td>
      <td>-0.775576</td>
    </tr>
    <tr>
      <th>PTRATIO</th>
      <td>-0.753753</td>
      <td>-0.761627</td>
      <td>-0.783677</td>
      <td>-0.794361</td>
      <td>-0.752705</td>
    </tr>
    <tr>
      <th>NOX</th>
      <td>-0.959210</td>
      <td>-0.019932</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>DIS</th>
      <td>-1.205832</td>
      <td>-1.176020</td>
      <td>-1.132253</td>
      <td>-0.987340</td>
      <td>-0.755423</td>
    </tr>
  </tbody>
</table>
</div>



**참고**

- 정규화 (0~1사이로 변환)


```python
def minmax_normalize(arr):
    return (arr-arr.min())/(arr.max()-arr.min())
```

- 표준화 (평균 0, 표준편차 1)


```python
def zscore_standize(arr): # 평균 0, 표준편차 1
    return (arr - arr.mean())/arr.std()
```


```python
X = np.arange(10)
X_normalized = minmax_normalize(X)
X_normalized.min(), X_normalized.max()
```




    (0.0, 1.0)




```python
X = np.arange(10)
X_standardized = zscore_standize(X)
X_standardized.mean(), X_standardized.std()
```




    (-6.661338147750939e-17, 1.0)


