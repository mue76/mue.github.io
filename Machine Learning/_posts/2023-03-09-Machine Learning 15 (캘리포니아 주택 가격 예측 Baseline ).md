---
tag: [python, machine learning, scikit-learn, california housing price]
---

# 캘리포니아 주택 가격 예측 모델 만들기


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
```

## 1. 데이터 가져오기


```python
housing = pd.read_csv('./datasets/housing.csv')
```

## 2. 데이터 훑어보기


```python
housing.head()
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41.0</td>
      <td>880.0</td>
      <td>129.0</td>
      <td>322.0</td>
      <td>126.0</td>
      <td>8.3252</td>
      <td>452600.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21.0</td>
      <td>7099.0</td>
      <td>1106.0</td>
      <td>2401.0</td>
      <td>1138.0</td>
      <td>8.3014</td>
      <td>358500.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1467.0</td>
      <td>190.0</td>
      <td>496.0</td>
      <td>177.0</td>
      <td>7.2574</td>
      <td>352100.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1274.0</td>
      <td>235.0</td>
      <td>558.0</td>
      <td>219.0</td>
      <td>5.6431</td>
      <td>341300.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1627.0</td>
      <td>280.0</td>
      <td>565.0</td>
      <td>259.0</td>
      <td>3.8462</td>
      <td>342200.0</td>
      <td>NEAR BAY</td>
    </tr>
  </tbody>
</table>
</div>




```python
housing.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20640 entries, 0 to 20639
    Data columns (total 10 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   longitude           20640 non-null  float64
     1   latitude            20640 non-null  float64
     2   housing_median_age  20640 non-null  float64
     3   total_rooms         20640 non-null  float64
     4   total_bedrooms      20433 non-null  float64
     5   population          20640 non-null  float64
     6   households          20640 non-null  float64
     7   median_income       20640 non-null  float64
     8   median_house_value  20640 non-null  float64
     9   ocean_proximity     20640 non-null  object 
    dtypes: float64(9), object(1)
    memory usage: 1.6+ MB
    

**범주형 특성 탐색**


```python
housing['ocean_proximity'].value_counts()
```




    <1H OCEAN     9136
    INLAND        6551
    NEAR OCEAN    2658
    NEAR BAY      2290
    ISLAND           5
    Name: ocean_proximity, dtype: int64



**수치형 특성 탐색**


```python
housing.describe()
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20433.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-119.569704</td>
      <td>35.631861</td>
      <td>28.639486</td>
      <td>2635.763081</td>
      <td>537.870553</td>
      <td>1425.476744</td>
      <td>499.539680</td>
      <td>3.870671</td>
      <td>206855.816909</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.003532</td>
      <td>2.135952</td>
      <td>12.585558</td>
      <td>2181.615252</td>
      <td>421.385070</td>
      <td>1132.462122</td>
      <td>382.329753</td>
      <td>1.899822</td>
      <td>115395.615874</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-124.350000</td>
      <td>32.540000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.499900</td>
      <td>14999.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-121.800000</td>
      <td>33.930000</td>
      <td>18.000000</td>
      <td>1447.750000</td>
      <td>296.000000</td>
      <td>787.000000</td>
      <td>280.000000</td>
      <td>2.563400</td>
      <td>119600.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-118.490000</td>
      <td>34.260000</td>
      <td>29.000000</td>
      <td>2127.000000</td>
      <td>435.000000</td>
      <td>1166.000000</td>
      <td>409.000000</td>
      <td>3.534800</td>
      <td>179700.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-118.010000</td>
      <td>37.710000</td>
      <td>37.000000</td>
      <td>3148.000000</td>
      <td>647.000000</td>
      <td>1725.000000</td>
      <td>605.000000</td>
      <td>4.743250</td>
      <td>264725.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-114.310000</td>
      <td>41.950000</td>
      <td>52.000000</td>
      <td>39320.000000</td>
      <td>6445.000000</td>
      <td>35682.000000</td>
      <td>6082.000000</td>
      <td>15.000100</td>
      <td>500001.000000</td>
    </tr>
  </tbody>
</table>
</div>



**수치형 특성별 히스토그램**


```python
housing.hist(bins=50, figsize=(20, 15))
plt.show()
```


    
![png](/assets/images/2023-03-09-Machine Learning 15 (캘리포니아 주택 가격 예측 Baseline )/output_12_0.png)
    


## 3. 데이터 세트 분리
- 훈련 데이터/ 테스트 데이터

**계층적 샘플링(Straityfied sampling)**


```python
bins = [0, 1.5, 3.0, 4.5, 6.0, np.inf]
labels = [1, 2, 3, 4, 5]
housing['income_cat'] = pd.cut(housing['median_income'], bins=bins, labels=labels)
```


```python
housing['income_cat'].value_counts() # 도수
```




    3    7236
    2    6581
    4    3639
    5    2362
    1     822
    Name: income_cat, dtype: int64




```python
housing['income_cat'].value_counts() / len(housing) # 상대도수
```




    3    0.350581
    2    0.318847
    4    0.176308
    5    0.114438
    1    0.039826
    Name: income_cat, dtype: float64




```python
from sklearn.model_selection import train_test_split
# 무작위 샘플링
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# 계층적 샘플링
strat_train_set, strat_test_set = train_test_split(housing, stratify= housing['income_cat'], test_size=0.2, random_state=42)
```


```python
strat_test_set['income_cat'].value_counts() / len(strat_test_set)
```




    3    0.350533
    2    0.318798
    4    0.176357
    5    0.114583
    1    0.039729
    Name: income_cat, dtype: float64



**데이터 되돌리기**


```python
strat_train_set = strat_train_set.drop('income_cat', axis=1)
strat_test_set = strat_test_set.drop('income_cat', axis=1)
```

## 4. 데이터 탐색


```python
# 훈련세트만을 대상으로 데이터 탐색할 예정 (strat_test_set는 최종 예측에 사용)
housing = strat_train_set.copy()
```

### 4.1 지리적 데이터 시각화


```python
# longitude(경도) : 동서
# latitude(위도) : 남북
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.3, grid=True)
```




    <AxesSubplot:xlabel='longitude', ylabel='latitude'>




    
![png](/assets/images/2023-03-09-Machine Learning 15 (캘리포니아 주택 가격 예측 Baseline )/output_25_1.png)
    



```python
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.3, grid=True,
             c='median_house_value', cmap='jet', colorbar=True, figsize=(10, 7), # color 를 통해서 주택가격 표시
             s= housing['population']/100, sharex=False) # size 를 통해서 상대적인 인구수를 표시
```




    <AxesSubplot:xlabel='longitude', ylabel='latitude'>




    
![png](/assets/images/2023-03-09-Machine Learning 15 (캘리포니아 주택 가격 예측 Baseline )/output_26_1.png)
    


**지리적 데이터 분석 결과** : 해안가이면서 밀집 지역일수록 주택 가격이 높음

### 4.2 상관관계 조사

- 상관계수


```python
# 모든 수치형 특성간의 상관계수 확인(타깃 포함)
corr_matrix = housing.corr()
corr_matrix
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>longitude</th>
      <td>1.000000</td>
      <td>-0.924478</td>
      <td>-0.105848</td>
      <td>0.048871</td>
      <td>0.076598</td>
      <td>0.108030</td>
      <td>0.063070</td>
      <td>-0.019583</td>
      <td>-0.047432</td>
    </tr>
    <tr>
      <th>latitude</th>
      <td>-0.924478</td>
      <td>1.000000</td>
      <td>0.005766</td>
      <td>-0.039184</td>
      <td>-0.072419</td>
      <td>-0.115222</td>
      <td>-0.077647</td>
      <td>-0.075205</td>
      <td>-0.142724</td>
    </tr>
    <tr>
      <th>housing_median_age</th>
      <td>-0.105848</td>
      <td>0.005766</td>
      <td>1.000000</td>
      <td>-0.364509</td>
      <td>-0.325047</td>
      <td>-0.298710</td>
      <td>-0.306428</td>
      <td>-0.111360</td>
      <td>0.114110</td>
    </tr>
    <tr>
      <th>total_rooms</th>
      <td>0.048871</td>
      <td>-0.039184</td>
      <td>-0.364509</td>
      <td>1.000000</td>
      <td>0.929379</td>
      <td>0.855109</td>
      <td>0.918392</td>
      <td>0.200087</td>
      <td>0.135097</td>
    </tr>
    <tr>
      <th>total_bedrooms</th>
      <td>0.076598</td>
      <td>-0.072419</td>
      <td>-0.325047</td>
      <td>0.929379</td>
      <td>1.000000</td>
      <td>0.876320</td>
      <td>0.980170</td>
      <td>-0.009740</td>
      <td>0.047689</td>
    </tr>
    <tr>
      <th>population</th>
      <td>0.108030</td>
      <td>-0.115222</td>
      <td>-0.298710</td>
      <td>0.855109</td>
      <td>0.876320</td>
      <td>1.000000</td>
      <td>0.904637</td>
      <td>0.002380</td>
      <td>-0.026920</td>
    </tr>
    <tr>
      <th>households</th>
      <td>0.063070</td>
      <td>-0.077647</td>
      <td>-0.306428</td>
      <td>0.918392</td>
      <td>0.980170</td>
      <td>0.904637</td>
      <td>1.000000</td>
      <td>0.010781</td>
      <td>0.064506</td>
    </tr>
    <tr>
      <th>median_income</th>
      <td>-0.019583</td>
      <td>-0.075205</td>
      <td>-0.111360</td>
      <td>0.200087</td>
      <td>-0.009740</td>
      <td>0.002380</td>
      <td>0.010781</td>
      <td>1.000000</td>
      <td>0.687160</td>
    </tr>
    <tr>
      <th>median_house_value</th>
      <td>-0.047432</td>
      <td>-0.142724</td>
      <td>0.114110</td>
      <td>0.135097</td>
      <td>0.047689</td>
      <td>-0.026920</td>
      <td>0.064506</td>
      <td>0.687160</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 중간 주택가격(타깃)가 특성들간의 상관관계 확인
corr_matrix['median_house_value'].sort_values(ascending=False)
```




    median_house_value    1.000000
    median_income         0.687160
    total_rooms           0.135097
    housing_median_age    0.114110
    households            0.064506
    total_bedrooms        0.047689
    population           -0.026920
    longitude            -0.047432
    latitude             -0.142724
    Name: median_house_value, dtype: float64



- 산점도


```python
attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
pd.plotting.scatter_matrix(housing[attributes], figsize=(12, 8), alpha=0.3)
plt.show()
```


    
![png](/assets/images/2023-03-09-Machine Learning 15 (캘리포니아 주택 가격 예측 Baseline )/output_33_0.png)
    



```python
# 중간 주택가격(타깃)과 중간소득의 산점도
housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1, grid=True)
```




    <AxesSubplot:xlabel='median_income', ylabel='median_house_value'>




    
![png](/assets/images/2023-03-09-Machine Learning 15 (캘리포니아 주택 가격 예측 Baseline )/output_34_1.png)
    


### 4.3 특성 조합을 실험


```python
# 가구당 방의갯수
# 전체방에서 침실방 차지 비율
# 가구당 인구수
housing['rooms_per_households']= housing['total_rooms'] / housing['households']
housing['bedrooms_per_rooms']= housing['total_bedrooms'] / housing['total_rooms']
housing['population_per_households']= housing['population'] / housing['households']
```


```python
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)
```




    median_house_value           1.000000
    median_income                0.687160
    rooms_per_households         0.146285
    total_rooms                  0.135097
    housing_median_age           0.114110
    households                   0.064506
    total_bedrooms               0.047689
    population_per_households   -0.021985
    population                  -0.026920
    longitude                   -0.047432
    latitude                    -0.142724
    bedrooms_per_rooms          -0.259984
    Name: median_house_value, dtype: float64



## 5. 데이터 전처리


```python
# strat_train_set (데이터 탐색, 데이터 전처리)
# strat_test_set (최종 예측측)
```


```python
# 특성(X)과 레이블(y)을 분리
housing = strat_train_set.drop('median_house_value', axis=1) # 특성 (X 데이터)
housing_label = strat_train_set['median_house_value'].copy() # 레이블 (y 데이터)
```


```python
housing.shape, housing_label.shape
```




    ((16512, 9), (16512,))



### 5.1 데이터 전처리(1) - 결손값 처리

**결손값(Null/NaN) 처리 방법**

- 옵션1 : 해당 구역 제거
- 옵션2 : 전체 특성 삭제
- 옵션3 : 어떤 값으로 대체(0, 평균, 중간값 등)


**scikit-learn의 전처리기를 이용하여 옵션3 을 처리**


```python
# <scikit-learn의 전처리기(변환기)들 예시>
# PolynomialFeatures : 다항 특성 추가
# StandardScaler : 표준화(평균 0, 분산 1)
# MinMaxScaler : 정규화(최소 0, 최대 1)
# LabelEncoder, OrdinalEncoder : 숫자로 변환
# OneHotEncoder : OneHot Encoding 
# SimpleImputer : 누락된 데이터 대체

# 함수를 이용한 전처리기
# 나만의 전처리기 
```


```python
# 수치형 데이터만 준비
housing_num = housing.drop('ocean_proximity', axis=1)
housing_num.columns
```




    Index(['longitude', 'latitude', 'housing_median_age', 'total_rooms',
           'total_bedrooms', 'population', 'households', 'median_income'],
          dtype='object')




```python
# 수치형 데이터만 준비(또다른 방법)
# housing_num = housing.select_dtypes(include=[np.number])
# housing_num.columns
```


```python
# SimpleImputer를 결측값을 대체(옵션3) 할 수 있음
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median') # 변환기 객체 생성 (중앙값을 대체)
imputer.fit(housing_num) # 변환할 준비 (중앙값 구하기)
```




    SimpleImputer(strategy='median')




```python
imputer.statistics_
```




    array([-118.51  ,   34.26  ,   29.    , 2119.5   ,  433.    , 1164.    ,
            408.    ,    3.5409])




```python
# 위에서 imputer가 구해준 중앙값과 동일일
housing_num.median()
```




    longitude             -118.5100
    latitude                34.2600
    housing_median_age      29.0000
    total_rooms           2119.5000
    total_bedrooms         433.0000
    population            1164.0000
    households             408.0000
    median_income            3.5409
    dtype: float64




```python
X = imputer.transform(housing_num) # 변환 (중앙값으로 대체)
```


```python
# transform의 결과는 numpy이므로 df로 바꿔서 확인인
# X_df = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
# X_df.info()
```

### 5.2 데이터 전처리(2) - 데이터 인코딩
- 데이터 인코딩을 하는 이유는 머신러닝에서 수치값만 기대하기 때문


```python
housing_cat = housing[['ocean_proximity']] # 2차원 데이터프레임으로 준비
```

#### (1) 레이블 인코딩


```python
# pandas
pd.factorize(housing['ocean_proximity'])
```




    (array([0, 0, 1, ..., 2, 0, 3], dtype=int64),
     Index(['<1H OCEAN', 'NEAR OCEAN', 'INLAND', 'NEAR BAY', 'ISLAND'], dtype='object'))




```python
# scikit-learn 변환기
from sklearn.preprocessing import OrdinalEncoder # LabelEncoder는 1차원 데이터를 기대대

ordinal_encoder = OrdinalEncoder()
ordinal_encoder.fit_transform(housing_cat)
```




    array([[0.],
           [0.],
           [4.],
           ...,
           [1.],
           [0.],
           [3.]])



#### (2) 원핫 인코딩
숫자의 크기가 모델 훈련과정에서 잘못된 영향을 줄 수 있으므로 원핫 인코딩


```python
# pandas
pd.get_dummies(housing_cat)
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
      <th>ocean_proximity_&lt;1H OCEAN</th>
      <th>ocean_proximity_INLAND</th>
      <th>ocean_proximity_ISLAND</th>
      <th>ocean_proximity_NEAR BAY</th>
      <th>ocean_proximity_NEAR OCEAN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17606</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18632</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14650</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3230</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3555</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6563</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12053</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13908</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11159</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15775</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>16512 rows × 5 columns</p>
</div>




```python
# scikit-learn 변환기
from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoder.fit_transform(housing_cat)
```




    array([[1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 1.],
           ...,
           [0., 1., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0.]])




```python
onehot_encoder.categories_
```




    [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
           dtype=object)]



### 5.3 데이터 전처리(3) - 특성 스케일링
- 표준화 (Z score Standardize) : 평균 0, 표준편차 1
- 정규화 (Min Max Scaling) : 0~1 사이로 정규화 (참고 : 특잇값에 영향을 받음)
- 로그 스케일링 : 데이터의 분포가 왜곡되어 있을때 주로 사용


```python
arr = np.arange(9).reshape(3, 3)
arr
```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])




```python
Z_arr = (arr - arr.mean())/arr.std()
Z_arr.mean(), Z_arr.std()
```




    (0.0, 1.0)




```python
M_arr = (arr - arr.min())/(arr.max()-arr.min())
M_arr.min(), M_arr.max()
```




    (0.0, 1.0)




```python
# pandas
def minmax_normalize(arr):
  return (arr - arr.min())/(arr.max()-arr.min())

def zscore_standardize(arr):
  return (arr - arr.mean())/arr.std()
```


```python
# scikit-learn 변환기

# (1) 표준화
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
housing_num_std = std_scaler.fit_transform(housing_num)
housing_num_std.mean(0), housing_num_std.std(0) # 컬럼별로 확인 axis=0
```




    (array([-4.35310702e-15,  2.28456358e-15, -4.70123509e-17,  7.58706190e-17,
                        nan, -3.70074342e-17,  2.07897868e-17, -2.07628918e-16]),
     array([ 1.,  1.,  1.,  1., nan,  1.,  1.,  1.]))




```python
# (2) 정규화
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
housing_num_mm = min_max_scaler.fit_transform(housing_num)
housing_num_mm.min(0), housing_num_mm.max(0)
```




    (array([ 0.,  0.,  0.,  0., nan,  0.,  0.,  0.]),
     array([ 1.,  1.,  1.,  1., nan,  1.,  1.,  1.]))




```python
# (3) 로그 스케일링 
from sklearn.preprocessing import FunctionTransformer
log_transformer = FunctionTransformer(np.log)
log_population = log_transformer.fit_transform(housing_num['population'])
```


```python
# 로그 변환전
housing_num['population'].hist(bins=50)
plt.show()
```


    
![png](/assets/images/2023-03-09-Machine Learning 15 (캘리포니아 주택 가격 예측 Baseline )/output_70_0.png)
    



```python
# 로그 변환후
log_population.hist(bins=50)
plt.show()
```


    
![png](/assets/images/2023-03-09-Machine Learning 15 (캘리포니아 주택 가격 예측 Baseline )/output_71_0.png)
    


### 5.4 데이터 전처리(4) - 변환 파이프라인


```python
housing.columns
```




    Index(['longitude', 'latitude', 'housing_median_age', 'total_rooms',
           'total_bedrooms', 'population', 'households', 'median_income',
           'ocean_proximity'],
          dtype='object')




```python
# 수치형 데이터
# (1) 누락된 데이터를 중앙값으로 대체
# (2) 표준화

from sklearn.pipeline import Pipeline
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')),
                         ('std_scaler', StandardScaler())
                        ])
# num_pipeline.fit_transform(housing_num) # pipeline의 중간결과를 확인하고 싶을 때때
```


```python
# 범주형 데이터
# (1) 원핫 인코딩
# oh_encoder = OneHotEncoder(sparse=False)
# oh_encoder.fit_transform(housing_cat)
```


```python
# 수치형 파이프라인과 범주형 변환기를 한번에 연결할 파이프라인인
from sklearn.compose import ColumnTransformer

num_attrib = list(housing_num.columns)
cat_attrib = ['ocean_proximity']

full_pipeline = ColumnTransformer([('num', num_pipeline, num_attrib),
                                   ('cat', OneHotEncoder(), cat_attrib)
                                  ])
```


```python
housing_prepared = full_pipeline.fit_transform(housing)
```


```python
housing.shape, housing_prepared.shape # 범주형 데이터의 OneHotEncoding으로 4 컬럼 추가가
```




    ((16512, 9), (16512, 13))



## 6. 모델 선택과 훈련


```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
```


```python
lin_reg = LinearRegression()
tree_reg = DecisionTreeRegressor(random_state=42)
rf_reg = RandomForestRegressor(random_state=42)
```


```python
# LinearRegression 교차검증
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_label, scoring="neg_mean_squared_error", cv=10, n_jobs=-1)
lin_rmse = np.sqrt(-lin_scores.mean())
lin_rmse
```




    69274.16940918249




```python
# DecisionTree 교차검증
tree_scores = cross_val_score(tree_reg, housing_prepared, housing_label, scoring="neg_mean_squared_error", cv=10, n_jobs=-1)
tree_rmse = np.sqrt(-tree_scores.mean())
tree_rmse
```




    69448.23452521549




```python
# RandomForest 교차검증
rf_scores = cross_val_score(rf_reg, housing_prepared, housing_label, scoring="neg_mean_squared_error", cv=10, n_jobs=-1)
rf_rmse = np.sqrt(-rf_scores.mean())
rf_rmse
```




    49640.00678301197



## 7. 모델 세부 튜닝

**그리드 탐색**


```python
from sklearn.model_selection import GridSearchCV

rf_reg = RandomForestRegressor(random_state=42)

param_grid = {'n_estimators' : [30, 50, 100], 'max_features' : [2, 4, 6, 8]} # 3 * 4 = 12가지 조합의 파라미터로 설정된 모델 준비

grid_search = GridSearchCV(rf_reg, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1) # 3 * 4 * 5 = 60번의 학습과 검증
%time grid_search.fit(housing_prepared, housing_label)
```

    Wall time: 2min 31s
    




    GridSearchCV(cv=5, estimator=RandomForestRegressor(random_state=42), n_jobs=-1,
                 param_grid={'max_features': [2, 4, 6, 8],
                             'n_estimators': [30, 50, 100]},
                 scoring='neg_mean_squared_error')




```python
grid_search.best_params_
```




    {'max_features': 8, 'n_estimators': 100}




```python
grid_search.best_estimator_
```




    RandomForestRegressor(max_features=8, random_state=42)




```python
cv_results = grid_search.cv_results_
for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
  print(np.sqrt(-mean_score), params)
```

    52745.33887865031 {'max_features': 2, 'n_estimators': 30}
    52214.445796615226 {'max_features': 2, 'n_estimators': 50}
    51781.43495872492 {'max_features': 2, 'n_estimators': 100}
    50663.79774079741 {'max_features': 4, 'n_estimators': 30}
    50277.936831978 {'max_features': 4, 'n_estimators': 50}
    50090.984409225 {'max_features': 4, 'n_estimators': 100}
    50028.060190761295 {'max_features': 6, 'n_estimators': 30}
    49701.69831473408 {'max_features': 6, 'n_estimators': 50}
    49565.52108533805 {'max_features': 6, 'n_estimators': 100}
    50165.81805010987 {'max_features': 8, 'n_estimators': 30}
    49679.042709503105 {'max_features': 8, 'n_estimators': 50}
    49532.45382221812 {'max_features': 8, 'n_estimators': 100}
    

**랜덤 탐색**


```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {'n_estimators' : randint(low=1, high=200),
                  'max_features' : randint(low=1, high=8)}

rnd_search = RandomizedSearchCV(rf_reg, param_distribs, n_iter=10, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)                  
%time rnd_search.fit(housing_prepared, housing_label)
```

    Wall time: 2min 45s
    




    RandomizedSearchCV(cv=5, estimator=RandomForestRegressor(random_state=42),
                       n_jobs=-1,
                       param_distributions={'max_features': <scipy.stats._distn_infrastructure.rv_frozen object at 0x000001CEA3CF0580>,
                                            'n_estimators': <scipy.stats._distn_infrastructure.rv_frozen object at 0x000001CEA3CF0550>},
                       scoring='neg_mean_squared_error')




```python
rnd_search.best_params_
```




    {'max_features': 7, 'n_estimators': 163}




```python
rnd_search.best_estimator_
```




    RandomForestRegressor(max_features=7, n_estimators=163, random_state=42)




```python
cv_results = rnd_search.cv_results_
for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
  print(np.sqrt(-mean_score), params)
```

    52439.59512716884 {'max_features': 7, 'n_estimators': 10}
    50628.08766555357 {'max_features': 3, 'n_estimators': 130}
    51431.57816434721 {'max_features': 6, 'n_estimators': 12}
    49890.12353477528 {'max_features': 5, 'n_estimators': 87}
    49489.55348866853 {'max_features': 7, 'n_estimators': 163}
    50015.728524855505 {'max_features': 4, 'n_estimators': 120}
    49566.82782355478 {'max_features': 6, 'n_estimators': 64}
    50417.44706394684 {'max_features': 4, 'n_estimators': 43}
    50019.98070924308 {'max_features': 4, 'n_estimators': 132}
    50810.67796991394 {'max_features': 3, 'n_estimators': 78}
    


```python
# 참고
# 지수분포
# https://en.wikipedia.org/wiki/Exponential_distribution

# 로그 유니폼 분포
# https://en.wikipedia.org/wiki/Reciprocal_distribution

# param_distribs = {
#         'kernel': ['linear', 'rbf'],
#         'C': reciprocal(20, 200000), # 로그유니폼분포
#         'gamma': expon(scale=1.0),   # 지수분포
#     }

# reciprocal : 주어진 범위 안에서 균등 분포로 샘플링. 하이파라미터의 스케일에 대해 잘 모를때 사용용
# expon : 하이파라미터의 스케일에대해 어느 정도 알고 있을 때 사용용
```


```python
best_model = rnd_search.best_estimator_
```

**모델의 특성 중요도**
- 특성 중요도는 트리 기반 모델만 제공


```python
feature_importances = best_model.feature_importances_
```


```python
onehot_encoder = full_pipeline.named_transformers_['cat']
cat_attrib = list(onehot_encoder.categories_[0])

attributes = num_attrib + cat_attrib
sorted(zip(feature_importances, attributes), reverse=True)
```




    [(0.4306509289965746, 'median_income'),
     (0.14767733038782704, 'INLAND'),
     (0.11659906700600976, 'longitude'),
     (0.10277782275663969, 'latitude'),
     (0.049473849699940384, 'housing_median_age'),
     (0.04047400008194098, 'population'),
     (0.03173798410316071, 'total_rooms'),
     (0.028947917625006446, 'total_bedrooms'),
     (0.027055836949122423, 'households'),
     (0.015431473611590581, '<1H OCEAN'),
     (0.006618272198336929, 'NEAR OCEAN'),
     (0.002492445670795039, 'NEAR BAY'),
     (6.307091305548657e-05, 'ISLAND')]



## 8. 모델 예측과 성능 평가

- 테스트 데이터 변환환


```python
X_test = strat_test_set.drop('median_house_value', axis=1)
y_test = strat_test_set['median_house_value'].copy()
X_test.shape, y_test.shape
```




    ((4128, 9), (4128,))




```python
# 훈련데이터에 대해서 전처리 했던 것들
# (1) 수치 데이터 -> 누락값처리/표준화
# (2) 범주 데이터 -> 원핫인코딩
# ==> 테스트에 대해서도 동일하게 처리해주자!
```


```python
# 훈련 데이터를 변경할때는 파이프라인의 fit_transform()을 사용
# 테스트 데이터를 변경할때는 파이프라인의 transform()을 사용

X_test_prepared = full_pipeline.transform(X_test)
X_test_prepared.shape
```




    (4128, 13)



- 예측과 평가


```python
from sklearn.metrics import mean_squared_error

final_predictions = best_model.predict(X_test_prepared)
final_rmse = mean_squared_error(y_test, final_predictions, squared=False) # RMSE
final_rmse
```




    46496.80161722716



- 테스트 데이터의 변환과 예측을 한번에


```python
# 전처리와와 모델을 파이프라인으로 연결해서 예측
full_pipeline_with_predictor = Pipeline([('preparation', full_pipeline),
                                         ('final', best_model)
                                         ])
final_predictions = full_pipeline_with_predictor.predict(X_test)
final_rmse = mean_squared_error(y_test, final_predictions, squared=False) # RMSE
final_rmse
```




    46496.80161722716



**일반화 오차 추정**
- 테스트 RMSE에 대한 95% 신뢰 구간

![image.png](data:image/png;base64,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)


```python
from scipy.stats import t

# 추정량 (오차의 제곱들의 합)
squared_erros = (final_predictions - y_test)**2

# 95% 신뢰구간
confidence = 0.95

# 표본의 크기
n = len(squared_erros)

# 자유도 (degree of freedom)
dof = n-1

# 추정량의 평균
m_squared_error = np.mean(squared_erros)

# 표본의 표준편차 (비편상 분산으로 구함)
sample_std = np.std(squared_erros, ddof=1) # n-1로 나눔 (그림에서 U)

# 표준 오차
std_err = sample_std/n**0.5 # (그림에서 U/n**0.5)

mse_ci = t.interval(confidence, dof, m_squared_error, std_err)
rmse_ci = np.sqrt(mse_ci)
rmse_ci
```




    array([44584.21439297, 48333.76612913])



## 9. 모델 저장


```python
import joblib

joblib.dump(full_pipeline_with_predictor, 'my_model.pkl')
```




    ['my_model.pkl']




```python
# 다시 불러오기
loaded_model = joblib.load('my_model.pkl')
final_predictions2 = loaded_model.predict(X_test)
mean_squared_error(y_test, final_predictions2, squared=False) # RMSE
```




    46496.80161722716

## Reference
- [핸즈온 머신러닝 (오렐리앙 제롱 저)](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=237677114)
