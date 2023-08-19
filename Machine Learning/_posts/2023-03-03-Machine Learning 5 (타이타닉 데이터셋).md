## 타이타닉 데이터셋 도전

- 승객의 나이, 성별, 승객 등급, 승선 위치 같은 속성을 기반으로 하여 승객의 생존 여부를 예측하는 것이 목표

- [캐글](https://www.kaggle.com)의 [타이타닉 챌린지](https://www.kaggle.com/c/titanic)에서 `train.csv`와 `test.csv`를 다운로드
- 두 파일을 각각 datasets 디렉토리에 titanic_train.csv titanic_test.csv로 저장

## 1. 데이터 탐색


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

## 1. 데이터 가져오기


```python
train_df = pd.read_csv("./datasets/titanic_train.csv")
test_df = pd.read_csv("./datasets/titanic_test.csv")
submission = pd.read_csv("./datasets/gender_submission.csv")
```

## 2. 데이터 훑어보기


```python
train_df.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



* **Survived**: 타깃. 0은 생존하지 못한 것이고 1은 생존을 의미
* **Pclass**: 승객 등급. 1, 2, 3등석.
* **Name**, **Sex**, **Age**: 이름 그대로의 의미
* **SibSp**: 함께 탑승한 형제, 배우자의 수
* **Parch**: 함께 탑승한 자녀, 부모의 수
* **Ticket**: 티켓 아이디
* **Fare**: 티켓 요금 (파운드)
* **Cabin**: 객실 번호
* **Embarked**: 승객이 탑승한 곳. C(Cherbourg), Q(Queenstown), S(Southampton)



```python
train_df.info()
```

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
    

**범주형 특성 탐색**

- **Pclass**, **Sex**, **Embarked**
- **Embarked** 특성은 승객이 탑승한 곳 : C=Cherbourg, Q=Queenstown, S=Southampton.


```python
train_df['Pclass'].value_counts(dropna=False)
```




    3    491
    1    216
    2    184
    Name: Pclass, dtype: int64




```python
train_df['Sex'].value_counts(dropna=False)
```




    male      577
    female    314
    Name: Sex, dtype: int64




```python
train_df['Embarked'].value_counts(dropna=False)
```




    S      644
    C      168
    Q       77
    NaN      2
    Name: Embarked, dtype: int64



**수치형 특성 탐색**


```python
train_df.hist(bins=50, figsize=(20, 10))
plt.show()
```


    
![png](/assets/images/2023-03-03-Machine Learning 5 (타이타닉 데이터셋)/output_17_0.png)
    



```python
train_df.corrwith(train_df['Survived']).sort_values(ascending=False)
```




    Survived       1.000000
    Fare           0.257307
    Parch          0.081629
    PassengerId   -0.005007
    SibSp         -0.035322
    Age           -0.077221
    Pclass        -0.338481
    dtype: float64



**특잇값 확인**


```python
train_df.describe()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



- Fare 특성을 boxplot으로 분포 확인하기


```python
train_df['Fare'].plot(kind='box', figsize=(6, 6))
plt.show()
```


    
![png](/assets/images/2023-03-03-Machine Learning 5 (타이타닉 데이터셋)/output_22_0.png)
    


- 숫자 특성들에 대한 분포 boxplot으로 확인하기


```python
cat_columns = ['Age', 'Fare', 'SibSp', 'Parch']
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 8))
for i, column in enumerate(cat_columns):
  train_df[column].plot(kind='box', ax=axes[i])
plt.show()
```


    
![png](/assets/images/2023-03-03-Machine Learning 5 (타이타닉 데이터셋)/output_24_0.png)
    



```python
# seaborn을 그리면 더 세련된 시각화
num_columns = ['Age', 'Fare', 'SibSp', 'Parch']
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 4))
for i, column in enumerate(num_columns):
  sns.boxplot(data=train_df, x=column, ax=axes[i])
plt.show()
```


    
![png](/assets/images/2023-03-03-Machine Learning 5 (타이타닉 데이터셋)/output_25_0.png)
    



```python
q1 = train_df["Fare"].quantile(0.25)
q3 = train_df["Fare"].quantile(0.75)
iqr = q3 -q1 # 박스의 길이이

cond = train_df["Fare"] >= q3 + (1.5 * iqr)
outliers = train_df.loc[cond]
max_wo_outlier = outliers['Fare'].min()
print(outliers.index, len(outliers))
```

    Int64Index([  1,  27,  31,  34,  52,  61,  62,  72,  88, 102,
                ...
                792, 802, 820, 829, 835, 846, 849, 856, 863, 879],
               dtype='int64', length=116) 116
    


```python
train_df.loc[outliers.index, 'Fare'] = max_wo_outlier
```


```python
train_df['Fare'].describe()
```




    count    891.000000
    mean      24.172526
    std       20.738142
    min        0.000000
    25%        7.910400
    50%       14.454200
    75%       31.000000
    max       66.600000
    Name: Fare, dtype: float64



**타깃에 영향을 미치는 특성 탐색**


```python
cat_columns = ['Pclass', 'Embarked', 'Sex', 'SibSp',	'Parch']
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 4))
for i, column in enumerate(cat_columns):
  sns.barplot(data=train_df, x=column, y='Survived', ax=axes[i])
plt.show()
```


    
![png](/assets/images/2023-03-03-Machine Learning 5 (타이타닉 데이터셋)/output_30_0.png)
    



```python
# 기존 특성을 바탕으로 새로운 특성 추가
train_df['Family_size'] = train_df['Parch'] + train_df['SibSp'] + 1
sns.barplot(data=train_df, x='Family_size', y='Survived')
plt.show()
```


    
![png](/assets/images/2023-03-03-Machine Learning 5 (타이타닉 데이터셋)/output_31_0.png)
    



```python
train_df['Family_size'].value_counts()
```




    1     537
    2     161
    3     102
    4      29
    6      22
    5      15
    7      12
    11      7
    8       6
    Name: Family_size, dtype: int64




```python
bins = [0, 1, 2, 4, 12]
train_df['Family_size'] = pd.cut(train_df['Family_size'], bins=bins, labels=['Single', 'SmallF', 'MedF', 'LargeF'])
```


```python
sns.barplot(data=train_df, x='Family_size', y='Survived')
plt.show()
```


    
![png](/assets/images/2023-03-03-Machine Learning 5 (타이타닉 데이터셋)/output_34_0.png)
    



```python
bins = [0, 18, 25, 60, 90]
group_names = ['Children', 'Youth', 'Adult', 'Senior']

train_df['Age_cat'] = pd.cut(train_df['Age'], bins, labels=group_names)
```


```python
sns.barplot(data=train_df, x='Age_cat', y='Survived')
plt.show()
```


    
![png](/assets/images/2023-03-03-Machine Learning 5 (타이타닉 데이터셋)/output_36_0.png)
    


**target 분포 확인**


```python
train_df['Survived'].value_counts()
```




    0    549
    1    342
    Name: Survived, dtype: int64




```python
sns.countplot(data=train_df, x='Survived')
plt.show()
```


    
![png](/assets/images/2023-03-03-Machine Learning 5 (타이타닉 데이터셋)/output_39_0.png)
    



```python
train_df.columns
```




    Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
           'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Family_size',
           'Age_cat'],
          dtype='object')




```python
# 현재까지 train_df에 적용했던 내용들을 test_df에도 적용
# 특잇값 처리, Family_size, Age_cat 열 추가

# (1)
cond = test_df["Fare"] >= q3 + (1.5 * iqr)
outliers = test_df.loc[cond]
max_wo_outlier = outliers['Fare'].min()
test_df.loc[outliers.index, 'Fare'] = max_wo_outlier

# (2)
test_df['Family_size'] = test_df['Parch'] + test_df['SibSp'] + 1
bins = [0, 1, 2, 4, 12]
test_df['Family_size'] = pd.cut(test_df['Family_size'], bins=bins, labels=['Single', 'SmallF', 'MedF', 'LargeF'])

# (3)
bins = [0, 18, 25, 60, 80]
group_names = ['Children', 'Youth', 'Adult', 'Senior']

test_df['Age_cat'] = pd.cut(test_df['Age'], bins, labels=group_names)

```


```python
test_df.columns
```




    Index(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
           'Ticket', 'Fare', 'Cabin', 'Embarked', 'Family_size', 'Age_cat'],
          dtype='object')



## 2. 데이터 전처리 (누락 데이터 처리, 범주화 등)

### 2.1 특성데이터와 레이블 데이터 분리


```python
X_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']
```

### 2.2 범주형 데이터 전처리


```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
```


```python
import sklearn
sklearn.__version__
```




    '0.24.1'




```python
cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')), # 누락된 데이터를 채워주는 변환기
                ('oh_encoder', OneHotEncoder(sparse=False)) # Onehot Encoding 해주는 변환기, sklearn 1.2 에서는 sparse_output
            ])
```

### 2.3 수치형 데이터 전처리


```python
num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')), # 누락된 데이터를 채워주는 변환기
                ('std_scaler', StandardScaler()) # 평균을 0, 표준편차1로 만들어주는 변환기기
            ])
```

### 2.4 수치형 데이터와 범주형 데이터 연결


```python
from sklearn.compose import ColumnTransformer

num_attribs = ['Age', 'Fare']
cat_attribs = ['Pclass', 'Sex', 'Embarked', 'Family_size'] #, 'Age_cat']

full_pipeline = ColumnTransformer([
                    #('num', SimpleImputer(strategy='median'), num_attribs), # 수치형
                    ('num', num_pipeline, num_attribs), # 수치형
                    ('cat', cat_pipeline, cat_attribs)  # 범주형
            ])

X_train_prepraed = full_pipeline.fit_transform(X_train)
```


```python
X_train.shape
```




    (891, 13)




```python
X_train_prepraed.shape
```




    (891, 14)



## 3. 모델 선택과 훈련

- Logistic Regression


```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
log_clf = LogisticRegression(max_iter=1000, random_state=42) # C가 작을 수록 규제가 강해짐
log_clf_scores = cross_val_score(log_clf, X_train_prepraed, y_train, scoring="accuracy", cv=3)
log_clf_scores
```




    array([0.7979798 , 0.81144781, 0.81481481])



- KNearestNeighbors


```python
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf_scores = cross_val_score(knn_clf, X_train_prepraed, y_train, scoring="accuracy", cv=3)
knn_clf_scores
```




    array([0.7979798, 0.8047138, 0.8047138])



- Decision Tree


```python
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier()
tree_clf_scores = cross_val_score(tree_clf, X_train_prepraed, y_train, scoring="accuracy", cv=3)
tree_clf_scores
```




    array([0.73400673, 0.78114478, 0.76430976])



- RandomForest


```python
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(random_state=42, n_jobs=-1)
rnd_clf_scores = cross_val_score(rnd_clf, X_train_prepraed, y_train, scoring="accuracy", cv=3)
rnd_clf_scores
```




    array([0.75757576, 0.81481481, 0.79124579])



- SVM


```python
from sklearn.svm import LinearSVC, SVC

# 1. LinearSVC
linear_svc = LinearSVC(random_state = 42)
linear_svc_scores = cross_val_score(linear_svc, X_train_prepraed, y_train, scoring="accuracy", cv=3)
print(linear_svc_scores)

# 2. SCV
linear_kernel_svc = SVC(kernel='linear')
linear_kernel_svc_scores = cross_val_score(linear_kernel_svc, X_train_prepraed, y_train, scoring="accuracy", cv=3)
print(linear_kernel_svc_scores)
```

    [0.7979798  0.81144781 0.80808081]
    [0.7979798  0.81481481 0.7979798 ]
    

## 4. 하이퍼파라미터 튜닝


```python
from sklearn.model_selection import GridSearchCV
```

- LogisticRegression


```python
param_grid = {'C':[0.1, 0.2, 0.5, 1, 10], 'max_iter':[1000, 2000]} # 5x2 = 10개의 조합

grid_search = GridSearchCV(log_clf, param_grid, cv=5, scoring="accuracy", n_jobs=-1) # 5x2x5 = 50번의 학습과 검증
grid_search.fit(X_train_prepraed, y_train)
```




    GridSearchCV(cv=5, estimator=LogisticRegression(max_iter=1000, random_state=42),
                 n_jobs=-1,
                 param_grid={'C': [0.1, 0.2, 0.5, 1, 10], 'max_iter': [1000, 2000]},
                 scoring='accuracy')




```python
grid_search.best_params_
```




    {'C': 1, 'max_iter': 1000}




```python
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
  print(mean_score, params)
```

    0.8047329106772958 {'C': 0.1, 'max_iter': 1000}
    0.8047329106772958 {'C': 0.1, 'max_iter': 2000}
    0.8103195028560668 {'C': 0.2, 'max_iter': 1000}
    0.8103195028560668 {'C': 0.2, 'max_iter': 2000}
    0.8114493754315486 {'C': 0.5, 'max_iter': 1000}
    0.8114493754315486 {'C': 0.5, 'max_iter': 2000}
    0.8136965664427844 {'C': 1, 'max_iter': 1000}
    0.8136965664427844 {'C': 1, 'max_iter': 2000}
    0.813684012303057 {'C': 10, 'max_iter': 1000}
    0.813684012303057 {'C': 10, 'max_iter': 2000}
    


```python
lr_final_model = grid_search.best_estimator_
lr_final_model.fit(X_train_prepraed, y_train)
```




    LogisticRegression(C=1, max_iter=1000, random_state=42)



- KNeighborsClassifier


```python
param_grid = {'n_neighbors':[3, 5, 7, 9, 11]}

grid_search = GridSearchCV(knn_clf, param_grid, cv=5, scoring="accuracy", n_jobs=-1) # 24 * 5 = 100번의 학습과 검증
grid_search.fit(X_train_prepraed, y_train)
```




    GridSearchCV(cv=5, estimator=KNeighborsClassifier(), n_jobs=-1,
                 param_grid={'n_neighbors': [3, 5, 7, 9, 11]}, scoring='accuracy')




```python
grid_search.best_params_
```




    {'n_neighbors': 5}




```python
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
  print(mean_score, params)
```

    0.7980101688531793 {'n_neighbors': 3}
    0.8148327160881299 {'n_neighbors': 5}
    0.7946582135459168 {'n_neighbors': 7}
    0.7968865733475614 {'n_neighbors': 9}
    0.7969305128366079 {'n_neighbors': 11}
    

- Decision Tree


```python
param_grid = {'max_depth':[3, 5, 7, 9, 11]}

grid_search = GridSearchCV(tree_clf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train_prepraed, y_train)
```




    GridSearchCV(cv=5, estimator=DecisionTreeClassifier(), n_jobs=-1,
                 param_grid={'max_depth': [3, 5, 7, 9, 11]}, scoring='accuracy')




```python
grid_search.best_params_
```




    {'max_depth': 7}




```python
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
  print(mean_score, params)
```

    0.8103132257862029 {'max_depth': 3}
    0.8181721172556651 {'max_depth': 5}
    0.8282719226664993 {'max_depth': 7}
    0.8036155922415418 {'max_depth': 9}
    0.7991212102190698 {'max_depth': 11}
    


```python
tree_final_model = grid_search.best_estimator_
tree_final_model.fit(X_train_prepraed, y_train)
```




    DecisionTreeClassifier(max_depth=7)




```python
num_attribs = ['Age', 'Fare']
cat_attribs = ['Pclass', 'Sex', 'Embarked', 'Family_size'] #, 'Age_cat']
cat_ohe_attribs = ['Pclass_1', 'Pclass_2', 'Pclass_3',
                   'Sex_1', 'Sex_2', 'Embarked_1', 'Embarked_2', 'Embarked_3',
                   'Family_size_1', 'Family_size_2', 'Family_size_3', 'Family_size_4']
                   #'Age_cat_1', 'Age_cat_2', 'Age_cat_3', 'Age_cat_4', 'Age_cat_5']

all_attribs = num_attribs + cat_ohe_attribs
target_names = ['Not survived', 'Survived']
```


```python
from sklearn import tree
plt.figure(figsize=(150, 50))
res = tree.plot_tree(tree_final_model,
               feature_names = all_attribs,
               class_names = target_names,
               rounded = True,
               filled = True,
               fontsize=25)
```


    
![png](/assets/images/2023-03-03-Machine Learning 5 (타이타닉 데이터셋)/output_84_0.png)
    



```python
sorted(zip(tree_final_model.feature_importances_, all_attribs), reverse=True)
```




    [(0.4799753034557417, 'Sex_1'),
     (0.1470328405953695, 'Fare'),
     (0.13179082253077198, 'Age'),
     (0.12021196678995298, 'Pclass_3'),
     (0.04869029678945934, 'Family_size_1'),
     (0.04661302686623672, 'Pclass_1'),
     (0.01001711785051913, 'Embarked_3'),
     (0.00633026610480512, 'Family_size_4'),
     (0.006172009452184985, 'Family_size_3'),
     (0.002922820151598823, 'Embarked_1'),
     (0.00024352941335955124, 'Family_size_2'),
     (0.0, 'Sex_2'),
     (0.0, 'Pclass_2'),
     (0.0, 'Embarked_2')]



- RandomForest


```python
param_grid = {'n_estimators':[100, 200, 300], 'max_depth':[5, 7, 9, 11]}

grid_search = GridSearchCV(rnd_clf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train_prepraed, y_train)
```




    GridSearchCV(cv=5, estimator=RandomForestClassifier(n_jobs=-1, random_state=42),
                 n_jobs=-1,
                 param_grid={'max_depth': [5, 7, 9, 11],
                             'n_estimators': [100, 200, 300]},
                 scoring='accuracy')




```python
print(grid_search.best_params_)
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
  print(mean_score, params)

rf_final_model = grid_search.best_estimator_
rf_final_model.fit(X_train_prepraed, y_train)

```

    {'max_depth': 7, 'n_estimators': 100}
    0.8193082669010107 {'max_depth': 5, 'n_estimators': 100}
    0.8193019898311468 {'max_depth': 5, 'n_estimators': 200}
    0.8170610758897746 {'max_depth': 5, 'n_estimators': 300}
    0.8226916075575922 {'max_depth': 7, 'n_estimators': 100}
    0.8226853304877284 {'max_depth': 7, 'n_estimators': 200}
    0.8182035026049841 {'max_depth': 7, 'n_estimators': 300}
    0.8215805661917017 {'max_depth': 9, 'n_estimators': 100}
    0.8215805661917017 {'max_depth': 9, 'n_estimators': 200}
    0.8193333751804659 {'max_depth': 9, 'n_estimators': 300}
    0.8159625886636119 {'max_depth': 11, 'n_estimators': 100}
    0.8159688657334756 {'max_depth': 11, 'n_estimators': 200}
    0.8170861841692298 {'max_depth': 11, 'n_estimators': 300}
    




    RandomForestClassifier(max_depth=7, n_jobs=-1, random_state=42)




```python
sorted(zip(rf_final_model.feature_importances_, all_attribs), reverse=True)
```




    [(0.22661319164248106, 'Sex_2'),
     (0.20598466532945145, 'Sex_1'),
     (0.15065113082226642, 'Fare'),
     (0.1476700317091729, 'Age'),
     (0.06749172988344757, 'Pclass_3'),
     (0.04309627388087689, 'Pclass_1'),
     (0.03810687703951553, 'Family_size_1'),
     (0.031037269118605965, 'Family_size_2'),
     (0.0198646408180714, 'Pclass_2'),
     (0.01908234726067075, 'Family_size_3'),
     (0.015539397763291867, 'Embarked_3'),
     (0.01481044882815969, 'Embarked_1'),
     (0.01111682809322282, 'Family_size_4'),
     (0.008935167810765757, 'Embarked_2')]



- SVM

- (1) linear_svc


```python
param_grid = {'C':[0.1, 1, 2, 5, 10]}

grid_search = GridSearchCV(linear_svc, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train_prepraed, y_train)

print(grid_search.best_params_)
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
  print(mean_score, params)
```

    {'C': 10}
    0.8080723118448307 {'C': 0.1}
    0.8080785889146945 {'C': 1}
    0.8080785889146945 {'C': 2}
    0.8080785889146945 {'C': 5}
    0.8092021844203126 {'C': 10}
    

    C:\Users\mue\anaconda3\lib\site-packages\sklearn\svm\_base.py:985: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      warnings.warn("Liblinear failed to converge, increase "
    

- (2) svc


```python
# 시간이 오래 걸림. 파라미터 줄여서 탐색하기기
# param_grid = [{'kernel': ['poly'], 'degree': [2, 3, 4], 'coef0':[1, 50, 100], 'C':[0.1, 1, 2, 5, 10, 50, 100, 500, 1000]},
#               {'kernel': ['rbf'], 'gamma': [0.1, 5, 10], 'C':[0.1, 1, 2, 5, 10, 100, 500, 1000]}]

# grid_search = GridSearchCV(linear_kernel_svc, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
# grid_search.fit(X_train_prepraed, y_train)

# print(grid_search.best_params_)
# cvres = grid_search.cv_results_

# for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
#   print(mean_score, params)
```

## 5. 예측과 성능 평가(by kaggle)


```python
X_test_preprocessed = full_pipeline.transform(test_df)
X_test_preprocessed
```




    array([[ 0.39488658, -0.78852313,  0.        , ...,  0.        ,
             1.        ,  0.        ],
           [ 1.35550962, -0.82852989,  0.        , ...,  0.        ,
             0.        ,  1.        ],
           [ 2.50825727, -0.69886497,  0.        , ...,  0.        ,
             1.        ,  0.        ],
           ...,
           [ 0.70228595, -0.81646803,  0.        , ...,  0.        ,
             1.        ,  0.        ],
           [-0.1046374 , -0.7778701 ,  0.        , ...,  0.        ,
             1.        ,  0.        ],
           [-0.1046374 , -0.08753169,  0.        , ...,  1.        ,
             0.        ,  0.        ]])




```python
test_df.shape, X_test_preprocessed.shape
```




    ((418, 13), (418, 14))




```python
final_pred = rf_final_model.predict(X_test_preprocessed)
```


```python
submission["Survived"] = final_pred
```


```python
ver = 10
submission.to_csv('./datasets/titanic_ver_{}_submission.csv'.format(ver), index=False)
```
