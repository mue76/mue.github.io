# PCA - 붓꽃 데이터


```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
```


```python
iris = load_iris()
```


```python
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df.head(3)
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
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



**sepal_length, sepal_width 두개의 특성으로 산점도**


```python
plt.scatter(iris_df['sepal length (cm)'], iris_df['sepal width (cm)'], c=iris_df['target'])
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()
```


    
![png](/assets/images/2023-03-09-Machine Learning 13 (PCA)/output_5_0.png)
    



```python
X = iris.data
y = iris.target
```


```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
iris_scaled = scaler.fit_transform(X)
```


```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2) # 4차원을 특성을 2차원으로 축소
iris_pca = pca.fit_transform(iris_scaled)
```


```python
iris_pca.shape
```




    (150, 2)




```python
pca_columns = ['pca_component1', 'pca_component2']
iris_pca_df = pd.DataFrame(iris_pca, columns=pca_columns)
iris_pca_df['target'] = iris.target
iris_pca_df.head(3)
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
      <th>pca_component1</th>
      <th>pca_component2</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.264703</td>
      <td>0.480027</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.080961</td>
      <td>-0.674134</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.364229</td>
      <td>-0.341908</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



**PCA로 차원축소된 새로운 특성으로 산점도**


```python
plt.scatter(iris_pca_df['pca_component1'], iris_pca_df['pca_component2'], c=iris_pca_df['target'])
plt.xlabel('pca component 1')
plt.ylabel('pca component 2')
plt.show()
```


    
![png](/assets/images/2023-03-09-Machine Learning 13 (PCA)/output_12_0.png)
    



```python
pca.explained_variance_ratio_ # 각 pca component 의 분산 비율
```




    array([0.72962445, 0.22850762])




```python
0.72962445 + 0.22850762
```




    0.95813207



**원본 데이터와 PCA 변환된 데이터로 모델 성능 측정**


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rf_clf = RandomForestClassifier(random_state=42)
scores = cross_val_score(rf_clf, X, y, scoring='accuracy', cv=3)
scores
```




    array([0.98, 0.94, 0.98])




```python
rf_clf = RandomForestClassifier(random_state=42)
scores = cross_val_score(rf_clf, iris_pca, y, scoring='accuracy', cv=3)
scores
```




    array([0.88, 0.88, 0.9 ])




```python
iris_pca.shape
```




    (150, 2)


