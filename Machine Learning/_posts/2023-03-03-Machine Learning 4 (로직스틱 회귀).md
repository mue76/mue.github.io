---
tag: [machine learning, scikit-learn]
toc: true
toc_sticky: true
toc_label: 목차
---

# 로지스틱 회귀


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
```


```python
iris = load_iris()
```


```python
iris.feature_names
```




    ['sepal length (cm)',
     'sepal width (cm)',
     'petal length (cm)',
     'petal width (cm)']




```python
(iris.target == 2).astype(np.int32) # Virginica이면 1, 아니면 0
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])




```python
X = iris.data[:, 3:] # 특성 1개만 사용('petal width (cm)')
y = (iris.target == 2).astype(np.int32) # Virginica이면 1, 아니면 0
```


```python
from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(random_state=42)
lr_clf.fit(X, y) # 훈련 : 비용함수(로그손실)를 최소화하는 모델 파라미터(theta 0, 1)를 찾기
```




    LogisticRegression(random_state=42)




```python
lr_clf.intercept_, lr_clf.coef_
```




    (array([-7.1947083]), array([[4.3330846]]))




```python
lr_clf.predict([[2.5]]) # 미지의 특성(petal width) 데이터를 2차원 데이터로 준비비
```




    array([1])




```python
lr_clf.predict_proba([[2.5]]) # Virginica일 확률(0.025 :Virginica 아닐 확률, 0.97:Virginica일 확률률)
```




    array([[0.02563061, 0.97436939]])




```python
X.shape
```




    (150, 1)




```python
#X_new = np.linspace(0, 3, 1000).reshape(1000,1)
X_new = np.linspace(0, 3, 1000).reshape(-1,1)
X_new.shape
```




    (1000, 1)




```python
y_prob = lr_clf.predict_proba(X_new)
```


```python
X_new[0], X_new[-1] # 0 ~ 3 cm 
```




    (array([0.]), array([3.]))




```python
y_prob[0] # X_new가 0cm 일 때 Virginica가 아닐 확률, Virginca일 확률률
```




    array([9.99250016e-01, 7.49984089e-04])




```python
decision_boundary = X_new[y_prob[:, 1] > 0.5][0]
```


```python
plt.figure(figsize=(8, 3))
plt.plot([decision_boundary, decision_boundary], [0, 1], 'k:')
plt.plot(X_new, y_prob[:, 0], label='Not Iris-Virginica') # Virginica가 아닐 확률
plt.plot(X_new, y_prob[:, 1], label='Virginica') # Virginica일 확률
plt.legend()
plt.show()
```


    
![png](/assets/images/2023-03-03-Machine Learning 4 (로직스틱 회귀)/output_16_0.png)
    



```python
lr_clf.predict([[1.7]])
```




    array([1])




```python
lr_clf.predict([[1.5]])
```




    array([0])



# 소프트맥스 회귀


```python
iris.feature_names
```




    ['sepal length (cm)',
     'sepal width (cm)',
     'petal length (cm)',
     'petal width (cm)']




```python
X = iris.data[:, 2:] # 특성 2개만 사용('petal length (cm)','petal width (cm))
y = iris.target # 3 품종(클래스)를 구분하는 문제 (다중 분류)
```


```python
sm_clf = LogisticRegression(multi_class ='multinomial', random_state=42)
sm_clf.fit(X, y) # 훈련 : 비용함수(크로스엔트로피)를 최소화하는 모델파라미터(theta 0, theta1, theta 2)*3 형태의 행렬을 찾기
```




    LogisticRegression(multi_class='multinomial', random_state=42)




```python
sm_clf.intercept_
```




    array([ 11.12767979,   3.22717485, -14.35485463])




```python
sm_clf.coef_
```




    array([[-2.74866104, -1.16890756],
           [ 0.08356447, -0.90803047],
           [ 2.66509657,  2.07693804]])




```python
# 예측 : 야생에서 채집해온 데이터의 꽃잎의 길이와 너비 특성을 예측으로 사용 (5cm, 2cm)
sm_clf.predict([[5, 2]]) # 2번 클래스(virginica로 예측측)
```




    array([2])




```python
iris.target_names
```




    array(['setosa', 'versicolor', 'virginica'], dtype='<U10')




```python
sm_clf.predict_proba([[5, 2]])
```




    array([[2.43559894e-04, 2.14859516e-01, 7.84896924e-01]])

## Reference
- [핸즈온 머신러닝 (오렐리앙 제롱 저)](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=237677114)
