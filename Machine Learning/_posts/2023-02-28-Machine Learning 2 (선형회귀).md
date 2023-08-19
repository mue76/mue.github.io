---
tag: [python, machine learning, scikit-learn]
---

# 선형회귀


```python
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
np.random.seed(42)
```

## 1. 정규 방정식을 사용한 선형회귀

- y = 3X + 4


```python
X = 2 * np.random.rand(100, 1)
y = 3 * X + 4 + np.random.randn(100, 1)
```


```python
plt.plot(X, y, '.')
plt.show()
```


    
![png](/assets/images/2023-02-28-Machine Learning 2 (선형회귀)/output_5_0.png)
    



```python
x0 = np.ones((100, 1))
```


```python
X.shape == x0.shape # (100, 1)
```




    True




```python
X_b = np.c_[x0, X] # (100, 2)
```


```python
X_b.shape
```




    (100, 2)




```python
Image('./images/img1.png')
```




    
![png](/assets/images/2023-02-28-Machine Learning 2 (선형회귀)/output_10_0.png)
    



- 정규방정식 공식을 np.linalg 함수를 이용해서 해를 구함


```python
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
```


```python
theta_best
```




    array([[4.21509616],
           [2.77011339]])




```python
# array([[4.21509616], # theta 0 : 실제 절편 4
#        [2.77011339]]) # theta 1 : 실제 기울기 3
# y = 3X + 4
```

- scikit-learn 제공 LinearRegression 사용


```python
from sklearn.linear_model import LinearRegression
```


```python
lin_reg = LinearRegression() # 모델 생성
```


```python
lin_reg.fit(X, y) # 모델 학습
```




    LinearRegression()




```python
lin_reg.intercept_, lin_reg.coef_
```




    (array([4.21509616]), array([[2.77011339]]))



## 2. 경사 하강법을 사용한 선형회귀

- 경사 하강법


```python
Image('./images/img2.png', width=300)
```




    
![png](/assets/images/2023-02-28-Machine Learning 2 (선형회귀)/output_22_0.png)
    



- 그레디언트


```python
Image('./images/img3.png', width=400)
```




    
![png](output_24_0.png)
    



- 배치 경사 하강법


```python
m = 100
theta = np.random.randn(2, 1)
eta = 0.1 # 학습률
n_iterations = 1000

for iteration in range(n_iterations):
  gradient = 2/m * X_b.T.dot(X_b.dot(theta) - y)
  theta = theta - eta*gradient
```


```python
theta # y = 3X + 4
```




    array([[4.21509616],
           [2.77011339]])



- 확률적 경사하강법


```python
np.random.seed(42)
```


```python
m = 100
theta = np.random.randn(2, 1)
n_epochs = 50

t0, t1 = 5, 50

def learning_schedule(t):
  return t0/(t+t1)

for epoch in range(50): #epoch : 모든 데이터가 학습에 참여했을 때 1 epoch
  for i in range(m):
    random_index = np.random.randint(m)
    xi = X_b[random_index:random_index+1]
    yi = y[random_index:random_index+1]
    gradient = 2 * xi.T.dot(xi.dot(theta) - yi)
    eta = learning_schedule(epoch*m + i)
    theta = theta - eta*gradient
```


```python
theta # y = 3X + 4
```




    array([[4.21076011],
           [2.74856079]])



- scikit-learn 제공하는 SGDRegressor()


```python
y.shape
```




    (100, 1)




```python
y.flatten().shape
```




    (100,)




```python
y.ravel().shape
```




    (100,)




```python
from sklearn.linear_model import SGDRegressor
```


```python
sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1, random_state=42) # 모델생성
sgd_reg.fit(X, y.ravel())
```




    SGDRegressor(eta0=0.1, max_iter=50, penalty=None, random_state=42)




```python
import sklearn
sklearn.__version__
```




    '0.24.1'




```python
sgd_reg.intercept_, sgd_reg.coef_ # y = 3X + 4
```




    (array([4.24365286]), array([2.8250878]))



**선형회귀 모델의 잠재적인 문제점**
- 데이터 자체가 선형적으로 표현이 안될 수 있음 -> 비선형 데이터를 표현할 수 있게 특성 추가(다항 회귀)
- 시간을 두고 쌓인 데이터, 지리적으로 가까운 데이터인 경우 잘 동작을 안함 -> 다른 모델 사용(ARIMA, ARMA 등)
- 데이터가 정규분포를 따르지 않으면 선형회귀 모델에서 성능이 떨어질 수 있음 -> 로그변환
- 바깥값이 있어도 잘 동작을 안함 -> 바깥값은 삭제
- 다중공선성 문제 -> 상관관계가 큰 특성을 삭제 또는 제 3의 특성을 추출

## 3. 다항 회귀


```python
# 비선형 데이터 준비
# 0.5*X**2 + X + 2 형태
# 모델이 훈련을 마친 후에 모델 파라미터 (0.5, 1, 2)에 근사하는지 확인
```


```python
np.random.seed(42)

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

plt.plot(X, y, '.')
plt.show()
```


    
![png](/assets/images/2023-02-28-Machine Learning 2 (선형회귀)/output_43_0.png)
    



```python
X.shape, y.shape
```




    ((100, 1), (100, 1))



- 원본 특성으로만 학습


```python
lin_reg = LinearRegression()
lin_reg.fit(X, y)
```




    LinearRegression()




```python
lin_reg.intercept_, lin_reg.coef_
```




    (array([3.56401543]), array([[0.84362064]]))



- 원본 특성 포함, 원본 특성에 제곱항을 추가한 데이터로 학습


```python
from sklearn.preprocessing import PolynomialFeatures
```


```python
# 변환기 = 변환기 객체 생성()
# 변환기.fit() # 변환할 준비
# 변환기.transform() # 실제 변환
```


```python
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
```


```python
X_poly.shape
```




    (100, 2)




```python
X_poly[0]
```




    array([-0.75275929,  0.56664654])




```python
(-0.75275929)**2
```




    0.566646548681304




```python
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
```




    LinearRegression()




```python
lin_reg.intercept_, lin_reg.coef_ # 0.5X**2 + X + 2
```




    (array([1.78134581]), array([[0.93366893, 0.56456263]]))



- 예측선 그리기


```python
X_new = np.linspace(-3, 3, 100).reshape(100, 1)

X_new_poly = poly_features.transform(X_new)
pred = lin_reg.predict(X_new_poly)

plt.plot(X_new, pred, 'r-')
plt.plot(X, y, 'b.')
plt.show()
```


    
![png](/assets/images/2023-02-28-Machine Learning 2 (선형회귀)/output_58_0.png)
    


## 4. 규제 모델

### 4.1 릿지 회귀 - L2 규제


```python
np.random.seed(42)

m = 20
X = 3 * np.random.rand(m, 1)
y = 0.5 * X + 1 + np.random.randn(m, 1)/1.5

plt.plot(X, y, '.')
plt.show()
```


    
![png](/assets/images/2023-02-28-Machine Learning 2 (선형회귀)/output_61_0.png)
    



```python
# 기본 선형 회귀 모델
lin_reg = LinearRegression()
lin_reg.fit(X, y)
```




    LinearRegression()




```python
lin_reg.intercept_, lin_reg.coef_ # y = 0.5X + 1
```




    (array([0.97573667]), array([[0.3852145]]))




```python
# 릿지회귀 모델(L2 규제) - 해석적으로 해를 구함

from sklearn.linear_model import Ridge

ridge_reg = Ridge()
ridge_reg.fit(X, y)
```




    Ridge()




```python
ridge_reg.intercept_, ridge_reg.coef_ # y = 0.5X + 1 , theta 0는 규제의 범위에 포함되지 않음음
```




    (array([1.00650911]), array([[0.36280369]]))




```python
# 릿지회귀 모델(L2 규제) - 경사 하강법으로 해를 구함

# 규제 없이
sgd_reg = SGDRegressor(penalty=None, random_state=42)
sgd_reg.fit(X, y.ravel())
print("규제 없음: ", sgd_reg.intercept_, sgd_reg.coef_)

# 규제 추가
sgd_reg_l2 = SGDRegressor(penalty='l2', alpha=0.1, random_state=42)
sgd_reg_l2.fit(X, y.ravel())
print("규제 추가: ", sgd_reg_l2.intercept_, sgd_reg_l2.coef_)
```

    규제 없음:  [0.53945658] [0.62046175]
    규제 추가:  [0.57901244] [0.58606577]
    

### 4.2 라쏘 회귀 - L1 규제


```python
from sklearn.linear_model import Lasso
```


```python
# 라쏘 회귀 모델(L1 규제) - 해석적으로 해를 구함
lasso_reg = Lasso(alpha=0.1, random_state=42)
lasso_reg.fit(X, y)
lasso_reg.intercept_, lasso_reg.coef_ # LinearRegression : (array([0.97573667]), array([[0.3852145]]))
```




    (array([1.14537356]), array([0.26167212]))




```python
# 라쏘 회귀 모델(L1 규제) - 경사하강법으로 해를 구함
# 규제 없이
sgd_reg = SGDRegressor(penalty=None, random_state=42)
sgd_reg.fit(X, y.ravel())
print("규제 없음: ", sgd_reg.intercept_, sgd_reg.coef_)

# 규제 추가
sgd_reg_l1= SGDRegressor(penalty='l1', alpha=0.1, random_state=42)
sgd_reg_l1.fit(X, y.ravel())
print("규제 추가: ", sgd_reg_l1.intercept_, sgd_reg_l1.coef_)
```

    규제 없음:  [0.53945658] [0.62046175]
    규제 추가:  [0.64450934] [0.54050476]
    

### 4.3 엘라스틱넷


```python
from sklearn.linear_model import ElasticNet
```


```python
# 엘라스틱넷 (l1 규제, l2 규제) - 해석적으로 해를 구함
elastic_reg = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_reg.fit(X, y)
elastic_reg.intercept_, elastic_reg.coef_
```




    (array([1.08639303]), array([0.30462619]))




```python
# 엘라스틱넷 모델(L1 규제, l2 규제제) - 경사하강법으로 해를 구함
# 규제 없이
sgd_reg = SGDRegressor(penalty='elasticnet', random_state=42)
sgd_reg.fit(X, y.ravel())
print("규제 없음: ", sgd_reg.intercept_, sgd_reg.coef_)

# 규제 추가
sgd_reg_l1l2= SGDRegressor(penalty='elasticnet', alpha=0.1, random_state=42)
sgd_reg_l1l2.fit(X, y.ravel())
print("규제 추가: ", sgd_reg_l1l2.intercept_, sgd_reg_l1l2.coef_)
```

    규제 없음:  [0.5394774] [0.62043094]
    규제 추가:  [0.59684835] [0.57598861]

## Reference
- [핸즈온 머신러닝 (오렐리앙 제롱 저)](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=237677114)    
    
