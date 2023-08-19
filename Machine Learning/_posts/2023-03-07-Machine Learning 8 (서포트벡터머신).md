---
tag: [machine learning, scikit-learn]
toc: true
toc_sticky: true
toc_label: 목차
---

# SVM 모델


```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC, SVC
```


```python
iris = load_iris()
X = iris['data'][:, 2:] # petal length, petal width
y = (iris['target'] == 2).astype(np.float64) # Virginica 이진 분류
```

```
pipeline1 = Pipeline([
                ('변환기이름1', 변환기1()),
                ('변환기이름2', 변환기2())
            ])
pipeline.fit_transfrom()

pipeline2 = Pipeline([
                ('변환기이름1', 변환기1()),
                ('변환기이름2', 변환기2()),
                ('모델이름', 모델객체())
            ])
pipeline2.fit()
```


```python
svm_clf = Pipeline([
            ('scaler', StandardScaler()),
            ('linear_svc', LinearSVC(C=1, loss='hinge', random_state=42))    
])
svm_clf.fit(X, y)
```




    Pipeline(steps=[('scaler', StandardScaler()),
                    ('linear_svc', LinearSVC(C=1, loss='hinge', random_state=42))])




```python
svm_clf.predict([[5.5, 1.7]]) # Virginca로 예측
```




    array([1.])



# 비선형 분류


```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'b.')
plt.plot(X[:, 0][y==1], X[:, 1][y==1], 'r.')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
```


    
![png](/assets/images/2023-03-07-Machine Learning 8 (서포트벡터머신)/output_7_0.png)
    



```python
polynomial_svm_clf = Pipeline([
            ('poly_features', PolynomialFeatures(degree=3)),
            ('scaler', StandardScaler()),
            ('linear_svc', LinearSVC(C=10, loss='hinge', max_iter=2000, random_state=42))    
])
polynomial_svm_clf.fit(X, y)
```




    Pipeline(steps=[('poly_features', PolynomialFeatures(degree=3)),
                    ('scaler', StandardScaler()),
                    ('linear_svc',
                     LinearSVC(C=10, loss='hinge', max_iter=2000,
                               random_state=42))])




```python
polynomial_svm_clf.predict([[5.5, 1.7]]) # Virginca로 예측
```




    array([1], dtype=int64)



## 다항식 커널


```python
poly_kernel_svm_clf = Pipeline([
            ('scaler', StandardScaler()),
            ('svm_clf', SVC(kernel='poly', degree=3, coef0=1, C=5))    
])
poly_kernel_svm_clf.fit(X, y)
```




    Pipeline(steps=[('scaler', StandardScaler()),
                    ('svm_clf', SVC(C=5, coef0=1, kernel='poly'))])



## 유사도 특성


```python
rbf_kernel_svm_clf = Pipeline([
            ('scaler', StandardScaler()),
            ('svm_clf', SVC(kernel='rbf', gamma=5, C=0.001))  # gamma, C 모두 값이 작아질수록 규제
])
rbf_kernel_svm_clf.fit(X, y)
```




    Pipeline(steps=[('scaler', StandardScaler()),
                    ('svm_clf', SVC(C=0.001, gamma=5))])


## Reference
- [핸즈온 머신러닝 (오렐리앙 제롱 저)](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=237677114)