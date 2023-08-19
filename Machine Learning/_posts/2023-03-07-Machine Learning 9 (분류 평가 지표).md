---
tag: [machine learning, scikit-learn]
toc: true
---

# 분류 평가 지표


```python
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score
# from sklearn.metrics import plot_roc_curve # not supported sklearn 1.2
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
```

### 데이터 가져오기
**MNIST Dataset**


```python
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
```

### 데이터 탐색하기


```python
type(mnist) # bunch type
```




    sklearn.utils.Bunch




```python
mnist.keys()
```




    dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])




```python
X = mnist['data'] # 특성
y = mnist['target'] # 정답
```


```python
X.shape, y.shape
```




    ((70000, 784), (70000,))




```python
X[0].shape # 0번째의 샘플의 크기
```




    (784,)




```python
28 * 28 # width * height
```




    784




```python
# 특성 데이터 탐색
some_digit = X[0]
some_digit_img = some_digit.reshape(28, 28)
plt.imshow(some_digit_img, cmap='binary')
```




    <matplotlib.image.AxesImage at 0x213785889d0>




    
![png](/assets/images/2023-03-07-Machine Learning 9 (분류 평가 지표)/output_11_1.png)
    



```python
# 정답 데이터 탐색
y[0] # 현재 타입이 문자열열
```




    '5'




```python
y = y.astype(np.uint8)
y[0]
```




    5



### 데이터 준비 (훈련데이터, 테스트데이터)


```python
y[:50]
```




    array([5, 0, 4, 1, 9, 2, 1, 3, 1, 4, 3, 5, 3, 6, 1, 7, 2, 8, 6, 9, 4, 0,
           9, 1, 1, 2, 4, 3, 2, 7, 3, 8, 6, 9, 0, 5, 6, 0, 7, 6, 1, 8, 7, 9,
           3, 9, 8, 5, 9, 3], dtype=uint8)




```python
# 70000만개의 데이터가 이미 뒤섞여 있으므로 train_test_split 없이
# 앞에서 60000개의 데이터를 훈련 데이터로 사용하고
# 뒤의 10000개의 데이터를 테스트 데이터로 사용

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```


```python
# 클래스별 분포
for i in range(10):
  print('클래스(레이블)별 데이터 개수 : ', i, (y_train == i).sum())
```

    클래스(레이블)별 데이터 개수 :  0 5923
    클래스(레이블)별 데이터 개수 :  1 6742
    클래스(레이블)별 데이터 개수 :  2 5958
    클래스(레이블)별 데이터 개수 :  3 6131
    클래스(레이블)별 데이터 개수 :  4 5842
    클래스(레이블)별 데이터 개수 :  5 5421
    클래스(레이블)별 데이터 개수 :  6 5918
    클래스(레이블)별 데이터 개수 :  7 6265
    클래스(레이블)별 데이터 개수 :  8 5851
    클래스(레이블)별 데이터 개수 :  9 5949
    

### 이진분류


```python
(y_train == 5).sum() # 5인 데이터는 대략 10%
```




    5421




```python
60000 - (y_train == 5).sum()  # 5가 아닌 데이터는 대략 90%
```




    54579




```python
y_train.shape, y_test.shape
```




    ((60000,), (10000,))




```python
y_train_5 = (y_train == 5).astype(np.uint8) # 정답이 5이면 1, 그렇지 않으면 0
y_test_5 = (y_test == 5).astype(np.uint8) # 정답이 5이면 1, 그렇지 않으면 0
```


```python
(y_train_5 == 1).sum(), (y_train_5 == 0).sum()
```




    (5421, 54579)




```python
sgd_clf = SGDClassifier(random_state=42)

sgd_clf_scores = cross_val_score(sgd_clf, X_train, y_train_5, scoring='accuracy', cv=3)
sgd_clf_scores
```




    array([0.95035, 0.96035, 0.9604 ])



### 오차행렬


```python
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
```


```python
y_train_pred.shape # (60000. )
```




    (60000,)




```python
# confusion_matrix(정답, 예측)
confusion_matrix(y_train_5, y_train_pred)
```




    array([[53892,   687],
           [ 1891,  3530]], dtype=int64)




```python
3530/(3530+687) # 정밀도
```




    0.8370879772350012




```python
3530/(3530+1891) # 재현율
```




    0.6511713705958311




```python
precision_score(y_train_5, y_train_pred) # 정밀도
```




    0.8370879772350012




```python
recall_score(y_train_5, y_train_pred) # 재현율
```




    0.6511713705958311




```python
f1_score(y_train_5, y_train_pred)
```




    0.7325171197343846




```python
y_train_scores = cross_val_predict(sgd_clf, X_train, y_train_5, method='decision_function', cv=3)
```


```python
y_train_scores # 60000개의 데이터에 대한 확신 점수 : 0보다 크면 양성(5), 0보다 작으면 음성(5 아님)
```




    array([  1200.93051237, -26883.79202424, -33072.03475406, ...,
            13272.12718981,  -7258.47203373, -16877.50840447])




```python
(y_train_scores > 0).sum()
```




    4217




```python
(y_train_pred == 1).sum()
```




    4217




```python
# 임계값 8000으로 변경 (확신 점수에 대한 임계값을 0에서 8000으로 올리면 '5'에 대한 예측이 신중, 정밀해짐)
(y_train_scores > 8000).sum()
```




    1667




```python
threshold = 8000
y_train_pred_th8000 = y_train_scores > threshold
```


```python
# 임계값을 8000으로 변경해서 나온 예측결과 오차행렬 구하기
confusion_matrix(y_train_5, y_train_pred_th8000)
```




    array([[54470,   109],
           [ 3863,  1558]], dtype=int64)




```python
# 기본 임계값(0)일 때의 오차행렬렬
# array([[53892,   687],
#        [ 1891,  3530]])
```


```python
precision_score(y_train_5, y_train_pred_th8000)
```




    0.9346130773845231




```python
recall_score(y_train_5, y_train_pred_th8000)
```




    0.2874008485519277




```python
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_train_scores)
```


```python
precisions
```




    array([0.09040123, 0.09038606, 0.09038757, ..., 1.        , 1.        ,
           1.        ])




```python
recalls
```




    array([1.00000000e+00, 9.99815532e-01, 9.99815532e-01, ...,
           3.68935621e-04, 1.84467810e-04, 0.00000000e+00])




```python
thresholds
```




    array([-106527.45300471, -105763.22240074, -105406.2965229 , ...,
             38871.26391927,   42216.05562787,   49441.43765905])




```python
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions[:-1], label='Precision')
plt.plot(thresholds, recalls[:-1], label='Recall')
plt.legend()
plt.show()
```


    
![png](/assets/images/2023-03-07-Machine Learning 9 (분류 평가 지표)/output_48_0.png)
    



```python
# 현재 목표하는 정밀도가 90%를 넘어야 한다면 아래와 같이 90%를 만족하는 threshold를 찾을 수 있음
precisions >= 0.9
```




    array([False, False, False, ...,  True,  True,  True])




```python
np.argmax(precisions >= 0.9)
```




    57075




```python
thresholds[np.argmax(precisions >= 0.9)]
```




    3370.0194991439557



### ROC 곡선


```python
fpr, tpr, thresholds = roc_curve(y_train_5, y_train_scores)
```


```python
# plot_roc_curve(sgd_clf, X_train, y_train_5)
```


```python
plt.plot(fpr, tpr)
plt.xlabel('fpr')
plt.ylabel('tpr(recall)')
plt.show()
```


    
![png](/assets/images/2023-03-07-Machine Learning 9 (분류 평가 지표)/output_55_0.png)
    



```python
roc_auc_score(y_train_5, y_train_scores) # AUC(위 그래프의 면적)가 1에 가까울수록 좋은 분류기
```




    0.9604938554008616



- 이진 분류에서 ROC_AUC를 성능 측정지표로 많이 사용하지만,
- 불균한 데이터셋에서는 재현율/정밀도, f1 score 로 확인하는게 더 바람직

### 다중분류

- svm classifier


```python
svm_clf = SVC(random_state=42)
svm_scores = cross_val_score(svm_clf, X_train[:1000], y_train[:1000], cv=3)
svm_scores
```




    array([0.89520958, 0.9009009 , 0.88288288])



- sgd classifier


```python
sgd_clf = SGDClassifier(random_state=42)
sgd_scores = cross_val_score(sgd_clf, X_train, y_train, cv=3)
sgd_scores
```




    array([0.87365, 0.85835, 0.8689 ])




```python
std_scaler = StandardScaler()
X_train_scaled = std_scaler.fit_transform(X_train)

sgd_scores = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3)
sgd_scores
```




    array([0.8983, 0.891 , 0.9018])




```python
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx
```




    array([[5577,    0,   22,    5,    8,   43,   36,    6,  225,    1],
           [   0, 6400,   37,   24,    4,   44,    4,    7,  212,   10],
           [  27,   27, 5220,   92,   73,   27,   67,   36,  378,   11],
           [  22,   17,  117, 5227,    2,  203,   27,   40,  403,   73],
           [  12,   14,   41,    9, 5182,   12,   34,   27,  347,  164],
           [  27,   15,   30,  168,   53, 4444,   75,   14,  535,   60],
           [  30,   15,   42,    3,   44,   97, 5552,    3,  131,    1],
           [  21,   10,   51,   30,   49,   12,    3, 5684,  195,  210],
           [  17,   63,   48,   86,    3,  126,   25,   10, 5429,   44],
           [  25,   18,   30,   64,  118,   36,    1,  179,  371, 5107]],
          dtype=int64)



### 다중 레이블 분류


```python
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7) # True, False
y_train_odd = (y_train % 2 == 1) # Odd, Even
y_multilabel = np.c_[y_train_large, y_train_odd]
```


```python
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
```




    KNeighborsClassifier()




```python
knn_clf.predict([some_digit]) # '5' image 에 대한 예측 : [False, True]
```




    array([[False,  True]])



### 다중 출력 분류


```python
# for train data
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise

# for test data
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise

# 깨긋한 원본 데이터를 정답 데이터로 만들기기
y_train_mod = X_train
y_test_mod = X_test
```


```python
plt.imshow(X_test_mod[0].reshape(28, 28), cmap='binary')
```




    <matplotlib.image.AxesImage at 0x213180758e0>




    
![png](/assets/images/2023-03-07-Machine Learning 9 (분류 평가 지표)/output_71_1.png)
    



```python
plt.imshow(y_test_mod[0].reshape(28, 28), cmap='binary')
```




    <matplotlib.image.AxesImage at 0x213180c5d30>




    
![png](/assets/images/2023-03-07-Machine Learning 9 (분류 평가 지표)/output_72_1.png)
    



```python
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)
```




    KNeighborsClassifier()




```python
clean_digit = knn_clf.predict([X_test_mod[0]]) # 깨끗한 7 이미지를 예측을 기대
```


```python
plt.imshow(clean_digit.reshape(28, 28), cmap='binary')
```




    <matplotlib.image.AxesImage at 0x213189f3790>




    
![png](/assets/images/2023-03-07-Machine Learning 9 (분류 평가 지표)/output_75_1.png)
    
## Reference
- [핸즈온 머신러닝 (오렐리앙 제롱 저)](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=237677114)

