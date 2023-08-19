### 랜덤포레스트 모델 만들기


```python
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
```

#### 1. 데이터 준비하기


```python
X, y = make_moons(n_samples = 10000, noise=0.4, random_state=42)

# X[:, 0] # X1 특성
# X[:, 1] # X2 특성

plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'b.') # 0번 클래스 blue
plt.plot(X[:, 0][y==1], X[:, 1][y==1], 'r.') # 1번 클래스 red
plt.xlabel("X1")
plt.ylabel("X2", rotation=0)
```




    Text(0, 0.5, 'X2')




    
![png](/assets/images/2023-03-06-Machine Learning 7 (랜덤포레스트 만들기)/output_3_1.png)
    



```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### 2. 결정 트리 모델 학습하기(기본 파라미터)
- 81.45% 정도의 정확도


```python
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred = tree_clf.predict(X_test)
accuracy_score(y_test, y_pred)
```




    0.8145



#### 3. 결정트리 모델에 최적의 하이퍼 파라미터를 탐색
- 'max_leaf_nodes': 17
- 86.95% 의 정확도 (기본 모델보다는 성능이 5.5% 증가)


```python
len(list(range(2, 100)))
```




    98




```python
from sklearn.model_selection import GridSearchCV
# 후보가 될 파라미터의 리스트
param_grid = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]} # 98*3


grid_search = GridSearchCV(tree_clf, param_grid, scoring="accuracy", cv=3 ) # 98*3*3
grid_search.fit(X_train, y_train) # 8000개의 훈련 데이터
```




    GridSearchCV(cv=3, estimator=DecisionTreeClassifier(random_state=42),
                 param_grid={'max_leaf_nodes': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                13, 14, 15, 16, 17, 18, 19, 20, 21,
                                                22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                31, ...],
                             'min_samples_split': [2, 3, 4]},
                 scoring='accuracy')




```python
grid_search.best_params_
```




    {'max_leaf_nodes': 17, 'min_samples_split': 2}




```python
grid_search.best_estimator_
```




    DecisionTreeClassifier(max_leaf_nodes=17, random_state=42)




```python
cvres = grid_search.cv_results_
```


```python
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(mean_score, params)
```

    0.7711257585675009 {'max_leaf_nodes': 2, 'min_samples_split': 2}
    0.7711257585675009 {'max_leaf_nodes': 2, 'min_samples_split': 3}
    0.7711257585675009 {'max_leaf_nodes': 2, 'min_samples_split': 4}
    0.8160007755969739 {'max_leaf_nodes': 3, 'min_samples_split': 2}
    0.8160007755969739 {'max_leaf_nodes': 3, 'min_samples_split': 3}
    0.8160007755969739 {'max_leaf_nodes': 3, 'min_samples_split': 4}
    0.8525004329447565 {'max_leaf_nodes': 4, 'min_samples_split': 2}
    0.8525004329447565 {'max_leaf_nodes': 4, 'min_samples_split': 3}
    0.8525004329447565 {'max_leaf_nodes': 4, 'min_samples_split': 4}
    0.8525004329447565 {'max_leaf_nodes': 5, 'min_samples_split': 2}
    0.8525004329447565 {'max_leaf_nodes': 5, 'min_samples_split': 3}
    0.8525004329447565 {'max_leaf_nodes': 5, 'min_samples_split': 4}
    0.8525004329447565 {'max_leaf_nodes': 6, 'min_samples_split': 2}
    0.8525004329447565 {'max_leaf_nodes': 6, 'min_samples_split': 3}
    0.8525004329447565 {'max_leaf_nodes': 6, 'min_samples_split': 4}
    0.8525004329447565 {'max_leaf_nodes': 7, 'min_samples_split': 2}
    0.8525004329447565 {'max_leaf_nodes': 7, 'min_samples_split': 3}
    0.8525004329447565 {'max_leaf_nodes': 7, 'min_samples_split': 4}
    0.8525004329447565 {'max_leaf_nodes': 8, 'min_samples_split': 2}
    0.8525004329447565 {'max_leaf_nodes': 8, 'min_samples_split': 3}
    0.8525004329447565 {'max_leaf_nodes': 8, 'min_samples_split': 4}
    0.8525004329447565 {'max_leaf_nodes': 9, 'min_samples_split': 2}
    0.8525004329447565 {'max_leaf_nodes': 9, 'min_samples_split': 3}
    0.8525004329447565 {'max_leaf_nodes': 9, 'min_samples_split': 4}
    0.8515005579291337 {'max_leaf_nodes': 10, 'min_samples_split': 2}
    0.8515005579291337 {'max_leaf_nodes': 10, 'min_samples_split': 3}
    0.8515005579291337 {'max_leaf_nodes': 10, 'min_samples_split': 4}
    0.8492499952884734 {'max_leaf_nodes': 11, 'min_samples_split': 2}
    0.8492499952884734 {'max_leaf_nodes': 11, 'min_samples_split': 3}
    0.8492499952884734 {'max_leaf_nodes': 11, 'min_samples_split': 4}
    0.8482498858685424 {'max_leaf_nodes': 12, 'min_samples_split': 2}
    0.8482498858685424 {'max_leaf_nodes': 12, 'min_samples_split': 3}
    0.8482498858685424 {'max_leaf_nodes': 12, 'min_samples_split': 4}
    0.847749948360731 {'max_leaf_nodes': 13, 'min_samples_split': 2}
    0.847749948360731 {'max_leaf_nodes': 13, 'min_samples_split': 3}
    0.847749948360731 {'max_leaf_nodes': 13, 'min_samples_split': 4}
    0.8482500733919887 {'max_leaf_nodes': 14, 'min_samples_split': 2}
    0.8482500733919887 {'max_leaf_nodes': 14, 'min_samples_split': 3}
    0.8482500733919887 {'max_leaf_nodes': 14, 'min_samples_split': 4}
    0.8509995890423675 {'max_leaf_nodes': 15, 'min_samples_split': 2}
    0.8509995890423675 {'max_leaf_nodes': 15, 'min_samples_split': 3}
    0.8509995890423675 {'max_leaf_nodes': 15, 'min_samples_split': 4}
    0.851499526550179 {'max_leaf_nodes': 16, 'min_samples_split': 2}
    0.851499526550179 {'max_leaf_nodes': 16, 'min_samples_split': 3}
    0.851499526550179 {'max_leaf_nodes': 16, 'min_samples_split': 4}
    0.8555001986342105 {'max_leaf_nodes': 17, 'min_samples_split': 2}
    0.8555001986342105 {'max_leaf_nodes': 17, 'min_samples_split': 3}
    0.8555001986342105 {'max_leaf_nodes': 17, 'min_samples_split': 4}
    0.855375167376396 {'max_leaf_nodes': 18, 'min_samples_split': 2}
    0.855375167376396 {'max_leaf_nodes': 18, 'min_samples_split': 3}
    0.855375167376396 {'max_leaf_nodes': 18, 'min_samples_split': 4}
    0.8550001673646759 {'max_leaf_nodes': 19, 'min_samples_split': 2}
    0.8550001673646759 {'max_leaf_nodes': 19, 'min_samples_split': 3}
    0.8550001673646759 {'max_leaf_nodes': 19, 'min_samples_split': 4}
    0.8550001673646759 {'max_leaf_nodes': 20, 'min_samples_split': 2}
    0.8550001673646759 {'max_leaf_nodes': 20, 'min_samples_split': 3}
    0.8550001673646759 {'max_leaf_nodes': 20, 'min_samples_split': 4}
    0.8545002298568644 {'max_leaf_nodes': 21, 'min_samples_split': 2}
    0.8545002298568644 {'max_leaf_nodes': 21, 'min_samples_split': 3}
    0.8545002298568644 {'max_leaf_nodes': 21, 'min_samples_split': 4}
    0.8546252142338172 {'max_leaf_nodes': 22, 'min_samples_split': 2}
    0.8546252142338172 {'max_leaf_nodes': 22, 'min_samples_split': 3}
    0.8546252142338172 {'max_leaf_nodes': 22, 'min_samples_split': 4}
    0.8541255580111752 {'max_leaf_nodes': 23, 'min_samples_split': 2}
    0.8541255580111752 {'max_leaf_nodes': 23, 'min_samples_split': 3}
    0.8541255580111752 {'max_leaf_nodes': 23, 'min_samples_split': 4}
    0.8535004486029644 {'max_leaf_nodes': 24, 'min_samples_split': 2}
    0.8535004486029644 {'max_leaf_nodes': 24, 'min_samples_split': 3}
    0.8535004486029644 {'max_leaf_nodes': 24, 'min_samples_split': 4}
    0.8538753079720999 {'max_leaf_nodes': 25, 'min_samples_split': 2}
    0.8538753079720999 {'max_leaf_nodes': 25, 'min_samples_split': 3}
    0.8538753079720999 {'max_leaf_nodes': 25, 'min_samples_split': 4}
    0.8533753704642884 {'max_leaf_nodes': 26, 'min_samples_split': 2}
    0.8533753704642884 {'max_leaf_nodes': 26, 'min_samples_split': 3}
    0.8533753704642884 {'max_leaf_nodes': 26, 'min_samples_split': 4}
    0.8518753704174076 {'max_leaf_nodes': 27, 'min_samples_split': 2}
    0.8518753704174076 {'max_leaf_nodes': 27, 'min_samples_split': 3}
    0.8518753704174076 {'max_leaf_nodes': 27, 'min_samples_split': 4}
    0.8518753704174076 {'max_leaf_nodes': 28, 'min_samples_split': 2}
    0.8518753704174076 {'max_leaf_nodes': 28, 'min_samples_split': 3}
    0.8518753704174076 {'max_leaf_nodes': 28, 'min_samples_split': 4}
    0.8510001516127064 {'max_leaf_nodes': 29, 'min_samples_split': 2}
    0.8510001516127064 {'max_leaf_nodes': 29, 'min_samples_split': 3}
    0.8510001516127064 {'max_leaf_nodes': 29, 'min_samples_split': 4}
    0.8512501203666121 {'max_leaf_nodes': 30, 'min_samples_split': 2}
    0.8512501203666121 {'max_leaf_nodes': 30, 'min_samples_split': 3}
    0.8512501203666121 {'max_leaf_nodes': 30, 'min_samples_split': 4}
    0.8506251047201245 {'max_leaf_nodes': 31, 'min_samples_split': 2}
    0.8506251047201245 {'max_leaf_nodes': 31, 'min_samples_split': 3}
    0.8506251047201245 {'max_leaf_nodes': 31, 'min_samples_split': 4}
    0.8522504641908508 {'max_leaf_nodes': 32, 'min_samples_split': 2}
    0.8522504641908508 {'max_leaf_nodes': 32, 'min_samples_split': 3}
    0.8522504641908508 {'max_leaf_nodes': 32, 'min_samples_split': 4}
    0.851875323536546 {'max_leaf_nodes': 33, 'min_samples_split': 2}
    0.851875323536546 {'max_leaf_nodes': 33, 'min_samples_split': 3}
    0.851875323536546 {'max_leaf_nodes': 33, 'min_samples_split': 4}
    0.853000276690845 {'max_leaf_nodes': 34, 'min_samples_split': 2}
    0.853000276690845 {'max_leaf_nodes': 34, 'min_samples_split': 3}
    0.853000276690845 {'max_leaf_nodes': 34, 'min_samples_split': 4}
    0.852000401675222 {'max_leaf_nodes': 35, 'min_samples_split': 2}
    0.852000401675222 {'max_leaf_nodes': 35, 'min_samples_split': 3}
    0.852000401675222 {'max_leaf_nodes': 35, 'min_samples_split': 4}
    0.8530004642142913 {'max_leaf_nodes': 36, 'min_samples_split': 2}
    0.8530004642142913 {'max_leaf_nodes': 36, 'min_samples_split': 3}
    0.8530004642142913 {'max_leaf_nodes': 36, 'min_samples_split': 4}
    0.8535004017221027 {'max_leaf_nodes': 37, 'min_samples_split': 2}
    0.8535004017221027 {'max_leaf_nodes': 37, 'min_samples_split': 3}
    0.8535004017221027 {'max_leaf_nodes': 37, 'min_samples_split': 4}
    0.8546252611146787 {'max_leaf_nodes': 38, 'min_samples_split': 2}
    0.8546252611146787 {'max_leaf_nodes': 38, 'min_samples_split': 3}
    0.8546252611146787 {'max_leaf_nodes': 38, 'min_samples_split': 4}
    0.8532501985638893 {'max_leaf_nodes': 39, 'min_samples_split': 2}
    0.8532501985638893 {'max_leaf_nodes': 39, 'min_samples_split': 3}
    0.8532501985638893 {'max_leaf_nodes': 39, 'min_samples_split': 4}
    0.8532501985638893 {'max_leaf_nodes': 40, 'min_samples_split': 2}
    0.8532501985638893 {'max_leaf_nodes': 40, 'min_samples_split': 3}
    0.8532501985638893 {'max_leaf_nodes': 40, 'min_samples_split': 4}
    0.8533751829408421 {'max_leaf_nodes': 41, 'min_samples_split': 2}
    0.8533751829408421 {'max_leaf_nodes': 41, 'min_samples_split': 3}
    0.8533751829408421 {'max_leaf_nodes': 41, 'min_samples_split': 4}
    0.853500261079518 {'max_leaf_nodes': 42, 'min_samples_split': 2}
    0.853500261079518 {'max_leaf_nodes': 42, 'min_samples_split': 3}
    0.853500261079518 {'max_leaf_nodes': 42, 'min_samples_split': 4}
    0.853250339206474 {'max_leaf_nodes': 43, 'min_samples_split': 2}
    0.853250339206474 {'max_leaf_nodes': 43, 'min_samples_split': 3}
    0.853250339206474 {'max_leaf_nodes': 43, 'min_samples_split': 4}
    0.8540003392299144 {'max_leaf_nodes': 44, 'min_samples_split': 2}
    0.8540003392299144 {'max_leaf_nodes': 44, 'min_samples_split': 3}
    0.8540003392299144 {'max_leaf_nodes': 44, 'min_samples_split': 4}
    0.8540003392299144 {'max_leaf_nodes': 45, 'min_samples_split': 2}
    0.8540003392299144 {'max_leaf_nodes': 45, 'min_samples_split': 3}
    0.8540003392299144 {'max_leaf_nodes': 45, 'min_samples_split': 4}
    0.8535004017221027 {'max_leaf_nodes': 46, 'min_samples_split': 2}
    0.8535004017221027 {'max_leaf_nodes': 46, 'min_samples_split': 3}
    0.8535004017221027 {'max_leaf_nodes': 46, 'min_samples_split': 4}
    0.8533754173451499 {'max_leaf_nodes': 47, 'min_samples_split': 2}
    0.8533754173451499 {'max_leaf_nodes': 47, 'min_samples_split': 3}
    0.8533754173451499 {'max_leaf_nodes': 47, 'min_samples_split': 4}
    0.8533754173451499 {'max_leaf_nodes': 48, 'min_samples_split': 2}
    0.8533754173451499 {'max_leaf_nodes': 48, 'min_samples_split': 3}
    0.8533754173451499 {'max_leaf_nodes': 48, 'min_samples_split': 4}
    0.8531254485912442 {'max_leaf_nodes': 49, 'min_samples_split': 2}
    0.8531254485912442 {'max_leaf_nodes': 49, 'min_samples_split': 3}
    0.8531254485912442 {'max_leaf_nodes': 49, 'min_samples_split': 4}
    0.8528754798373385 {'max_leaf_nodes': 50, 'min_samples_split': 2}
    0.8528754798373385 {'max_leaf_nodes': 50, 'min_samples_split': 3}
    0.8528754798373385 {'max_leaf_nodes': 50, 'min_samples_split': 4}
    0.8528754798373385 {'max_leaf_nodes': 51, 'min_samples_split': 2}
    0.8528754798373385 {'max_leaf_nodes': 51, 'min_samples_split': 3}
    0.8528754798373385 {'max_leaf_nodes': 51, 'min_samples_split': 4}
    0.8528754798373385 {'max_leaf_nodes': 52, 'min_samples_split': 2}
    0.8528754798373385 {'max_leaf_nodes': 52, 'min_samples_split': 3}
    0.8528754798373385 {'max_leaf_nodes': 52, 'min_samples_split': 4}
    0.8520005891986683 {'max_leaf_nodes': 53, 'min_samples_split': 2}
    0.8520005891986683 {'max_leaf_nodes': 53, 'min_samples_split': 3}
    0.8520005891986683 {'max_leaf_nodes': 53, 'min_samples_split': 4}
    0.8517506204447626 {'max_leaf_nodes': 54, 'min_samples_split': 2}
    0.8517506204447626 {'max_leaf_nodes': 54, 'min_samples_split': 3}
    0.8517506204447626 {'max_leaf_nodes': 54, 'min_samples_split': 4}
    0.8515006048099951 {'max_leaf_nodes': 55, 'min_samples_split': 2}
    0.8515006048099951 {'max_leaf_nodes': 55, 'min_samples_split': 3}
    0.8515006048099951 {'max_leaf_nodes': 55, 'min_samples_split': 4}
    0.851875651702577 {'max_leaf_nodes': 56, 'min_samples_split': 2}
    0.851875651702577 {'max_leaf_nodes': 56, 'min_samples_split': 3}
    0.851875651702577 {'max_leaf_nodes': 56, 'min_samples_split': 4}
    0.8520006360795298 {'max_leaf_nodes': 57, 'min_samples_split': 2}
    0.8520006360795298 {'max_leaf_nodes': 57, 'min_samples_split': 3}
    0.8520006360795298 {'max_leaf_nodes': 57, 'min_samples_split': 4}
    0.8518756048217154 {'max_leaf_nodes': 58, 'min_samples_split': 2}
    0.8518756048217154 {'max_leaf_nodes': 58, 'min_samples_split': 3}
    0.8518756048217154 {'max_leaf_nodes': 58, 'min_samples_split': 4}
    0.8512505891752277 {'max_leaf_nodes': 59, 'min_samples_split': 2}
    0.8512505891752277 {'max_leaf_nodes': 59, 'min_samples_split': 3}
    0.8512505891752277 {'max_leaf_nodes': 59, 'min_samples_split': 4}
    0.8511256047982748 {'max_leaf_nodes': 60, 'min_samples_split': 2}
    0.8511256047982748 {'max_leaf_nodes': 60, 'min_samples_split': 3}
    0.8511256047982748 {'max_leaf_nodes': 60, 'min_samples_split': 4}
    0.8503756516556961 {'max_leaf_nodes': 61, 'min_samples_split': 2}
    0.8503756516556961 {'max_leaf_nodes': 61, 'min_samples_split': 3}
    0.8503756516556961 {'max_leaf_nodes': 61, 'min_samples_split': 4}
    0.8506256204096019 {'max_leaf_nodes': 62, 'min_samples_split': 2}
    0.8506256204096019 {'max_leaf_nodes': 62, 'min_samples_split': 3}
    0.8506256204096019 {'max_leaf_nodes': 62, 'min_samples_split': 4}
    0.8497507297709316 {'max_leaf_nodes': 63, 'min_samples_split': 2}
    0.8497507297709316 {'max_leaf_nodes': 63, 'min_samples_split': 3}
    0.8497507297709316 {'max_leaf_nodes': 63, 'min_samples_split': 4}
    0.8496256985131172 {'max_leaf_nodes': 64, 'min_samples_split': 2}
    0.8496256985131172 {'max_leaf_nodes': 64, 'min_samples_split': 3}
    0.8496256985131172 {'max_leaf_nodes': 64, 'min_samples_split': 4}
    0.8496256985131172 {'max_leaf_nodes': 65, 'min_samples_split': 2}
    0.8496256985131172 {'max_leaf_nodes': 65, 'min_samples_split': 3}
    0.8496256985131172 {'max_leaf_nodes': 65, 'min_samples_split': 4}
    0.8496256985131172 {'max_leaf_nodes': 66, 'min_samples_split': 2}
    0.8496256985131172 {'max_leaf_nodes': 66, 'min_samples_split': 3}
    0.8496256985131172 {'max_leaf_nodes': 66, 'min_samples_split': 4}
    0.8495006672553028 {'max_leaf_nodes': 67, 'min_samples_split': 2}
    0.8495006672553028 {'max_leaf_nodes': 67, 'min_samples_split': 3}
    0.8495006672553028 {'max_leaf_nodes': 67, 'min_samples_split': 4}
    0.84937568287835 {'max_leaf_nodes': 68, 'min_samples_split': 2}
    0.84937568287835 {'max_leaf_nodes': 68, 'min_samples_split': 3}
    0.84937568287835 {'max_leaf_nodes': 68, 'min_samples_split': 4}
    0.8482505890814661 {'max_leaf_nodes': 69, 'min_samples_split': 2}
    0.8482505890814661 {'max_leaf_nodes': 69, 'min_samples_split': 3}
    0.8482505890814661 {'max_leaf_nodes': 69, 'min_samples_split': 4}
    0.8488756984896767 {'max_leaf_nodes': 70, 'min_samples_split': 2}
    0.8488756984896767 {'max_leaf_nodes': 70, 'min_samples_split': 3}
    0.8488756984896767 {'max_leaf_nodes': 70, 'min_samples_split': 4}
    0.8486256359740478 {'max_leaf_nodes': 71, 'min_samples_split': 2}
    0.8486256359740478 {'max_leaf_nodes': 71, 'min_samples_split': 3}
    0.8486256359740478 {'max_leaf_nodes': 71, 'min_samples_split': 4}
    0.8490006359857681 {'max_leaf_nodes': 72, 'min_samples_split': 2}
    0.8490006359857681 {'max_leaf_nodes': 72, 'min_samples_split': 3}
    0.8490006359857681 {'max_leaf_nodes': 72, 'min_samples_split': 4}
    0.8488756516088153 {'max_leaf_nodes': 73, 'min_samples_split': 2}
    0.8488756516088153 {'max_leaf_nodes': 73, 'min_samples_split': 3}
    0.8488756516088153 {'max_leaf_nodes': 73, 'min_samples_split': 4}
    0.848500651597095 {'max_leaf_nodes': 74, 'min_samples_split': 2}
    0.848500651597095 {'max_leaf_nodes': 74, 'min_samples_split': 3}
    0.848500651597095 {'max_leaf_nodes': 74, 'min_samples_split': 4}
    0.848500651597095 {'max_leaf_nodes': 75, 'min_samples_split': 2}
    0.848500651597095 {'max_leaf_nodes': 75, 'min_samples_split': 3}
    0.848500651597095 {'max_leaf_nodes': 75, 'min_samples_split': 4}
    0.8487506203510007 {'max_leaf_nodes': 76, 'min_samples_split': 2}
    0.8487506203510007 {'max_leaf_nodes': 76, 'min_samples_split': 3}
    0.8487506203510007 {'max_leaf_nodes': 76, 'min_samples_split': 4}
    0.8488755578470921 {'max_leaf_nodes': 77, 'min_samples_split': 2}
    0.8488755578470921 {'max_leaf_nodes': 77, 'min_samples_split': 3}
    0.8488755578470921 {'max_leaf_nodes': 77, 'min_samples_split': 4}
    0.8486254953314631 {'max_leaf_nodes': 78, 'min_samples_split': 2}
    0.8486254953314631 {'max_leaf_nodes': 78, 'min_samples_split': 3}
    0.8486254953314631 {'max_leaf_nodes': 78, 'min_samples_split': 4}
    0.8491254797201363 {'max_leaf_nodes': 79, 'min_samples_split': 2}
    0.8491254797201363 {'max_leaf_nodes': 79, 'min_samples_split': 3}
    0.8491254797201363 {'max_leaf_nodes': 79, 'min_samples_split': 4}
    0.8490004953431832 {'max_leaf_nodes': 80, 'min_samples_split': 2}
    0.8490004953431832 {'max_leaf_nodes': 80, 'min_samples_split': 3}
    0.8490004953431832 {'max_leaf_nodes': 80, 'min_samples_split': 4}
    0.8486255422123247 {'max_leaf_nodes': 81, 'min_samples_split': 2}
    0.8486255422123247 {'max_leaf_nodes': 81, 'min_samples_split': 3}
    0.8485005578353718 {'max_leaf_nodes': 81, 'min_samples_split': 4}
    0.8485005578353718 {'max_leaf_nodes': 82, 'min_samples_split': 2}
    0.8485005578353718 {'max_leaf_nodes': 82, 'min_samples_split': 3}
    0.8485005578353718 {'max_leaf_nodes': 82, 'min_samples_split': 4}
    0.848375573458419 {'max_leaf_nodes': 83, 'min_samples_split': 2}
    0.848375573458419 {'max_leaf_nodes': 83, 'min_samples_split': 3}
    0.848375573458419 {'max_leaf_nodes': 83, 'min_samples_split': 4}
    0.8482504953197431 {'max_leaf_nodes': 84, 'min_samples_split': 2}
    0.8482504953197431 {'max_leaf_nodes': 84, 'min_samples_split': 3}
    0.8480005265658371 {'max_leaf_nodes': 84, 'min_samples_split': 4}
    0.8475005890580256 {'max_leaf_nodes': 85, 'min_samples_split': 2}
    0.8475005890580256 {'max_leaf_nodes': 85, 'min_samples_split': 3}
    0.8472506203041199 {'max_leaf_nodes': 85, 'min_samples_split': 4}
    0.8477505578119314 {'max_leaf_nodes': 86, 'min_samples_split': 2}
    0.8477505578119314 {'max_leaf_nodes': 86, 'min_samples_split': 3}
    0.8482504953197431 {'max_leaf_nodes': 86, 'min_samples_split': 4}
    0.8482504953197431 {'max_leaf_nodes': 87, 'min_samples_split': 2}
    0.8482504953197431 {'max_leaf_nodes': 87, 'min_samples_split': 3}
    0.8482504953197431 {'max_leaf_nodes': 87, 'min_samples_split': 4}
    0.8481255578236516 {'max_leaf_nodes': 88, 'min_samples_split': 2}
    0.8481255578236516 {'max_leaf_nodes': 88, 'min_samples_split': 3}
    0.8481255578236516 {'max_leaf_nodes': 88, 'min_samples_split': 4}
    0.848750479708416 {'max_leaf_nodes': 89, 'min_samples_split': 2}
    0.848750479708416 {'max_leaf_nodes': 89, 'min_samples_split': 3}
    0.848750479708416 {'max_leaf_nodes': 89, 'min_samples_split': 4}
    0.8486254484506016 {'max_leaf_nodes': 90, 'min_samples_split': 2}
    0.8486254484506016 {'max_leaf_nodes': 90, 'min_samples_split': 3}
    0.8482504953197431 {'max_leaf_nodes': 90, 'min_samples_split': 4}
    0.8481255109427902 {'max_leaf_nodes': 91, 'min_samples_split': 2}
    0.8481255109427902 {'max_leaf_nodes': 91, 'min_samples_split': 3}
    0.8477505578119314 {'max_leaf_nodes': 91, 'min_samples_split': 4}
    0.8475005890580256 {'max_leaf_nodes': 92, 'min_samples_split': 2}
    0.8475005890580256 {'max_leaf_nodes': 92, 'min_samples_split': 3}
    0.8473756046810728 {'max_leaf_nodes': 92, 'min_samples_split': 4}
    0.8471255421654439 {'max_leaf_nodes': 93, 'min_samples_split': 2}
    0.8471255421654439 {'max_leaf_nodes': 93, 'min_samples_split': 3}
    0.8471255421654439 {'max_leaf_nodes': 93, 'min_samples_split': 4}
    0.847000557788491 {'max_leaf_nodes': 94, 'min_samples_split': 2}
    0.847000557788491 {'max_leaf_nodes': 94, 'min_samples_split': 3}
    0.8468755734115382 {'max_leaf_nodes': 94, 'min_samples_split': 4}
    0.847000557788491 {'max_leaf_nodes': 95, 'min_samples_split': 2}
    0.847000557788491 {'max_leaf_nodes': 95, 'min_samples_split': 3}
    0.847000557788491 {'max_leaf_nodes': 95, 'min_samples_split': 4}
    0.8467505890345853 {'max_leaf_nodes': 96, 'min_samples_split': 2}
    0.8467505890345853 {'max_leaf_nodes': 96, 'min_samples_split': 3}
    0.8467505890345853 {'max_leaf_nodes': 96, 'min_samples_split': 4}
    0.8465006202806794 {'max_leaf_nodes': 97, 'min_samples_split': 2}
    0.8465006202806794 {'max_leaf_nodes': 97, 'min_samples_split': 3}
    0.8466256046576324 {'max_leaf_nodes': 97, 'min_samples_split': 4}
    0.8465006202806794 {'max_leaf_nodes': 98, 'min_samples_split': 2}
    0.8465006202806794 {'max_leaf_nodes': 98, 'min_samples_split': 3}
    0.8463756359037266 {'max_leaf_nodes': 98, 'min_samples_split': 4}
    0.8461255733880977 {'max_leaf_nodes': 99, 'min_samples_split': 2}
    0.8461255733880977 {'max_leaf_nodes': 99, 'min_samples_split': 3}
    0.8460005890111448 {'max_leaf_nodes': 99, 'min_samples_split': 4}
    


```python
best_tree_clf = grid_search.best_estimator_
y_pred = best_tree_clf.predict(X_test)
accuracy_score(y_test, y_pred)
```




    0.8695



#### 4. 랜덤포레스트 모델 만들기(결정트리의 앙상블 학습)

- 훈련 데이터 세트를 1000세트 생성, 1000개의 트리모델 준비
- 각 모델(best_tree_clf)은 무작위 샘플링된 100개의 데이터로 학습


```python
from sklearn.model_selection import ShuffleSplit
n_trees = 1000
n_samples = 100
mini_sets = []

split = ShuffleSplit(n_splits = n_trees, test_size = len(X_train) - n_samples, random_state=42)
for mini_train_index, mini_test_index in split.split(X_train): # 8000개의 데이터에서 100개씩 샘플링
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))
```


```python
len(mini_sets)
```




    1000




```python
mini_sets[0][0].shape, mini_sets[0][1].shape # 1000개의 데이터셋중 첫번째 데이터의 특성과 정답
```




    ((100, 2), (100,))




```python
X_train.shape, X_test.shape
```




    ((8000, 2), (2000, 2))




```python
# 1000개의 트리 모델이 각각 100개의 작은 데이터로 학습을 한뒤
# 테스트 데이터(X_test)로 예측을 했을 때
# 개별 모델들의 평균값이 80.54 %
# 개별 모델은 10000개로 학습했을때 보다는 안좋은 결과
```


```python
import numpy as np
```


```python
from sklearn.base import clone

accuracy_scores = []
forest = [clone(best_tree_clf) for _ in range(n_trees)]

for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets): # 1000개의 모델과 1000개의 데이터셋
    tree.fit(X_mini_train, y_mini_train) # 100개의 샘플링 데이터로 훈련
    y_pred = tree.predict(X_test) # 1000개의 예측
    accuracy_scores.append(accuracy_score(y_test, y_pred)) # 1000개의 정확도

np.mean(accuracy_scores)
```




    0.8054499999999999




```python
Y_pred = np.empty((n_trees, len(X_test)), dtype=np.uint8)
Y_pred.shape
```




    (1000, 2000)




```python
for tree_index, tree in enumerate(forest): # 1000개의 모델이 1000번의 예측
    y_pred = tree.predict(X_test) # 테스트 데이터 2000개의 예측
    Y_pred[tree_index] = y_pred
```


```python
Y_pred
```




    array([[0, 1, 0, ..., 0, 0, 1],
           [1, 1, 1, ..., 0, 0, 0],
           [1, 1, 0, ..., 0, 0, 1],
           ...,
           [1, 1, 0, ..., 0, 0, 0],
           [1, 1, 0, ..., 0, 0, 1],
           [1, 1, 0, ..., 0, 0, 0]], dtype=uint8)




```python
# scipy mode() 함수 : 최다 빈도수 값을 반환
```


```python
from scipy.stats import mode
y_pred_major_votes, n_votes = mode(Y_pred, axis=0)
```


```python
y_pred_major_votes # 최종 예측 결과로 사용하자
```




    array([[1, 1, 0, ..., 0, 0, 0]], dtype=uint8)




```python
y_test.shape, y_pred_major_votes.shape
```




    ((2000,), (1, 2000))




```python
# 개별 학습기 1000개의 정확도 평균은 80.54%
# 최다빈도수 투표를 통해 예측한 결과로 성능 측정했을 때 87.2%
# 기반모델(결정트리)로 예측했을 때의 결과 86.95% 보다도 살짝 더 나은 성능
```


```python
accuracy_score(y_test, y_pred_major_votes.reshape(2000,))
```




    0.872


