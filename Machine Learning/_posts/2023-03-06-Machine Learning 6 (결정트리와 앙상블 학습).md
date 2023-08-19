# 결정트리


```python
from sklearn.tree import export_graphviz
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
```


```python
import graphviz
```

## 1. 분류


```python
iris = load_iris()
X = iris.data[:, 2:] # 꽃잎의 길이, 너비('petal length (cm)', 'petal width (cm)')
y = iris.target 
```


```python
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X, y)
```




    DecisionTreeClassifier(max_depth=2, random_state=42)




```python
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html
from sklearn import tree
res = tree.plot_tree(tree_clf,
               feature_names = iris.feature_names[2:],
               class_names = iris.target_names,
               rounded = True,
               filled = True)
```


    
![png](/assets/images/2023-03-06-Machine Learning 6 (결정트리와 앙상블 학습)/output_6_0.png)
    



```python
export_graphviz(tree_clf,
               out_file = 'iris_tree.dot',
               feature_names = iris.feature_names[2:],
               class_names = iris.target_names,
               rounded = True,
               filled = True)
```


```python
with open('iris_tree.dot') as f:
    dot_graph = f.read()
    
graphviz.Source(dot_graph)    
```




    
![svg](/assets/images/2023-03-06-Machine Learning 6 (결정트리와 앙상블 학습)/output_8_0.svg)
    




```python
tree_clf.feature_importances_ # petal length(56%), petal width(43%)
```




    array([0.56199095, 0.43800905])



## 2. 회귀(수치예측)


```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
np.random.seed(42)

m = 200
X = np.random.rand(m, 1)
# y = 4(X-0.5)**2
y = 4 * (X-0.5)**2 + np.random.randn(m, 1)/10
```


```python
plt.plot(X, y, 'b.')
plt.show()
```


    
![png](/assets/images/2023-03-06-Machine Learning 6 (결정트리와 앙상블 학습)/output_13_0.png)
    



```python
from sklearn.tree import DecisionTreeRegressor
```


```python
tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(X,y)
```




    DecisionTreeRegressor(max_depth=2, random_state=42)




```python
plt.figure(figsize=(14, 8))
res = tree.plot_tree(tree_reg,
               feature_names = ['x1'],
               class_names = ['y'],
               rounded = True,
               filled = True,
               fontsize=14)
```


    
![png](/assets/images/2023-03-06-Machine Learning 6 (결정트리와 앙상블 학습)/output_16_0.png)
    



```python
tree_reg.predict([[0.6]])
```




    array([0.11063973])



# 앙상블 학습

### 1. 투표기반 분류기


```python
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import VotingClassifier

# 개별별 분류기
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
```


```python
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
```


```python
X.shape, y.shape
```




    ((500, 2), (500,))




```python
plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'b.')
plt.plot(X[:, 0][y==1], X[:, 1][y==1], 'r.')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
```


    
![png](/assets/images/2023-03-06-Machine Learning 6 (결정트리와 앙상블 학습)/output_23_0.png)
    



```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

    (375, 2) (125, 2) (375,) (125,)
    

#### 1.1 하드 보팅


```python
lr_clf = LogisticRegression(random_state=42)
rnd_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC(random_state=42)


voting_clf = VotingClassifier(
                    estimators=[('lr', lr_clf), ('rf', rnd_clf), ('svm', svm_clf)],
                    voting = 'hard'
              )
```


```python
from sklearn.metrics import accuracy_score
for clf in (lr_clf, rnd_clf, svm_clf, voting_clf):
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
```

    LogisticRegression 0.864
    RandomForestClassifier 0.896
    SVC 0.896
    VotingClassifier 0.912
    

#### 1.2 소프트 보팅


```python
lr_clf = LogisticRegression(random_state=42)
rnd_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC(probability=True, random_state=42)


voting_clf = VotingClassifier(
                    estimators=[('lr', lr_clf), ('rf', rnd_clf), ('svm', svm_clf)],
                    voting = 'soft'
              )
```


```python
from sklearn.metrics import accuracy_score
for clf in (lr_clf, rnd_clf, svm_clf, voting_clf):
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
```

    LogisticRegression 0.864
    RandomForestClassifier 0.896
    SVC 0.896
    VotingClassifier 0.92
    

## 2. 배깅 앙상블


```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
```


```python
bag_clf = BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=500,
                            max_samples=100, bootstrap=True, random_state=42, n_jobs=-1)

bag_clf.fit(X_train, y_train)
```




    BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=42),
                      max_samples=100, n_estimators=500, n_jobs=-1,
                      random_state=42)




```python
y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred) # 정확도도
```




    0.904




```python
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

y_pred = tree_clf.predict(X_test)
accuracy_score(y_test, y_pred)
```




    0.856



**oob 평가**


```python
bag_clf = BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=500,
                            max_samples=100, bootstrap=True, oob_score=True, random_state=42, n_jobs=-1)

bag_clf.fit(X_train, y_train)
```




    BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=42),
                      max_samples=100, n_estimators=500, n_jobs=-1, oob_score=True,
                      random_state=42)




```python
bag_clf.oob_score_ # 교차검증 대용용
```




    0.9253333333333333




```python
y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred) # 정확도
```




    0.904




```python
bag_clf = BaggingClassifier(DecisionTreeClassifier(max_leaf_nodes=16, random_state=42), n_estimators=500,
                            bootstrap=True, oob_score=True, random_state=42, n_jobs=-1)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred)
```




    0.912




```python
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42, n_jobs=-1)
rnd_clf.fit(X_train, y_train)
y_pred = rnd_clf.predict(X_test)
accuracy_score(y_test, y_pred)
```




    0.912




```python
rnd_clf.feature_importances_ # x1 : 42%, x2 : 57% 중요도
```




    array([0.42253629, 0.57746371])


