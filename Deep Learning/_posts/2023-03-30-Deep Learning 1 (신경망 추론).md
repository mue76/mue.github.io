---
tag: [Deep Learning, 딥러닝, from scratch, 밑바닥부터시작하는딥러닝]
toc: true
toc_sticky: true
toc_label: 목차
author_profile: false

---

# 신경망 


```python
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
```

## 퍼셉트론으로 논리 회로 표현

**AND**


```python
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])   
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1
```


```python
AND(0, 1)
```




    0




```python
AND(1, 0)
```




    0




```python
AND(1, 0)
```




    0




```python
AND(1, 1)
```




    1



**NAND**


```python
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7    
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1
```


```python
NAND(1, 0)
```




    1




```python
NAND(0, 1)
```




    1




```python
NAND(0, 0)
```




    1




```python
NAND(1, 1)
```




    0



**OR**


```python
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([1, 1])
    b = -0.5    
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1
```


```python
OR(1, 0)
```




    1




```python
OR(0, 1)
```




    1




```python
OR(1, 1)
```




    1




```python
OR(0, 0)
```




    0



**XOR**


```python
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
```


```python
XOR(0, 0)
```




    0




```python
XOR(1, 0)
```




    1




```python
XOR(0, 1)
```




    1




```python
XOR(1, 1)
```




    0



## 활성화 함수

**계단함수**


```python
def step_function(x):
    return np.array(x>0, dtype=np.int32)

X = np.arange(-5.0, 5.0 , 0.1)
Y = step_function(X)

plt.plot(X, Y)
plt.ylim(-0.1, 1.1)
plt.show()
```


    
![png](/assets/images/2023-03-30-Deep Learning 1 (신경망 추론)/output_29_0.png)
    


**시그모이드 함수**


```python
def sigmoid(x):
    return 1 / (1+np.exp(-x))

X = np.arange(-5.0, 5.0, 0.1)
Y = sigmoid(X)

plt.plot(X, Y)
plt.ylim(-0.1, 1.1)
plt.show()
```


    
![png](/assets/images/2023-03-30-Deep Learning 1 (신경망 추론)/output_31_0.png)
    


**ReLU**


```python
def relu(x):
    return np.maximum(0, x)

X = np.arange(-5.0, 5.0, 0.1)
Y = relu(X)

plt.plot(X, Y)
plt.ylim(-1.0, 5.5)
plt.show()
```


    
![png](/assets/images/2023-03-30-Deep Learning 1 (신경망 추론)/output_33_0.png)
    


## 다차원 배열 계산


```python
from IPython.display import Image
Image('./deep_learning_images/fig 3-11.png', width=400)
```




    
![png](/assets/images/2023-03-30-Deep Learning 1 (신경망 추론)/output_35_0.png)
    




```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
np.dot(A, B)
```




    array([[19, 22],
           [43, 50]])




```python
Image('./deep_learning_images/fig 3-12.png', width=400)
```




    
![png](/assets/images/2023-03-30-Deep Learning 1 (신경망 추론)/output_37_0.png)
    




```python
A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[7, 8, 9, 10], [11, 12, 13, 14]])
C = np.dot(A, B)
print(A.shape, B.shape, C.shape)
```

    (3, 2) (2, 4) (3, 4)
    


```python
Image('./deep_learning_images/fig 3-13.png', width=350)
```




    
![png](/assets/images/2023-03-30-Deep Learning 1 (신경망 추론)/output_39_0.png)
    




```python
A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([1, 2])
C = np.dot(A, B)
print(A.shape, B.shape, C.shape)
```

    (3, 2) (2,) (3,)
    

## 신경망에서의 행렬곱


```python
Image('./deep_learning_images/fig 3-14.png', width=420)
```




    
![png](/assets/images/2023-03-30-Deep Learning 1 (신경망 추론)/output_42_0.png)
    




```python
X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])
Y = np.dot(X, W)
print(X.shape, W.shape, Y.shape)
```

    (2,) (2, 3) (3,)
    


```python
Image('./deep_learning_images/fig 3-17.png', width=400)
```




    
![png](/assets/images/2023-03-30-Deep Learning 1 (신경망 추론)/output_44_0.png)
    




```python
X = np.array([1, 2])
W1 = np.array([[3], [4]])
B1 = np.array([5])
a1 = np.dot(X, W1) + B1
```


```python
print(X.shape, W1.shape, B1.shape, a1.shape)
```

    (2,) (2, 1) (1,) (1,)
    


```python
Image('./deep_learning_images/e 3.9.png', width=200)
```




    
![png](/assets/images/2023-03-30-Deep Learning 1 (신경망 추론)/output_47_0.png)
    




```python
X = np.array([1.0, 0.5]) # (2, )
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) # (2, 3)
B1 = np.array([0.1, 0.2, 0.3]) # (3,)

A1 = np.dot(X, W1) + B1
print(X.shape, W1.shape, B1.shape, A1.shape)
```

    (2,) (2, 3) (3,) (3,)
    


```python
Image('./deep_learning_images/fig 3-18.png', width=400)
```




    
![png](/assets/images/2023-03-30-Deep Learning 1 (신경망 추론)/output_49_0.png)
    




```python
Z1 = sigmoid(A1) # (3, )
```


```python
Image('./deep_learning_images/fig 3-19.png', width=400)
```




    
![png](/assets/images/2023-03-30-Deep Learning 1 (신경망 추론)/output_51_0.png)
    




```python
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]) # (3, 2)
B2 = np.array([0.1, 0.2]) # (2,)
A2 = np.dot(Z1, W2) + B2  # (2,)
Z2 = sigmoid(A2) # (2, )
print(W2.shape, B2.shape, A2.shape, Z2.shape)
```

    (3, 2) (2,) (2,) (2,)
    


```python
Image('./deep_learning_images/fig 3-20.png', width=400)
```




    
![png](/assets/images/2023-03-30-Deep Learning 1 (신경망 추론)/output_53_0.png)
    




```python
def identity_function(x):
    return x
```


```python
W3 = np.array([[0.1, 0.3], [0.2, 0.4]]) # (2, 2)
B3 = np.array([0.1, 0.2]) # (2,)
A3 = np.dot(Z2, W3) + B3 # (2,)
Y = identity_function(A3) # (2,)
print(W3.shape, B3.shape, A3.shape, Y.shape)
```

    (2, 2) (2,) (2,) (2,)
    

## 출력층(Softmax 함수)


```python
# overflow 문제가 있음
# def softmax(a):
#     exp_a = np.exp(a)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a/sum_exp_a
#     return y

# a = np.array([1010, 1000, 990])
# softmax(a)
```


```python
# overflow 문제 해결
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y

softmax(a)
```




    array([9.99954600e-01, 4.53978686e-05, 2.06106005e-09])



## 3층 신경망 구현


```python
X = np.array([1.0, 0.5]) # (2, )
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) # (2, 3)
B1 = np.array([0.1, 0.2, 0.3]) # (3,)

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1) # (3, )

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]) # (3, 2)
B2 = np.array([0.1, 0.2]) # (2,)

A2 = np.dot(Z1, W2) + B2  # (2,)
Z2 = sigmoid(A2) # (2, )

W3 = np.array([[0.1, 0.3], [0.2, 0.4]]) # (2, 2)
B3 = np.array([0.1, 0.2]) # (2,)
A3 = np.dot(Z2, W3) + B3 # (2,)
Y = identity_function(A3) # (2,)
Y
```




    array([0.31682708, 0.69627909])




```python
# 초기 모델 파라미터 설정
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) # (2, 3)
    network['B1'] = np.array([0.1, 0.2, 0.3]) # (3,)

    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]) # (3, 2)
    network['B2'] = np.array([0.1, 0.2]) # (2,)

    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]]) # (2, 2)
    network['B3'] = np.array([0.1, 0.2]) # (2,)
    
    return network
```


```python
# 전방향(forward) 연산
def forward(network, X): # predict
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    B1, B2, B3 = network['B1'], network['B2'], network['B3']

    A1 = np.dot(X, W1) + B1
    Z1 = sigmoid(A1) # (3, )    

    A2 = np.dot(Z1, W2) + B2  # (2,)
    Z2 = sigmoid(A2) # (2, )
    
    A3 = np.dot(Z2, W3) + B3 # (2,)
    Y = identity_function(A3) # (2,)
    
    return Y
```


```python
network = init_network()
X = np.array([1.0, 0.5]) # (2, )
Y = forward(network, X)
Y
```




    array([0.31682708, 0.69627909])



## 손글씨 숫자(MNIST)


```python
from dataset.mnist import load_mnist
```


```python
def get_data():
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return X_test, y_test
```


```python
X_test, y_test = get_data()
print(X_test.shape, y_test.shape)
```

    (10000, 784) (10000,)
    


```python
print(y_test[0])
plt.imshow(X_test[0].reshape(28, 28), cmap="binary")
```

    7
    




    <matplotlib.image.AxesImage at 0x1d4c86ccee0>




    
![png](/assets/images/2023-03-30-Deep Learning 1 (신경망 추론)/output_68_2.png)
    



```python
import pickle

def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network
```


```python
network = init_network()
network.keys()
```




    dict_keys(['b2', 'W1', 'b1', 'W2', 'W3', 'b3'])




```python
network['W1'].shape, network['W2'].shape, network['W3'].shape
```




    ((784, 50), (50, 100), (100, 10))




```python
network['b1'].shape, network['b2'].shape, network['b3'].shape
```




    ((50,), (100,), (10,))



**image 한장에 대한 신경망 예측 과정**


```python
Image('./deep_learning_images/fig 3-26.png', width=430)
```




    
![png](/assets/images/2023-03-30-Deep Learning 1 (신경망 추론)/output_74_0.png)
    




```python
# 전방향(forward) 연산
def predict(network, X): # predict
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    B1, B2, B3 = network['b1'], network['b2'], network['b3']

    A1 = np.dot(X, W1) + B1
    Z1 = sigmoid(A1) # (3, )    

    A2 = np.dot(Z1, W2) + B2  # (2,)
    Z2 = sigmoid(A2) # (2, )
    
    A3 = np.d) # (2,)
    ot(Z2, W3) + B3 # (2,)
    Y = softmax(A3
    return Y
```


```python
y = predict(network, X_test[0]) # 첫번째 이미지에 대한 예측
np.round(y, 3) # y는 소프트맥스 함수의 결과로서 확률로 표현
```




    array([0.   , 0.   , 0.001, 0.001, 0.   , 0.   , 0.   , 0.997, 0.   ,
           0.001], dtype=float32)




```python
accuracy_cnt = 0
for i in range(len(X_test)): # 10000 iterations
    prob = predict(network, X_test[i])
    pred = np.argmax(prob) # (0~9)까지의 값이 예측
    if pred == y_test[i]:
        accuracy_cnt += 1
        
print("Accuracy : ", accuracy_cnt / len(y_test))
```

    Accuracy :  0.9352
    

**image 배치에 대한 신경망 예측 과정**


```python
Image('./deep_learning_images/fig 3-27.png', width=430)
```




    
![png](/assets/images/2023-03-30-Deep Learning 1 (신경망 추론)/output_79_0.png)
    




```python
batch_size = 100
accuracy_cnt = 0
for i in range(0, len(X_test), batch_size): # 100장씩 100번 iterations
    x_batch = X_test[i:i+batch_size] # 100장씩 슬라이싱 (0~99번이미지, 100~199번 이미지.....9900~9999 이미지)
    prob = predict(network, x_batch) # 100장의 이미지에 대한 결과 probability(10개)
    pred = np.argmax(prob, axis=1) # pred : 100장의 이미지에 대한 최종 예측값
    accuracy_cnt += np.sum(pred == y_test[i:i+batch_size])
    
print("Accuracy:", accuracy_cnt/len(y_test))
```

    Accuracy: 0.9352
    
