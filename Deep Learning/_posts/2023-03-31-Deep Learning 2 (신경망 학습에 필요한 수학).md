---
tag: [Deep Learning, 딥러닝, from scratch, 밑바닥부터시작하는딥러닝]
toc: true
toc_sticky: true
toc_label: 목차
author_profile: false

---

# 신경망 학습


```python
import numpy as np
import matplotlib.pyplot as plt
```

## 교차 엔트로피

**샘플 한개에 대한 손실값**


```python
from IPython.display import Image
Image('./deep_learning_images/e 4.2.png', width=180)
```




    
![png](/assets/images/2023-03-31-Deep Learning 2 (신경망 학습에 필요한 수학)/output_4_0.png)
    




```python
def cross_entropy_error(y, t): # y,t는 단일 샘플 (1차원 배열)
    delta = 1e-7 # 0.0000001
    return -np.sum(t*np.log(y+delta))
```


```python
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]) # 정답 : 2 (one-hot encoding)
y = np.array([0.1, 0.05, 0.6, 0.0, 0.5, 0.0, 0.1, 0.0, 0.0, 0.0]) # 예측 확률 : 2에 가깝게 예측

cross_entropy_error(y, t)
```




    0.510825457099338




```python
# t가 현재 one-hot encoding 되어 있으므로 정답이 1인 값만 적용
-np.log(0.6+1e-7)
```




    0.510825457099338



**예측을 잘못했을 수록 cross_entropy의 값은 크다** 
- 손실함수로의 자격을 만족


```python
Image('./deep_learning_images/e 4.3.png', width=260)
```




    
![png](/assets/images/2023-03-31-Deep Learning 2 (신경망 학습에 필요한 수학)/output_9_0.png)
    




```python
def cross_entropy_error(y, t): # y, t는 배치 단위의 샘플들 (2차원 배열)
    delta = 1e-7 # 0.0000001    
    
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+delta))/batch_size
```


```python
cross_entropy_error(y, t)
```




    0.510825457099338



## 미분

**모델파라미터가 하나인 손실함수의 예**
- x를 모델파라미터로 간주


```python
Image('./deep_learning_images/e 4.5.png', width=170)
```




    
![png](/assets/images/2023-03-31-Deep Learning 2 (신경망 학습에 필요한 수학)/output_14_0.png)
    




```python
def function_1(x):
    return 0.01*x**2 + 0.1*x
```


```python
x = np.arange(0, 10.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()
```


    
![png](/assets/images/2023-03-31-Deep Learning 2 (신경망 학습에 필요한 수학)/output_16_0.png)
    


**중앙차분에 의한 수치미분**


```python
def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h))/(2*h)
```


```python
x = np.arange(0, 10.0, 0.1)
y = function_1(x)

a = numerical_diff(function_1, 5) # 약 0.2가 미분값으로 나올지 관찰
b = function_1(5) - (a*5)
y2= a*x + b

plt.xlabel('x')
plt.ylabel('f(x)')

plt.plot(x, y)
plt.plot(x, y2)
plt.show()
```


    
![png](/assets/images/2023-03-31-Deep Learning 2 (신경망 학습에 필요한 수학)/output_19_0.png)
    


## 편미분

**모델파라미터가 두개인 손실함수의 예**
- x0, x1을 모델파라미터로 간주


```python
Image('./deep_learning_images/e 4.6.png', width=200)
```




    
![png](output_22_0.png)
    




```python
def function_2(x): # x는 x0와 x1으로 구성된 벡터
    result = x[0]**2 + x[1]**2
    return result
```


```python
def numerical_graident(f, x): # x는 x0와 x1으로 구성된 벡터
    h = 1e-4 # 0.0001
    grads = np.zeros_like(x)
    
    for idx in range(x.size): # 0 -> 1 
        tmp_val = x[idx] # x[0] -> x[1]            
        
        x[idx] = tmp_val + h        
        fxh1 = f(x) # f(x+h)
        x[idx] = tmp_val - h        
        fxh2 = f(x) # f(x-h)
        
        grads[idx] = (fxh1 - fxh2) / (2*h)        
        
        x[idx] = tmp_val # backup        
    return grads # x0의 미분과 x1의 미분을 담고있는 벡터
```


```python
numerical_graident(function_2, np.array([3.0, 4.0]))
```




    array([6., 8.])



## 경사하강법


```python
Image('./deep_learning_images/e 4.7.png', width=150)
```




    
![png](output_27_0.png)
    



- init_x : 초기 랜덤하게 설정한 weight 벡터 (w1, w2....)
- function_2 : 손실 함수 (loss function)    
- gradient_descent : 손실함수에 대한 weight(w1, w2...)의 미분(기울기)를 단서로 해서 손실의 최소값을 찾아가는 과정


```python
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x # 임의로 설정한 값에서 시작
    
    for i in range(step_num):
        grad = numerical_graident(f, x) # grad : x0와 x1에 대해 편미분한 벡터
        x = x - (lr * grad) # 벡터화된 연산
    
    # 경사하강법을 100번 반복한 후의 x값(x0, x1)
    return x
```


```python
init_x = np.array([3.0, 4.0])
x = gradient_descent(function_2, init_x, lr=0.1, step_num=100)
x # x0, x1이 0에 가까운 값으로 수렴
```




    array([6.11110793e-10, 8.14814391e-10])



## 신경망에서의 기울기

- (참고) np.nditer 사용법


```python
import numpy as np
x=np.array([[1,2,3],[4,5,6]])
it=np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])
while not it.finished:
    print(it.multi_index) #(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)
    print(x[it.multi_index]) # 1,2,3,4,5,6
    it.iternext()
```

    (0, 0)
    1
    (0, 1)
    2
    (0, 2)
    3
    (1, 0)
    4
    (1, 1)
    5
    (1, 2)
    6
    


```python
# 신경망에서 사용한 W(행렬 형태)의 편미분 행렬을 구하는 함수

def numerical_graident(f, x): # x는 행렬 형태
    h = 1e-4 # 0.0001
    grads = np.zeros_like(x)
    
    it=np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index

        tmp_val = x[idx] # x[0] -> x[1]            
        
        x[idx] = tmp_val + h        
        fxh1 = f(x) # f(x+h)
        x[idx] = tmp_val - h        
        fxh2 = f(x) # f(x-h)
        
        grads[idx] = (fxh1 - fxh2) / (2*h)        
        
        x[idx] = tmp_val # backup                
        it.iternext()
        
    return grads # 편미분 행렬이 반환
```


```python
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y
```


```python
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # (2, 3)
        
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z) # y : 예측, t: 정답
        loss = cross_entropy_error(y, t)
        return loss    
```


```python
x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

f = lambda w: net.loss(x, t) # f: Loss 함수
```


```python
dW = numerical_graident(f, net.W) # net.W.shape : (2, 3) -> dW.shape : (2, 3)
```


```python
dW # net.W 의 편미분 행렬
```




    array([[ 0.24049883,  0.16957396, -0.41007279],
           [ 0.36074824,  0.25436094, -0.61510918]])


