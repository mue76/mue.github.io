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
# 신경망에서 사용한 W(행렬 형태)의 편미분 행렬을 구하는 함수

def numerical_gradient(f, x): # x는 행렬 형태
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
# Softmax
def softmax(x):
    if x.ndim == 2:
        x = x.T # 10*100
        x = x - np.max(x, axis=0) # 10*100 - 100 = 10*100
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))
```


```python
# Sigmoid
def sigmoid(x):
    return 1 / (1+np.exp(-x))
```

## 2층 신경망 구현하기


```python
!dir images
```

     C 드라이브의 볼륨에는 이름이 없습니다.
     볼륨 일련 번호: 308F-3770
    
     C:\Users\Playdata\Documents\2022 인공지능 28기\Deep Learning\images 디렉터리
    
    2023-04-04  오전 09:10    <DIR>          .
    2023-04-04  오전 09:10    <DIR>          ..
    2023-04-04  오전 09:08           407,028 backprop.jpg
    2023-03-29  오후 09:05           790,450 fashion-mnist-sprite.png
    2023-04-01  오전 08:38            66,202 image1.png
    2023-04-01  오전 11:15            48,501 image2.png
    2023-04-01  오전 11:10            20,796 image3.png
    2023-04-01  오전 11:08            27,425 image4.png
    2023-04-01  오후 12:18            75,489 image5.png
    2023-04-01  오후 01:01             7,976 image6.png
    2023-03-29  오후 09:05            54,373 mlp_mnist.png
    2023-03-29  오후 09:05            86,550 tensor_examples.svg
                  10개 파일           1,584,790 바이트
                   2개 디렉터리  69,680,427,008 바이트 남음
    


```python
from IPython.display import Image
Image('./images/backprop.jpg')
```




    
![jpeg](/assets/images/2023-04-04-Deep Learning 4 (오차역전파 이용한 신경망 학습)/output_8_0.jpg)
    




```python
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01): # 모델 파라미터 초기화
        # input_size : 784, hidden_size : 20, output_size : 10        
        # W1.shape (784, 20), b1.shape(20, ), W2.shape(20, 10), b2.shape(10,)
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * weight_init_std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * weight_init_std
        self.params['b2'] = np.zeros(output_size)    
        
    def predict(self, x): # 입력에 대한 전방향 연산 결과(확률)
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1 # (batch_size, 784) x (784, 20) + (20, ) = (batch_size, 20, )
        z1 = sigmoid(a1)
        
        a2 = np.dot(z1, W2) + b2 # (batch_size, 20) x (20, 10) + (10, ) = (batch_size, 10)
        y = softmax(a2) # 예측 확률
        return y
        
    def loss(self, x, t): # 예측과 정답값에 대한 손실
        y = self.predict(x) # 예측 확률
        loss = cross_entropy_error(y, t)
        return loss
    
    def numerical_gradient(self, x, t):
        f = lambda w : self.loss(x, t) # 손실 함수
        grads = {}
        grads['W1'] = numerical_gradient(f, self.params['W1'])
        grads['b1'] = numerical_gradient(f, self.params['b1'])
        grads['W2'] = numerical_gradient(f, self.params['W2'])
        grads['b2'] = numerical_gradient(f, self.params['b2'])        
        return grads
    
    def gradient(self, x, t): # 오차역전파로 미분 구하기        
        grads = {}
        
        # 아래 코드는 self.predict(x) 와 동일
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']      
        a1 = np.dot(x, W1) + b1 # (batch_size, 784) x (784, 20) + (20, ) = (batch_size, 20, )
        z1 = sigmoid(a1)        
        a2 = np.dot(z1, W2) + b2 # (batch_size, 20) x (20, 10) + (10, ) = (batch_size, 10)
        y = softmax(a2) # 예측 확률
        # ------------------------------------        
        
        batch_size = x.shape[0]
        da2 = (y-t)/batch_size
        
        grads['W2'] = np.dot(z1.T, da2)
        grads['b2'] = np.sum(da2, axis=0)
        
        dz1 = np.dot(da2, W2.T)
        da1 = z1*(1-z1)*dz1
        
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)
        
        return grads        
            
    def accuracy(self, x, t):
        prob = self.predict(x) # 확률
        pred = np.argmax(prob, axis=1) # 최종예측
        t = np.argmax(t, axis=1)
        accuracy = np.sum(pred == t)/x.shape[0]
        return accuracy        
```


```python
from dataset.mnist import load_mnist

(X_train, y_train), (X_test, y_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=20, output_size=10)

# 히스토리를 위한 리스트
train_loss_list = []
train_acc_list = []
test_acc_list = []

# 파라미터 설정
iters_num = 10000 # 모델 파라미터 업데이트(경사하강법) 횟수
train_size = X_train.shape[0] # 훈련데이터 사이즈
batch_size = 100 # 미니 배치 사이즈
learning_rate = 0.1 # 학습률

# 에폭(epoch) : 훈련데이터를 모두 소진했을 때의 횟수
# 60000개의 데이터를 100개씩 가져다 쓰면 600번 반복 : 1 epoch

iter_per_epoch = train_size/batch_size # 600

for i in range(iters_num):
    # 1 단계 - 미니배치
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = X_train[batch_mask] # 100장의 이미지
    t_batch = y_train[batch_mask] # 100개의 정답

    # 2 단계 - 기울기 산출
    # grads = network.numerical_gradient(x_batch, t_batch)
    grads = network.gradient(x_batch, t_batch)

    # 3 단계 - 매개변수 갱신
    # W(new) <- W(old) - (lr * gradient) : 경사하강법
    for key in ('W1','b1', 'W2', 'b2'):
        network.params[key] = network.params[key] - (learning_rate * grads[key])
        
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if (i % iter_per_epoch) == 0 :# 0, 600, 1200, 1800... (1 epoch 마다)
        train_accuracy = network.accuracy(X_train, y_train) # 60000개
        test_accuracy = network.accuracy(X_test, y_test) # 10000개
        
        train_acc_list.append(train_accuracy)
        test_acc_list.append(test_accuracy)
        
        print("Train Accuracy :" + str(train_accuracy) + "  Test Accuracy:" + str(test_accuracy))    
```

    Train Accuracy :0.09871666666666666  Test Accuracy:0.098
    Train Accuracy :0.72665  Test Accuracy:0.7342
    Train Accuracy :0.8658  Test Accuracy:0.8705
    Train Accuracy :0.8949666666666667  Test Accuracy:0.8981
    Train Accuracy :0.9050333333333334  Test Accuracy:0.9056
    Train Accuracy :0.9122  Test Accuracy:0.9134
    Train Accuracy :0.9165333333333333  Test Accuracy:0.9186
    Train Accuracy :0.9208666666666666  Test Accuracy:0.9226
    Train Accuracy :0.9249833333333334  Test Accuracy:0.926
    Train Accuracy :0.92795  Test Accuracy:0.9284
    Train Accuracy :0.9313666666666667  Test Accuracy:0.9312
    Train Accuracy :0.9332666666666667  Test Accuracy:0.9326
    Train Accuracy :0.9352166666666667  Test Accuracy:0.9353
    Train Accuracy :0.9373333333333334  Test Accuracy:0.9361
    Train Accuracy :0.9387833333333333  Test Accuracy:0.9371
    Train Accuracy :0.93995  Test Accuracy:0.9391
    Train Accuracy :0.9418333333333333  Test Accuracy:0.9389
    


```python
# keras
# model.fit(.....)


# pytorch
# 1. 미니배치 구성
# 2. 전방향 연산 (예측, 로스값) : pred = model(), loss = cross_entropy(pred, true)
# 3. 역방향 연산 (gradient가 구해짐) : loss.backward()
# 4. 경사하강법으로 매개변수 갱신하기 : optimizer.step()
```


```python
# Loss
x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list)
```




    [<matplotlib.lines.Line2D at 0x1d413d03eb0>]




    
![png](/assets/images/2023-04-04-Deep Learning 4 (오차역전파 이용한 신경망 학습)/output_12_1.png)
    



```python
# Accuracy
x1 = np.arange(len(train_acc_list))
y1 = train_acc_list
plt.plot(x1, y1, label="train_accuracy")
x2 = np.arange(len(test_acc_list))
y2 = test_acc_list
plt.plot(x2, y2, label="test_accuracy")

plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend(loc="best")
```




    <matplotlib.legend.Legend at 0x1d406f98df0>




    
![png](/assets/images/2023-04-04-Deep Learning 4 (오차역전파 이용한 신경망 학습)/output_13_1.png)
    

