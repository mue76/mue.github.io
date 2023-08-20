---
tag: [Deep Learning, 딥러닝, pytorch, 파이토치]
toc: true
toc_sticky: true
toc_label: 목차
author_profile: false

---

# 신경망 학습 (MNIST in Pytorch)


```python
import torch # 파이토치 기본 라이브러리

# torchvision : 데이터셋, 모델 아키텍처, 컴퓨터 비전의 이미지 변환 기능 제공
from torchvision import datasets # torchvision에서 제공하는 데이터셋
from torchvision import transforms # 이미지 변환기능을 제공하는 패키지

# torch.utils.data : 파이토치 데이터 로딩 유틸리티
from torch.utils.data import DataLoader # 모델 훈련에 사용할 수 있는 미니 배치 구성하고
                                        # 매 epoch마다 데이터를 샘플링, 병렬처리 등의 일을 해주는 함수

from torch.utils.data import random_split

import numpy as np
import matplotlib.pyplot as plt
```

## 1. 데이터 불러오기


```python
# Compose를 통해 원하는 전처리기를 차례대로 넣을 수 있음음
# mnist_transform = transforms.Compose([transforms.Resize(16), transforms.ToTensor()])
# trainset = datasets.MNIST('./datasets/', download=True, train=True, transform = mnist_transform)
```


```python
# dataset = datasets.MNIST(다운받을 디렉토리, 다운로드여부, 학습용여부, 전처리방법)
trainset = datasets.MNIST('./datasets/', download=True, train=True, transform = transforms.ToTensor())
testset = datasets.MNIST('./datasets/', download=True, train=False, transform = transforms.ToTensor())
```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ./datasets/MNIST/raw/train-images-idx3-ubyte.gz
    

    100%|██████████| 9912422/9912422 [00:00<00:00, 133144530.98it/s]
    

    Extracting ./datasets/MNIST/raw/train-images-idx3-ubyte.gz to ./datasets/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ./datasets/MNIST/raw/train-labels-idx1-ubyte.gz
    

    100%|██████████| 28881/28881 [00:00<00:00, 73953415.03it/s]
    

    Extracting ./datasets/MNIST/raw/train-labels-idx1-ubyte.gz to ./datasets/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ./datasets/MNIST/raw/t10k-images-idx3-ubyte.gz
    

    100%|██████████| 1648877/1648877 [00:00<00:00, 42539436.30it/s]

    Extracting ./datasets/MNIST/raw/t10k-images-idx3-ubyte.gz to ./datasets/MNIST/raw
    

    
    

    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ./datasets/MNIST/raw/t10k-labels-idx1-ubyte.gz
    

    100%|██████████| 4542/4542 [00:00<00:00, 8308124.19it/s]

    Extracting ./datasets/MNIST/raw/t10k-labels-idx1-ubyte.gz to ./datasets/MNIST/raw
    
    

    
    


```python
# trainset을 다시 train용과 valid 용으로 나누고자 할 때
trainset, validset = random_split(trainset, [50000, 10000])
```


```python
print(type(trainset), len(trainset))
print(type(validset), len(validset))
print(type(testset), len(testset))
```

    <class 'torch.utils.data.dataset.Subset'> 50000
    <class 'torch.utils.data.dataset.Subset'> 10000
    <class 'torchvision.datasets.mnist.MNIST'> 10000
    


```python
# 0번째 샘플에 2개의 원소가 있는데, 그중 첫번째 원소는 이미지, 두번째 원소는 정답
# 그러나 파이토치로 읽어들인 이미지 텐서의 형상이 channels * height * width 임
# 그에 비해 opencv, matplotlib으로 읽어들인 이미지 array의 형상은 height * width * channels
print(trainset[0][0].size(), trainset[0][1])
```

    torch.Size([1, 28, 28]) 2
    

## 2. 데이터 시각화


```python
figure, axes = plt.subplots(nrows=1, ncols=8, figsize=(22, 6))

for i in range(8):
  image, label= trainset[i][0], trainset[i][1]
  axes[i].imshow(image.squeeze(), cmap='gray')
  axes[i].set_title('Class : ' + str(label))
```


    
![png](/assets/images/2023-04-07-Deep Learning 8 (MNIST in Pytorch (3) 모델 수정)/output_9_0.png)
    


## 3. 데이터 적재


```python
# DataLoader
# 모델 훈련에 사용할 수 있는 미니 배치 구성하고
# 매 epoch마다 데이터를 샘플링, 병렬처리 등의 일을 해주는 함수
```


```python
batch_size = 100
# dataloader = DataLoader(데이터셋, 배치사이즈, 셔플여부.....)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True) # 훈련용 50000개의 데이터를 100개씩 준비
validloader = DataLoader(validset, batch_size=batch_size, shuffle=False) # 검증용 10000개의 데이터를 100개씩 준비
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False) # 테스트용 10000개의 데이터를 100개씩 준비
```


```python
print(type(trainloader), len(trainloader))
print(type(validloader), len(validloader))
print(type(testloader), len(testloader))
```

    <class 'torch.utils.data.dataloader.DataLoader'> 500
    <class 'torch.utils.data.dataloader.DataLoader'> 100
    <class 'torch.utils.data.dataloader.DataLoader'> 100
    


```python
train_iter = iter(trainloader)
images, labels = next(train_iter)
images.size(), labels.size()
```




    (torch.Size([100, 1, 28, 28]), torch.Size([100]))



## 4. 모델 생성


```python
from IPython.display import Image
Image('./images/mlp_mnist.png', width=500)
```




    
![png](/assets/images/2023-04-07-Deep Learning 8 (MNIST in Pytorch (3) 모델 수정)/output_16_0.png)
    




```python
import torch.nn as nn # 파이토치에서 제공하는 다양한 계층 (Linear Layer, ....)
import torch.optim as optim # 옵티마이저 (경사하강법...)
import torch.nn.functional as F # 파이토치에서 제공하는 함수(활성화 함수...)
```

**option 4** : nn.Module 서브클래싱하기
- 파라미터 관리가 필요없는 기능(활성화 함수, ...) 함수형(functional)으로 작성
- 함수형이란 출력이 입력에 의해 결정


```python
class Mnist_DNN(nn.Module):
  def __init__(self):
    super().__init__()
    # linear layer, fully connected layer, affine layer, dense layer : np.dot(x, w) + b
    self.hidden_linear1 = nn.Linear(in_features=784, out_features=128)    
    self.hidden_linear2 = nn.Linear(in_features=128, out_features=64)    
    self.ouput_linear = nn.Linear(in_features=64, out_features=10)

  def forward(self, x):
    x = self.hidden_linear1(x)    
    x = F.relu(x)
    x = self.hidden_linear2(x)  
    x = F.relu(x)  
    x = self.ouput_linear(x)    
    return x
```


```python
model = Mnist_DNN()
model
```




    Mnist_DNN(
      (hidden_linear1): Linear(in_features=784, out_features=128, bias=True)
      (hidden_linear2): Linear(in_features=128, out_features=64, bias=True)
      (ouput_linear): Linear(in_features=64, out_features=10, bias=True)
    )




```python
model.ouput_linear.bias 
```




    Parameter containing:
    tensor([ 0.0361,  0.0344, -0.0412,  0.0781, -0.0987, -0.0746, -0.0694,  0.1225,
             0.0778,  0.0395], requires_grad=True)




```python
for name, parameter in model.named_parameters():
  print(name, parameter.size())
```

    hidden_linear1.weight torch.Size([128, 784])
    hidden_linear1.bias torch.Size([128])
    hidden_linear2.weight torch.Size([64, 128])
    hidden_linear2.bias torch.Size([64])
    ouput_linear.weight torch.Size([10, 64])
    ouput_linear.bias torch.Size([10])
    

## 5. 모델 컴파일 (손실함수, 옵티마이저 선택)


```python
# Note. CrossEntropyLoss 관련
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss
# Note that this case is equivalent to the combination of LogSoftmax and NLLLoss.
# CrossEntropy를 손실함수로 사용하게 되면 forward() 계산시에 softmax() 함수를 사용하면 안됨(otherwise 중복)
# softmax 를 사용하면 부동 소수점 부정확성으로 인해 정확도가 떨어지고 불안정해질 수 있음
# forward()의 마지막 출력은 확률값이 아닌 score(logit)이어야 함
```


```python
learning_rate = 0.1
# 손실함수
loss_fn = nn.CrossEntropyLoss()

# 옵티마이저(최적화함수, 예:경사하강법)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
```


```python
from torchsummary import summary
```


```python
# summary(모델, (채널, 인풋사이즈))
summary(model, (1, 784))
```

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Linear-1               [-1, 1, 128]         100,480
                Linear-2                [-1, 1, 64]           8,256
                Linear-3                [-1, 1, 10]             650
    ================================================================
    Total params: 109,386
    Trainable params: 109,386
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.00
    Params size (MB): 0.42
    Estimated Total Size (MB): 0.42
    ----------------------------------------------------------------
    


```python
784*128 + 128
```




    100480




```python
128*64 + 64
```




    8256




```python
64*10 + 10
```




    650



## 6. 모델 훈련


```python
# torch.no_grad()
# https://pytorch.org/docs/stable/generated/torch.no_grad.html
# Context-manager that disabled gradient calculation.

# Disabling gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward(). 
# It will reduce memory consumption for computations that would otherwise have requires_grad=True.
```


```python
def validate(model, validloader, loss_fn):
  total = 0   
  correct = 0
  valid_loss = 0
  valid_accuracy = 0

  # 전방향 예측을 구할 때는 gradient가 필요가 없음음
  with torch.no_grad():
    for images, labels in validloader: # 이터레이터로부터 next()가 호출되며 미니배치 100개씩을 반환(images, labels)
      # 1. 입력 데이터 준비
      images.resize_(images.size()[0], 784)
      # 2. 전방향(Forward) 예측
      logit = model(images) # 예측 점수
      _, preds = torch.max(logit, 1) # 100개에 대한 최종 예측
      # preds = logit.max(dim=1)[1] 
      correct += int((preds == labels).sum()) # 100개 중 맞은 것의 개수가 coorect에 누적
      total += labels.shape[0] # 배치 사이즈만큼씩 total에 누적

      loss = loss_fn(logit, labels)
      valid_loss += loss.item() # tensor에서 값을 꺼내와서, 100개의 loss 평균값을 valid_loss에 누적

    valid_accuracy = correct / total
  
  return valid_loss, valid_accuracy
```


```python
epochs = 17
steps = 0

steps_per_epoch = len(trainloader) # 500 iterations


for epoch in range(epochs):
  model.train() # 훈련 모드드
  train_loss = 0
  for images, labels in trainloader: # 이터레이터로부터 next()가 호출되며 미니배치 100개씩을 반환(images, labels)
    steps += 1
  # 1. 입력 데이터 준비
    # before resize : (100, 1, 28, 28)
    images.resize_(images.shape[0], 784) 
    # after resize : (100, 784)

  # 2. 전방향(forward) 예측
    predict = model(images) # 예측 점수
    loss = loss_fn(predict, labels) # 예측 점수와 정답을 CrossEntropyLoss에 넣어 Loss값 반환

  # 3. 역방향(backward) 오차(Gradient) 전파
    optimizer.zero_grad() # Gradient가 누적되지 않게 하기 위해
    loss.backward() # 모델파리미터들의 Gradient 전파

  # 4. 경사 하강법으로 모델 파라미터 업데이트
    optimizer.step() # W <- W -lr*Gradient

    train_loss += loss.item()
    if (steps % steps_per_epoch) == 0 : # 500, 1000, 1500, ....(epoch)
      model.eval() # 평가 모드 : 평가에서 사용하지 않을 계층(배치 정규화, 드롭아웃)들을 수행하지 않게 하기 위해서
      valid_loss, valid_accuracy = validate(model, validloader, loss_fn)

      print('Epoch : {}/{}.......'.format(epoch+1, epochs),            
            'Train Loss : {:.3f}'.format(train_loss/len(trainloader)), # train_loss 500개 분량의 loss
            'Valid Loss : {:.3f}'.format(valid_loss/len(validloader)), # valid_loss 100개 분량의 loss
            'Valid Accuracy : {:.3f}'.format(valid_accuracy)            
            )
      
```

    Epoch : 1/17....... Train Loss : 0.704 Valid Loss : 0.309 Valid Accuracy : 0.906
    Epoch : 2/17....... Train Loss : 0.273 Valid Loss : 0.237 Valid Accuracy : 0.929
    Epoch : 3/17....... Train Loss : 0.200 Valid Loss : 0.175 Valid Accuracy : 0.948
    Epoch : 4/17....... Train Loss : 0.157 Valid Loss : 0.153 Valid Accuracy : 0.955
    Epoch : 5/17....... Train Loss : 0.129 Valid Loss : 0.128 Valid Accuracy : 0.962
    Epoch : 6/17....... Train Loss : 0.110 Valid Loss : 0.123 Valid Accuracy : 0.964
    Epoch : 7/17....... Train Loss : 0.096 Valid Loss : 0.106 Valid Accuracy : 0.968
    Epoch : 8/17....... Train Loss : 0.083 Valid Loss : 0.097 Valid Accuracy : 0.972
    Epoch : 9/17....... Train Loss : 0.073 Valid Loss : 0.094 Valid Accuracy : 0.971
    Epoch : 10/17....... Train Loss : 0.064 Valid Loss : 0.091 Valid Accuracy : 0.973
    Epoch : 11/17....... Train Loss : 0.057 Valid Loss : 0.085 Valid Accuracy : 0.975
    Epoch : 12/17....... Train Loss : 0.050 Valid Loss : 0.088 Valid Accuracy : 0.974
    Epoch : 13/17....... Train Loss : 0.045 Valid Loss : 0.088 Valid Accuracy : 0.975
    Epoch : 14/17....... Train Loss : 0.040 Valid Loss : 0.082 Valid Accuracy : 0.976
    Epoch : 15/17....... Train Loss : 0.036 Valid Loss : 0.083 Valid Accuracy : 0.975
    Epoch : 16/17....... Train Loss : 0.032 Valid Loss : 0.081 Valid Accuracy : 0.977
    Epoch : 17/17....... Train Loss : 0.028 Valid Loss : 0.081 Valid Accuracy : 0.977
    


```python
epochs = 17
steps = 0
# steps_per_epoch = len(trainloader) # 600 iterations
steps_per_epoch = len(trainset)/batch_size # 600 iterations

for epoch in range(epochs):
  model.train()
  train_loss = 0
  for images, labels in trainloader: # 이터레이터로부터 next()가 호출되며 미니배치 100개씩을 반환(images, labels)
    steps += 1
  # 1. 입력 데이터 준비
    # before resize : (100, 1, 28, 28)
    images.resize_(images.shape[0], 784) 
    # after resize : (100, 784)

  # 2. 전방향(forward) 예측
    predict = model(images) # 예측 점수
    loss = loss_fn(predict, labels) # 예측 점수와 정답을 CrossEntropyLoss에 넣어 Loss값 반환

  # 3. 역방향(backward) 오차(Gradient) 전파
    optimizer.zero_grad() # Gradient가 누적되지 않게 하기 위해
    loss.backward() # 모델파리미터들의 Gradient 전파

  # 4. 경사 하강법으로 모델 파라미터 업데이트
    optimizer.step() # W <- W -lr*Gradient

    train_loss += loss.item()
  
  # 매 epoch 마다 loss값 출력
  model.eval()
  valid_loss, valid_accuracy = validate(model, validloader, loss_fn)

  print('Epoch : {}/{}.......'.format(epoch+1, epochs),            
        'Train Loss : {:.3f}'.format(train_loss/len(trainloader)), # train_loss 500개 분량의 loss
        'Valid Loss : {:.3f}'.format(valid_loss/len(validloader)), # valid_loss 100개 분량의 loss
        'Valid Accuracy : {:.3f}'.format(valid_accuracy)            
         )
```

    Epoch : 1/17....... Train Loss : 0.026 Valid Loss : 0.079 Valid Accuracy : 0.977
    Epoch : 2/17....... Train Loss : 0.023 Valid Loss : 0.083 Valid Accuracy : 0.976
    Epoch : 3/17....... Train Loss : 0.020 Valid Loss : 0.079 Valid Accuracy : 0.978
    Epoch : 4/17....... Train Loss : 0.018 Valid Loss : 0.082 Valid Accuracy : 0.978
    Epoch : 5/17....... Train Loss : 0.016 Valid Loss : 0.083 Valid Accuracy : 0.978
    Epoch : 6/17....... Train Loss : 0.013 Valid Loss : 0.085 Valid Accuracy : 0.977
    Epoch : 7/17....... Train Loss : 0.012 Valid Loss : 0.084 Valid Accuracy : 0.977
    Epoch : 8/17....... Train Loss : 0.011 Valid Loss : 0.081 Valid Accuracy : 0.978
    Epoch : 9/17....... Train Loss : 0.009 Valid Loss : 0.083 Valid Accuracy : 0.978
    Epoch : 10/17....... Train Loss : 0.008 Valid Loss : 0.082 Valid Accuracy : 0.978
    Epoch : 11/17....... Train Loss : 0.007 Valid Loss : 0.083 Valid Accuracy : 0.978
    Epoch : 12/17....... Train Loss : 0.007 Valid Loss : 0.083 Valid Accuracy : 0.979
    Epoch : 13/17....... Train Loss : 0.006 Valid Loss : 0.083 Valid Accuracy : 0.978
    Epoch : 14/17....... Train Loss : 0.005 Valid Loss : 0.084 Valid Accuracy : 0.979
    Epoch : 15/17....... Train Loss : 0.005 Valid Loss : 0.084 Valid Accuracy : 0.979
    Epoch : 16/17....... Train Loss : 0.004 Valid Loss : 0.084 Valid Accuracy : 0.978
    Epoch : 17/17....... Train Loss : 0.004 Valid Loss : 0.087 Valid Accuracy : 0.979
    

## 7. 모델 예측


```python
# testloader에서 미니 배치(100개) 가져오기
test_iter = iter(testloader)
images, labels = next(test_iter)
print(images.size(), labels.size())

# random한 index로 이미지 한장 준비하기
rnd_idx = 10
print(images[rnd_idx].shape, labels[rnd_idx])
flattend_img = images[rnd_idx].view(1, 784)

# 준비된 이미지로 예측하기
model.eval()
with torch.no_grad():
  logit = model(flattend_img)

pred = logit.max(dim=1)[1]
print(pred == labels[rnd_idx]) # True : 잘 예측
```

    torch.Size([100, 1, 28, 28]) torch.Size([100])
    torch.Size([1, 28, 28]) tensor(0)
    tensor([True])
    


```python
plt.imshow(images[rnd_idx].squeeze(), cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7f6b71844e20>




    
![png](/assets/images/2023-04-07-Deep Learning 8 (MNIST in Pytorch (3) 모델 수정)/output_38_1.png)
    


## 8. 모델 평가


```python
def evaluation(model, testloader, loss_fn):
  total = 0   
  correct = 0
  test_loss = 0
  test_accuracy = 0

  # 전방향 예측을 구할 때는 gradient가 필요가 없음음
  with torch.no_grad():
    for images, labels in testloader: # 이터레이터로부터 next()가 호출되며 미니배치 100개씩을 반환(images, labels)
      # 1. 입력 데이터 준비
      images.resize_(images.size()[0], 784)
      # 2. 전방향(Forward) 예측
      logit = model(images) # 예측 점수
      _, preds = torch.max(logit, 1) # 100개에 대한 최종 예측
      # preds = logit.max(dim=1)[1] 
      correct += int((preds == labels).sum()) # 100개 중 맞은 것의 개수가 coorect에 누적
      total += labels.shape[0] # 배치 사이즈만큼씩 total에 누적

      loss = loss_fn(logit, labels)
      test_loss += loss.item() # tensor에서 값을 꺼내와서, 100개의 loss 평균값을 valid_loss에 누적

    test_accuracy = correct / total
  
  print('Test Loss : {:.3f}'.format(test_loss/len(testloader)), 
        'Test Accuracy : {:.3f}'.format(test_accuracy))

model.eval()
evaluation(model, testloader, loss_fn)  
```

    Test Loss : 0.077 Test Accuracy : 0.977
    


```python

```
