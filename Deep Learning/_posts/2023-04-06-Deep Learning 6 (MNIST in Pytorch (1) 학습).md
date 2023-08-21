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
    

    100%|██████████| 9912422/9912422 [00:00<00:00, 224742078.05it/s]

    Extracting ./datasets/MNIST/raw/train-images-idx3-ubyte.gz to ./datasets/MNIST/raw
    

    
    

    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ./datasets/MNIST/raw/train-labels-idx1-ubyte.gz
    

    100%|██████████| 28881/28881 [00:00<00:00, 37772277.46it/s]
    

    Extracting ./datasets/MNIST/raw/train-labels-idx1-ubyte.gz to ./datasets/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ./datasets/MNIST/raw/t10k-images-idx3-ubyte.gz
    

    100%|██████████| 1648877/1648877 [00:00<00:00, 111429813.85it/s]

    Extracting ./datasets/MNIST/raw/t10k-images-idx3-ubyte.gz to ./datasets/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ./datasets/MNIST/raw/t10k-labels-idx1-ubyte.gz
    

    
    100%|██████████| 4542/4542 [00:00<00:00, 12274825.24it/s]
    

    Extracting ./datasets/MNIST/raw/t10k-labels-idx1-ubyte.gz to ./datasets/MNIST/raw
    
    


```python
print(type(trainset), len(trainset))
print(type(testset), len(testset))
```

    <class 'torchvision.datasets.mnist.MNIST'> 60000
    <class 'torchvision.datasets.mnist.MNIST'> 10000
    


```python
# 0번째 샘플에 2개의 원소가 있는데, 그중 첫번째 원소는 이미지, 두번째 원소는 정답
# 그러나 파이토치로 읽어들인 이미지 텐서의 형상이 channels * height * width 임
# 그에 비해 opencv, matplotlib으로 읽어들인 이미지 array의 형상은 height * width * channels

print(trainset[0][0].size(), trainset[0][1])
```

    torch.Size([1, 28, 28]) 5
    

## 2. 데이터 시각화

- tensor로 시각화화


```python
sample_img = trainset[0][0]
sample_img.size()
```




    torch.Size([1, 28, 28])




```python
sample_img.squeeze().size()
```




    torch.Size([28, 28])




```python
plt.imshow(sample_img.squeeze(), cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7fea3e6ac070>




    
![png](/assets/images/2023-04-06-Deep Learning 6 (MNIST in Pytorch (1) 학습)/output_11_1.png)
    


- numpy로 시각화


```python
sample_img_np = sample_img.numpy()
sample_img_np.shape
```




    (1, 28, 28)




```python
sample_img_np.squeeze().shape
```




    (28, 28)




```python
plt.imshow(sample_img_np.squeeze(), cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7fea3bc4f3d0>




    
![png](/assets/images/2023-04-06-Deep Learning 6 (MNIST in Pytorch (1) 학습)/output_15_1.png)
    


- permute -> squeeze : sample_img가 현재는 gray 


```python
# sample_img가 컬러이미지라면 squeeze 할 필요 없음
# color_sample = sample_img.permute(1, 2, 0)
```


```python
sample_img.size()
```




    torch.Size([1, 28, 28])




```python
sample_img_permute = sample_img.permute(1, 2, 0)
sample_img_permute.size()
```




    torch.Size([28, 28, 1])




```python
sample_img_permute.squeeze(axis=2).size()
```




    torch.Size([28, 28])




```python
plt.imshow(sample_img_permute.squeeze(), cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7fea3bc60520>




    
![png](/assets/images/2023-04-06-Deep Learning 6 (MNIST in Pytorch (1) 학습)/output_21_1.png)
    



```python
figure, axes = plt.subplots(nrows=1, ncols=8, figsize=(22, 6))

for i in range(8):
  image, label= trainset[i][0], trainset[i][1]
  axes[i].imshow(image.squeeze(), cmap='gray')
  axes[i].set_title('Class : ' + str(label))
```


    
![png](/assets/images/2023-04-06-Deep Learning 6 (MNIST in Pytorch (1) 학습)/output_22_0.png)
    


## 3. 데이터 적재


```python
# DataLoader
# 모델 훈련에 사용할 수 있는 미니 배치 구성하고
# 매 epoch마다 데이터를 샘플링, 병렬처리 등의 일을 해주는 함수
```


```python
batch_size = 100
# dataloader = DataLoader(데이터셋, 배치사이즈, 셔플여부.....)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True) # 훈련용 60000개의 데이터를 100개씩 준비
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False) # 테스트용 10000개의 데이터를 100개씩 준비
```


```python
print(type(trainloader), len(trainloader))
print(type(testloader), len(testloader))
```

    <class 'torch.utils.data.dataloader.DataLoader'> 600
    <class 'torch.utils.data.dataloader.DataLoader'> 100
    


```python
trainloader
```




    <torch.utils.data.dataloader.DataLoader at 0x7fea3b28e9d0>




```python
dir(trainloader) # trainloader는 iterator
```


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




    
![png](/assets/images/2023-04-06-Deep Learning 6 (MNIST in Pytorch (1) 학습)/output_31_0.png)
    




```python
import torch.nn as nn # 파이토치에서 제공하는 다양한 계층 (Linear Layer, ....)
import torch.optim as optim # 옵티마이저 (경사하강법...)
import torch.nn.functional as F # 파이토치에서 제공하는 함수(활성화 함수...)
```

**option 1** : nn.Sequential 사용하기


```python
model = nn.Sequential(
                      nn.Linear(784, 20),
                      nn.Sigmoid(),
                      nn.Linear(20, 10)
                      )
model
```




    Sequential(
      (0): Linear(in_features=784, out_features=20, bias=True)
      (1): Sigmoid()
      (2): Linear(in_features=20, out_features=10, bias=True)
    )




```python
# nn.Module을 상속받아서 사용하는 서브모듈(nn.Linear, ...)들은 모델 파라미터들이 관리되고 있음을 알 수 있음
for parameter in model.parameters():
  print(parameter.size())
```

    torch.Size([20, 784])
    torch.Size([20])
    torch.Size([10, 20])
    torch.Size([10])
    


```python
# nn.Sequential을 구성하는 계층들의 이름까지 출력
for name, parameter in model.named_parameters():
  print(name, parameter.size())
```

    0.weight torch.Size([20, 784])
    0.bias torch.Size([20])
    2.weight torch.Size([10, 20])
    2.bias torch.Size([10])
    

**option 2** : nn.Sequential에 OrderedDict 적용하기


```python
from collections import OrderedDict
```


```python
model = nn.Sequential(OrderedDict([
                        ('hidden_linear', nn.Linear(784, 20)),
                        ('hidden_activation', nn.Sigmoid()),
                        ('ouput_linear', nn.Linear(20, 10))
                      ]))
model
```




    Sequential(
      (hidden_linear): Linear(in_features=784, out_features=20, bias=True)
      (hidden_activation): Sigmoid()
      (ouput_linear): Linear(in_features=20, out_features=10, bias=True)
    )




```python
for name, parameter in model.named_parameters():
  print(name, parameter.size())
```

    hidden_linear.weight torch.Size([20, 784])
    hidden_linear.bias torch.Size([20])
    ouput_linear.weight torch.Size([10, 20])
    ouput_linear.bias torch.Size([10])
    

**option 3** : nn.Module 서브클래싱하기
- 파라미터 관리가 필요없는 기능(활성화 함수, ...)도 서브모듈(nn.Module로부터 상속)로 작성


```python
class TwoLayerNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.hidden_linear = nn.Linear(in_features=784, out_features=20)
    self.hidden_activation = nn.Sigmoid()
    self.ouput_linear = nn.Linear(in_features=20, out_features=10)

  def forward(self, x):
    x = self.hidden_linear(x)
    x = self.hidden_activation(x)
    x = self.ouput_linear(x)    
    return x
```


```python
model = TwoLayerNet()
model
```




    TwoLayerNet(
      (hidden_linear): Linear(in_features=784, out_features=20, bias=True)
      (hidden_activation): Sigmoid()
      (ouput_linear): Linear(in_features=20, out_features=10, bias=True)
    )




```python
for name, parameter in model.named_parameters():
  print(name, parameter.size())
```

    hidden_linear.weight torch.Size([20, 784])
    hidden_linear.bias torch.Size([20])
    ouput_linear.weight torch.Size([10, 20])
    ouput_linear.bias torch.Size([10])
    

**option 4** : nn.Module 서브클래싱하기
- 파라미터 관리가 필요없는 기능(활성화 함수, ...) 함수형(functional)으로 작성
- 함수형이란 출력이 입력에 의해 결정


```python
class TwoLayerNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.hidden_linear = nn.Linear(in_features=784, out_features=20)    
    self.ouput_linear = nn.Linear(in_features=20, out_features=10)

  def forward(self, x):
    x = self.hidden_linear(x)    
    x = F.sigmoid(x)
    x = self.ouput_linear(x)    
    return x
```


```python
model = TwoLayerNet()
model
```




    TwoLayerNet(
      (hidden_linear): Linear(in_features=784, out_features=20, bias=True)
      (ouput_linear): Linear(in_features=20, out_features=10, bias=True)
    )




```python
for name, parameter in model.named_parameters():
  print(name, parameter.size())
```

    hidden_linear.weight torch.Size([20, 784])
    hidden_linear.bias torch.Size([20])
    ouput_linear.weight torch.Size([10, 20])
    ouput_linear.bias torch.Size([10])
    

## 5. 모델 설정 (손실함수, 옵티마이저 선택)


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
                Linear-1                [-1, 1, 20]          15,700
                Linear-2                [-1, 1, 10]             210
    ================================================================
    Total params: 15,910
    Trainable params: 15,910
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.00
    Params size (MB): 0.06
    Estimated Total Size (MB): 0.06
    ----------------------------------------------------------------
    


```python
784*20 + 20
```




    15700




```python
10 * 20 + 10
```




    210



## 6. 모델 훈련


```python
trainloader
```




    <torch.utils.data.dataloader.DataLoader at 0x7fea3b28e9d0>




```python
len(trainloader)
```




    600




```python
epochs = 17
steps = 0
# steps_per_epoch = len(trainloader) # 600 iterations
steps_per_epoch = len(trainset)/batch_size # 600 iterations

for epoch in range(epochs):
  running_loss = 0
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

    running_loss += loss.item()
    if (steps % steps_per_epoch) == 0 : # 600, 1200, 1800, ....(epoch)
      print('Epoch : {}/{}.......'.format(epoch+1, epochs),
            'Loss : {}'.format(running_loss/steps_per_epoch))
      running_loss = 0
```

    Epoch : 1/17....... Loss : 0.5584875592092673
    Epoch : 2/17....... Loss : 0.4106673734138409
    Epoch : 3/17....... Loss : 0.3518453801671664
    Epoch : 4/17....... Loss : 0.3187762003143628
    Epoch : 5/17....... Loss : 0.29627135639389357
    Epoch : 6/17....... Loss : 0.27978881236165765
    Epoch : 7/17....... Loss : 0.2663819943368435
    Epoch : 8/17....... Loss : 0.255467306189239
    Epoch : 9/17....... Loss : 0.24600239269435406
    Epoch : 10/17....... Loss : 0.23775523462643225
    Epoch : 11/17....... Loss : 0.2302229973177115
    Epoch : 12/17....... Loss : 0.22369915128995974
    Epoch : 13/17....... Loss : 0.21776643681029478
    Epoch : 14/17....... Loss : 0.212262866931657
    Epoch : 15/17....... Loss : 0.20721533005436263
    Epoch : 16/17....... Loss : 0.20254455306877692
    Epoch : 17/17....... Loss : 0.1984483090043068
    


```python
epochs = 17
steps = 0
# steps_per_epoch = len(trainloader) # 600 iterations
steps_per_epoch = len(trainset)/batch_size # 600 iterations

for epoch in range(epochs):
  running_loss = 0
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

    running_loss += loss.item()
  
  # 매 epoch 마다 loss값 출력
  print('Epoch : {}/{}.......'.format(epoch+1, epochs),
        'Loss : {}'.format(running_loss/steps_per_epoch))
  # running_loss = 0
```

    Epoch : 1/17....... Loss : 1.3314239980777105
    Epoch : 2/17....... Loss : 0.5544554363191128
    Epoch : 3/17....... Loss : 0.4085979046920935
    Epoch : 4/17....... Loss : 0.35159790456295015
    Epoch : 5/17....... Loss : 0.3199071667591731
    Epoch : 6/17....... Loss : 0.2987711794177691
    Epoch : 7/17....... Loss : 0.2830567279458046
    Epoch : 8/17....... Loss : 0.2703934657449524
    Epoch : 9/17....... Loss : 0.2597169236342112
    Epoch : 10/17....... Loss : 0.2505294536675016
    Epoch : 11/17....... Loss : 0.24248206504931052
    Epoch : 12/17....... Loss : 0.23508482793966928
    Epoch : 13/17....... Loss : 0.228595398205022
    Epoch : 14/17....... Loss : 0.22253441113978625
    Epoch : 15/17....... Loss : 0.21717012982815503
    Epoch : 16/17....... Loss : 0.21197644542902708
    Epoch : 17/17....... Loss : 0.2072841609393557
    

