# 신경망 학습 (CIFAR10 in Pytorch)


```python
from IPython.display import Image
Image('./images/cifar10.png', width=600)
```




    
![png](/assets/images/CNN 3 (CIFAR10 (1) Dataset)/output_1_0.png)
    




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

from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
```


```python
!nvidia-smi
```

    Mon Apr 10 02:21:21 2023       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  NVIDIA A100-SXM...  Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   32C    P0    43W / 400W |      0MiB / 40960MiB |      0%      Default |
    |                               |                      |             Disabled |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+
    


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
```




    device(type='cuda')



## 1. 데이터 불러오기


```python
# Compose를 통해 원하는 전처리기를 차례대로 넣을 수 있음음
# mnist_transform = transforms.Compose([transforms.Resize(16), transforms.ToTensor()])
# trainset = datasets.MNIST('./datasets/', download=True, train=True, transform = mnist_transform)
```


```python
# dataset = datasets.MNIST(다운받을 디렉토리, 다운로드여부, 학습용여부, 전처리방법)
trainset = datasets.CIFAR10('./datasets/', download=True, train=True, transform = transforms.ToTensor())
testset = datasets.CIFAR10('./datasets/', download=True, train=False, transform = transforms.ToTensor())
```

    Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./datasets/cifar-10-python.tar.gz
    

    100%|██████████| 170498071/170498071 [00:05<00:00, 32944383.80it/s]
    

    Extracting ./datasets/cifar-10-python.tar.gz to ./datasets/
    Files already downloaded and verified
    


```python
print(type(trainset), len(trainset))
print(type(testset), len(testset))
```

    <class 'torchvision.datasets.cifar.CIFAR10'> 50000
    <class 'torchvision.datasets.cifar.CIFAR10'> 10000
    


```python
print(type(trainset.targets), len(trainset.targets), trainset.targets[:5])
```

    <class 'list'> 50000 [6, 9, 9, 4, 1]
    


```python
# 클래스별 분포
for i in range(10): # 클래스별 순회
  print('클래스(레이블)별 데이터 개수 : ', i, (np.array(trainset.targets) == i).sum())
```

    클래스(레이블)별 데이터 개수 :  0 5000
    클래스(레이블)별 데이터 개수 :  1 5000
    클래스(레이블)별 데이터 개수 :  2 5000
    클래스(레이블)별 데이터 개수 :  3 5000
    클래스(레이블)별 데이터 개수 :  4 5000
    클래스(레이블)별 데이터 개수 :  5 5000
    클래스(레이블)별 데이터 개수 :  6 5000
    클래스(레이블)별 데이터 개수 :  7 5000
    클래스(레이블)별 데이터 개수 :  8 5000
    클래스(레이블)별 데이터 개수 :  9 5000
    


```python
from sklearn.model_selection import train_test_split

train_indices, valid_indices, _, _ = train_test_split(
                            range(len(trainset)), # X의 index
                            trainset.targets, # y
                            stratify=trainset.targets, # target의 비율이 train과 valid에 그대로 반영되게
                            test_size= 0.2, random_state=42)
```


```python
len(train_indices), len(valid_indices) # 80%, 20%
```




    (40000, 10000)




```python
from torch.utils.data import Subset
train_set = Subset(trainset, train_indices)
valid_set = Subset(trainset, valid_indices)
```


```python
valid_set[0][1] # 0번째 샘플의 정답답
```




    2




```python
# 클래스별 분포
class_list = []
for i in range(10): # 클래스별 순회
  s = 0
  for j in range(len(valid_set)): # valid_data 10000개 순회
    if valid_set[j][1] == i :
      s += 1
  class_list.append(s)
class_list
```




    [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]




```python
# trainset을 다시 train용과 valid 용으로 나누고자 할 때
# trainset, validset = random_split(trainset, [50000, 10000])
```


```python
print(type(train_set), len(train_set))
print(type(valid_set), len(valid_set))
print(type(testset), len(testset))
```

    <class 'torch.utils.data.dataset.Subset'> 40000
    <class 'torch.utils.data.dataset.Subset'> 10000
    <class 'torchvision.datasets.cifar.CIFAR10'> 10000
    


```python
# 0번째 샘플에 2개의 원소가 있는데, 그중 첫번째 원소는 이미지, 두번째 원소는 정답
# 그러나 파이토치로 읽어들인 이미지 텐서의 형상이 channels * height * width 임
# 그에 비해 opencv, matplotlib으로 읽어들인 이미지 array의 형상은 height * width * channels
print(train_set[0][0].size(), train_set[0][1])
```

    torch.Size([3, 32, 32]) 6
    

## 2. 데이터 시각화


```python
labels_map = {0 : 'airplane', 1 : 'automobile', 2 : 'bird', 3 : 'cat', 4 : 'deer', 5 : 'dog', 6 : 'frog',
             7 : 'horse', 8 : 'ship', 9 : 'truck'}  # for cifar   

figure, axes = plt.subplots(nrows=4, ncols=8, figsize=(14, 8))
axes = axes.flatten()

for i in range(32):
  rand_i = np.random.randint(0, len(trainset))
  image, label= trainset[rand_i][0].permute(1, 2, 0), trainset[rand_i][1]
  axes[i].axis('off')
  axes[i].imshow(image)
  axes[i].set_title(labels_map[label])
```


    
![png](/assets/images/CNN 3 (CIFAR10 (1) Dataset)/output_20_0.png)
    


## 3. 데이터 적재

**DataLoader**
- 모델 훈련에 사용할 수 있는 미니 배치 구성하고
- 매 epoch마다 데이터를 샘플링, 병렬처리 등의 일을 해주는 함수

**배치 사이즈**
- 배치 사이즈 중요한 하이퍼 파라미터
- 16 이하로 사용하는것이 성능에 좋다고 알려져 있음

- 배치 사이즈가 크다는 것은 실제 Loss, Gradient, Weight를 구하는 데 참여하는 데이타가 많다라는 뜻
- 배치 사이즈가 작을 수록 모델이 학습을 하는데 한번도 보지 않은 신선한 데이터가 제공될 확률이 큼

- 배치 사이즈가 크면 학습시간은 줄일 수 있으나 적절한 배치사이즈로 학습을 해야 성능을 높일 수 있음
- (60000개의 데이터를 100개의 미니배치로 학습하면 1 epoch당 걸리는 횟수가 600번인데, 10개의 미니배치로 학습하면 1 epoch당 걸리는 횟수가 6000번)


```python
batch_size = 16 # 100 -> 16
# dataloader = DataLoader(데이터셋, 배치사이즈, 셔플여부.....)
trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True) # 훈련용 50000개의 데이터를 100개씩 준비
validloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False) # 검증용 10000개의 데이터를 100개씩 준비
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False) # 테스트용 10000개의 데이터를 100개씩 준비
```


```python
40000/16, 10000/16, 10000/16
```




    (2500.0, 625.0, 625.0)




```python
print(type(trainloader), len(trainloader))
print(type(validloader), len(validloader))
print(type(testloader), len(testloader))
```

    <class 'torch.utils.data.dataloader.DataLoader'> 2500
    <class 'torch.utils.data.dataloader.DataLoader'> 625
    <class 'torch.utils.data.dataloader.DataLoader'> 625
    


```python
train_iter = iter(trainloader)
images, labels = next(train_iter)
images.size(), labels.size()
```




    (torch.Size([16, 3, 32, 32]), torch.Size([16]))



## 4. 모델 생성


```python
from IPython.display import Image
Image('./images/cnn architecture 2.png', width=700)
```




    
![png](/assets/images/CNN 3 (CIFAR10 (1) Dataset)/output_29_0.png)
    




```python
import torch.nn as nn # 파이토치에서 제공하는 다양한 계층 (Linear Layer, ....)
import torch.optim as optim # 옵티마이저 (경사하강법...)
import torch.nn.functional as F # 파이토치에서 제공하는 함수(활성화 함수...)
```


```python
# 가중치 초기화
# https://pytorch.org/docs/stable/nn.init.html

# 현재 default 값
# Linear :
# https://github.com/pytorch/pytorch/blob/9cf62a4b5d3b287442e70c0c560a8e21d8c3b189/torch/nn/modules/linear.py#L168
# Conv :
# https://github.com/pytorch/pytorch/blob/9cf62a4b5d3b287442e70c0c560a8e21d8c3b189/torch/nn/modules/conv.py#L111
```


```python
# 가중치 초기화시 고려할 사항
# 1. 값이 충분히 작아야 함
# 2. 값이 하나로 치우쳐선 안됨
# 3. 적당한 분산으로 골고루 분포가 되어야 함
```

**option 4** : nn.Module 서브클래싱하기
- 파라미터 관리가 필요없는 기능(활성화 함수, ...) 함수형(functional)으로 작성
- 함수형이란 출력이 입력에 의해 결정

**모델 바뀐점**
- conv_block 추가
- batch noramalization 추가
- drop out 추가

**완전 연결망과 CNN망과의 차이점**
- 지역 연산
- 가중치 공유(적은 파라미터)
- 평행 이동 불변성


```python
class CIFAR10_CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_block1 = nn.Sequential(
                                      nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(num_features=32),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2)
                                     ) # batch_size x 32 x 16 x 16
    self.conv_block2 = nn.Sequential(
                                      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(num_features=64),
                                      nn.Dropout(0.2),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2)
                                     ) # batch_size x 64 x 8 x 8

    self.conv_block3 = nn.Sequential(
                                      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(num_features=128),
                                      nn.Dropout(0.4),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2)
                                     ) # batch_size x 128 x 4 x 4                                   

    self.linear1 = nn.Linear(in_features=128*4*4, out_features=128)
    self.batch_norm = nn.BatchNorm1d(num_features=128)
    self.linear2 = nn.Linear(in_features=128, out_features=10)

  def forward(self, x):
    x = self.conv_block1(x) # batch_size x 32 x 16 x 16
    x = self.conv_block2(x) # batch_size x 64 x 8 x 8
    x = self.conv_block3(x) # batch_size x 128 x 4 x 4
    
    # reshape할 형상 : (batch_size x 128*4*4)
    # x = x.view(-1, 128*4*4) # option 1 : view
    x = torch.flatten(x, 1) # option 2 : flatten 
    # x = x.reshape(x.shape[0], -1) # option 3 : reshape

    x = self.linear1(x)
    x = self.batch_norm(x)
    x = F.dropout(x, 0.5)
    x = F.relu(x)
    x = self.linear2(x)
    return x
```


```python
model = CIFAR10_CNN()
model.to(device)
model
```




    CIFAR10_CNN(
      (conv_block1): Sequential(
        (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (conv_block2): Sequential(
        (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): Dropout(p=0.2, inplace=False)
        (3): ReLU()
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (conv_block3): Sequential(
        (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): Dropout(p=0.4, inplace=False)
        (3): ReLU()
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (linear1): Linear(in_features=2048, out_features=128, bias=True)
      (batch_norm): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (linear2): Linear(in_features=128, out_features=10, bias=True)
    )




```python
for name, parameter in model.named_parameters():
  print(name, parameter.size())
```

    conv_block1.0.weight torch.Size([32, 3, 3, 3])
    conv_block1.0.bias torch.Size([32])
    conv_block1.1.weight torch.Size([32])
    conv_block1.1.bias torch.Size([32])
    conv_block2.0.weight torch.Size([64, 32, 3, 3])
    conv_block2.0.bias torch.Size([64])
    conv_block2.1.weight torch.Size([64])
    conv_block2.1.bias torch.Size([64])
    conv_block3.0.weight torch.Size([128, 64, 3, 3])
    conv_block3.0.bias torch.Size([128])
    conv_block3.1.weight torch.Size([128])
    conv_block3.1.bias torch.Size([128])
    linear1.weight torch.Size([128, 2048])
    linear1.bias torch.Size([128])
    batch_norm.weight torch.Size([128])
    batch_norm.bias torch.Size([128])
    linear2.weight torch.Size([10, 128])
    linear2.bias torch.Size([10])
    

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
learning_rate = 0.001
# 손실함수
loss_fn = nn.CrossEntropyLoss()

# 옵티마이저(최적화함수, 예:경사하강법)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 규제의 강도 설정 weight_decay
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

# Learning Rate Schedule
# https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html

# 모니터링하고 있는 값(예:valid_loss)의 최소값(min) 또는 최대값(max) patience 기간동안 줄어들지 않을 때(OnPlateau) lr에 factor(0.1)를 곱해주는 전략
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
```


```python
from torchsummary import summary
```


```python
# summary(모델, (채널, 인풋사이즈))
summary(model, (3, 32, 32))
```

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 32, 32, 32]             896
           BatchNorm2d-2           [-1, 32, 32, 32]              64
                  ReLU-3           [-1, 32, 32, 32]               0
             MaxPool2d-4           [-1, 32, 16, 16]               0
                Conv2d-5           [-1, 64, 16, 16]          18,496
           BatchNorm2d-6           [-1, 64, 16, 16]             128
               Dropout-7           [-1, 64, 16, 16]               0
                  ReLU-8           [-1, 64, 16, 16]               0
             MaxPool2d-9             [-1, 64, 8, 8]               0
               Conv2d-10            [-1, 128, 8, 8]          73,856
          BatchNorm2d-11            [-1, 128, 8, 8]             256
              Dropout-12            [-1, 128, 8, 8]               0
                 ReLU-13            [-1, 128, 8, 8]               0
            MaxPool2d-14            [-1, 128, 4, 4]               0
               Linear-15                  [-1, 128]         262,272
          BatchNorm1d-16                  [-1, 128]             256
               Linear-17                   [-1, 10]           1,290
    ================================================================
    Total params: 357,514
    Trainable params: 357,514
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 1.61
    Params size (MB): 1.36
    Estimated Total Size (MB): 2.99
    ----------------------------------------------------------------
    


```python
# 첫번째 conv layer의 모델 파라미터 수
# 필터수 x (필터) + bias
32 * (3*3*3) + 32
```




    896




```python
# 마지막 출력 feature map의 사이즈
128 * 4 * 4
```




    2048




```python
# linear 1 layer
2048 * 128 + 128
```




    262272




```python
# linear 2 layer
128 * 10 + 10
```




    1290



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
      # images, labels : (torch.Size([16, 3, 32, 32]), torch.Size([16]))
      # 0. Data를 GPU로 보내기
      images, labels = images.to(device), labels.to(device)

      # 1. 입력 데이터 준비
      # not Flatten !!
      # images.resize_(images.size()[0], 784)

      # 2. 전방향(Forward) 예측
      logit = model(images) # 예측 점수
      _, preds = torch.max(logit, 1) # 배치에 대한 최종 예측
      # preds = logit.max(dim=1)[1] 
      correct += int((preds == labels).sum()) # 배치 중 맞은 것의 개수가 correct에 누적
      total += labels.shape[0] # 배치 사이즈만큼씩 total에 누적

      loss = loss_fn(logit, labels)
      valid_loss += loss.item() # tensor에서 값을 꺼내와서, 배치의 loss 평균값을 valid_loss에 누적

    valid_accuracy = correct / total
  
  return valid_loss, valid_accuracy
```


```python
# 파이토치에서 텐서보드 사용하기
# https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
```


```python
writer = SummaryWriter()

def train_loop(model, trainloader, loss_fn, epochs, optimizer):  
  steps = 0
  steps_per_epoch = len(trainloader) 
  min_loss = 1000000
  max_accuracy = 0
  trigger = 0
  patience = 7 

  for epoch in range(epochs):
    model.train() # 훈련 모드
    train_loss = 0
    for images, labels in trainloader: # 이터레이터로부터 next()가 호출되며 미니배치를 반환(images, labels)
      steps += 1
      # images, labels : (torch.Size([16, 3, 32, 32]), torch.Size([16]))
      # 0. Data를 GPU로 보내기
      images, labels = images.to(device), labels.to(device)

      # 1. 입력 데이터 준비
      # not Flatten !!
      # images.resize_(images.shape[0], 784) 

      # 2. 전방향(forward) 예측
      predict = model(images) # 예측 점수
      loss = loss_fn(predict, labels) # 예측 점수와 정답을 CrossEntropyLoss에 넣어 Loss값 반환

      # 3. 역방향(backward) 오차(Gradient) 전파
      optimizer.zero_grad() # Gradient가 누적되지 않게 하기 위해
      loss.backward() # 모델파리미터들의 Gradient 전파

      # 4. 경사 하강법으로 모델 파라미터 업데이트
      optimizer.step() # W <- W -lr*Gradient

      train_loss += loss.item()
      if (steps % steps_per_epoch) == 0 : 
        model.eval() # 평가 모드 : 평가에서 사용하지 않을 계층(배치 정규화, 드롭아웃)들을 수행하지 않게 하기 위해서
        valid_loss, valid_accuracy = validate(model, validloader, loss_fn)

        # tensorboard 시각화를 위한 로그 이벤트 등록
        writer.add_scalar('Train Loss', train_loss/len(trainloader), epoch+1)
        writer.add_scalar('Valid Loss', valid_loss/len(validloader), epoch+1)
        writer.add_scalars('Train Loss and Valid Loss',
                          {'Train' : train_loss/len(trainloader),
                            'Valid' : valid_loss/len(validloader)}, epoch+1)
        writer.add_scalar('Valid Accuracy', valid_accuracy, epoch+1)
        # -------------------------------------------

        print('Epoch : {}/{}.......'.format(epoch+1, epochs),            
              'Train Loss : {:.3f}'.format(train_loss/len(trainloader)), 
              'Valid Loss : {:.3f}'.format(valid_loss/len(validloader)), 
              'Valid Accuracy : {:.3f}'.format(valid_accuracy)            
              )
        
        # Best model 저장    
        # option 1 : valid_loss 모니터링
        # if valid_loss < min_loss: # 바로 이전 epoch의 loss보다 작으면 저장하기
        #   min_loss = valid_loss
        #   best_model_state = deepcopy(model.state_dict())          
        #   torch.save(best_model_state, 'best_checkpoint.pth')     
        
        # option 2 : valid_accuracy 모니터링      
        if valid_accuracy > max_accuracy : # 바로 이전 epoch의 accuracy보다 크면 저장하기
          max_accuracy = valid_accuracy
          best_model_state = deepcopy(model.state_dict())          
          torch.save(best_model_state, 'best_checkpoint.pth')  
        # -------------------------------------------

        # Early Stopping (조기 종료)
        if valid_loss > min_loss: # valid_loss가 min_loss를 갱신하지 못하면
          trigger += 1
          print('trigger : ', trigger)
          if trigger > patience:
            print('Early Stopping !!!')
            print('Training loop is finished !!')
            writer.flush()   
            return
        else:
          trigger = 0
          min_loss = valid_loss
        # -------------------------------------------

        # Learning Rate Scheduler
        scheduler.step(valid_loss)
        # -------------------------------------------
        
  writer.flush()
  return  
```


```python
epochs = 50
train_loop(model, trainloader, loss_fn, epochs, optimizer)
writer.close()
```

    Epoch : 1/50....... Train Loss : 1.392 Valid Loss : 1.412 Valid Accuracy : 0.483
    Epoch : 2/50....... Train Loss : 1.090 Valid Loss : 1.187 Valid Accuracy : 0.592
    Epoch : 3/50....... Train Loss : 0.985 Valid Loss : 1.153 Valid Accuracy : 0.597
    Epoch : 4/50....... Train Loss : 0.906 Valid Loss : 0.957 Valid Accuracy : 0.669
    Epoch : 5/50....... Train Loss : 0.855 Valid Loss : 0.955 Valid Accuracy : 0.665
    Epoch : 6/50....... Train Loss : 0.813 Valid Loss : 0.893 Valid Accuracy : 0.697
    Epoch : 7/50....... Train Loss : 0.769 Valid Loss : 0.955 Valid Accuracy : 0.667
    trigger :  1
    Epoch : 8/50....... Train Loss : 0.738 Valid Loss : 0.901 Valid Accuracy : 0.688
    trigger :  2
    Epoch : 9/50....... Train Loss : 0.709 Valid Loss : 0.943 Valid Accuracy : 0.678
    trigger :  3
    Epoch : 10/50....... Train Loss : 0.681 Valid Loss : 0.805 Valid Accuracy : 0.725
    Epoch : 11/50....... Train Loss : 0.666 Valid Loss : 0.749 Valid Accuracy : 0.747
    Epoch : 12/50....... Train Loss : 0.645 Valid Loss : 0.813 Valid Accuracy : 0.723
    trigger :  1
    Epoch : 13/50....... Train Loss : 0.625 Valid Loss : 0.729 Valid Accuracy : 0.752
    Epoch : 14/50....... Train Loss : 0.617 Valid Loss : 0.734 Valid Accuracy : 0.754
    trigger :  1
    Epoch : 15/50....... Train Loss : 0.600 Valid Loss : 0.804 Valid Accuracy : 0.720
    trigger :  2
    Epoch : 16/50....... Train Loss : 0.587 Valid Loss : 0.706 Valid Accuracy : 0.766
    Epoch : 17/50....... Train Loss : 0.567 Valid Loss : 0.880 Valid Accuracy : 0.704
    trigger :  1
    Epoch : 18/50....... Train Loss : 0.568 Valid Loss : 0.682 Valid Accuracy : 0.771
    Epoch : 19/50....... Train Loss : 0.555 Valid Loss : 0.760 Valid Accuracy : 0.741
    trigger :  1
    Epoch : 20/50....... Train Loss : 0.537 Valid Loss : 0.696 Valid Accuracy : 0.770
    trigger :  2
    Epoch : 21/50....... Train Loss : 0.527 Valid Loss : 0.691 Valid Accuracy : 0.769
    trigger :  3
    Epoch : 22/50....... Train Loss : 0.523 Valid Loss : 0.667 Valid Accuracy : 0.779
    Epoch : 23/50....... Train Loss : 0.509 Valid Loss : 0.707 Valid Accuracy : 0.759
    trigger :  1
    Epoch : 24/50....... Train Loss : 0.512 Valid Loss : 0.715 Valid Accuracy : 0.766
    trigger :  2
    Epoch : 25/50....... Train Loss : 0.494 Valid Loss : 0.647 Valid Accuracy : 0.786
    Epoch : 26/50....... Train Loss : 0.495 Valid Loss : 0.672 Valid Accuracy : 0.776
    trigger :  1
    Epoch : 27/50....... Train Loss : 0.484 Valid Loss : 0.652 Valid Accuracy : 0.779
    trigger :  2
    Epoch : 28/50....... Train Loss : 0.485 Valid Loss : 0.713 Valid Accuracy : 0.765
    trigger :  3
    Epoch : 29/50....... Train Loss : 0.469 Valid Loss : 0.671 Valid Accuracy : 0.771
    trigger :  4
    Epoch : 30/50....... Train Loss : 0.469 Valid Loss : 0.653 Valid Accuracy : 0.784
    trigger :  5
    Epoch 00030: reducing learning rate of group 0 to 1.0000e-04.
    Epoch : 31/50....... Train Loss : 0.412 Valid Loss : 0.612 Valid Accuracy : 0.795
    Epoch : 32/50....... Train Loss : 0.401 Valid Loss : 0.619 Valid Accuracy : 0.793
    trigger :  1
    Epoch : 33/50....... Train Loss : 0.384 Valid Loss : 0.601 Valid Accuracy : 0.801
    Epoch : 34/50....... Train Loss : 0.381 Valid Loss : 0.614 Valid Accuracy : 0.794
    trigger :  1
    Epoch : 35/50....... Train Loss : 0.372 Valid Loss : 0.593 Valid Accuracy : 0.803
    Epoch : 36/50....... Train Loss : 0.368 Valid Loss : 0.601 Valid Accuracy : 0.797
    trigger :  1
    Epoch : 37/50....... Train Loss : 0.365 Valid Loss : 0.596 Valid Accuracy : 0.800
    trigger :  2
    Epoch : 38/50....... Train Loss : 0.361 Valid Loss : 0.586 Valid Accuracy : 0.807
    Epoch : 39/50....... Train Loss : 0.358 Valid Loss : 0.580 Valid Accuracy : 0.811
    Epoch : 40/50....... Train Loss : 0.357 Valid Loss : 0.588 Valid Accuracy : 0.802
    trigger :  1
    Epoch : 41/50....... Train Loss : 0.356 Valid Loss : 0.591 Valid Accuracy : 0.802
    trigger :  2
    Epoch : 42/50....... Train Loss : 0.350 Valid Loss : 0.587 Valid Accuracy : 0.805
    trigger :  3
    Epoch : 43/50....... Train Loss : 0.350 Valid Loss : 0.589 Valid Accuracy : 0.807
    trigger :  4
    Epoch : 44/50....... Train Loss : 0.349 Valid Loss : 0.621 Valid Accuracy : 0.789
    trigger :  5
    Epoch 00044: reducing learning rate of group 0 to 1.0000e-05.
    Epoch : 45/50....... Train Loss : 0.344 Valid Loss : 0.592 Valid Accuracy : 0.805
    trigger :  6
    Epoch : 46/50....... Train Loss : 0.340 Valid Loss : 0.589 Valid Accuracy : 0.801
    trigger :  7
    Epoch : 47/50....... Train Loss : 0.341 Valid Loss : 0.583 Valid Accuracy : 0.807
    trigger :  8
    Early Stopping !!!
    Training loop is finished !!
    


```python
%load_ext tensorboard
```


```python
%tensorboard --logdir=runs
```


    Output hidden; open in https://colab.research.google.com to view.


## 7. 모델 예측


```python
# testloader에서 미니 배치 가져오기
test_iter = iter(testloader)
images, labels = next(test_iter)
images, labels = images.to(device), labels.to(device)
print(images.size(), labels.size())

# random한 index로 이미지 한장 준비하기
rnd_idx = 10
print(images[rnd_idx].shape, labels[rnd_idx])
```

    torch.Size([16, 3, 32, 32]) torch.Size([16])
    torch.Size([3, 32, 32]) tensor(0, device='cuda:0')
    


```python
images[rnd_idx].shape
```




    torch.Size([3, 32, 32])




```python
# not Flatten!
# flattend_img = images[rnd_idx].view(1, 784)

# 준비된 이미지로 예측하기
model.eval()
with torch.no_grad():
  logit = model(images[rnd_idx].unsqueeze(0)) # model.forward()에서는 배치가 적용된 4차원 입력 기대

pred = logit.max(dim=1)[1]
print(pred == labels[rnd_idx]) # True : 잘 예측
```

    tensor([False], device='cuda:0')
    


```python
print("pred:", pred, "labels:", labels[rnd_idx])
print(labels_map[pred.cpu().item()], labels_map[labels[rnd_idx].cpu().item()])
plt.imshow(images[rnd_idx].permute(1, 2, 0).cpu())
```

    pred: tensor([5], device='cuda:0') labels: tensor(0, device='cuda:0')
    dog airplane
    




    <matplotlib.image.AxesImage at 0x7f6fae2e0490>




    
![png](/assets/images/CNN 3 (CIFAR10 (1) Dataset)/output_60_2.png)
    


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
      # 0. Data를 GPU로 보내기
      images, labels = images.to(device), labels.to(device)

      # 1. 입력 데이터 준비
      # not Flatten
      # images.resize_(images.size()[0], 784)
      
      # 2. 전방향(Forward) 예측
      logit = model(images) # 예측 점수
      _, preds = torch.max(logit, 1) # 배치에 대한 최종 예측
      # preds = logit.max(dim=1)[1] 
      correct += int((preds == labels).sum()) # 배치치 중 맞은 것의 개수가 correct에 누적
      total += labels.shape[0] # 배치 사이즈만큼씩 total에 누적

      loss = loss_fn(logit, labels)
      test_loss += loss.item() # tensor에서 값을 꺼내와서, 배치의 loss 평균값을 valid_loss에 누적

    test_accuracy = correct / total
   
  print('Test Loss : {:.3f}'.format(test_loss/len(testloader)), 
        'Test Accuracy : {:.3f}'.format(test_accuracy))

model.eval()
evaluation(model, testloader, loss_fn)  
```

    Test Loss : 0.590 Test Accuracy : 0.806
    

## 9. 모델 저장


```python
# 모델을 저장하는 이유
# 1. 예측을 할 때마다 훈련시키는것은 비효율적
# 2. 기존 훈련 결과에 이어서 학습을 하고자 할 때

# 파이토치에서 모델 저장하기
# https://pytorch.org/tutorials/beginner/saving_loading_models.html
```


```python
# 현재 모델에 저장되어 있는 모델 파라미터터
model.state_dict().keys()
```




    odict_keys(['conv_block1.0.weight', 'conv_block1.0.bias', 'conv_block1.1.weight', 'conv_block1.1.bias', 'conv_block1.1.running_mean', 'conv_block1.1.running_var', 'conv_block1.1.num_batches_tracked', 'conv_block2.0.weight', 'conv_block2.0.bias', 'conv_block2.1.weight', 'conv_block2.1.bias', 'conv_block2.1.running_mean', 'conv_block2.1.running_var', 'conv_block2.1.num_batches_tracked', 'conv_block3.0.weight', 'conv_block3.0.bias', 'conv_block3.1.weight', 'conv_block3.1.bias', 'conv_block3.1.running_mean', 'conv_block3.1.running_var', 'conv_block3.1.num_batches_tracked', 'linear1.weight', 'linear1.bias', 'batch_norm.weight', 'batch_norm.bias', 'batch_norm.running_mean', 'batch_norm.running_var', 'batch_norm.num_batches_tracked', 'linear2.weight', 'linear2.bias'])




```python
torch.save(model.state_dict(), 'last_checkpoint.pth')
```


```python
# 시간이 흐른뒤 다시 모델 가져오기
last_state_dict = torch.load('last_checkpoint.pth')
```


```python
last_state_dict.keys()
```




    odict_keys(['conv_block1.0.weight', 'conv_block1.0.bias', 'conv_block1.1.weight', 'conv_block1.1.bias', 'conv_block1.1.running_mean', 'conv_block1.1.running_var', 'conv_block1.1.num_batches_tracked', 'conv_block2.0.weight', 'conv_block2.0.bias', 'conv_block2.1.weight', 'conv_block2.1.bias', 'conv_block2.1.running_mean', 'conv_block2.1.running_var', 'conv_block2.1.num_batches_tracked', 'conv_block3.0.weight', 'conv_block3.0.bias', 'conv_block3.1.weight', 'conv_block3.1.bias', 'conv_block3.1.running_mean', 'conv_block3.1.running_var', 'conv_block3.1.num_batches_tracked', 'linear1.weight', 'linear1.bias', 'batch_norm.weight', 'batch_norm.bias', 'batch_norm.running_mean', 'batch_norm.running_var', 'batch_norm.num_batches_tracked', 'linear2.weight', 'linear2.bias'])




```python
# 읽어들인 모델 파라미터는 모델 아키텍처에 연결을 시켜줘야 함
# load_state_dict() 사용
last_model = CIFAR10_CNN()
last_model.to(device)
last_model.load_state_dict(last_state_dict)
```




    <All keys matched successfully>




```python
last_model.eval()
evaluation(last_model, testloader, loss_fn)  
```

    Test Loss : 0.594 Test Accuracy : 0.803
    


```python
# valid loss or accuracy 기준 best model
best_state_dict = torch.load('best_checkpoint.pth')
best_model = CIFAR10_CNN()
best_model.to(device)
best_model.load_state_dict(best_state_dict)
```




    <All keys matched successfully>




```python
best_model.eval()
evaluation(best_model, testloader, loss_fn)
```

    Test Loss : 0.593 Test Accuracy : 0.804
    


```python
# best_state_dict['conv_block1.0.weight']
```


```python
# last_state_dict['conv_block1.0.weight']
```


```python

```
