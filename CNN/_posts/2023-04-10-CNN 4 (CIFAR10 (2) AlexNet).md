---
tag: [Deep Learning, 딥러닝, pytorch, 파이토치, CNN, CIFAR10, AlexNet]
toc: true
toc_sticky: true
toc_label: 목차
author_profile: false

---

# 신경망 학습 (CIFAR10 in Pytorch)


```python
from IPython.display import Image
Image('./images/cifar10.png', width=600)
```




    
![png](/assets/images/2023-04-10-CNN 4 (CIFAR10 (2) AlexNet)/output_1_0.png)
    




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

    Mon Apr 10 05:26:18 2023       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   51C    P8    10W /  70W |      0MiB / 15360MiB |      0%      Default |
    |                               |                      |                  N/A |
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
    

    100%|██████████| 170498071/170498071 [00:10<00:00, 15742754.02it/s]
    

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


    
![png](/assets/images/2023-04-10-CNN 4 (CIFAR10 (2) AlexNet)/output_20_0.png)
    


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
Image('./images/알렉스넷.jpg', width=700)
```




    
![jpeg](/assets/images/2023-04-10-CNN 4 (CIFAR10 (2) AlexNet)/output_29_0.jpg)
    




```python
Image('./images/알렉스넷2.jpg')
```




    
![jpeg](/assets/images/2023-04-10-CNN 4 (CIFAR10 (2) AlexNet)/output_30_0.jpg)
    




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

**완전 연결망과 CNN망과의 차이점**
- 지역 연산
- 가중치 공유(적은 파라미터)
- 평행 이동 불변성


```python
images.shape, labels.shape
```




    (torch.Size([16, 3, 32, 32]), torch.Size([16]))




```python
Image('./images/알렉스넷2.jpg', width=600)
```




    
![jpeg](/assets/images/2023-04-10-CNN 4 (CIFAR10 (2) AlexNet)/output_36_0.jpg)
    



**conv block 별 사이즈 확인**


```python
conv_block1 = nn.Sequential(
                            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(num_features=96),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=3, stride=2)
                            ) # [16, 96, 15, 15]
conv_block1_out = conv_block1(images)                                     
conv_block1_out.shape
```




    torch.Size([16, 96, 15, 15])




```python
conv_block2 = nn.Sequential(
                            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(num_features=256),                                      
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=3, stride=2)
                            ) # [16, 256, 7, 7]
conv_block2_out = conv_block2(conv_block1_out)                                     
conv_block2_out.shape                                     
```




    torch.Size([16, 256, 7, 7])




```python
conv_block3 = nn.Sequential(
                            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
                            nn.BatchNorm2d(num_features=384),                                      
                            nn.ReLU(),                                      
                            ) # [16, 384, 7, 7] 
conv_block3_out = conv_block3(conv_block2_out)                                     
conv_block3_out.shape                                       
```




    torch.Size([16, 384, 7, 7])




```python
conv_block4 = nn.Sequential(
                            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
                            nn.BatchNorm2d(num_features=384),                                      
                            nn.ReLU(),                                      
                            ) # [16, 384, 7, 7] 
conv_block4_out = conv_block4(conv_block3_out)                                     
conv_block4_out.shape  
```




    torch.Size([16, 384, 7, 7])




```python
conv_block5 = nn.Sequential(
                            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
                            nn.BatchNorm2d(num_features=256),                                      
                            nn.ReLU(),   
                            nn.MaxPool2d(kernel_size=3, stride=2)                                   
                            ) # [16, 256, 3, 3]
conv_block5_out = conv_block5(conv_block4_out)                                     
conv_block5_out.shape  
```




    torch.Size([16, 256, 3, 3])




```python
class AlexNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_block1 = nn.Sequential(
                                      nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(num_features=96),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=3, stride=2)
                                     ) # [16, 96, 15, 15]
    self.conv_block2 = nn.Sequential(
                                      nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(num_features=256),                                      
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=3, stride=2)
                                     ) # [16, 256, 7, 7]

    self.conv_block3 = nn.Sequential(
                                      nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(num_features=384), 
                                      nn.Dropout(0.1),                                    
                                      nn.ReLU(),                                      
                                     ) # [16, 384, 7, 7]     

    self.conv_block4 = nn.Sequential(
                                      nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(num_features=384),  
                                      nn.Dropout(0.3),                                    
                                      nn.ReLU(),                                      
                                     ) # [16, 384, 7, 7]   

    self.conv_block5 = nn.Sequential(
                                      nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(num_features=256),
                                      nn.Dropout(0.1), 
                                      nn.ReLU(),   
                                      nn.MaxPool2d(kernel_size=3, stride=2)                                   
                                     ) # [16, 256, 3, 3]                                                                                                       

    self.linear1 = nn.Linear(in_features=256*3*3, out_features=512)
    self.batch_norm = nn.BatchNorm1d(num_features=512)
    self.linear2 = nn.Linear(in_features=512, out_features=10)

  def forward(self, x):
    x = self.conv_block1(x) 
    x = self.conv_block2(x) 
    x = self.conv_block3(x) 
    x = self.conv_block4(x) 
    x = self.conv_block5(x) 
    
    # reshape할 형상 : (batch_size x 256*3*3)
    # x = x.view(-1, 256*3*3) # option 1 : view
    x = torch.flatten(x, 1) # option 2 : flatten 
    # x = x.reshape(x.shape[0], -1) # option 3 : reshape
    x = F.dropout(x, 0.3)
    x = self.linear1(x)
    x = self.batch_norm(x)
    x = F.dropout(x, 0.1)
    x = F.relu(x)
    x = self.linear2(x)
    return x
```


```python
model = AlexNet()
model.to(device)
model
```




    AlexNet(
      (conv_block1): Sequential(
        (0): Conv2d(3, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (conv_block2): Sequential(
        (0): Conv2d(96, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (conv_block3): Sequential(
        (0): Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): Dropout(p=0.1, inplace=False)
        (3): ReLU()
      )
      (conv_block4): Sequential(
        (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): Dropout(p=0.3, inplace=False)
        (3): ReLU()
      )
      (conv_block5): Sequential(
        (0): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): Dropout(p=0.1, inplace=False)
        (3): ReLU()
        (4): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (linear1): Linear(in_features=2304, out_features=512, bias=True)
      (batch_norm): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (linear2): Linear(in_features=512, out_features=10, bias=True)
    )




```python
for name, parameter in model.named_parameters():
  print(name, parameter.size())
```

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
                Conv2d-1           [-1, 96, 32, 32]           2,688
           BatchNorm2d-2           [-1, 96, 32, 32]             192
                  ReLU-3           [-1, 96, 32, 32]               0
             MaxPool2d-4           [-1, 96, 15, 15]               0
                Conv2d-5          [-1, 256, 15, 15]         221,440
           BatchNorm2d-6          [-1, 256, 15, 15]             512
                  ReLU-7          [-1, 256, 15, 15]               0
             MaxPool2d-8            [-1, 256, 7, 7]               0
                Conv2d-9            [-1, 384, 7, 7]         885,120
          BatchNorm2d-10            [-1, 384, 7, 7]             768
              Dropout-11            [-1, 384, 7, 7]               0
                 ReLU-12            [-1, 384, 7, 7]               0
               Conv2d-13            [-1, 384, 7, 7]       1,327,488
          BatchNorm2d-14            [-1, 384, 7, 7]             768
              Dropout-15            [-1, 384, 7, 7]               0
                 ReLU-16            [-1, 384, 7, 7]               0
               Conv2d-17            [-1, 256, 7, 7]         884,992
          BatchNorm2d-18            [-1, 256, 7, 7]             512
              Dropout-19            [-1, 256, 7, 7]               0
                 ReLU-20            [-1, 256, 7, 7]               0
            MaxPool2d-21            [-1, 256, 3, 3]               0
               Linear-22                  [-1, 512]       1,180,160
          BatchNorm1d-23                  [-1, 512]           1,024
               Linear-24                   [-1, 10]           5,130
    ================================================================
    Total params: 4,510,794
    Trainable params: 4,510,794
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 5.39
    Params size (MB): 17.21
    Estimated Total Size (MB): 22.60
    ----------------------------------------------------------------
    


```python
# 첫번째 conv layer의 모델 파라미터 수
# 필터수 x (필터) + bias
96 * (3*3*3) + 96
```




    2688




```python
# 마지막 출력 feature map의 사이즈
256 * 3 * 3
```




    2304




```python
# linear 1 layer
2304 * 512 + 512
```




    1180160




```python
# linear 2 layer
512 * 10 + 10
```




    5130



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
epochs = 55
%time train_loop(model, trainloader, loss_fn, epochs, optimizer)
writer.close()
```

    Epoch : 1/55....... Train Loss : 1.341 Valid Loss : 1.158 Valid Accuracy : 0.588
    Epoch : 2/55....... Train Loss : 0.928 Valid Loss : 0.802 Valid Accuracy : 0.717
    Epoch : 3/55....... Train Loss : 0.767 Valid Loss : 0.686 Valid Accuracy : 0.762
    Epoch : 4/55....... Train Loss : 0.664 Valid Loss : 0.734 Valid Accuracy : 0.754
    trigger :  1
    Epoch : 5/55....... Train Loss : 0.578 Valid Loss : 0.803 Valid Accuracy : 0.724
    trigger :  2
    Epoch : 6/55....... Train Loss : 0.500 Valid Loss : 0.600 Valid Accuracy : 0.795
    Epoch : 7/55....... Train Loss : 0.436 Valid Loss : 0.550 Valid Accuracy : 0.820
    Epoch : 8/55....... Train Loss : 0.387 Valid Loss : 0.515 Valid Accuracy : 0.830
    Epoch : 9/55....... Train Loss : 0.341 Valid Loss : 0.534 Valid Accuracy : 0.827
    trigger :  1
    Epoch : 10/55....... Train Loss : 0.295 Valid Loss : 0.491 Valid Accuracy : 0.837
    Epoch : 11/55....... Train Loss : 0.261 Valid Loss : 0.560 Valid Accuracy : 0.819
    trigger :  1
    Epoch : 12/55....... Train Loss : 0.231 Valid Loss : 0.561 Valid Accuracy : 0.828
    trigger :  2
    Epoch : 13/55....... Train Loss : 0.208 Valid Loss : 0.564 Valid Accuracy : 0.828
    trigger :  3
    Epoch : 14/55....... Train Loss : 0.184 Valid Loss : 0.570 Valid Accuracy : 0.831
    trigger :  4
    Epoch : 15/55....... Train Loss : 0.161 Valid Loss : 0.580 Valid Accuracy : 0.839
    trigger :  5
    Epoch 00015: reducing learning rate of group 0 to 1.0000e-04.
    Epoch : 16/55....... Train Loss : 0.084 Valid Loss : 0.487 Valid Accuracy : 0.864
    Epoch : 17/55....... Train Loss : 0.058 Valid Loss : 0.502 Valid Accuracy : 0.868
    trigger :  1
    Epoch : 18/55....... Train Loss : 0.052 Valid Loss : 0.500 Valid Accuracy : 0.869
    trigger :  2
    Epoch : 19/55....... Train Loss : 0.044 Valid Loss : 0.528 Valid Accuracy : 0.868
    trigger :  3
    Epoch : 20/55....... Train Loss : 0.040 Valid Loss : 0.533 Valid Accuracy : 0.869
    trigger :  4
    Epoch : 21/55....... Train Loss : 0.034 Valid Loss : 0.551 Valid Accuracy : 0.868
    trigger :  5
    Epoch 00021: reducing learning rate of group 0 to 1.0000e-05.
    Epoch : 22/55....... Train Loss : 0.028 Valid Loss : 0.536 Valid Accuracy : 0.874
    trigger :  6
    Epoch : 23/55....... Train Loss : 0.028 Valid Loss : 0.559 Valid Accuracy : 0.870
    trigger :  7
    Epoch : 24/55....... Train Loss : 0.028 Valid Loss : 0.544 Valid Accuracy : 0.872
    trigger :  8
    Early Stopping !!!
    Training loop is finished !!
    CPU times: user 10min 55s, sys: 5.11 s, total: 11min
    Wall time: 10min 57s
    


```python
# %load_ext tensorboard
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

    tensor([True], device='cuda:0')
    


```python
print("pred:", pred, "labels:", labels[rnd_idx])
print(labels_map[pred.cpu().item()], labels_map[labels[rnd_idx].cpu().item()])
plt.imshow(images[rnd_idx].permute(1, 2, 0).cpu())
```

    pred: tensor([0], device='cuda:0') labels: tensor(0, device='cuda:0')
    airplane airplane
    




    <matplotlib.image.AxesImage at 0x7f93c5bac4f0>




    
![png](/assets/images/2023-04-10-CNN 4 (CIFAR10 (2) AlexNet)/output_67_2.png)
    


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

    Test Loss : 0.555 Test Accuracy : 0.864
    

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




    odict_keys(['conv_block1.0.weight', 'conv_block1.0.bias', 'conv_block1.1.weight', 'conv_block1.1.bias', 'conv_block1.1.running_mean', 'conv_block1.1.running_var', 'conv_block1.1.num_batches_tracked', 'conv_block2.0.weight', 'conv_block2.0.bias', 'conv_block2.1.weight', 'conv_block2.1.bias', 'conv_block2.1.running_mean', 'conv_block2.1.running_var', 'conv_block2.1.num_batches_tracked', 'conv_block3.0.weight', 'conv_block3.0.bias', 'conv_block3.1.weight', 'conv_block3.1.bias', 'conv_block3.1.running_mean', 'conv_block3.1.running_var', 'conv_block3.1.num_batches_tracked', 'conv_block4.0.weight', 'conv_block4.0.bias', 'conv_block4.1.weight', 'conv_block4.1.bias', 'conv_block4.1.running_mean', 'conv_block4.1.running_var', 'conv_block4.1.num_batches_tracked', 'conv_block5.0.weight', 'conv_block5.0.bias', 'conv_block5.1.weight', 'conv_block5.1.bias', 'conv_block5.1.running_mean', 'conv_block5.1.running_var', 'conv_block5.1.num_batches_tracked', 'linear1.weight', 'linear1.bias', 'batch_norm.weight', 'batch_norm.bias', 'batch_norm.running_mean', 'batch_norm.running_var', 'batch_norm.num_batches_tracked', 'linear2.weight', 'linear2.bias'])




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




    odict_keys(['conv_block1.0.weight', 'conv_block1.0.bias', 'conv_block1.1.weight', 'conv_block1.1.bias', 'conv_block1.1.running_mean', 'conv_block1.1.running_var', 'conv_block1.1.num_batches_tracked', 'conv_block2.0.weight', 'conv_block2.0.bias', 'conv_block2.1.weight', 'conv_block2.1.bias', 'conv_block2.1.running_mean', 'conv_block2.1.running_var', 'conv_block2.1.num_batches_tracked', 'conv_block3.0.weight', 'conv_block3.0.bias', 'conv_block3.1.weight', 'conv_block3.1.bias', 'conv_block3.1.running_mean', 'conv_block3.1.running_var', 'conv_block3.1.num_batches_tracked', 'conv_block4.0.weight', 'conv_block4.0.bias', 'conv_block4.1.weight', 'conv_block4.1.bias', 'conv_block4.1.running_mean', 'conv_block4.1.running_var', 'conv_block4.1.num_batches_tracked', 'conv_block5.0.weight', 'conv_block5.0.bias', 'conv_block5.1.weight', 'conv_block5.1.bias', 'conv_block5.1.running_mean', 'conv_block5.1.running_var', 'conv_block5.1.num_batches_tracked', 'linear1.weight', 'linear1.bias', 'batch_norm.weight', 'batch_norm.bias', 'batch_norm.running_mean', 'batch_norm.running_var', 'batch_norm.num_batches_tracked', 'linear2.weight', 'linear2.bias'])




```python
# 읽어들인 모델 파라미터는 모델 아키텍처에 연결을 시켜줘야 함
# load_state_dict() 사용
last_model = AlexNet()
last_model.to(device)
last_model.load_state_dict(last_state_dict)
```




    <All keys matched successfully>




```python
last_model.eval()
evaluation(last_model, testloader, loss_fn)  
```

    Test Loss : 0.562 Test Accuracy : 0.865
    


```python
# valid loss or accuracy 기준 best model
best_state_dict = torch.load('best_checkpoint.pth')
best_model = AlexNet()
best_model.to(device)
best_model.load_state_dict(best_state_dict)
```




    <All keys matched successfully>




```python
best_model.eval()
evaluation(best_model, testloader, loss_fn)
```

    Test Loss : 0.549 Test Accuracy : 0.869
    


```python
#best_state_dict['conv_block1.0.weight']
```


```python
#last_state_dict['conv_block1.0.weight']
```


```python

```
