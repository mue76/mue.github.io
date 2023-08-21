---
tag: [Deep Learning, 딥러닝, pytorch, 파이토치, CNN]
toc: true
toc_sticky: true
toc_label: 목차
author_profile: false

---
# 신경망 학습 (Fashion MNIST in Pytorch)


```python
from IPython.display import Image
Image('./images/fashion-mnist-sprite.png', width=600)
```




    
![png](/assets/images/2023-04-10-CNN 2 (Fashion MNIST (2) CNN 성능 개선)/output_1_0.png)
    




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


```python
!nvidia-smi
```

    Fri Apr  7 06:40:29 2023       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   54C    P8    10W /  70W |      0MiB / 15360MiB |      0%      Default |
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
trainset = datasets.FashionMNIST('./datasets/', download=True, train=True, transform = transforms.ToTensor())
testset = datasets.FashionMNIST('./datasets/', download=True, train=False, transform = transforms.ToTensor())
```

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to ./datasets/FashionMNIST/raw/train-images-idx3-ubyte.gz
    

    100%|██████████| 26421880/26421880 [00:01<00:00, 15166229.33it/s]
    

    Extracting ./datasets/FashionMNIST/raw/train-images-idx3-ubyte.gz to ./datasets/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to ./datasets/FashionMNIST/raw/train-labels-idx1-ubyte.gz
    

    100%|██████████| 29515/29515 [00:00<00:00, 269254.52it/s]
    

    Extracting ./datasets/FashionMNIST/raw/train-labels-idx1-ubyte.gz to ./datasets/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to ./datasets/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
    

    100%|██████████| 4422102/4422102 [00:00<00:00, 5091846.11it/s]
    

    Extracting ./datasets/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to ./datasets/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to ./datasets/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
    

    100%|██████████| 5148/5148 [00:00<00:00, 6036420.74it/s]

    Extracting ./datasets/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to ./datasets/FashionMNIST/raw
    
    

    
    


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
    <class 'torchvision.datasets.mnist.FashionMNIST'> 10000
    


```python
# 0번째 샘플에 2개의 원소가 있는데, 그중 첫번째 원소는 이미지, 두번째 원소는 정답
# 그러나 파이토치로 읽어들인 이미지 텐서의 형상이 channels * height * width 임
# 그에 비해 opencv, matplotlib으로 읽어들인 이미지 array의 형상은 height * width * channels
print(trainset[0][0].size(), trainset[0][1])
```

    torch.Size([1, 28, 28]) 1
    

## 2. 데이터 시각화


```python
labels_map = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt',
              7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'}

figure, axes = plt.subplots(nrows=4, ncols=8, figsize=(16, 8))
axes = axes.flatten()

for i in range(32):
  rand_i = np.random.randint(0, len(trainset))
  image, label= trainset[rand_i][0], trainset[rand_i][1]
  axes[i].axis('off')
  axes[i].imshow(image.squeeze(), cmap='gray')
  axes[i].set_title(labels_map[label])
```


    
![png](/assets/images/2023-04-10-CNN 2 (Fashion MNIST (2) CNN 성능 개선)/output_12_0.png)
    


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
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True) # 훈련용 50000개의 데이터를 100개씩 준비
validloader = DataLoader(validset, batch_size=batch_size, shuffle=False) # 검증용 10000개의 데이터를 100개씩 준비
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False) # 테스트용 10000개의 데이터를 100개씩 준비
```


```python
print(type(trainloader), len(trainloader))
print(type(validloader), len(validloader))
print(type(testloader), len(testloader))
```

    <class 'torch.utils.data.dataloader.DataLoader'> 3125
    <class 'torch.utils.data.dataloader.DataLoader'> 625
    <class 'torch.utils.data.dataloader.DataLoader'> 625
    


```python
train_iter = iter(trainloader)
images, labels = next(train_iter)
images.size(), labels.size()
```




    (torch.Size([16, 1, 28, 28]), torch.Size([16]))



## 4. 모델 생성


```python
from IPython.display import Image
Image('./images/cnn architecture 2.png', width=700)
```




    
![png](/assets/images/2023-04-10-CNN 2 (Fashion MNIST (2) CNN 성능 개선)/output_20_0.png)
    




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


```python
class FMnist_CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_block1 = nn.Sequential(
                                      nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(num_features=32),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2)
                                     ) # batch_size x 32 x 14 x 14
    self.conv_block2 = nn.Sequential(
                                      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(num_features=64),
                                      nn.Dropout(0.2),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2)
                                     ) # batch_size x 64 x 7 x 7

    self.conv_block3 = nn.Sequential(
                                      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(num_features=128),
                                      nn.Dropout(0.4),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2)
                                     ) # batch_size x 128 x 3 x 3                                   

    self.linear1 = nn.Linear(in_features=128*3*3, out_features=128)
    self.batch_norm = nn.BatchNorm1d(num_features=128)
    self.linear2 = nn.Linear(in_features=128, out_features=10)

  def forward(self, x):
    x = self.conv_block1(x) # batch_size x 32 x 14 x 14
    x = self.conv_block2(x) # batch_size x 64 x 7 x 7
    x = self.conv_block3(x) # batch_size x 64 x 7 x 7
    x = x.view(-1, 128*3*3) # flatten
    x = self.linear1(x)
    x = self.batch_norm(x)
    x = F.dropout(x, 0.5)
    x = F.relu(x)
    x = self.linear2(x)
    return x
```


```python
model = FMnist_CNN()
model.to(device)
model
```




    FMnist_CNN(
      (conv_block1): Sequential(
        (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
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
      (linear1): Linear(in_features=1152, out_features=128, bias=True)
      (batch_norm): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (linear2): Linear(in_features=128, out_features=10, bias=True)
    )




```python
for name, parameter in model.named_parameters():
  print(name, parameter.size())
```

    conv_block1.0.weight torch.Size([32, 1, 3, 3])
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
    linear1.weight torch.Size([128, 1152])
    linear1.bias torch.Size([128])
    batch_norm.weight torch.Size([128])
    batch_norm.bias torch.Size([128])
    linear2.weight torch.Size([10, 128])
    linear2.bias torch.Size([10])
    

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
summary(model, (1, 28, 28))
```

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 32, 28, 28]             320
           BatchNorm2d-2           [-1, 32, 28, 28]              64
                  ReLU-3           [-1, 32, 28, 28]               0
             MaxPool2d-4           [-1, 32, 14, 14]               0
                Conv2d-5           [-1, 64, 14, 14]          18,496
           BatchNorm2d-6           [-1, 64, 14, 14]             128
               Dropout-7           [-1, 64, 14, 14]               0
                  ReLU-8           [-1, 64, 14, 14]               0
             MaxPool2d-9             [-1, 64, 7, 7]               0
               Conv2d-10            [-1, 128, 7, 7]          73,856
          BatchNorm2d-11            [-1, 128, 7, 7]             256
              Dropout-12            [-1, 128, 7, 7]               0
                 ReLU-13            [-1, 128, 7, 7]               0
            MaxPool2d-14            [-1, 128, 3, 3]               0
               Linear-15                  [-1, 128]         147,584
          BatchNorm1d-16                  [-1, 128]             256
               Linear-17                   [-1, 10]           1,290
    ================================================================
    Total params: 242,250
    Trainable params: 242,250
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 1.23
    Params size (MB): 0.92
    Estimated Total Size (MB): 2.16
    ----------------------------------------------------------------
    


```python
32 * 3 * 3 + 32
```




    320




```python
64 * (32 * 3 * 3) + 64
```




    18496




```python
3136 * 128 + 128
```




    401536




```python
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
      # images, labels : (torch.Size([16, 1, 28, 28]), torch.Size([16]))
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
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
```


```python
50000/16
```




    3125.0




```python
from copy import deepcopy
```


```python
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
      # images, labels : (torch.Size([16, 1, 28, 28]), torch.Size([16]))
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
```

    Epoch : 1/50....... Train Loss : 0.490 Valid Loss : 0.377 Valid Accuracy : 0.864
    Epoch : 2/50....... Train Loss : 0.352 Valid Loss : 0.335 Valid Accuracy : 0.880
    Epoch : 3/50....... Train Loss : 0.318 Valid Loss : 0.289 Valid Accuracy : 0.898
    Epoch : 4/50....... Train Loss : 0.292 Valid Loss : 0.287 Valid Accuracy : 0.898
    Epoch : 5/50....... Train Loss : 0.276 Valid Loss : 0.257 Valid Accuracy : 0.914
    Epoch : 6/50....... Train Loss : 0.265 Valid Loss : 0.242 Valid Accuracy : 0.918
    Epoch : 7/50....... Train Loss : 0.254 Valid Loss : 0.246 Valid Accuracy : 0.916
    trigger :  1
    Epoch : 8/50....... Train Loss : 0.249 Valid Loss : 0.248 Valid Accuracy : 0.917
    trigger :  2
    Epoch : 9/50....... Train Loss : 0.238 Valid Loss : 0.244 Valid Accuracy : 0.918
    trigger :  3
    Epoch : 10/50....... Train Loss : 0.231 Valid Loss : 0.234 Valid Accuracy : 0.917
    Epoch : 11/50....... Train Loss : 0.225 Valid Loss : 0.229 Valid Accuracy : 0.925
    Epoch : 12/50....... Train Loss : 0.216 Valid Loss : 0.232 Valid Accuracy : 0.919
    trigger :  1
    Epoch : 13/50....... Train Loss : 0.214 Valid Loss : 0.218 Valid Accuracy : 0.925
    Epoch : 14/50....... Train Loss : 0.211 Valid Loss : 0.243 Valid Accuracy : 0.918
    trigger :  1
    Epoch : 15/50....... Train Loss : 0.204 Valid Loss : 0.232 Valid Accuracy : 0.917
    trigger :  2
    Epoch : 16/50....... Train Loss : 0.203 Valid Loss : 0.247 Valid Accuracy : 0.913
    trigger :  3
    Epoch : 17/50....... Train Loss : 0.197 Valid Loss : 0.253 Valid Accuracy : 0.912
    trigger :  4
    Epoch : 18/50....... Train Loss : 0.196 Valid Loss : 0.225 Valid Accuracy : 0.918
    trigger :  5
    Epoch 00018: reducing learning rate of group 0 to 1.0000e-04.
    Epoch : 19/50....... Train Loss : 0.174 Valid Loss : 0.212 Valid Accuracy : 0.928
    Epoch : 20/50....... Train Loss : 0.161 Valid Loss : 0.208 Valid Accuracy : 0.930
    Epoch : 21/50....... Train Loss : 0.156 Valid Loss : 0.202 Valid Accuracy : 0.931
    Epoch : 22/50....... Train Loss : 0.157 Valid Loss : 0.198 Valid Accuracy : 0.932
    Epoch : 23/50....... Train Loss : 0.151 Valid Loss : 0.200 Valid Accuracy : 0.933
    trigger :  1
    Epoch : 24/50....... Train Loss : 0.149 Valid Loss : 0.202 Valid Accuracy : 0.932
    trigger :  2
    Epoch : 25/50....... Train Loss : 0.148 Valid Loss : 0.201 Valid Accuracy : 0.929
    trigger :  3
    Epoch : 26/50....... Train Loss : 0.147 Valid Loss : 0.198 Valid Accuracy : 0.933
    Epoch : 27/50....... Train Loss : 0.144 Valid Loss : 0.201 Valid Accuracy : 0.932
    trigger :  1
    Epoch : 28/50....... Train Loss : 0.143 Valid Loss : 0.199 Valid Accuracy : 0.931
    trigger :  2
    Epoch : 29/50....... Train Loss : 0.144 Valid Loss : 0.195 Valid Accuracy : 0.935
    Epoch : 30/50....... Train Loss : 0.143 Valid Loss : 0.197 Valid Accuracy : 0.936
    trigger :  1
    Epoch : 31/50....... Train Loss : 0.140 Valid Loss : 0.197 Valid Accuracy : 0.932
    trigger :  2
    Epoch : 32/50....... Train Loss : 0.139 Valid Loss : 0.194 Valid Accuracy : 0.932
    Epoch : 33/50....... Train Loss : 0.140 Valid Loss : 0.203 Valid Accuracy : 0.929
    trigger :  1
    Epoch : 34/50....... Train Loss : 0.139 Valid Loss : 0.191 Valid Accuracy : 0.937
    Epoch : 35/50....... Train Loss : 0.135 Valid Loss : 0.192 Valid Accuracy : 0.937
    trigger :  1
    Epoch : 36/50....... Train Loss : 0.134 Valid Loss : 0.202 Valid Accuracy : 0.929
    trigger :  2
    Epoch : 37/50....... Train Loss : 0.134 Valid Loss : 0.201 Valid Accuracy : 0.931
    trigger :  3
    Epoch : 38/50....... Train Loss : 0.133 Valid Loss : 0.193 Valid Accuracy : 0.935
    trigger :  4
    Epoch : 39/50....... Train Loss : 0.132 Valid Loss : 0.200 Valid Accuracy : 0.930
    trigger :  5
    Epoch 00039: reducing learning rate of group 0 to 1.0000e-05.
    Epoch : 40/50....... Train Loss : 0.131 Valid Loss : 0.201 Valid Accuracy : 0.931
    trigger :  6
    Epoch : 41/50....... Train Loss : 0.129 Valid Loss : 0.207 Valid Accuracy : 0.929
    trigger :  7
    Epoch : 42/50....... Train Loss : 0.130 Valid Loss : 0.193 Valid Accuracy : 0.935
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



```python
writer.close()
```

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

    torch.Size([16, 1, 28, 28]) torch.Size([16])
    torch.Size([1, 28, 28]) tensor(4, device='cuda:0')
    


```python
images[rnd_idx].shape
```




    torch.Size([1, 28, 28])




```python
# not Flatten!
# flattend_img = images[rnd_idx].view(1, 784)

# 준비된 이미지로 예측하기
model.eval()
with torch.no_grad():
  logit = model(images[rnd_idx].unsqueeze(0))

pred = logit.max(dim=1)[1]
print(pred == labels[rnd_idx]) # True : 잘 예측
```

    tensor([True], device='cuda:0')
    


```python
plt.imshow(images[rnd_idx].squeeze().cpu(), cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7fa6a6469970>




    
![png](/assets/images/2023-04-10-CNN 2 (Fashion MNIST (2) CNN 성능 개선)/output_54_1.png)
    


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

    Test Loss : 0.214 Test Accuracy : 0.923
    

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
last_model = FMnist_CNN()
last_model.to(device)
last_model.load_state_dict(last_state_dict)
```




    <All keys matched successfully>




```python
last_model.eval()
evaluation(last_model, testloader, loss_fn)  
```

    Test Loss : 0.204 Test Accuracy : 0.929
    


```python
# valid loss or accuracy 기준 best model
best_state_dict = torch.load('best_checkpoint.pth')
best_model = FMnist_CNN()
best_model.to(device)
best_model.load_state_dict(best_state_dict)
```




    <All keys matched successfully>




```python
best_model.eval()
evaluation(best_model, testloader, loss_fn)
```

    Test Loss : 0.198 Test Accuracy : 0.929
    


```python
# best_state_dict['conv_block1.0.weight']
```


```python
# last_state_dict['conv_block1.0.weight']
```
