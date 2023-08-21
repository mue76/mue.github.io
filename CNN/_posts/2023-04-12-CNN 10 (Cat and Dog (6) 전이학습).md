---
tag: [Deep Learning, 딥러닝, pytorch, 파이토치, CNN, Cat and Dog Dataset, 전이학습]
toc: true
toc_sticky: true
toc_label: 목차
author_profile: false

---

# Cat and Dog Classifier


```python
from IPython.display import Image
Image('./images/dog_cat.png', width=600)
```




    
![png](/assets/images/2023-04-12-CNN 10 (Cat and Dog (6) 전이학습)/output_1_0.png)
    




```python
import torch # 파이토치 기본 라이브러리

# torchvision : 데이터셋, 모델 아키텍처, 컴퓨터 비전의 이미지 변환 기능 제공
import torchvision
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

    Wed Apr 12 04:55:14 2023       
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



## 0. 데이터 다운로드


```python
# option 1
# from google.colab import files

# files.upload()
```


```python
%pwd
```




    '/content'




```python
# option 2
# from google.colab import drive
# drive.mount('/content/drive')
# !cp '/content/drive/MyDrive/Classroom/Playdata 인공지능 28기/CNN/kaggle_catanddog.zip' './'
# !unzip -q kaggle_catanddog.zip -d catanddog/
```


```python
# option 3

# kaggle api를 사용할 수 있는 패키지 설치
!pip install kaggle

# kaggle.json upload
from google.colab import files
files.upload()

# permmision warning 방지
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# download
kaggle datasets download -d amananandrai/ag-news-classification-dataset
# unzip(압축풀기)
!unzip -q cat-and-dog.zip -d catanddog/
```


    Saving kaggle.json to kaggle.json
    Downloading cat-and-dog.zip to /content
     99% 215M/218M [00:07<00:00, 32.8MB/s]
    100% 218M/218M [00:07<00:00, 28.8MB/s]
    


```python
data_dir = './catanddog/'
```

## 1. 데이터 불러오기


```python
# Compose를 통해 원하는 전처리기를 차례대로 넣을 수 있음
transform = transforms.Compose([transforms.Resize([224, 224]), 
                                transforms.RandomHorizontalFlip(p=0.3),
                                transforms.ToTensor()])
```


```python
trainset = datasets.ImageFolder(data_dir+ 'training_set/training_set', transform = transform)
testset = datasets.ImageFolder(data_dir+ 'test_set/test_set', transform = transform)
```


```python
print(type(trainset), len(trainset))
print(type(testset), len(testset))
```

    <class 'torchvision.datasets.folder.ImageFolder'> 8005
    <class 'torchvision.datasets.folder.ImageFolder'> 2023
    


```python
print(type(trainset.targets), len(trainset.targets), trainset.targets[:5], trainset.targets[-5:])
```

    <class 'list'> 8005 [0, 0, 0, 0, 0] [1, 1, 1, 1, 1]
    


```python
# 클래스별 분포
for i in range(2): # 클래스별 순회
  print('클래스(레이블)별 데이터 개수 : ', i, (np.array(trainset.targets) == i).sum())
```

    클래스(레이블)별 데이터 개수 :  0 4000
    클래스(레이블)별 데이터 개수 :  1 4005
    


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




    (6404, 1601)




```python
from torch.utils.data import Subset
train_set = Subset(trainset, train_indices)
valid_set = Subset(trainset, valid_indices)
```


```python
valid_set[0][1] # 0번째 샘플의 정답
```




    1




```python
# 클래스별 분포
class_list = []
for i in range(2): # 클래스별 순회
  s = 0
  for j in range(len(valid_set)): # valid_data 1601개 순회
    if valid_set[j][1] == i :
      s += 1
  class_list.append(s)
class_list
```




    [800, 801]




```python
# trainset을 다시 train용과 valid 용으로 나누고자 할 때
# trainset, validset = random_split(trainset, [50000, 10000])
```


```python
print(type(train_set), len(train_set))
print(type(valid_set), len(valid_set))
print(type(testset), len(testset))
```

    <class 'torch.utils.data.dataset.Subset'> 6404
    <class 'torch.utils.data.dataset.Subset'> 1601
    <class 'torchvision.datasets.folder.ImageFolder'> 2023
    


```python
# 0번째 샘플에 2개의 원소가 있는데, 그중 첫번째 원소는 이미지, 두번째 원소는 정답
# 그러나 파이토치로 읽어들인 이미지 텐서의 형상이 channels * height * width 임
# 그에 비해 opencv, matplotlib으로 읽어들인 이미지 array의 형상은 height * width * channels
print(train_set[0][0].size(), train_set[0][1])
```

    torch.Size([3, 224, 224]) 0
    

## 2. 데이터 시각화


```python
labels_map = {0 : 'cat', 1 : 'dog'}  # for cat and dog

figure, axes = plt.subplots(nrows=4, ncols=8, figsize=(14, 8))
axes = axes.flatten()

for i in range(32):
  rand_i = np.random.randint(0, len(trainset))
  image, label= trainset[rand_i][0].permute(1, 2, 0), trainset[rand_i][1]
  axes[i].axis('off')
  axes[i].imshow(image)
  axes[i].set_title(labels_map[label])
```


    
![png](/assets/images/2023-04-12-CNN 10 (Cat and Dog (6) 전이학습)/output_26_0.png)
    


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
batch_size = 4 # 16 -> 4
# dataloader = DataLoader(데이터셋, 배치사이즈, 셔플여부.....)
trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True) # 훈련용 50000개의 데이터를 100개씩 준비
validloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False) # 검증용 10000개의 데이터를 100개씩 준비
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False) # 테스트용 10000개의 데이터를 100개씩 준비
```


```python
6404/4, 1601/4, 2023/4
```




    (1601.0, 400.25, 505.75)




```python
print(type(trainloader), len(trainloader))
print(type(validloader), len(validloader))
print(type(testloader), len(testloader))
```

    <class 'torch.utils.data.dataloader.DataLoader'> 1601
    <class 'torch.utils.data.dataloader.DataLoader'> 401
    <class 'torch.utils.data.dataloader.DataLoader'> 506
    


```python
train_iter = iter(trainloader)
images, labels = next(train_iter)
images.size(), labels.size()
```




    (torch.Size([4, 3, 224, 224]), torch.Size([4]))




```python
grid_img = torchvision.utils.make_grid(images)
plt.imshow(grid_img.permute(1, 2, 0))
```




    <matplotlib.image.AxesImage at 0x7fbcaeb4adf0>




    
![png](/assets/images/2023-04-12-CNN 10 (Cat and Dog (6) 전이학습)/output_34_1.png)
    


**데이터 전처리기 예시(transforms)**


```python
# 샘플의 원본 이미지
sample_image = images[0]
type(sample_image), sample_image.shape
```




    (torch.Tensor, torch.Size([3, 224, 224]))




```python
plt.imshow(sample_image.permute(1, 2, 0))
```




    <matplotlib.image.AxesImage at 0x7fbcc8283ca0>




    
![png](/assets/images/2023-04-12-CNN 10 (Cat and Dog (6) 전이학습)/output_37_1.png)
    



```python
# 샘플의 그레이스케일 이미지
trans = transforms.Grayscale()
gray_image = trans(sample_image)
type(gray_image), gray_image.shape
plt.imshow(gray_image.squeeze(), cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7fbcae3866d0>




    
![png](/assets/images/2023-04-12-CNN 10 (Cat and Dog (6) 전이학습)/output_38_1.png)
    



```python
# 샘플 이미지를 random하게 rotation
trans = transforms.RandomRotation(degrees=(0, 180))
rotated_image = trans(sample_image)
plt.imshow(rotated_image.permute(1, 2, 0))
```




    <matplotlib.image.AxesImage at 0x7fbcae45b0d0>




    
![png](/assets/images/2023-04-12-CNN 10 (Cat and Dog (6) 전이학습)/output_39_1.png)
    



```python
# 샘플 이미지를 random하게 crop
trans = transforms.RandomCrop(size=(128, 128))
cropped_image = trans(sample_image)
plt.imshow(cropped_image.permute(1, 2, 0))
```




    <matplotlib.image.AxesImage at 0x7fbcae486be0>




    
![png](/assets/images/2023-04-12-CNN 10 (Cat and Dog (6) 전이학습)/output_40_1.png)
    



```python
# 샘플 이미지를 random하게 horizontal flip
trans = transforms.RandomHorizontalFlip(p=0.3)
horizontal_flip_image = trans(sample_image)
plt.imshow(horizontal_flip_image.permute(1, 2, 0))
```




    <matplotlib.image.AxesImage at 0x7fbcae7f1130>




    
![png](/assets/images/2023-04-12-CNN 10 (Cat and Dog (6) 전이학습)/output_41_1.png)
    


## 4. 모델 생성


```python
from IPython.display import Image
Image('./images/전이학습.jpg', width=500)
```




    
![jpeg](/assets/images/2023-04-12-CNN 10 (Cat and Dog (6) 전이학습)/output_43_0.jpg)
    




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

**변경 사항**
- 이전 VGGNet 노트북에서는 83~84% 정확도

**Gloabal Average Pooling**
- FC layer를를 없애기 위한 방법으로 GooLenet에서 도입, 
- 최종 feature map 한장 한장에 대해 평균을 출력하여 모델 파라미터 수를 대폭 줄이게 됨
- 4200만개의 모델 파라미터가 Global Average Pooling으로1600만개로 줄었음
- 현재 노트북에서는 FC를 제거하기 전의 성능보다 좋음(90~91%)

**Learning Rate 조정**
- 학습 초반에 Loss가 안정적으로 줄지 않는 현상이 있어서 0.001에서 0.0001로 변경
- 조정 후 Loss가 안정적으로 줄어들며 학습도 빨리 진행됨. 93~94% 로 성능 향상

**데이터 증강(Data Augmentation)**
- RandomHorizontalFlip(p=0.3) 적용
- 모델에 입력되는 (셔플된) 미니 배치 단위로 매번 RandomHorizontalFlip이 30% 확률로 적용되므로 결국 어떤 배치에서는 원본 데이터가 그대로 사용되기도 하고 또다른 배치에서는 적용이 안되기도 해서 결국 데이터가 증강(추가)된 효과
- 이로 인해 모델 과적합을 막아주고 성능도 개선됨
- kaggle 커널에서 학습한 결과 95~96% 정확도

**ResNet50 적용**
- 배치 사이즈 4로 조정
- 성능 93~94%

**전이학습**
- torchvision.models 에서 제공하는 pretrained vgg16_bn model 사용하여 학습
- 약 98% 정확도


```python
import torchvision.models as models

# https://github.com/pytorch/vision/tree/6db1569c89094cf23f3bc41f79275c45e9fcb3f3/torchvision/models

model = models.vgg16_bn(weights=True)
```


```python
for parameter in model.parameters():
  print(parameter.requires_grad)
# parameter들의 requires_grad 속성이 True라는 것은 
# 오차 역전파를 통해 gradient를 전달할 수 있는 상태(즉, 학습이 가능한 상태태)  
```


```python
for parameter in model.parameters():
  parameter.requires_grad = False # 학습이 안되게 고정

for parameter in model.classifier.parameters():
  parameter.requires_grad = True # 학습이 가능한 상태
```


```python
model.classifier
```




    Sequential(
      (0): Linear(in_features=25088, out_features=4096, bias=True)
      (1): ReLU(inplace=True)
      (2): Dropout(p=0.5, inplace=False)
      (3): Linear(in_features=4096, out_features=4096, bias=True)
      (4): ReLU(inplace=True)
      (5): Dropout(p=0.5, inplace=False)
      (6): Linear(in_features=4096, out_features=1000, bias=True)
    )




```python
model.classifier[3] = nn.Linear(in_features=4096, out_features=512, bias=True)
model.classifier[6] = nn.Linear(in_features=512, out_features=2, bias=True)
model.classifier
```




    Sequential(
      (0): Linear(in_features=25088, out_features=4096, bias=True)
      (1): ReLU(inplace=True)
      (2): Dropout(p=0.5, inplace=False)
      (3): Linear(in_features=4096, out_features=512, bias=True)
      (4): ReLU(inplace=True)
      (5): Dropout(p=0.5, inplace=False)
      (6): Linear(in_features=512, out_features=2, bias=True)
    )




```python
model.to(device)
```


```python
out = model(images.to(device))
out.shape
```




    torch.Size([4, 2])




```python
for name, parameter in model.named_parameters():
  print(name, parameter.size())
```

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
learning_rate = 0.0001
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
summary(model, (3, 224, 224))
```

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1         [-1, 64, 224, 224]           1,792
           BatchNorm2d-2         [-1, 64, 224, 224]             128
                  ReLU-3         [-1, 64, 224, 224]               0
                Conv2d-4         [-1, 64, 224, 224]          36,928
           BatchNorm2d-5         [-1, 64, 224, 224]             128
                  ReLU-6         [-1, 64, 224, 224]               0
             MaxPool2d-7         [-1, 64, 112, 112]               0
                Conv2d-8        [-1, 128, 112, 112]          73,856
           BatchNorm2d-9        [-1, 128, 112, 112]             256
                 ReLU-10        [-1, 128, 112, 112]               0
               Conv2d-11        [-1, 128, 112, 112]         147,584
          BatchNorm2d-12        [-1, 128, 112, 112]             256
                 ReLU-13        [-1, 128, 112, 112]               0
            MaxPool2d-14          [-1, 128, 56, 56]               0
               Conv2d-15          [-1, 256, 56, 56]         295,168
          BatchNorm2d-16          [-1, 256, 56, 56]             512
                 ReLU-17          [-1, 256, 56, 56]               0
               Conv2d-18          [-1, 256, 56, 56]         590,080
          BatchNorm2d-19          [-1, 256, 56, 56]             512
                 ReLU-20          [-1, 256, 56, 56]               0
               Conv2d-21          [-1, 256, 56, 56]         590,080
          BatchNorm2d-22          [-1, 256, 56, 56]             512
                 ReLU-23          [-1, 256, 56, 56]               0
            MaxPool2d-24          [-1, 256, 28, 28]               0
               Conv2d-25          [-1, 512, 28, 28]       1,180,160
          BatchNorm2d-26          [-1, 512, 28, 28]           1,024
                 ReLU-27          [-1, 512, 28, 28]               0
               Conv2d-28          [-1, 512, 28, 28]       2,359,808
          BatchNorm2d-29          [-1, 512, 28, 28]           1,024
                 ReLU-30          [-1, 512, 28, 28]               0
               Conv2d-31          [-1, 512, 28, 28]       2,359,808
          BatchNorm2d-32          [-1, 512, 28, 28]           1,024
                 ReLU-33          [-1, 512, 28, 28]               0
            MaxPool2d-34          [-1, 512, 14, 14]               0
               Conv2d-35          [-1, 512, 14, 14]       2,359,808
          BatchNorm2d-36          [-1, 512, 14, 14]           1,024
                 ReLU-37          [-1, 512, 14, 14]               0
               Conv2d-38          [-1, 512, 14, 14]       2,359,808
          BatchNorm2d-39          [-1, 512, 14, 14]           1,024
                 ReLU-40          [-1, 512, 14, 14]               0
               Conv2d-41          [-1, 512, 14, 14]       2,359,808
          BatchNorm2d-42          [-1, 512, 14, 14]           1,024
                 ReLU-43          [-1, 512, 14, 14]               0
            MaxPool2d-44            [-1, 512, 7, 7]               0
    AdaptiveAvgPool2d-45            [-1, 512, 7, 7]               0
               Linear-46                 [-1, 4096]     102,764,544
                 ReLU-47                 [-1, 4096]               0
              Dropout-48                 [-1, 4096]               0
               Linear-49                  [-1, 512]       2,097,664
                 ReLU-50                  [-1, 512]               0
              Dropout-51                  [-1, 512]               0
               Linear-52                    [-1, 2]           1,026
    ================================================================
    Total params: 119,586,370
    Trainable params: 104,863,234
    Non-trainable params: 14,723,136
    ----------------------------------------------------------------
    Input size (MB): 0.57
    Forward/backward pass size (MB): 322.05
    Params size (MB): 456.19
    Estimated Total Size (MB): 778.81
    ----------------------------------------------------------------
    


```python
# 첫번째 conv layer의 모델 파라미터 수
# 필터수 x (필터) + bias
64 * (3*3*3) + 64
```




    1792




```python
# 마지막 출력 feature map의 사이즈
512 * 1 * 1
```




    512




```python
# linear 1 layer
512 * 64 + 64
```




    32832



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
    for images, labels in validloader: # 이터레이터로부터 next()가 호출되며 미니배치를 반환(images, labels)      
      # images, labels : (torch.Size([16, 3, 224, 224]), torch.Size([16]))
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
      # images, labels : (torch.Size([16, 3, 224, 224]), torch.Size([16]))
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

    Epoch : 1/55....... Train Loss : 0.164 Valid Loss : 0.145 Valid Accuracy : 0.963
    Epoch : 2/55....... Train Loss : 0.096 Valid Loss : 0.064 Valid Accuracy : 0.976
    Epoch : 3/55....... Train Loss : 0.067 Valid Loss : 0.119 Valid Accuracy : 0.966
    trigger :  1
    Epoch : 4/55....... Train Loss : 0.068 Valid Loss : 0.063 Valid Accuracy : 0.978
    Epoch : 5/55....... Train Loss : 0.063 Valid Loss : 0.231 Valid Accuracy : 0.963
    trigger :  1
    Epoch : 6/55....... Train Loss : 0.049 Valid Loss : 0.091 Valid Accuracy : 0.972
    trigger :  2
    Epoch : 7/55....... Train Loss : 0.046 Valid Loss : 0.190 Valid Accuracy : 0.968
    trigger :  3
    Epoch : 8/55....... Train Loss : 0.051 Valid Loss : 0.064 Valid Accuracy : 0.976
    trigger :  4
    Epoch : 9/55....... Train Loss : 0.044 Valid Loss : 0.150 Valid Accuracy : 0.973
    trigger :  5
    Epoch 00009: reducing learning rate of group 0 to 1.0000e-05.
    Epoch : 10/55....... Train Loss : 0.024 Valid Loss : 0.069 Valid Accuracy : 0.981
    trigger :  6
    Epoch : 11/55....... Train Loss : 0.021 Valid Loss : 0.092 Valid Accuracy : 0.979
    trigger :  7
    Epoch : 12/55....... Train Loss : 0.023 Valid Loss : 0.085 Valid Accuracy : 0.980
    trigger :  8
    Early Stopping !!!
    Training loop is finished !!
    CPU times: user 41min 22s, sys: 11.9 s, total: 41min 34s
    Wall time: 30min 13s
    


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
rnd_idx = 1
print(images[rnd_idx].shape, labels[rnd_idx])
```

    torch.Size([4, 3, 224, 224]) torch.Size([4])
    torch.Size([3, 224, 224]) tensor(0, device='cuda:0')
    


```python
images[rnd_idx].shape
```




    torch.Size([3, 224, 224])




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
    cat cat
    




    <matplotlib.image.AxesImage at 0x7fbcace0dc10>




    
![png](/assets/images/2023-04-12-CNN 10 (Cat and Dog (6) 전이학습)/output_77_2.png)
    


## 8. 모델 평가


```python
def evaluation(model, testloader, loss_fn):
  total = 0   
  correct = 0
  test_loss = 0
  test_accuracy = 0

  # 전방향 예측을 구할 때는 gradient가 필요가 없음음
  with torch.no_grad():
    for images, labels in testloader: # 이터레이터로부터 next()가 호출되며 미니배치를 반환(images, labels)
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

    Test Loss : 0.125 Test Accuracy : 0.979
    

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




    odict_keys(['features.0.weight', 'features.0.bias', 'features.1.weight', 'features.1.bias', 'features.1.running_mean', 'features.1.running_var', 'features.1.num_batches_tracked', 'features.3.weight', 'features.3.bias', 'features.4.weight', 'features.4.bias', 'features.4.running_mean', 'features.4.running_var', 'features.4.num_batches_tracked', 'features.7.weight', 'features.7.bias', 'features.8.weight', 'features.8.bias', 'features.8.running_mean', 'features.8.running_var', 'features.8.num_batches_tracked', 'features.10.weight', 'features.10.bias', 'features.11.weight', 'features.11.bias', 'features.11.running_mean', 'features.11.running_var', 'features.11.num_batches_tracked', 'features.14.weight', 'features.14.bias', 'features.15.weight', 'features.15.bias', 'features.15.running_mean', 'features.15.running_var', 'features.15.num_batches_tracked', 'features.17.weight', 'features.17.bias', 'features.18.weight', 'features.18.bias', 'features.18.running_mean', 'features.18.running_var', 'features.18.num_batches_tracked', 'features.20.weight', 'features.20.bias', 'features.21.weight', 'features.21.bias', 'features.21.running_mean', 'features.21.running_var', 'features.21.num_batches_tracked', 'features.24.weight', 'features.24.bias', 'features.25.weight', 'features.25.bias', 'features.25.running_mean', 'features.25.running_var', 'features.25.num_batches_tracked', 'features.27.weight', 'features.27.bias', 'features.28.weight', 'features.28.bias', 'features.28.running_mean', 'features.28.running_var', 'features.28.num_batches_tracked', 'features.30.weight', 'features.30.bias', 'features.31.weight', 'features.31.bias', 'features.31.running_mean', 'features.31.running_var', 'features.31.num_batches_tracked', 'features.34.weight', 'features.34.bias', 'features.35.weight', 'features.35.bias', 'features.35.running_mean', 'features.35.running_var', 'features.35.num_batches_tracked', 'features.37.weight', 'features.37.bias', 'features.38.weight', 'features.38.bias', 'features.38.running_mean', 'features.38.running_var', 'features.38.num_batches_tracked', 'features.40.weight', 'features.40.bias', 'features.41.weight', 'features.41.bias', 'features.41.running_mean', 'features.41.running_var', 'features.41.num_batches_tracked', 'classifier.0.weight', 'classifier.0.bias', 'classifier.3.weight', 'classifier.3.bias', 'classifier.6.weight', 'classifier.6.bias'])




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




    odict_keys(['features.0.weight', 'features.0.bias', 'features.1.weight', 'features.1.bias', 'features.1.running_mean', 'features.1.running_var', 'features.1.num_batches_tracked', 'features.3.weight', 'features.3.bias', 'features.4.weight', 'features.4.bias', 'features.4.running_mean', 'features.4.running_var', 'features.4.num_batches_tracked', 'features.7.weight', 'features.7.bias', 'features.8.weight', 'features.8.bias', 'features.8.running_mean', 'features.8.running_var', 'features.8.num_batches_tracked', 'features.10.weight', 'features.10.bias', 'features.11.weight', 'features.11.bias', 'features.11.running_mean', 'features.11.running_var', 'features.11.num_batches_tracked', 'features.14.weight', 'features.14.bias', 'features.15.weight', 'features.15.bias', 'features.15.running_mean', 'features.15.running_var', 'features.15.num_batches_tracked', 'features.17.weight', 'features.17.bias', 'features.18.weight', 'features.18.bias', 'features.18.running_mean', 'features.18.running_var', 'features.18.num_batches_tracked', 'features.20.weight', 'features.20.bias', 'features.21.weight', 'features.21.bias', 'features.21.running_mean', 'features.21.running_var', 'features.21.num_batches_tracked', 'features.24.weight', 'features.24.bias', 'features.25.weight', 'features.25.bias', 'features.25.running_mean', 'features.25.running_var', 'features.25.num_batches_tracked', 'features.27.weight', 'features.27.bias', 'features.28.weight', 'features.28.bias', 'features.28.running_mean', 'features.28.running_var', 'features.28.num_batches_tracked', 'features.30.weight', 'features.30.bias', 'features.31.weight', 'features.31.bias', 'features.31.running_mean', 'features.31.running_var', 'features.31.num_batches_tracked', 'features.34.weight', 'features.34.bias', 'features.35.weight', 'features.35.bias', 'features.35.running_mean', 'features.35.running_var', 'features.35.num_batches_tracked', 'features.37.weight', 'features.37.bias', 'features.38.weight', 'features.38.bias', 'features.38.running_mean', 'features.38.running_var', 'features.38.num_batches_tracked', 'features.40.weight', 'features.40.bias', 'features.41.weight', 'features.41.bias', 'features.41.running_mean', 'features.41.running_var', 'features.41.num_batches_tracked', 'classifier.0.weight', 'classifier.0.bias', 'classifier.3.weight', 'classifier.3.bias', 'classifier.6.weight', 'classifier.6.bias'])




```python
# 읽어들인 모델 파라미터는 모델 아키텍처에 연결을 시켜줘야 함
# load_state_dict() 사용
last_model = model
last_model.to(device)
last_model.load_state_dict(last_state_dict)
```




    <All keys matched successfully>




```python
last_model.eval()
evaluation(last_model, testloader, loss_fn)  
```

    Test Loss : 0.114 Test Accuracy : 0.979
    


```python
# valid loss or accuracy 기준 best model
best_state_dict = torch.load('best_checkpoint.pth')
best_model = model
best_model.to(device)
best_model.load_state_dict(best_state_dict)
```




    <All keys matched successfully>




```python
best_model.eval()
evaluation(best_model, testloader, loss_fn)
```

    Test Loss : 0.083 Test Accuracy : 0.982
    


```python
#best_state_dict['conv_block1.0.weight']
```


```python
#last_state_dict['conv_block1.0.weight']
```

## Reference
- [딥러닝 파이토치 교과서 (서지영 저)](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=289661077)
