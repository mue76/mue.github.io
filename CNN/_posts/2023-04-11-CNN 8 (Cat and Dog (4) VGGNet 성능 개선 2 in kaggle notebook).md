---
tag: [Deep Learning, 딥러닝, pytorch, 파이토치, CNN, Cat and Dog Dataset, VGG]
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




    
![png](/assets/images/2023-04-11-CNN 8 (Cat and Dog (4) VGGNet 성능 개선 2 in kaggle notebook)/output_1_0.png)
    




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

    Tue Apr 11 07:30:31 2023       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 470.161.03   Driver Version: 470.161.03   CUDA Version: 11.4     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla P100-PCIE...  Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   40C    P0    26W / 250W |      0MiB / 16280MiB |      0%      Default |
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



# 0. 데이터 다운로드


```python
data_dir = '/kaggle/input/cat-and-dog/'
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


    
![png](/assets/images/2023-04-11-CNN 8 (Cat and Dog (4) VGGNet 성능 개선 2 in kaggle notebook)/output_22_0.png)
    


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
6404/16, 1601/16, 2023/16
```




    (400.25, 100.0625, 126.4375)




```python
print(type(trainloader), len(trainloader))
print(type(validloader), len(validloader))
print(type(testloader), len(testloader))
```

    <class 'torch.utils.data.dataloader.DataLoader'> 401
    <class 'torch.utils.data.dataloader.DataLoader'> 101
    <class 'torch.utils.data.dataloader.DataLoader'> 127
    


```python
train_iter = iter(trainloader)
images, labels = next(train_iter)
images.size(), labels.size()
```




    (torch.Size([16, 3, 224, 224]), torch.Size([16]))




```python
# 샘플의 원본 이미지
sample_image = images[0]
type(sample_image), sample_image.shape
```




    (torch.Tensor, torch.Size([3, 224, 224]))




```python
plt.imshow(sample_image.permute(1, 2, 0))
```




    <matplotlib.image.AxesImage at 0x7c4d764d2b90>




    
![png](/assets/images/2023-04-11-CNN 8 (Cat and Dog (4) VGGNet 성능 개선 2 in kaggle notebook)/output_31_1.png)
    



```python
# 샘플의 그레이스케일 이미지
trans = transforms.Grayscale()
gray_image = trans(sample_image)
type(gray_image), gray_image.shape
plt.imshow(gray_image.squeeze(), cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7c4d7646cb10>




    
![png](/assets/images/2023-04-11-CNN 8 (Cat and Dog (4) VGGNet 성능 개선 2 in kaggle notebook)/output_32_1.png)
    



```python
# 샘플 이미지를 random하게 rotation
trans = transforms.RandomRotation(degrees=(0, 180))
rotated_image = trans(sample_image)
plt.imshow(rotated_image.permute(1, 2, 0))
```




    <matplotlib.image.AxesImage at 0x7c4d76384950>




    
![png](/assets/images/2023-04-11-CNN 8 (Cat and Dog (4) VGGNet 성능 개선 2 in kaggle notebook)/output_33_1.png)
    



```python
# 샘플 이미지를 random하게 crop
trans = transforms.RandomCrop(size=(128, 128))
cropped_image = trans(sample_image)
plt.imshow(cropped_image.permute(1, 2, 0))
```




    <matplotlib.image.AxesImage at 0x7c4d7630d450>




    
![png](/assets/images/2023-04-11-CNN 8 (Cat and Dog (4) VGGNet 성능 개선 2 in kaggle notebook)/output_34_1.png)
    



```python
# 샘플 이미지를 random하게 horizontal flip
trans = transforms.RandomHorizontalFlip(p=0.3)
horizontal_flip_image = trans(sample_image)
plt.imshow(horizontal_flip_image.permute(1, 2, 0))
```




    <matplotlib.image.AxesImage at 0x7c4d76288990>




    
![png](/assets/images/2023-04-11-CNN 8 (Cat and Dog (4) VGGNet 성능 개선 2 in kaggle notebook)/output_35_1.png)
    


## 4. 모델 생성


```python
from IPython.display import Image
Image('./images/VGG1.jpg', width=700)
```




    
![jpeg](/assets/images/2023-04-11-CNN 8 (Cat and Dog (4) VGGNet 성능 개선 2 in kaggle notebook)/output_37_0.jpg)
    




```python
Image('./images/VGG5.jpg')
```




    
![jpeg](/assets/images/2023-04-11-CNN 8 (Cat and Dog (4) VGGNet 성능 개선 2 in kaggle notebook)/output_38_0.jpg)
    




```python
Image('./images/vgg16.png', width=350)
```




    
![png](/assets/images/2023-04-11-CNN 8 (Cat and Dog (4) VGGNet 성능 개선 2 in kaggle notebook)/output_39_0.png)
    




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

**더 해볼만한 것**
- 배치 사이즈 조정, 다양한 데이터 증강, Dropout 실험

**conv block 별 사이즈 확인**


```python
conv_block1 = nn.Sequential(
                            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(num_features=64),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(num_features=64),
                            nn.ReLU(),                            
                            nn.MaxPool2d(kernel_size=2, stride=2)
                            ) # [16, 64, 112, 112]
conv_block1_out = conv_block1(images)                                     
conv_block1_out.shape
```




    torch.Size([16, 64, 112, 112])




```python
conv_block2 = nn.Sequential(
                            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(num_features=128),                                      
                            nn.ReLU(),
                            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(num_features=128),                                      
                            nn.ReLU(),                            
                            nn.MaxPool2d(kernel_size=2, stride=2)
                            ) # [16, 128, 56, 56]
conv_block2_out = conv_block2(conv_block1_out)                                     
conv_block2_out.shape                                     
```




    torch.Size([16, 128, 56, 56])




```python
conv_block3 = nn.Sequential(
                            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(num_features=256),                                      
                            nn.ReLU(),     
                            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(num_features=256),                                      
                            nn.ReLU(),     
                            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(num_features=512),                                      
                            nn.ReLU(),     
                            nn.MaxPool2d(kernel_size=2, stride=2)                                                                                         
                            ) # [16, 512, 28, 28] 
conv_block3_out = conv_block3(conv_block2_out)                                     
conv_block3_out.shape                                       
```




    torch.Size([16, 512, 28, 28])




```python
conv_block4 = nn.Sequential(
                            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(num_features=512),                                      
                            nn.ReLU(),   
                            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(num_features=512),                                      
                            nn.ReLU(),  
                            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(num_features=512),                                      
                            nn.ReLU(),   
                            nn.MaxPool2d(kernel_size=2, stride=2)                                                                                           
                            ) # [16, 512, 14, 14]
conv_block4_out = conv_block4(conv_block3_out)                                     
conv_block4_out.shape  
```




    torch.Size([16, 512, 14, 14])




```python
conv_block5 = nn.Sequential(
                            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(num_features=512),                                      
                            nn.ReLU(),   
                            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(num_features=512),                                      
                            nn.ReLU(),   
                            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(num_features=512),                                      
                            nn.ReLU(),                                                           
                            nn.MaxPool2d(kernel_size=2, stride=2)                                   
                            ) # [16, 512, 7, 7]
conv_block5_out = conv_block5(conv_block4_out)                                     
conv_block5_out.shape  
```




    torch.Size([16, 512, 7, 7])




```python
avg_pool = nn.AdaptiveAvgPool2d((1, 1)) 
avg_pool_out = avg_pool(conv_block5_out)
avg_pool_out.shape
```




    torch.Size([16, 512, 1, 1])




```python
class VGG16(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_block1 = nn.Sequential(
                                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(num_features=64),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(num_features=64),
                                nn.ReLU(),                            
                                nn.MaxPool2d(kernel_size=2, stride=2)
                                ) # [16, 64, 112, 112]
    self.conv_block2 = nn.Sequential(
                                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(num_features=128),                                      
                                nn.ReLU(),
                                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(num_features=128),                                      
                                nn.ReLU(),                            
                                nn.MaxPool2d(kernel_size=2, stride=2)
                                ) # [16, 128, 56, 56]

    self.conv_block3 = nn.Sequential(
                                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(num_features=256),                                      
                                nn.ReLU(),     
                                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(num_features=256),                                      
                                nn.ReLU(),     
                                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(num_features=512),                                      
                                nn.ReLU(),     
                                nn.MaxPool2d(kernel_size=2, stride=2)                                                                                         
                                ) # [16, 512, 28, 28]   

    self.conv_block4 = nn.Sequential(
                                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(num_features=512),                                      
                                nn.ReLU(),   
                                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(num_features=512),                                      
                                nn.ReLU(),  
                                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(num_features=512),                                      
                                nn.ReLU(),   
                                nn.MaxPool2d(kernel_size=2, stride=2)                                                                                           
                                ) # [16, 512, 14, 14] 

    self.conv_block5 = nn.Sequential(
                                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(num_features=512),                                      
                                nn.ReLU(),   
                                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(num_features=512),                                      
                                nn.ReLU(),   
                                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(num_features=512),                                      
                                nn.ReLU(),                                                           
                                nn.MaxPool2d(kernel_size=2, stride=2)                                   
                                ) # [16, 512, 7, 7]   

    self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # [16, 512, 1, 1]                                                                                                                              

    self.classifier = nn.Sequential(
                                nn.Linear(in_features=512, out_features=64),
                                nn.BatchNorm1d(num_features=64),
                                nn.ReLU(),
                                nn.Linear(in_features=64, out_features=2)
                                )

  def forward(self, x):
    x = self.conv_block1(x) # [16, 64, 112, 112]
    x = self.conv_block2(x) # [16, 128, 56, 56]
    x = self.conv_block3(x) # [16, 512, 28, 28] 
    x = self.conv_block4(x) # [16, 512, 14, 14]
    x = self.conv_block5(x) # [16, 512, 7, 7]
    x = self.avg_pool(x) # [16, 512, 1, 1] 
    
    # reshape할 형상 : (batch_size x 512)
    # x = x.view(-1, 512) # option 1 : view
    x = torch.flatten(x, 1) # option 2 : flatten 
    # x = x.reshape(x.shape[0], -1) # option 3 : reshape

    x = self.classifier(x)    
    return x
```


```python
model = VGG16()
model.to(device)
model
```




    VGG16(
      (conv_block1): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (conv_block2): Sequential(
        (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (conv_block3): Sequential(
        (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (7): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (8): ReLU()
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (conv_block4): Sequential(
        (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (7): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (8): ReLU()
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (conv_block5): Sequential(
        (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (7): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (8): ReLU()
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (avg_pool): AdaptiveAvgPool2d(output_size=(1, 1))
      (classifier): Sequential(
        (0): Linear(in_features=512, out_features=64, bias=True)
        (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=64, out_features=2, bias=True)
      )
    )




```python
out = model(images.to(device))
out.shape
```




    torch.Size([16, 2])




```python
for name, parameter in model.named_parameters():
  print(name, parameter.size())
```

    conv_block1.0.weight torch.Size([64, 3, 3, 3])
    conv_block1.0.bias torch.Size([64])
    conv_block1.1.weight torch.Size([64])
    conv_block1.1.bias torch.Size([64])
    conv_block1.3.weight torch.Size([64, 64, 3, 3])
    conv_block1.3.bias torch.Size([64])
    conv_block1.4.weight torch.Size([64])
    conv_block1.4.bias torch.Size([64])
    conv_block2.0.weight torch.Size([128, 64, 3, 3])
    conv_block2.0.bias torch.Size([128])
    conv_block2.1.weight torch.Size([128])
    conv_block2.1.bias torch.Size([128])
    conv_block2.3.weight torch.Size([128, 128, 3, 3])
    conv_block2.3.bias torch.Size([128])
    conv_block2.4.weight torch.Size([128])
    conv_block2.4.bias torch.Size([128])
    conv_block3.0.weight torch.Size([256, 128, 3, 3])
    conv_block3.0.bias torch.Size([256])
    conv_block3.1.weight torch.Size([256])
    conv_block3.1.bias torch.Size([256])
    conv_block3.3.weight torch.Size([256, 256, 3, 3])
    conv_block3.3.bias torch.Size([256])
    conv_block3.4.weight torch.Size([256])
    conv_block3.4.bias torch.Size([256])
    conv_block3.6.weight torch.Size([512, 256, 3, 3])
    conv_block3.6.bias torch.Size([512])
    conv_block3.7.weight torch.Size([512])
    conv_block3.7.bias torch.Size([512])
    conv_block4.0.weight torch.Size([512, 512, 3, 3])
    conv_block4.0.bias torch.Size([512])
    conv_block4.1.weight torch.Size([512])
    conv_block4.1.bias torch.Size([512])
    conv_block4.3.weight torch.Size([512, 512, 3, 3])
    conv_block4.3.bias torch.Size([512])
    conv_block4.4.weight torch.Size([512])
    conv_block4.4.bias torch.Size([512])
    conv_block4.6.weight torch.Size([512, 512, 3, 3])
    conv_block4.6.bias torch.Size([512])
    conv_block4.7.weight torch.Size([512])
    conv_block4.7.bias torch.Size([512])
    conv_block5.0.weight torch.Size([512, 512, 3, 3])
    conv_block5.0.bias torch.Size([512])
    conv_block5.1.weight torch.Size([512])
    conv_block5.1.bias torch.Size([512])
    conv_block5.3.weight torch.Size([512, 512, 3, 3])
    conv_block5.3.bias torch.Size([512])
    conv_block5.4.weight torch.Size([512])
    conv_block5.4.bias torch.Size([512])
    conv_block5.6.weight torch.Size([512, 512, 3, 3])
    conv_block5.6.bias torch.Size([512])
    conv_block5.7.weight torch.Size([512])
    conv_block5.7.bias torch.Size([512])
    classifier.0.weight torch.Size([64, 512])
    classifier.0.bias torch.Size([64])
    classifier.1.weight torch.Size([64])
    classifier.1.bias torch.Size([64])
    classifier.3.weight torch.Size([2, 64])
    classifier.3.bias torch.Size([2])
    

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
    
               Conv2d-21          [-1, 512, 56, 56]       1,180,160
    
          BatchNorm2d-22          [-1, 512, 56, 56]           1,024
    
                 ReLU-23          [-1, 512, 56, 56]               0
    
            MaxPool2d-24          [-1, 512, 28, 28]               0
    
               Conv2d-25          [-1, 512, 28, 28]       2,359,808
    
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
    
    AdaptiveAvgPool2d-45            [-1, 512, 1, 1]               0
    
               Linear-46                   [-1, 64]          32,832
    
          BatchNorm1d-47                   [-1, 64]             128
    
                 ReLU-48                   [-1, 64]               0
    
               Linear-49                    [-1, 2]             130
    
    ================================================================
    
    Total params: 16,526,466
    
    Trainable params: 16,526,466
    
    Non-trainable params: 0
    
    ----------------------------------------------------------------
    
    Input size (MB): 0.57
    
    Forward/backward pass size (MB): 341.67
    
    Params size (MB): 63.04
    
    Estimated Total Size (MB): 405.28
    
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

    Epoch : 1/55....... Train Loss : 0.645 Valid Loss : 0.593 Valid Accuracy : 0.667
    Epoch : 2/55....... Train Loss : 0.588 Valid Loss : 0.604 Valid Accuracy : 0.681
    trigger :  1
    Epoch : 3/55....... Train Loss : 0.552 Valid Loss : 0.609 Valid Accuracy : 0.668
    trigger :  2
    Epoch : 4/55....... Train Loss : 0.509 Valid Loss : 0.444 Valid Accuracy : 0.786
    Epoch : 5/55....... Train Loss : 0.462 Valid Loss : 0.489 Valid Accuracy : 0.765
    trigger :  1
    Epoch : 6/55....... Train Loss : 0.400 Valid Loss : 0.350 Valid Accuracy : 0.855
    Epoch : 7/55....... Train Loss : 0.356 Valid Loss : 0.460 Valid Accuracy : 0.781
    trigger :  1
    Epoch : 8/55....... Train Loss : 0.309 Valid Loss : 0.313 Valid Accuracy : 0.861
    Epoch : 9/55....... Train Loss : 0.280 Valid Loss : 0.230 Valid Accuracy : 0.900
    Epoch : 10/55....... Train Loss : 0.239 Valid Loss : 0.226 Valid Accuracy : 0.895
    Epoch : 11/55....... Train Loss : 0.215 Valid Loss : 0.198 Valid Accuracy : 0.920
    Epoch : 12/55....... Train Loss : 0.199 Valid Loss : 0.189 Valid Accuracy : 0.919
    Epoch : 13/55....... Train Loss : 0.200 Valid Loss : 0.292 Valid Accuracy : 0.883
    trigger :  1
    Epoch : 14/55....... Train Loss : 0.179 Valid Loss : 0.185 Valid Accuracy : 0.924
    Epoch : 15/55....... Train Loss : 0.192 Valid Loss : 0.205 Valid Accuracy : 0.914
    trigger :  1
    Epoch : 16/55....... Train Loss : 0.164 Valid Loss : 0.162 Valid Accuracy : 0.930
    Epoch : 17/55....... Train Loss : 0.151 Valid Loss : 0.149 Valid Accuracy : 0.938
    Epoch : 18/55....... Train Loss : 0.132 Valid Loss : 0.213 Valid Accuracy : 0.913
    trigger :  1
    Epoch : 19/55....... Train Loss : 0.132 Valid Loss : 0.135 Valid Accuracy : 0.942
    Epoch : 20/55....... Train Loss : 0.100 Valid Loss : 0.190 Valid Accuracy : 0.921
    trigger :  1
    Epoch : 21/55....... Train Loss : 0.115 Valid Loss : 0.183 Valid Accuracy : 0.923
    trigger :  2
    Epoch : 22/55....... Train Loss : 0.105 Valid Loss : 0.185 Valid Accuracy : 0.929
    trigger :  3
    Epoch : 23/55....... Train Loss : 0.114 Valid Loss : 0.157 Valid Accuracy : 0.941
    trigger :  4
    Epoch : 24/55....... Train Loss : 0.095 Valid Loss : 0.136 Valid Accuracy : 0.943
    trigger :  5
    Epoch 00024: reducing learning rate of group 0 to 1.0000e-05.
    Epoch : 25/55....... Train Loss : 0.060 Valid Loss : 0.139 Valid Accuracy : 0.944
    trigger :  6
    Epoch : 26/55....... Train Loss : 0.044 Valid Loss : 0.123 Valid Accuracy : 0.954
    Epoch : 27/55....... Train Loss : 0.046 Valid Loss : 0.137 Valid Accuracy : 0.945
    trigger :  1
    Epoch : 28/55....... Train Loss : 0.041 Valid Loss : 0.127 Valid Accuracy : 0.949
    trigger :  2
    Epoch : 29/55....... Train Loss : 0.043 Valid Loss : 0.130 Valid Accuracy : 0.948
    trigger :  3
    Epoch : 30/55....... Train Loss : 0.033 Valid Loss : 0.135 Valid Accuracy : 0.951
    trigger :  4
    Epoch : 31/55....... Train Loss : 0.028 Valid Loss : 0.126 Valid Accuracy : 0.951
    trigger :  5
    Epoch 00031: reducing learning rate of group 0 to 1.0000e-06.
    Epoch : 32/55....... Train Loss : 0.027 Valid Loss : 0.129 Valid Accuracy : 0.952
    trigger :  6
    Epoch : 33/55....... Train Loss : 0.025 Valid Loss : 0.115 Valid Accuracy : 0.954
    Epoch : 34/55....... Train Loss : 0.025 Valid Loss : 0.134 Valid Accuracy : 0.948
    trigger :  1
    Epoch : 35/55....... Train Loss : 0.027 Valid Loss : 0.112 Valid Accuracy : 0.957
    Epoch : 36/55....... Train Loss : 0.027 Valid Loss : 0.118 Valid Accuracy : 0.953
    trigger :  1
    Epoch : 37/55....... Train Loss : 0.025 Valid Loss : 0.120 Valid Accuracy : 0.953
    trigger :  2
    Epoch : 38/55....... Train Loss : 0.027 Valid Loss : 0.117 Valid Accuracy : 0.952
    trigger :  3
    Epoch : 39/55....... Train Loss : 0.029 Valid Loss : 0.124 Valid Accuracy : 0.951
    trigger :  4
    Epoch : 40/55....... Train Loss : 0.026 Valid Loss : 0.136 Valid Accuracy : 0.948
    trigger :  5
    Epoch 00040: reducing learning rate of group 0 to 1.0000e-07.
    Epoch : 41/55....... Train Loss : 0.021 Valid Loss : 0.129 Valid Accuracy : 0.950
    trigger :  6
    Epoch : 42/55....... Train Loss : 0.029 Valid Loss : 0.122 Valid Accuracy : 0.951
    trigger :  7
    Epoch : 43/55....... Train Loss : 0.021 Valid Loss : 0.116 Valid Accuracy : 0.953
    trigger :  8
    Early Stopping !!!
    Training loop is finished !!
    CPU times: user 1h 10min 35s, sys: 29.8 s, total: 1h 11min 5s
    Wall time: 1h 15min 3s
    


```python
%load_ext tensorboard
```


```python
%tensorboard --logdir=runs
```


    <IPython.core.display.Javascript object>


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

    torch.Size([16, 3, 224, 224]) torch.Size([16])
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
    




    <matplotlib.image.AxesImage at 0x7c4b8cc49bd0>




    
![png](/assets/images/2023-04-11-CNN 8 (Cat and Dog (4) VGGNet 성능 개선 2 in kaggle notebook)/output_76_2.png)
    


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

    Test Loss : 0.126 Test Accuracy : 0.953
    

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




    odict_keys(['conv_block1.0.weight', 'conv_block1.0.bias', 'conv_block1.1.weight', 'conv_block1.1.bias', 'conv_block1.1.running_mean', 'conv_block1.1.running_var', 'conv_block1.1.num_batches_tracked', 'conv_block1.3.weight', 'conv_block1.3.bias', 'conv_block1.4.weight', 'conv_block1.4.bias', 'conv_block1.4.running_mean', 'conv_block1.4.running_var', 'conv_block1.4.num_batches_tracked', 'conv_block2.0.weight', 'conv_block2.0.bias', 'conv_block2.1.weight', 'conv_block2.1.bias', 'conv_block2.1.running_mean', 'conv_block2.1.running_var', 'conv_block2.1.num_batches_tracked', 'conv_block2.3.weight', 'conv_block2.3.bias', 'conv_block2.4.weight', 'conv_block2.4.bias', 'conv_block2.4.running_mean', 'conv_block2.4.running_var', 'conv_block2.4.num_batches_tracked', 'conv_block3.0.weight', 'conv_block3.0.bias', 'conv_block3.1.weight', 'conv_block3.1.bias', 'conv_block3.1.running_mean', 'conv_block3.1.running_var', 'conv_block3.1.num_batches_tracked', 'conv_block3.3.weight', 'conv_block3.3.bias', 'conv_block3.4.weight', 'conv_block3.4.bias', 'conv_block3.4.running_mean', 'conv_block3.4.running_var', 'conv_block3.4.num_batches_tracked', 'conv_block3.6.weight', 'conv_block3.6.bias', 'conv_block3.7.weight', 'conv_block3.7.bias', 'conv_block3.7.running_mean', 'conv_block3.7.running_var', 'conv_block3.7.num_batches_tracked', 'conv_block4.0.weight', 'conv_block4.0.bias', 'conv_block4.1.weight', 'conv_block4.1.bias', 'conv_block4.1.running_mean', 'conv_block4.1.running_var', 'conv_block4.1.num_batches_tracked', 'conv_block4.3.weight', 'conv_block4.3.bias', 'conv_block4.4.weight', 'conv_block4.4.bias', 'conv_block4.4.running_mean', 'conv_block4.4.running_var', 'conv_block4.4.num_batches_tracked', 'conv_block4.6.weight', 'conv_block4.6.bias', 'conv_block4.7.weight', 'conv_block4.7.bias', 'conv_block4.7.running_mean', 'conv_block4.7.running_var', 'conv_block4.7.num_batches_tracked', 'conv_block5.0.weight', 'conv_block5.0.bias', 'conv_block5.1.weight', 'conv_block5.1.bias', 'conv_block5.1.running_mean', 'conv_block5.1.running_var', 'conv_block5.1.num_batches_tracked', 'conv_block5.3.weight', 'conv_block5.3.bias', 'conv_block5.4.weight', 'conv_block5.4.bias', 'conv_block5.4.running_mean', 'conv_block5.4.running_var', 'conv_block5.4.num_batches_tracked', 'conv_block5.6.weight', 'conv_block5.6.bias', 'conv_block5.7.weight', 'conv_block5.7.bias', 'conv_block5.7.running_mean', 'conv_block5.7.running_var', 'conv_block5.7.num_batches_tracked', 'classifier.0.weight', 'classifier.0.bias', 'classifier.1.weight', 'classifier.1.bias', 'classifier.1.running_mean', 'classifier.1.running_var', 'classifier.1.num_batches_tracked', 'classifier.3.weight', 'classifier.3.bias'])




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




    odict_keys(['conv_block1.0.weight', 'conv_block1.0.bias', 'conv_block1.1.weight', 'conv_block1.1.bias', 'conv_block1.1.running_mean', 'conv_block1.1.running_var', 'conv_block1.1.num_batches_tracked', 'conv_block1.3.weight', 'conv_block1.3.bias', 'conv_block1.4.weight', 'conv_block1.4.bias', 'conv_block1.4.running_mean', 'conv_block1.4.running_var', 'conv_block1.4.num_batches_tracked', 'conv_block2.0.weight', 'conv_block2.0.bias', 'conv_block2.1.weight', 'conv_block2.1.bias', 'conv_block2.1.running_mean', 'conv_block2.1.running_var', 'conv_block2.1.num_batches_tracked', 'conv_block2.3.weight', 'conv_block2.3.bias', 'conv_block2.4.weight', 'conv_block2.4.bias', 'conv_block2.4.running_mean', 'conv_block2.4.running_var', 'conv_block2.4.num_batches_tracked', 'conv_block3.0.weight', 'conv_block3.0.bias', 'conv_block3.1.weight', 'conv_block3.1.bias', 'conv_block3.1.running_mean', 'conv_block3.1.running_var', 'conv_block3.1.num_batches_tracked', 'conv_block3.3.weight', 'conv_block3.3.bias', 'conv_block3.4.weight', 'conv_block3.4.bias', 'conv_block3.4.running_mean', 'conv_block3.4.running_var', 'conv_block3.4.num_batches_tracked', 'conv_block3.6.weight', 'conv_block3.6.bias', 'conv_block3.7.weight', 'conv_block3.7.bias', 'conv_block3.7.running_mean', 'conv_block3.7.running_var', 'conv_block3.7.num_batches_tracked', 'conv_block4.0.weight', 'conv_block4.0.bias', 'conv_block4.1.weight', 'conv_block4.1.bias', 'conv_block4.1.running_mean', 'conv_block4.1.running_var', 'conv_block4.1.num_batches_tracked', 'conv_block4.3.weight', 'conv_block4.3.bias', 'conv_block4.4.weight', 'conv_block4.4.bias', 'conv_block4.4.running_mean', 'conv_block4.4.running_var', 'conv_block4.4.num_batches_tracked', 'conv_block4.6.weight', 'conv_block4.6.bias', 'conv_block4.7.weight', 'conv_block4.7.bias', 'conv_block4.7.running_mean', 'conv_block4.7.running_var', 'conv_block4.7.num_batches_tracked', 'conv_block5.0.weight', 'conv_block5.0.bias', 'conv_block5.1.weight', 'conv_block5.1.bias', 'conv_block5.1.running_mean', 'conv_block5.1.running_var', 'conv_block5.1.num_batches_tracked', 'conv_block5.3.weight', 'conv_block5.3.bias', 'conv_block5.4.weight', 'conv_block5.4.bias', 'conv_block5.4.running_mean', 'conv_block5.4.running_var', 'conv_block5.4.num_batches_tracked', 'conv_block5.6.weight', 'conv_block5.6.bias', 'conv_block5.7.weight', 'conv_block5.7.bias', 'conv_block5.7.running_mean', 'conv_block5.7.running_var', 'conv_block5.7.num_batches_tracked', 'classifier.0.weight', 'classifier.0.bias', 'classifier.1.weight', 'classifier.1.bias', 'classifier.1.running_mean', 'classifier.1.running_var', 'classifier.1.num_batches_tracked', 'classifier.3.weight', 'classifier.3.bias'])




```python
# 읽어들인 모델 파라미터는 모델 아키텍처에 연결을 시켜줘야 함
# load_state_dict() 사용
last_model = VGG16()
last_model.to(device)
last_model.load_state_dict(last_state_dict)
```




    <All keys matched successfully>




```python
last_model.eval()
evaluation(last_model, testloader, loss_fn)  
```

    Test Loss : 0.119 Test Accuracy : 0.953
    


```python
# valid loss or accuracy 기준 best model
best_state_dict = torch.load('best_checkpoint.pth')
best_model = VGG16()
best_model.to(device)
best_model.load_state_dict(best_state_dict)
```




    <All keys matched successfully>




```python
best_model.eval()
evaluation(best_model, testloader, loss_fn)
```

    Test Loss : 0.120 Test Accuracy : 0.959
    


```python
#best_state_dict['conv_block1.0.weight']
```


```python
#last_state_dict['conv_block1.0.weight']
```


```python

```
