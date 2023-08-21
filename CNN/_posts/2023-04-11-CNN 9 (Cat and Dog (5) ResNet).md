---
tag: [Deep Learning, 딥러닝, pytorch, 파이토치, CNN, Cat and Dog Dataset, ResNet]
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




    
![png](/assets/images/2023-04-11-CNN 9 (Cat and Dog (5) ResNet)/output_1_0.png)
    




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

    Wed Apr 12 01:06:40 2023       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   65C    P8    10W /  70W |      0MiB / 15360MiB |      0%      Default |
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
!kaggle datasets download -d tongpython/cat-and-dog
# unzip(압축풀기)
!unzip -q cat-and-dog.zip -d catanddog/
```


    Saving kaggle.json to kaggle.json
    Downloading cat-and-dog.zip to /content
    100% 217M/218M [00:12<00:00, 20.2MB/s]
    100% 218M/218M [00:12<00:00, 18.4MB/s]
    


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


    
![png](/assets/images/2023-04-11-CNN 9 (Cat and Dog (5) ResNet)/output_26_0.png)
    


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




    <matplotlib.image.AxesImage at 0x7f7020a373d0>




    
![png](/assets/images/2023-04-11-CNN 9 (Cat and Dog (5) ResNet)/output_36_1.png)
    



```python
# 샘플의 그레이스케일 이미지
trans = transforms.Grayscale()
gray_image = trans(sample_image)
type(gray_image), gray_image.shape
plt.imshow(gray_image.squeeze(), cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7f6fff316d30>




    
![png](/assets/images/2023-04-11-CNN 9 (Cat and Dog (5) ResNet)/output_37_1.png)
    



```python
# 샘플 이미지를 random하게 rotation
trans = transforms.RandomRotation(degrees=(0, 180))
rotated_image = trans(sample_image)
plt.imshow(rotated_image.permute(1, 2, 0))
```




    <matplotlib.image.AxesImage at 0x7f6fff29b250>




    
![png](/assets/images/2023-04-11-CNN 9 (Cat and Dog (5) ResNet)/output_38_1.png)
    



```python
# 샘플 이미지를 random하게 crop
trans = transforms.RandomCrop(size=(128, 128))
cropped_image = trans(sample_image)
plt.imshow(cropped_image.permute(1, 2, 0))
```




    <matplotlib.image.AxesImage at 0x7f6fff26eb50>




    
![png](/assets/images/2023-04-11-CNN 9 (Cat and Dog (5) ResNet)/output_39_1.png)
    



```python
# 샘플 이미지를 random하게 horizontal flip
trans = transforms.RandomHorizontalFlip(p=0.3)
horizontal_flip_image = trans(sample_image)
plt.imshow(horizontal_flip_image.permute(1, 2, 0))
```




    <matplotlib.image.AxesImage at 0x7f6ffcab6640>




    
![png](/assets/images/2023-04-11-CNN 9 (Cat and Dog (5) ResNet)/output_40_1.png)
    


## 4. 모델 생성


```python
from IPython.display import Image
Image('./images/resnet.png', width=700)
```




    
![png](/assets/images/2023-04-11-CNN 9 (Cat and Dog (5) ResNet)/output_42_0.png)
    




```python
Image('./images/shortcut1.png', width=700)
```




    
![png](/assets/images/2023-04-11-CNN 9 (Cat and Dog (5) ResNet)/output_43_0.png)
    




```python
Image('./images/shortcut2.png', width=700)
```




    
![png](/assets/images/2023-04-11-CNN 9 (Cat and Dog (5) ResNet)/output_44_0.png)
    




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

**conv block 별 사이즈 확인**


```python
# conv1
conv1 = nn.Sequential(
                            # BatchNorm 계층은 편향값의 효과를 보완해주므로 관례상 생략략
                            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False), # [16, 65, 112, 112]
                            nn.BatchNorm2d(num_features=64),
                            nn.ReLU(),                        
                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                            ) # [16, 64, 56, 56]
conv1_out = conv1(images)                                     
conv1_out.shape                             
```




    torch.Size([4, 64, 56, 56])




```python
# conv2_x      
shortcut2 = nn.Sequential(
                                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1), 
                                nn.BatchNorm2d(num_features=256)                                  
                              )  
conv2_x = nn.Sequential(
                              ResBlock(in_channels=64, out_channels=64, shortcut=shortcut2, stride=1),                                 
                              ResBlock(in_channels=256, out_channels=64, shortcut=None, stride=1),
                              ResBlock(in_channels=256, out_channels=64, shortcut=None, stride=1)
                            ) # [16, 256, 56, 56]
conv2_x_out = conv2_x(conv1_out)                                     
conv2_x_out.shape                                     
```




    torch.Size([4, 256, 56, 56])




```python
# conv3_x
shortcut3 = nn.Sequential(
                                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2), 
                                nn.BatchNorm2d(num_features=512)                                  
                              )      
conv3_x = nn.Sequential(
                              ResBlock(in_channels=256, out_channels=128, shortcut=shortcut3, stride=2),
                              ResBlock(in_channels=512, out_channels=128, shortcut=None, stride=1),
                              ResBlock(in_channels=512, out_channels=128, shortcut=None, stride=1),
                              ResBlock(in_channels=512, out_channels=128, shortcut=None, stride=1)

                            ) # [16, 512, 28, 28] 
conv3_x_out = conv3_x(conv2_x_out)                                     
conv3_x_out.shape                                       
```




    torch.Size([4, 512, 28, 28])




```python
shortcut4 = nn.Sequential(
                                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=2), 
                                nn.BatchNorm2d(num_features=1024)                                  
                              )      
conv4_x = nn.Sequential(
                              ResBlock(in_channels=512, out_channels=256, shortcut=shortcut4, stride=2),
                              ResBlock(in_channels=1024, out_channels=256, shortcut=None, stride=1),
                              ResBlock(in_channels=1024, out_channels=256, shortcut=None, stride=1),
                              ResBlock(in_channels=1024, out_channels=256, shortcut=None, stride=1),
                              ResBlock(in_channels=1024, out_channels=256, shortcut=None, stride=1),
                              ResBlock(in_channels=1024, out_channels=256, shortcut=None, stride=1),                                                                                        
                            ) # [16, 1024, 14, 14]
conv4_x_out = conv4_x(conv3_x_out)                                     
conv4_x_out.shape  
```




    torch.Size([4, 1024, 14, 14])




```python
# conv5_x
shortcut5 = nn.Sequential(
                                nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=2), 
                                nn.BatchNorm2d(num_features=2048)                                  
                              )    
conv5_x = nn.Sequential(
                              ResBlock(in_channels=1024, out_channels=512, shortcut=shortcut5, stride=2),
                              ResBlock(in_channels=2048, out_channels=512, shortcut=None, stride=1),
                              ResBlock(in_channels=2048, out_channels=512, shortcut=None, stride=1),                                                                
                            ) # [16, 2048, 7, 7]  
conv5_x_out = conv5_x(conv4_x_out)                                     
conv5_x_out.shape  
```




    torch.Size([4, 2048, 7, 7])




```python
avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # [16, 2048, 1, 1]  
avg_pool_out = avg_pool(conv5_x_out)
avg_pool_out.shape
```




    torch.Size([4, 2048, 1, 1])




```python
class ResBlock(nn.Module):
  def __init__(self, in_channels, out_channels, shortcut=None, stride=1): # shortcut에 계층을 설정되어 있다면 그 계층을 통과한뒤 Add
    super().__init__()
    
    # 1x1 conv
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    self.batch_norm1 = nn.BatchNorm2d(out_channels)

    # 3x3 conv 
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1) # stride=2일 경우에는 downsampling
    self.batch_norm2 = nn.BatchNorm2d(out_channels)

    # 1x1 conv
    self.conv3 = nn.Conv2d(out_channels, out_channels*4, kernel_size=1, stride=1, padding=0)
    self.batch_norm3 = nn.BatchNorm2d(out_channels*4)

    self.shortcut = shortcut
    self.stride = stride
    self.relu = nn.ReLU()

  def forward(self, x):
    identity = x.clone()
    x = self.relu(self.batch_norm1(self.conv1(x)))
    x = self.relu(self.batch_norm2(self.conv2(x)))
    x = self.batch_norm3(self.conv3(x))
    
    # shortcut 계층을 바깥에서 설정한것을 적용할 때
    if self.shortcut is not None:
      identity = self.shortcut(identity)

    x += identity  # x = x+identity
    x = self.relu(x)

    return x

```


```python
class ResNet50(nn.Module):
  def __init__(self):
    super().__init__()
    # conv1
    self.conv1 = nn.Sequential(
                                # BatchNorm 계층은 편향값의 효과를 보완해주므로 관례상 생략략
                                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
                                nn.BatchNorm2d(num_features=64),
                                nn.ReLU(),                        
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                ) # [16, 64, 56, 56]
    # conv2_x      
    self.shortcut2 = nn.Sequential(
                                    nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1), 
                                    nn.BatchNorm2d(num_features=256)                                  
                                  )              
    self.conv2_x = nn.Sequential(
                                  ResBlock(in_channels=64, out_channels=64, shortcut=self.shortcut2, stride=1),                                 
                                  ResBlock(in_channels=256, out_channels=64, shortcut=None, stride=1),
                                  ResBlock(in_channels=256, out_channels=64, shortcut=None, stride=1)
                                ) # [16, 256, 56, 56]
    # conv3_x
    self.shortcut3 = nn.Sequential(
                                    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2), 
                                    nn.BatchNorm2d(num_features=512)                                  
                                  )      
    self.conv3_x = nn.Sequential(
                                  ResBlock(in_channels=256, out_channels=128, shortcut=self.shortcut3, stride=2),
                                  ResBlock(in_channels=512, out_channels=128, shortcut=None, stride=1),
                                  ResBlock(in_channels=512, out_channels=128, shortcut=None, stride=1),
                                  ResBlock(in_channels=512, out_channels=128, shortcut=None, stride=1)

                                ) # [16, 512, 28, 28]   
    # conv4_x
    self.shortcut4 = nn.Sequential(
                                    nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=2), 
                                    nn.BatchNorm2d(num_features=1024)                                  
                                  )      
    self.conv4_x = nn.Sequential(
                                 ResBlock(in_channels=512, out_channels=256, shortcut=self.shortcut4, stride=2),
                                 ResBlock(in_channels=1024, out_channels=256, shortcut=None, stride=1),
                                 ResBlock(in_channels=1024, out_channels=256, shortcut=None, stride=1),
                                 ResBlock(in_channels=1024, out_channels=256, shortcut=None, stride=1),
                                 ResBlock(in_channels=1024, out_channels=256, shortcut=None, stride=1),
                                 ResBlock(in_channels=1024, out_channels=256, shortcut=None, stride=1),                                                                                        
                                ) # [16, 1024, 14, 14] 
    # conv5_x
    self.shortcut5 = nn.Sequential(
                                    nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=2), 
                                    nn.BatchNorm2d(num_features=2048)                                  
                                  )    
    self.conv5_x = nn.Sequential(
                                 ResBlock(in_channels=1024, out_channels=512, shortcut=self.shortcut5, stride=2),
                                 ResBlock(in_channels=2048, out_channels=512, shortcut=None, stride=1),
                                 ResBlock(in_channels=2048, out_channels=512, shortcut=None, stride=1),                                                                
                                ) # [16, 2048, 7, 7]  

    self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # [16, 2048, 1, 1]                                                                                                                              

    self.classifier = nn.Sequential(
                                nn.Linear(in_features=2048, out_features=2),
                                # nn.BatchNorm1d(num_features=64),
                                # nn.ReLU(),
                                # nn.Linear(in_features=64, out_features=2)
                                )

  def forward(self, x):
    x = self.conv1(x) # [16, 64, 56, 56]
    x = self.conv2_x(x) # [16, 256, 56, 56]
    x = self.conv3_x(x) # [16, 512, 28, 28] 
    x = self.conv4_x(x) # [16, 1024, 14, 14] 
    x = self.conv5_x(x) # [16, 2048, 7, 7] 
    x = self.avg_pool(x) # [16, 2048, 1, 1] 
    
    # reshape할 형상 : (batch_size x 2048)
    # x = x.view(-1, 2048) # option 1 : view
    x = torch.flatten(x, 1) # option 2 : flatten 
    # x = x.reshape(x.shape[0], -1) # option 3 : reshape

    x = self.classifier(x)    
    return x
```


```python
model = ResNet50()
model.to(device)
model
```




    ResNet50(
      (conv1): Sequential(
        (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      )
      (shortcut2): Sequential(
        (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv2_x): Sequential(
        (0): ResBlock(
          (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (batch_norm2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential(
            (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (relu): ReLU()
        )
        (1): ResBlock(
          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (batch_norm2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU()
        )
        (2): ResBlock(
          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (batch_norm2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU()
        )
      )
      (shortcut3): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2))
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv3_x): Sequential(
        (0): ResBlock(
          (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (batch_norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2))
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (relu): ReLU()
        )
        (1): ResBlock(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (batch_norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU()
        )
        (2): ResBlock(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (batch_norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU()
        )
        (3): ResBlock(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (batch_norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU()
        )
      )
      (shortcut4): Sequential(
        (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2))
        (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv4_x): Sequential(
        (0): ResBlock(
          (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (batch_norm2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential(
            (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2))
            (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (relu): ReLU()
        )
        (1): ResBlock(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (batch_norm2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU()
        )
        (2): ResBlock(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (batch_norm2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU()
        )
        (3): ResBlock(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (batch_norm2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU()
        )
        (4): ResBlock(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (batch_norm2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU()
        )
        (5): ResBlock(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (batch_norm2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU()
        )
      )
      (shortcut5): Sequential(
        (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2))
        (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv5_x): Sequential(
        (0): ResBlock(
          (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (batch_norm2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential(
            (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2))
            (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (relu): ReLU()
        )
        (1): ResBlock(
          (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (batch_norm2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU()
        )
        (2): ResBlock(
          (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (batch_norm2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1))
          (batch_norm3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU()
        )
      )
      (avg_pool): AdaptiveAvgPool2d(output_size=(1, 1))
      (classifier): Sequential(
        (0): Linear(in_features=2048, out_features=2, bias=True)
      )
    )




```python
out = model(images.to(device))
out.shape
```




    torch.Size([4, 2])




```python
for name, parameter in model.named_parameters():
  print(name, parameter.size())
```

    conv1.0.weight torch.Size([64, 3, 7, 7])
    conv1.1.weight torch.Size([64])
    conv1.1.bias torch.Size([64])
    shortcut2.0.weight torch.Size([256, 64, 1, 1])
    shortcut2.0.bias torch.Size([256])
    shortcut2.1.weight torch.Size([256])
    shortcut2.1.bias torch.Size([256])
    conv2_x.0.conv1.weight torch.Size([64, 64, 1, 1])
    conv2_x.0.conv1.bias torch.Size([64])
    conv2_x.0.batch_norm1.weight torch.Size([64])
    conv2_x.0.batch_norm1.bias torch.Size([64])
    conv2_x.0.conv2.weight torch.Size([64, 64, 3, 3])
    conv2_x.0.conv2.bias torch.Size([64])
    conv2_x.0.batch_norm2.weight torch.Size([64])
    conv2_x.0.batch_norm2.bias torch.Size([64])
    conv2_x.0.conv3.weight torch.Size([256, 64, 1, 1])
    conv2_x.0.conv3.bias torch.Size([256])
    conv2_x.0.batch_norm3.weight torch.Size([256])
    conv2_x.0.batch_norm3.bias torch.Size([256])
    conv2_x.1.conv1.weight torch.Size([64, 256, 1, 1])
    conv2_x.1.conv1.bias torch.Size([64])
    conv2_x.1.batch_norm1.weight torch.Size([64])
    conv2_x.1.batch_norm1.bias torch.Size([64])
    conv2_x.1.conv2.weight torch.Size([64, 64, 3, 3])
    conv2_x.1.conv2.bias torch.Size([64])
    conv2_x.1.batch_norm2.weight torch.Size([64])
    conv2_x.1.batch_norm2.bias torch.Size([64])
    conv2_x.1.conv3.weight torch.Size([256, 64, 1, 1])
    conv2_x.1.conv3.bias torch.Size([256])
    conv2_x.1.batch_norm3.weight torch.Size([256])
    conv2_x.1.batch_norm3.bias torch.Size([256])
    conv2_x.2.conv1.weight torch.Size([64, 256, 1, 1])
    conv2_x.2.conv1.bias torch.Size([64])
    conv2_x.2.batch_norm1.weight torch.Size([64])
    conv2_x.2.batch_norm1.bias torch.Size([64])
    conv2_x.2.conv2.weight torch.Size([64, 64, 3, 3])
    conv2_x.2.conv2.bias torch.Size([64])
    conv2_x.2.batch_norm2.weight torch.Size([64])
    conv2_x.2.batch_norm2.bias torch.Size([64])
    conv2_x.2.conv3.weight torch.Size([256, 64, 1, 1])
    conv2_x.2.conv3.bias torch.Size([256])
    conv2_x.2.batch_norm3.weight torch.Size([256])
    conv2_x.2.batch_norm3.bias torch.Size([256])
    shortcut3.0.weight torch.Size([512, 256, 1, 1])
    shortcut3.0.bias torch.Size([512])
    shortcut3.1.weight torch.Size([512])
    shortcut3.1.bias torch.Size([512])
    conv3_x.0.conv1.weight torch.Size([128, 256, 1, 1])
    conv3_x.0.conv1.bias torch.Size([128])
    conv3_x.0.batch_norm1.weight torch.Size([128])
    conv3_x.0.batch_norm1.bias torch.Size([128])
    conv3_x.0.conv2.weight torch.Size([128, 128, 3, 3])
    conv3_x.0.conv2.bias torch.Size([128])
    conv3_x.0.batch_norm2.weight torch.Size([128])
    conv3_x.0.batch_norm2.bias torch.Size([128])
    conv3_x.0.conv3.weight torch.Size([512, 128, 1, 1])
    conv3_x.0.conv3.bias torch.Size([512])
    conv3_x.0.batch_norm3.weight torch.Size([512])
    conv3_x.0.batch_norm3.bias torch.Size([512])
    conv3_x.1.conv1.weight torch.Size([128, 512, 1, 1])
    conv3_x.1.conv1.bias torch.Size([128])
    conv3_x.1.batch_norm1.weight torch.Size([128])
    conv3_x.1.batch_norm1.bias torch.Size([128])
    conv3_x.1.conv2.weight torch.Size([128, 128, 3, 3])
    conv3_x.1.conv2.bias torch.Size([128])
    conv3_x.1.batch_norm2.weight torch.Size([128])
    conv3_x.1.batch_norm2.bias torch.Size([128])
    conv3_x.1.conv3.weight torch.Size([512, 128, 1, 1])
    conv3_x.1.conv3.bias torch.Size([512])
    conv3_x.1.batch_norm3.weight torch.Size([512])
    conv3_x.1.batch_norm3.bias torch.Size([512])
    conv3_x.2.conv1.weight torch.Size([128, 512, 1, 1])
    conv3_x.2.conv1.bias torch.Size([128])
    conv3_x.2.batch_norm1.weight torch.Size([128])
    conv3_x.2.batch_norm1.bias torch.Size([128])
    conv3_x.2.conv2.weight torch.Size([128, 128, 3, 3])
    conv3_x.2.conv2.bias torch.Size([128])
    conv3_x.2.batch_norm2.weight torch.Size([128])
    conv3_x.2.batch_norm2.bias torch.Size([128])
    conv3_x.2.conv3.weight torch.Size([512, 128, 1, 1])
    conv3_x.2.conv3.bias torch.Size([512])
    conv3_x.2.batch_norm3.weight torch.Size([512])
    conv3_x.2.batch_norm3.bias torch.Size([512])
    conv3_x.3.conv1.weight torch.Size([128, 512, 1, 1])
    conv3_x.3.conv1.bias torch.Size([128])
    conv3_x.3.batch_norm1.weight torch.Size([128])
    conv3_x.3.batch_norm1.bias torch.Size([128])
    conv3_x.3.conv2.weight torch.Size([128, 128, 3, 3])
    conv3_x.3.conv2.bias torch.Size([128])
    conv3_x.3.batch_norm2.weight torch.Size([128])
    conv3_x.3.batch_norm2.bias torch.Size([128])
    conv3_x.3.conv3.weight torch.Size([512, 128, 1, 1])
    conv3_x.3.conv3.bias torch.Size([512])
    conv3_x.3.batch_norm3.weight torch.Size([512])
    conv3_x.3.batch_norm3.bias torch.Size([512])
    shortcut4.0.weight torch.Size([1024, 512, 1, 1])
    shortcut4.0.bias torch.Size([1024])
    shortcut4.1.weight torch.Size([1024])
    shortcut4.1.bias torch.Size([1024])
    conv4_x.0.conv1.weight torch.Size([256, 512, 1, 1])
    conv4_x.0.conv1.bias torch.Size([256])
    conv4_x.0.batch_norm1.weight torch.Size([256])
    conv4_x.0.batch_norm1.bias torch.Size([256])
    conv4_x.0.conv2.weight torch.Size([256, 256, 3, 3])
    conv4_x.0.conv2.bias torch.Size([256])
    conv4_x.0.batch_norm2.weight torch.Size([256])
    conv4_x.0.batch_norm2.bias torch.Size([256])
    conv4_x.0.conv3.weight torch.Size([1024, 256, 1, 1])
    conv4_x.0.conv3.bias torch.Size([1024])
    conv4_x.0.batch_norm3.weight torch.Size([1024])
    conv4_x.0.batch_norm3.bias torch.Size([1024])
    conv4_x.1.conv1.weight torch.Size([256, 1024, 1, 1])
    conv4_x.1.conv1.bias torch.Size([256])
    conv4_x.1.batch_norm1.weight torch.Size([256])
    conv4_x.1.batch_norm1.bias torch.Size([256])
    conv4_x.1.conv2.weight torch.Size([256, 256, 3, 3])
    conv4_x.1.conv2.bias torch.Size([256])
    conv4_x.1.batch_norm2.weight torch.Size([256])
    conv4_x.1.batch_norm2.bias torch.Size([256])
    conv4_x.1.conv3.weight torch.Size([1024, 256, 1, 1])
    conv4_x.1.conv3.bias torch.Size([1024])
    conv4_x.1.batch_norm3.weight torch.Size([1024])
    conv4_x.1.batch_norm3.bias torch.Size([1024])
    conv4_x.2.conv1.weight torch.Size([256, 1024, 1, 1])
    conv4_x.2.conv1.bias torch.Size([256])
    conv4_x.2.batch_norm1.weight torch.Size([256])
    conv4_x.2.batch_norm1.bias torch.Size([256])
    conv4_x.2.conv2.weight torch.Size([256, 256, 3, 3])
    conv4_x.2.conv2.bias torch.Size([256])
    conv4_x.2.batch_norm2.weight torch.Size([256])
    conv4_x.2.batch_norm2.bias torch.Size([256])
    conv4_x.2.conv3.weight torch.Size([1024, 256, 1, 1])
    conv4_x.2.conv3.bias torch.Size([1024])
    conv4_x.2.batch_norm3.weight torch.Size([1024])
    conv4_x.2.batch_norm3.bias torch.Size([1024])
    conv4_x.3.conv1.weight torch.Size([256, 1024, 1, 1])
    conv4_x.3.conv1.bias torch.Size([256])
    conv4_x.3.batch_norm1.weight torch.Size([256])
    conv4_x.3.batch_norm1.bias torch.Size([256])
    conv4_x.3.conv2.weight torch.Size([256, 256, 3, 3])
    conv4_x.3.conv2.bias torch.Size([256])
    conv4_x.3.batch_norm2.weight torch.Size([256])
    conv4_x.3.batch_norm2.bias torch.Size([256])
    conv4_x.3.conv3.weight torch.Size([1024, 256, 1, 1])
    conv4_x.3.conv3.bias torch.Size([1024])
    conv4_x.3.batch_norm3.weight torch.Size([1024])
    conv4_x.3.batch_norm3.bias torch.Size([1024])
    conv4_x.4.conv1.weight torch.Size([256, 1024, 1, 1])
    conv4_x.4.conv1.bias torch.Size([256])
    conv4_x.4.batch_norm1.weight torch.Size([256])
    conv4_x.4.batch_norm1.bias torch.Size([256])
    conv4_x.4.conv2.weight torch.Size([256, 256, 3, 3])
    conv4_x.4.conv2.bias torch.Size([256])
    conv4_x.4.batch_norm2.weight torch.Size([256])
    conv4_x.4.batch_norm2.bias torch.Size([256])
    conv4_x.4.conv3.weight torch.Size([1024, 256, 1, 1])
    conv4_x.4.conv3.bias torch.Size([1024])
    conv4_x.4.batch_norm3.weight torch.Size([1024])
    conv4_x.4.batch_norm3.bias torch.Size([1024])
    conv4_x.5.conv1.weight torch.Size([256, 1024, 1, 1])
    conv4_x.5.conv1.bias torch.Size([256])
    conv4_x.5.batch_norm1.weight torch.Size([256])
    conv4_x.5.batch_norm1.bias torch.Size([256])
    conv4_x.5.conv2.weight torch.Size([256, 256, 3, 3])
    conv4_x.5.conv2.bias torch.Size([256])
    conv4_x.5.batch_norm2.weight torch.Size([256])
    conv4_x.5.batch_norm2.bias torch.Size([256])
    conv4_x.5.conv3.weight torch.Size([1024, 256, 1, 1])
    conv4_x.5.conv3.bias torch.Size([1024])
    conv4_x.5.batch_norm3.weight torch.Size([1024])
    conv4_x.5.batch_norm3.bias torch.Size([1024])
    shortcut5.0.weight torch.Size([2048, 1024, 1, 1])
    shortcut5.0.bias torch.Size([2048])
    shortcut5.1.weight torch.Size([2048])
    shortcut5.1.bias torch.Size([2048])
    conv5_x.0.conv1.weight torch.Size([512, 1024, 1, 1])
    conv5_x.0.conv1.bias torch.Size([512])
    conv5_x.0.batch_norm1.weight torch.Size([512])
    conv5_x.0.batch_norm1.bias torch.Size([512])
    conv5_x.0.conv2.weight torch.Size([512, 512, 3, 3])
    conv5_x.0.conv2.bias torch.Size([512])
    conv5_x.0.batch_norm2.weight torch.Size([512])
    conv5_x.0.batch_norm2.bias torch.Size([512])
    conv5_x.0.conv3.weight torch.Size([2048, 512, 1, 1])
    conv5_x.0.conv3.bias torch.Size([2048])
    conv5_x.0.batch_norm3.weight torch.Size([2048])
    conv5_x.0.batch_norm3.bias torch.Size([2048])
    conv5_x.1.conv1.weight torch.Size([512, 2048, 1, 1])
    conv5_x.1.conv1.bias torch.Size([512])
    conv5_x.1.batch_norm1.weight torch.Size([512])
    conv5_x.1.batch_norm1.bias torch.Size([512])
    conv5_x.1.conv2.weight torch.Size([512, 512, 3, 3])
    conv5_x.1.conv2.bias torch.Size([512])
    conv5_x.1.batch_norm2.weight torch.Size([512])
    conv5_x.1.batch_norm2.bias torch.Size([512])
    conv5_x.1.conv3.weight torch.Size([2048, 512, 1, 1])
    conv5_x.1.conv3.bias torch.Size([2048])
    conv5_x.1.batch_norm3.weight torch.Size([2048])
    conv5_x.1.batch_norm3.bias torch.Size([2048])
    conv5_x.2.conv1.weight torch.Size([512, 2048, 1, 1])
    conv5_x.2.conv1.bias torch.Size([512])
    conv5_x.2.batch_norm1.weight torch.Size([512])
    conv5_x.2.batch_norm1.bias torch.Size([512])
    conv5_x.2.conv2.weight torch.Size([512, 512, 3, 3])
    conv5_x.2.conv2.bias torch.Size([512])
    conv5_x.2.batch_norm2.weight torch.Size([512])
    conv5_x.2.batch_norm2.bias torch.Size([512])
    conv5_x.2.conv3.weight torch.Size([2048, 512, 1, 1])
    conv5_x.2.conv3.bias torch.Size([2048])
    conv5_x.2.batch_norm3.weight torch.Size([2048])
    conv5_x.2.batch_norm3.bias torch.Size([2048])
    classifier.0.weight torch.Size([2, 2048])
    classifier.0.bias torch.Size([2])
    

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
                Conv2d-1         [-1, 64, 112, 112]           9,408
           BatchNorm2d-2         [-1, 64, 112, 112]             128
                  ReLU-3         [-1, 64, 112, 112]               0
             MaxPool2d-4           [-1, 64, 56, 56]               0
                Conv2d-5           [-1, 64, 56, 56]           4,160
           BatchNorm2d-6           [-1, 64, 56, 56]             128
                  ReLU-7           [-1, 64, 56, 56]               0
                Conv2d-8           [-1, 64, 56, 56]          36,928
           BatchNorm2d-9           [-1, 64, 56, 56]             128
                 ReLU-10           [-1, 64, 56, 56]               0
               Conv2d-11          [-1, 256, 56, 56]          16,640
          BatchNorm2d-12          [-1, 256, 56, 56]             512
               Conv2d-13          [-1, 256, 56, 56]          16,640
               Conv2d-14          [-1, 256, 56, 56]          16,640
          BatchNorm2d-15          [-1, 256, 56, 56]             512
          BatchNorm2d-16          [-1, 256, 56, 56]             512
                 ReLU-17          [-1, 256, 56, 56]               0
             ResBlock-18          [-1, 256, 56, 56]               0
               Conv2d-19           [-1, 64, 56, 56]          16,448
          BatchNorm2d-20           [-1, 64, 56, 56]             128
                 ReLU-21           [-1, 64, 56, 56]               0
               Conv2d-22           [-1, 64, 56, 56]          36,928
          BatchNorm2d-23           [-1, 64, 56, 56]             128
                 ReLU-24           [-1, 64, 56, 56]               0
               Conv2d-25          [-1, 256, 56, 56]          16,640
          BatchNorm2d-26          [-1, 256, 56, 56]             512
                 ReLU-27          [-1, 256, 56, 56]               0
             ResBlock-28          [-1, 256, 56, 56]               0
               Conv2d-29           [-1, 64, 56, 56]          16,448
          BatchNorm2d-30           [-1, 64, 56, 56]             128
                 ReLU-31           [-1, 64, 56, 56]               0
               Conv2d-32           [-1, 64, 56, 56]          36,928
          BatchNorm2d-33           [-1, 64, 56, 56]             128
                 ReLU-34           [-1, 64, 56, 56]               0
               Conv2d-35          [-1, 256, 56, 56]          16,640
          BatchNorm2d-36          [-1, 256, 56, 56]             512
                 ReLU-37          [-1, 256, 56, 56]               0
             ResBlock-38          [-1, 256, 56, 56]               0
               Conv2d-39          [-1, 128, 56, 56]          32,896
          BatchNorm2d-40          [-1, 128, 56, 56]             256
                 ReLU-41          [-1, 128, 56, 56]               0
               Conv2d-42          [-1, 128, 28, 28]         147,584
          BatchNorm2d-43          [-1, 128, 28, 28]             256
                 ReLU-44          [-1, 128, 28, 28]               0
               Conv2d-45          [-1, 512, 28, 28]          66,048
          BatchNorm2d-46          [-1, 512, 28, 28]           1,024
               Conv2d-47          [-1, 512, 28, 28]         131,584
               Conv2d-48          [-1, 512, 28, 28]         131,584
          BatchNorm2d-49          [-1, 512, 28, 28]           1,024
          BatchNorm2d-50          [-1, 512, 28, 28]           1,024
                 ReLU-51          [-1, 512, 28, 28]               0
             ResBlock-52          [-1, 512, 28, 28]               0
               Conv2d-53          [-1, 128, 28, 28]          65,664
          BatchNorm2d-54          [-1, 128, 28, 28]             256
                 ReLU-55          [-1, 128, 28, 28]               0
               Conv2d-56          [-1, 128, 28, 28]         147,584
          BatchNorm2d-57          [-1, 128, 28, 28]             256
                 ReLU-58          [-1, 128, 28, 28]               0
               Conv2d-59          [-1, 512, 28, 28]          66,048
          BatchNorm2d-60          [-1, 512, 28, 28]           1,024
                 ReLU-61          [-1, 512, 28, 28]               0
             ResBlock-62          [-1, 512, 28, 28]               0
               Conv2d-63          [-1, 128, 28, 28]          65,664
          BatchNorm2d-64          [-1, 128, 28, 28]             256
                 ReLU-65          [-1, 128, 28, 28]               0
               Conv2d-66          [-1, 128, 28, 28]         147,584
          BatchNorm2d-67          [-1, 128, 28, 28]             256
                 ReLU-68          [-1, 128, 28, 28]               0
               Conv2d-69          [-1, 512, 28, 28]          66,048
          BatchNorm2d-70          [-1, 512, 28, 28]           1,024
                 ReLU-71          [-1, 512, 28, 28]               0
             ResBlock-72          [-1, 512, 28, 28]               0
               Conv2d-73          [-1, 128, 28, 28]          65,664
          BatchNorm2d-74          [-1, 128, 28, 28]             256
                 ReLU-75          [-1, 128, 28, 28]               0
               Conv2d-76          [-1, 128, 28, 28]         147,584
          BatchNorm2d-77          [-1, 128, 28, 28]             256
                 ReLU-78          [-1, 128, 28, 28]               0
               Conv2d-79          [-1, 512, 28, 28]          66,048
          BatchNorm2d-80          [-1, 512, 28, 28]           1,024
                 ReLU-81          [-1, 512, 28, 28]               0
             ResBlock-82          [-1, 512, 28, 28]               0
               Conv2d-83          [-1, 256, 28, 28]         131,328
          BatchNorm2d-84          [-1, 256, 28, 28]             512
                 ReLU-85          [-1, 256, 28, 28]               0
               Conv2d-86          [-1, 256, 14, 14]         590,080
          BatchNorm2d-87          [-1, 256, 14, 14]             512
                 ReLU-88          [-1, 256, 14, 14]               0
               Conv2d-89         [-1, 1024, 14, 14]         263,168
          BatchNorm2d-90         [-1, 1024, 14, 14]           2,048
               Conv2d-91         [-1, 1024, 14, 14]         525,312
               Conv2d-92         [-1, 1024, 14, 14]         525,312
          BatchNorm2d-93         [-1, 1024, 14, 14]           2,048
          BatchNorm2d-94         [-1, 1024, 14, 14]           2,048
                 ReLU-95         [-1, 1024, 14, 14]               0
             ResBlock-96         [-1, 1024, 14, 14]               0
               Conv2d-97          [-1, 256, 14, 14]         262,400
          BatchNorm2d-98          [-1, 256, 14, 14]             512
                 ReLU-99          [-1, 256, 14, 14]               0
              Conv2d-100          [-1, 256, 14, 14]         590,080
         BatchNorm2d-101          [-1, 256, 14, 14]             512
                ReLU-102          [-1, 256, 14, 14]               0
              Conv2d-103         [-1, 1024, 14, 14]         263,168
         BatchNorm2d-104         [-1, 1024, 14, 14]           2,048
                ReLU-105         [-1, 1024, 14, 14]               0
            ResBlock-106         [-1, 1024, 14, 14]               0
              Conv2d-107          [-1, 256, 14, 14]         262,400
         BatchNorm2d-108          [-1, 256, 14, 14]             512
                ReLU-109          [-1, 256, 14, 14]               0
              Conv2d-110          [-1, 256, 14, 14]         590,080
         BatchNorm2d-111          [-1, 256, 14, 14]             512
                ReLU-112          [-1, 256, 14, 14]               0
              Conv2d-113         [-1, 1024, 14, 14]         263,168
         BatchNorm2d-114         [-1, 1024, 14, 14]           2,048
                ReLU-115         [-1, 1024, 14, 14]               0
            ResBlock-116         [-1, 1024, 14, 14]               0
              Conv2d-117          [-1, 256, 14, 14]         262,400
         BatchNorm2d-118          [-1, 256, 14, 14]             512
                ReLU-119          [-1, 256, 14, 14]               0
              Conv2d-120          [-1, 256, 14, 14]         590,080
         BatchNorm2d-121          [-1, 256, 14, 14]             512
                ReLU-122          [-1, 256, 14, 14]               0
              Conv2d-123         [-1, 1024, 14, 14]         263,168
         BatchNorm2d-124         [-1, 1024, 14, 14]           2,048
                ReLU-125         [-1, 1024, 14, 14]               0
            ResBlock-126         [-1, 1024, 14, 14]               0
              Conv2d-127          [-1, 256, 14, 14]         262,400
         BatchNorm2d-128          [-1, 256, 14, 14]             512
                ReLU-129          [-1, 256, 14, 14]               0
              Conv2d-130          [-1, 256, 14, 14]         590,080
         BatchNorm2d-131          [-1, 256, 14, 14]             512
                ReLU-132          [-1, 256, 14, 14]               0
              Conv2d-133         [-1, 1024, 14, 14]         263,168
         BatchNorm2d-134         [-1, 1024, 14, 14]           2,048
                ReLU-135         [-1, 1024, 14, 14]               0
            ResBlock-136         [-1, 1024, 14, 14]               0
              Conv2d-137          [-1, 256, 14, 14]         262,400
         BatchNorm2d-138          [-1, 256, 14, 14]             512
                ReLU-139          [-1, 256, 14, 14]               0
              Conv2d-140          [-1, 256, 14, 14]         590,080
         BatchNorm2d-141          [-1, 256, 14, 14]             512
                ReLU-142          [-1, 256, 14, 14]               0
              Conv2d-143         [-1, 1024, 14, 14]         263,168
         BatchNorm2d-144         [-1, 1024, 14, 14]           2,048
                ReLU-145         [-1, 1024, 14, 14]               0
            ResBlock-146         [-1, 1024, 14, 14]               0
              Conv2d-147          [-1, 512, 14, 14]         524,800
         BatchNorm2d-148          [-1, 512, 14, 14]           1,024
                ReLU-149          [-1, 512, 14, 14]               0
              Conv2d-150            [-1, 512, 7, 7]       2,359,808
         BatchNorm2d-151            [-1, 512, 7, 7]           1,024
                ReLU-152            [-1, 512, 7, 7]               0
              Conv2d-153           [-1, 2048, 7, 7]       1,050,624
         BatchNorm2d-154           [-1, 2048, 7, 7]           4,096
              Conv2d-155           [-1, 2048, 7, 7]       2,099,200
              Conv2d-156           [-1, 2048, 7, 7]       2,099,200
         BatchNorm2d-157           [-1, 2048, 7, 7]           4,096
         BatchNorm2d-158           [-1, 2048, 7, 7]           4,096
                ReLU-159           [-1, 2048, 7, 7]               0
            ResBlock-160           [-1, 2048, 7, 7]               0
              Conv2d-161            [-1, 512, 7, 7]       1,049,088
         BatchNorm2d-162            [-1, 512, 7, 7]           1,024
                ReLU-163            [-1, 512, 7, 7]               0
              Conv2d-164            [-1, 512, 7, 7]       2,359,808
         BatchNorm2d-165            [-1, 512, 7, 7]           1,024
                ReLU-166            [-1, 512, 7, 7]               0
              Conv2d-167           [-1, 2048, 7, 7]       1,050,624
         BatchNorm2d-168           [-1, 2048, 7, 7]           4,096
                ReLU-169           [-1, 2048, 7, 7]               0
            ResBlock-170           [-1, 2048, 7, 7]               0
              Conv2d-171            [-1, 512, 7, 7]       1,049,088
         BatchNorm2d-172            [-1, 512, 7, 7]           1,024
                ReLU-173            [-1, 512, 7, 7]               0
              Conv2d-174            [-1, 512, 7, 7]       2,359,808
         BatchNorm2d-175            [-1, 512, 7, 7]           1,024
                ReLU-176            [-1, 512, 7, 7]               0
              Conv2d-177           [-1, 2048, 7, 7]       1,050,624
         BatchNorm2d-178           [-1, 2048, 7, 7]           4,096
                ReLU-179           [-1, 2048, 7, 7]               0
            ResBlock-180           [-1, 2048, 7, 7]               0
    AdaptiveAvgPool2d-181           [-1, 2048, 1, 1]               0
              Linear-182                    [-1, 2]           4,098
    ================================================================
    Total params: 26,319,042
    Trainable params: 26,319,042
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.57
    Forward/backward pass size (MB): 309.52
    Params size (MB): 100.40
    Estimated Total Size (MB): 410.49
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

    Epoch : 1/55....... Train Loss : 0.708 Valid Loss : 0.693 Valid Accuracy : 0.628
    Epoch : 2/55....... Train Loss : 0.645 Valid Loss : 0.838 Valid Accuracy : 0.690
    trigger :  1
    Epoch : 3/55....... Train Loss : 0.606 Valid Loss : 0.588 Valid Accuracy : 0.698
    Epoch : 4/55....... Train Loss : 0.568 Valid Loss : 0.629 Valid Accuracy : 0.703
    trigger :  1
    Epoch : 5/55....... Train Loss : 0.521 Valid Loss : 0.656 Valid Accuracy : 0.691
    trigger :  2
    Epoch : 6/55....... Train Loss : 0.479 Valid Loss : 0.484 Valid Accuracy : 0.774
    Epoch : 7/55....... Train Loss : 0.458 Valid Loss : 0.504 Valid Accuracy : 0.780
    trigger :  1
    Epoch : 8/55....... Train Loss : 0.411 Valid Loss : 0.390 Valid Accuracy : 0.837
    Epoch : 9/55....... Train Loss : 0.382 Valid Loss : 0.480 Valid Accuracy : 0.800
    trigger :  1
    Epoch : 10/55....... Train Loss : 0.340 Valid Loss : 0.354 Valid Accuracy : 0.848
    Epoch : 11/55....... Train Loss : 0.307 Valid Loss : 0.331 Valid Accuracy : 0.861
    Epoch : 12/55....... Train Loss : 0.278 Valid Loss : 0.376 Valid Accuracy : 0.836
    trigger :  1
    Epoch : 13/55....... Train Loss : 0.251 Valid Loss : 0.270 Valid Accuracy : 0.881
    Epoch : 14/55....... Train Loss : 0.230 Valid Loss : 0.253 Valid Accuracy : 0.903
    Epoch : 15/55....... Train Loss : 0.211 Valid Loss : 0.268 Valid Accuracy : 0.896
    trigger :  1
    Epoch : 16/55....... Train Loss : 0.199 Valid Loss : 0.283 Valid Accuracy : 0.879
    trigger :  2
    Epoch : 17/55....... Train Loss : 0.188 Valid Loss : 0.246 Valid Accuracy : 0.904
    Epoch : 18/55....... Train Loss : 0.178 Valid Loss : 0.378 Valid Accuracy : 0.844
    trigger :  1
    Epoch : 19/55....... Train Loss : 0.164 Valid Loss : 0.197 Valid Accuracy : 0.921
    Epoch : 20/55....... Train Loss : 0.150 Valid Loss : 0.255 Valid Accuracy : 0.894
    trigger :  1
    Epoch : 21/55....... Train Loss : 0.138 Valid Loss : 0.229 Valid Accuracy : 0.904
    trigger :  2
    Epoch : 22/55....... Train Loss : 0.136 Valid Loss : 0.262 Valid Accuracy : 0.904
    trigger :  3
    Epoch : 23/55....... Train Loss : 0.108 Valid Loss : 0.267 Valid Accuracy : 0.901
    trigger :  4
    Epoch : 24/55....... Train Loss : 0.114 Valid Loss : 0.225 Valid Accuracy : 0.919
    trigger :  5
    Epoch 00024: reducing learning rate of group 0 to 1.0000e-05.
    Epoch : 25/55....... Train Loss : 0.062 Valid Loss : 0.175 Valid Accuracy : 0.935
    Epoch : 26/55....... Train Loss : 0.042 Valid Loss : 0.175 Valid Accuracy : 0.940
    Epoch : 27/55....... Train Loss : 0.036 Valid Loss : 0.168 Valid Accuracy : 0.942
    Epoch : 28/55....... Train Loss : 0.030 Valid Loss : 0.166 Valid Accuracy : 0.944
    Epoch : 29/55....... Train Loss : 0.025 Valid Loss : 0.170 Valid Accuracy : 0.939
    trigger :  1
    Epoch : 30/55....... Train Loss : 0.017 Valid Loss : 0.178 Valid Accuracy : 0.933
    trigger :  2
    Epoch : 31/55....... Train Loss : 0.019 Valid Loss : 0.189 Valid Accuracy : 0.938
    trigger :  3
    Epoch : 32/55....... Train Loss : 0.020 Valid Loss : 0.171 Valid Accuracy : 0.936
    trigger :  4
    Epoch : 33/55....... Train Loss : 0.015 Valid Loss : 0.212 Valid Accuracy : 0.931
    trigger :  5
    Epoch 00033: reducing learning rate of group 0 to 1.0000e-06.
    Epoch : 34/55....... Train Loss : 0.014 Valid Loss : 0.190 Valid Accuracy : 0.939
    trigger :  6
    Epoch : 35/55....... Train Loss : 0.011 Valid Loss : 0.194 Valid Accuracy : 0.939
    trigger :  7
    Epoch : 36/55....... Train Loss : 0.011 Valid Loss : 0.206 Valid Accuracy : 0.932
    trigger :  8
    Early Stopping !!!
    Training loop is finished !!
    CPU times: user 1h 46min 17s, sys: 24.8 s, total: 1h 46min 42s
    Wall time: 1h 14min 2s
    


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
    


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-99-96e05baa1841> in <cell line: 9>()
          7 # random한 index로 이미지 한장 준비하기
          8 rnd_idx = 10
    ----> 9 print(images[rnd_idx].shape, labels[rnd_idx])
    

    IndexError: index 10 is out of bounds for dimension 0 with size 4



```python
images[rnd_idx].shape
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-100-81630fd2f131> in <cell line: 1>()
    ----> 1 images[rnd_idx].shape
    

    IndexError: index 10 is out of bounds for dimension 0 with size 4



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


```python
print("pred:", pred, "labels:", labels[rnd_idx])
print(labels_map[pred.cpu().item()], labels_map[labels[rnd_idx].cpu().item()])
plt.imshow(images[rnd_idx].permute(1, 2, 0).cpu())
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-101-3b818cac3b8f> in <cell line: 1>()
    ----> 1 print("pred:", pred, "labels:", labels[rnd_idx])
          2 print(labels_map[pred.cpu().item()], labels_map[labels[rnd_idx].cpu().item()])
          3 plt.imshow(images[rnd_idx].permute(1, 2, 0).cpu())
    

    NameError: name 'pred' is not defined


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

    Test Loss : 0.206 Test Accuracy : 0.936
    

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


```python
# 읽어들인 모델 파라미터는 모델 아키텍처에 연결을 시켜줘야 함
# load_state_dict() 사용
last_model = ResNet50()
last_model.to(device)
last_model.load_state_dict(last_state_dict)
```




    <All keys matched successfully>




```python
last_model.eval()
evaluation(last_model, testloader, loss_fn)  
```

    Test Loss : 0.216 Test Accuracy : 0.937
    


```python
# valid loss or accuracy 기준 best model
best_state_dict = torch.load('best_checkpoint.pth')
best_model = ResNet50()
best_model.to(device)
best_model.load_state_dict(best_state_dict)
```




    <All keys matched successfully>




```python
best_model.eval()
evaluation(best_model, testloader, loss_fn)
```

    Test Loss : 0.197 Test Accuracy : 0.939
    


```python
#best_state_dict['conv_block1.0.weight']
```


```python
#last_state_dict['conv_block1.0.weight']
```


```python

```
