---
tag: [Deep Learning, 딥러닝, pytorch, 파이토치]
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




    
![png](/assets/images/2023-04-08-Deep Learning 10 (Fashion MNIST in Pytorch (2) 배치정규화, 규제, 모델저장)/output_1_0.png)
    




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
trainset = datasets.FashionMNIST('./datasets/', download=True, train=True, transform = transforms.ToTensor())
testset = datasets.FashionMNIST('./datasets/', download=True, train=False, transform = transforms.ToTensor())
```

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to ./datasets/FashionMNIST/raw/train-images-idx3-ubyte.gz
    

    100%|██████████| 26421880/26421880 [00:14<00:00, 1809538.75it/s]
    

    Extracting ./datasets/FashionMNIST/raw/train-images-idx3-ubyte.gz to ./datasets/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to ./datasets/FashionMNIST/raw/train-labels-idx1-ubyte.gz
    

    100%|██████████| 29515/29515 [00:00<00:00, 119076.02it/s]
    

    Extracting ./datasets/FashionMNIST/raw/train-labels-idx1-ubyte.gz to ./datasets/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to ./datasets/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
    

    100%|██████████| 4422102/4422102 [00:09<00:00, 484805.94it/s] 
    

    Extracting ./datasets/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to ./datasets/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to ./datasets/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
    

    100%|██████████| 5148/5148 [00:00<00:00, 3797445.83it/s]
    

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

    torch.Size([1, 28, 28]) 3
    

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


    
![png](/assets/images/2023-04-08-Deep Learning 10 (Fashion MNIST in Pytorch (2) 배치정규화, 규제, 모델저장)/output_10_0.png)
    


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
Image('./images/mlp_mnist.png', width=500)
```




    
![png](/assets/images/2023-04-08-Deep Learning 10 (Fashion MNIST in Pytorch (2) 배치정규화, 규제, 모델저장)/output_18_0.png)
    




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


```python
class FMnist_DNN(nn.Module):
  def __init__(self):
    super().__init__()
    # linear layer, fully connected layer, affine layer, dense layer : np.dot(x, w) + b
    self.hidden_linear1 = nn.Linear(in_features=784, out_features=128)    
    self.batch_norm1 = nn.BatchNorm1d(num_features=128)
    self.hidden_linear2 = nn.Linear(in_features=128, out_features=64)    
    self.batch_norm2 = nn.BatchNorm1d(num_features=64)
    self.ouput_linear = nn.Linear(in_features=64, out_features=10)

    # 가중치 초기화 (he초기화)
    # nn.init.kaiming_normal_(self.hidden_linear1.weight, mode='fan_in', nonlinearity='relu')
    # nn.init.kaiming_normal_(self.hidden_linear2.weight, mode='fan_in', nonlinearity='relu')
    # nn.init.kaiming_normal_(self.ouput_linear.weight, mode='fan_in', nonlinearity='relu')

  def forward(self, x):
    x = self.hidden_linear1(x)    
    x = self.batch_norm1(x)
    x = F.relu(x)
    x = F.dropout(x, 0.3)
    x = self.hidden_linear2(x)  
    x = self.batch_norm2(x)
    x = F.relu(x) 
    x = F.dropout(x, 0.2) 
    x = self.ouput_linear(x)    
    return x
```


```python
model = FMnist_DNN()
model
```




    FMnist_DNN(
      (hidden_linear1): Linear(in_features=784, out_features=128, bias=True)
      (batch_norm1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (hidden_linear2): Linear(in_features=128, out_features=64, bias=True)
      (batch_norm2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (ouput_linear): Linear(in_features=64, out_features=10, bias=True)
    )




```python
model.ouput_linear.bias 
```




    Parameter containing:
    tensor([ 0.0612,  0.0867, -0.1179, -0.0624, -0.0185,  0.0005, -0.1098,  0.0449,
             0.0565, -0.0743], requires_grad=True)




```python
for name, parameter in model.named_parameters():
  print(name, parameter.size())
```

    hidden_linear1.weight torch.Size([128, 784])
    hidden_linear1.bias torch.Size([128])
    batch_norm1.weight torch.Size([128])
    batch_norm1.bias torch.Size([128])
    hidden_linear2.weight torch.Size([64, 128])
    hidden_linear2.bias torch.Size([64])
    batch_norm2.weight torch.Size([64])
    batch_norm2.bias torch.Size([64])
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
learning_rate = 0.001
# 손실함수
loss_fn = nn.CrossEntropyLoss()

# 옵티마이저(최적화함수, 예:경사하강법)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 규제의 강도 설정 weight_decay
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
```


```python
from torchsummary import summary
```


```python
# summary(모델, (채널, 인풋사이즈))
# summary(model, (1, 784))
```


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
epochs = 17
steps = 0
steps_per_epoch = len(trainloader) 
min_loss = 1000000
max_accuracy = 0

for epoch in range(epochs):
  model.train() # 훈련 모드
  train_loss = 0
  for images, labels in trainloader: # 이터레이터로부터 next()가 호출되며 미니배치를 반환(images, labels)
    steps += 1
  # 1. 입력 데이터 준비
    images.resize_(images.shape[0], 784) 

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
      if valid_loss < min_loss: # 바로 이전 epoch의 loss보다 작으면 저장하기
        min_loss = valid_loss
        torch.save(model.state_dict(), 'best_checkpoint.pth')      
      
      # option 2 : valid_accuracy 모니터링      
      # if valid_accuracy > max_accuracy : # 바로 이전 epoch의 accuracy보다 크면 저장하기
      #   max_accuracy = valid_accuracy
      #   torch.save(model.state_dict(), 'best_checkpoint.pth')  
      # -------------------------------------------
      
writer.flush()      
```

    Epoch : 1/17....... Train Loss : 0.635 Valid Loss : 0.467 Valid Accuracy : 0.835
    Epoch : 2/17....... Train Loss : 0.495 Valid Loss : 0.427 Valid Accuracy : 0.848
    Epoch : 3/17....... Train Loss : 0.458 Valid Loss : 0.409 Valid Accuracy : 0.856
    Epoch : 4/17....... Train Loss : 0.433 Valid Loss : 0.409 Valid Accuracy : 0.852
    Epoch : 5/17....... Train Loss : 0.420 Valid Loss : 0.381 Valid Accuracy : 0.865
    Epoch : 6/17....... Train Loss : 0.405 Valid Loss : 0.381 Valid Accuracy : 0.861
    Epoch : 7/17....... Train Loss : 0.394 Valid Loss : 0.373 Valid Accuracy : 0.868
    Epoch : 8/17....... Train Loss : 0.387 Valid Loss : 0.367 Valid Accuracy : 0.872
    Epoch : 9/17....... Train Loss : 0.380 Valid Loss : 0.363 Valid Accuracy : 0.871
    Epoch : 10/17....... Train Loss : 0.374 Valid Loss : 0.363 Valid Accuracy : 0.871
    Epoch : 11/17....... Train Loss : 0.366 Valid Loss : 0.364 Valid Accuracy : 0.869
    Epoch : 12/17....... Train Loss : 0.359 Valid Loss : 0.362 Valid Accuracy : 0.870
    Epoch : 13/17....... Train Loss : 0.358 Valid Loss : 0.367 Valid Accuracy : 0.869
    Epoch : 14/17....... Train Loss : 0.350 Valid Loss : 0.349 Valid Accuracy : 0.879
    Epoch : 15/17....... Train Loss : 0.346 Valid Loss : 0.353 Valid Accuracy : 0.872
    Epoch : 16/17....... Train Loss : 0.339 Valid Loss : 0.357 Valid Accuracy : 0.870
    Epoch : 17/17....... Train Loss : 0.340 Valid Loss : 0.353 Valid Accuracy : 0.875
    


```python
#%load_ext tensorboard
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

    torch.Size([16, 1, 28, 28]) torch.Size([16])
    torch.Size([1, 28, 28]) tensor(4)
    tensor([True])
    


```python
plt.imshow(images[rnd_idx].squeeze(), cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7f0784f48df0>




    
![png](/assets/images/2023-04-08-Deep Learning 10 (Fashion MNIST in Pytorch (2) 배치정규화, 규제, 모델저장)/output_47_1.png)
    


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

    Test Loss : 0.383 Test Accuracy : 0.864
    

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




    odict_keys(['hidden_linear1.weight', 'hidden_linear1.bias', 'batch_norm1.weight', 'batch_norm1.bias', 'batch_norm1.running_mean', 'batch_norm1.running_var', 'batch_norm1.num_batches_tracked', 'hidden_linear2.weight', 'hidden_linear2.bias', 'batch_norm2.weight', 'batch_norm2.bias', 'batch_norm2.running_mean', 'batch_norm2.running_var', 'batch_norm2.num_batches_tracked', 'ouput_linear.weight', 'ouput_linear.bias'])




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




    odict_keys(['hidden_linear1.weight', 'hidden_linear1.bias', 'batch_norm1.weight', 'batch_norm1.bias', 'batch_norm1.running_mean', 'batch_norm1.running_var', 'batch_norm1.num_batches_tracked', 'hidden_linear2.weight', 'hidden_linear2.bias', 'batch_norm2.weight', 'batch_norm2.bias', 'batch_norm2.running_mean', 'batch_norm2.running_var', 'batch_norm2.num_batches_tracked', 'ouput_linear.weight', 'ouput_linear.bias'])




```python
# 읽어들인 모델 파라미터는 모델 아키텍처에 연결을 시켜줘야 함
# load_state_dict() 사용
last_model = FMnist_DNN()
last_model.load_state_dict(last_state_dict)
```




    <All keys matched successfully>




```python
last_model.eval()
evaluation(last_model, testloader, loss_fn)  
```

    Test Loss : 0.380 Test Accuracy : 0.862
    


```python
# valid loss or accuracy 기준 best model
best_state_dict = torch.load('best_checkpoint.pth')
best_model = FMnist_DNN()
best_model.load_state_dict(best_state_dict)
```




    <All keys matched successfully>




```python
best_model.eval()
evaluation(best_model, testloader, loss_fn)  
```

    Test Loss : 0.380 Test Accuracy : 0.863
    
