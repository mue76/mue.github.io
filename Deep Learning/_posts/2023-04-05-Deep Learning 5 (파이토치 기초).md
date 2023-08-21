---
tag: [Deep Learning, 딥러닝, pytorch, 파이토치]
toc: true
toc_sticky: true
toc_label: 목차
author_profile: false

---


```python
from IPython.display import Image
```

# 1. 실습 환경


```python
Image('./images/image1.png', width=500)
```




    
![png](/assets/images/2023-04-05-Deep Learning 5 (파이토치 기초)/output_2_0.png)
    



**Colab 구조**

- Colab은 구글 클라우드에서
구동되는 가상 컨테이너
- 주피터 노트북과 매우 유사한
환경 제공
- 개인 계정의 구글 드라이브를
통해 데이터 파일 접근 가능


```python
Image('./images/image2.png', width=300)
```




    
![png](/assets/images/2023-04-05-Deep Learning 5 (파이토치 기초)/output_5_0.png)
    



- 일정 시간동안 Colab에 키보드로 아무 입력이 없을 경우 자동으로 런타임 연결 종료함
- 1차는 Colab jupyter 커널만 shutdown 됨
- 2차는 아예 VM Container가 종료됨
- Colab jupyter 커널만 종료된 경우엔 설치된 패키지나 데이터들이 남아 있으나
- VM Container까지 종료된 경우에는 실습 환경을 처음부터 다시 설정해줘야 함
- 메뉴에서 **런타임 다시시작**은 Colab juputer 커널만 restart
- 메뉴에서 **런타임 연결 해제 및 삭제**는 VM Conatainer까지 종료


**런타임 유형에서 하드웨어 가속기를 GPU로 설정한 경우 리소스**


```python
Image('./images/image3.png', width=270)
```




    
![png](/assets/images/2023-04-05-Deep Learning 5 (파이토치 기초)/output_8_0.png)
    



**런타임 유형에서 하드웨어 가속기를 None으로 설정한 경우 리소스**


```python
Image('./images/image4.png', width=400)
```




    
![png](/assets/images/2023-04-05-Deep Learning 5 (파이토치 기초)/output_10_0.png)
    



**현재 Colab에서 사용할 수 있는 GPU 확인하기**
- 무료로 지원되는 GPU는 Tesla T4 인 경우가 많음
- GPU를 자주 사용하게 되면 할당을 받지 못하는 경우 발생


```python
!nvidia-smi
```

    Tue Apr  4 00:46:02 2023       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   44C    P8    10W /  70W |      0MiB / 15360MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+
    

**Colab에 이미 설치된 파이토치**


```python
import torch

print(torch.__version__)
```

    2.0.0+cu118
    

**cuda (Compute Unified Device Architecture)**
- cuda는 그래픽 처리 장치(GPU)에서 수행하는 (병렬 처리) 알고리즘을 C 프로그래밍 언어를 비롯한 산업 표준 언어를 사용하여 작성할 수 있도록 하는 기술


**cuDNN**
- cuDNN 은 cuda의 딥러닝 라이브러리. 딥 뉴럴 네트워크를 구성하고
학습시킬 수 있는 기능으로 구성
- 컨볼루션, 풀링, 소프트맥스, ReLU, Sigmoid, TANH, 배치노멀라이제이션 기능등이 포함

**cuda/cuDNN 기반의 Deep Learning Framework**


```python
Image('./images/image6.png')
```




    
![png](/assets/images/2023-04-05-Deep Learning 5 (파이토치 기초)/output_18_0.png)
    



**현재 cuda가 사용가능한지 알아보기**


```python
torch.cuda.is_available()
```




    True




```python
torch.version.cuda
```




    '11.8'



# 2. 파이토치
- 파이토치 공식 문서 : [https://pytorch.org/docs/2.0/](https://pytorch.org/docs/2.0/)
- Numpy 배열과 유사한 Tensor
- GPU에서 빠른 속도로 처리되는 Tensor
- 자동 미분 계산 기능(오차역전파용)
- 그 외 신경망을 구축하기 위한 모듈들
- 직관적인 인터페이스

## 1. 텐서 구조체

### 0. 텐서 : 다차원 배열


```python
Image('./images/image5.png', width=550)
```




    
![png](/assets/images/2023-04-05-Deep Learning 5 (파이토치 기초)/output_25_0.png)
    




```python
import torch
import numpy as np
```

### 1. 텐서 생성


```python
torch.empty(5, 4)
```




    tensor([[5.3898e+32, 4.5825e-41, 3.3098e+13, 4.5825e-41],
            [1.3447e-07, 4.5825e-41, 5.4259e+32, 4.5825e-41],
            [1.3344e-07, 4.5825e-41, 1.3365e-07, 4.5825e-41],
            [1.3369e-07, 4.5825e-41, 1.8230e-05, 4.5825e-41],
            [5.3401e+32, 4.5825e-41, 1.3366e-07, 4.5825e-41]])




```python
torch.ones(3, 3)
```




    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]])




```python
torch.zeros(2)
```




    tensor([0., 0.])




```python
torch.rand(5, 6) # 0~1에서 균등분포
```




    tensor([[0.8015, 0.1638, 0.5732, 0.7477, 0.1916, 0.3382],
            [0.0230, 0.7552, 0.8278, 0.6550, 0.9480, 0.6142],
            [0.6860, 0.2296, 0.5043, 0.7061, 0.3346, 0.3387],
            [0.7346, 0.8134, 0.9133, 0.0311, 0.8562, 0.9337],
            [0.2271, 0.6794, 0.3783, 0.3509, 0.1312, 0.1676]])




```python
torch.randn(5, 6) # 표준 정규분포
```




    tensor([[ 1.7821,  0.8871,  0.0995,  0.3637, -1.0591, -0.3674],
            [-0.5858,  0.4990, -1.0360,  0.7538,  0.0192, -1.3922],
            [ 0.8734,  0.0394, -1.3596, -0.0257, -3.4435,  0.2082],
            [ 1.9193, -0.8630,  0.3942, -0.8747, -1.4567,  0.6254],
            [-0.9484,  1.6873, -0.4991,  1.5598, -0.7632, -0.4801]])



- tensor 생성자에 파이썬 list나 numpy array를 넘김


```python
# from list
l = [23, 25]
torch.tensor(l)
```




    tensor([23, 25])




```python
# from numpy array
r = np.array([4, 5, 6])
torch.tensor(r)
```




    tensor([4, 5, 6])



### 2. 텐서 사이즈


```python
x = torch.empty(5, 4)
x.size()
```




    torch.Size([5, 4])




```python
x.size()[0], x.size()[1]
```




    (5, 4)




```python
x.size(0), x.size(1)
```




    (5, 4)




```python
x.shape
```




    torch.Size([5, 4])



### 3. 텐서 타입


```python
x = torch.empty(5, 4)
```


```python
type(x)
```




    torch.Tensor




```python
x.dtype # 개별 원소의 타입
```




    torch.float32



- torch.float32 혹은 torch.float : 32비트 단정밀도 부동소수점
- torch.float64 혹은 torch.double : 64비트 배정밀도 부동소수점
- torch.float16 혹은 torch.half : 16비트 반정밀도 부동소수점
- torch.int8 : 부호있는 8비트 정수
- torch.uint8 : 부호 없는 8비트 정수
- torch.int16 혹은 torch.short : 부호 있는 16비트 정수
- torch.int32 혹은 torch.int : 부호 있는 32비트 정수
- torch.int64 혹은 torch.long : 부호 있는 64비트 정수
- torch.bool : 불리언
- **텐서의 기본 타입은 32비트 부동소수점**


```python
a = torch.tensor([1]) # 소문자 tensor는 데이터를 자동으로 추론해서 dtype 설정정
a.dtype
```




    torch.int64




```python
b = torch.tensor([1.2])
b.dtype
```




    torch.float32




```python
c = torch.zeros(10, 2)
c.dtype
```




    torch.float32




```python
d = torch.ones(10, 2, dtype=torch.uint8)
d.dtype
```




    torch.uint8



**참고**


```python
a = torch.Tensor([1]) # 대문자 Tensor는 기본 데이터 타입이 float32, 데이터 타입을 함수내에서 지정 안됨
a.dtype
```




    torch.float32




```python
b = torch.LongTensor([1]) # torch.int64
b.dtype
```




    torch.int64




```python
c = torch.FloatTensor([1]) # torch.float32
c.dtype
```




    torch.float32




```python
d = torch.DoubleTensor([1]) # torch.float64
d.dtype
```




    torch.float64



### 4. 텐서 형변환


```python

```


```python
# option 1 : 생성시 형 지정
double_points = torch.ones(10, 2, dtype=torch.double)
short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short)
```


```python
# option 2 : 데이터 타입과 동일한 함수 호출출
double_points = torch.ones(10, 2).double()
short_points = torch.tensor([[1, 2], [3, 4]]).short()
```




    torch.int16




```python
# option 3 : to 함수 이용
double_points = torch.ones(10, 2).to(torch.double)
short_points = torch.tensor([[1, 2], [3, 4]]).to(torch.short)
```


```python
# option 4 : type 함수 이용
float_points = double_points.type(torch.float32)
long_points = short_points.type(torch.int64)
```


```python
# 서로 다른 dtype의 tensor끼리 연산
points_64 = torch.rand(5, dtype=torch.double)
points_16 = points_64.to(torch.short)

points_64 * points_16
```




    tensor([0., 0., 0., 0., 0.], dtype=torch.float64)



### 5. 텐서 연산


```python
x = torch.rand(2, 2)
y = torch.rand(2, 2)
```


```python
x
```




    tensor([[0.6164, 0.5263],
            [0.6439, 0.8302]])




```python
y
```




    tensor([[0.1032, 0.1465],
            [0.4994, 0.2463]])




```python
z = x + y
z
```




    tensor([[0.7195, 0.6728],
            [1.1433, 1.0765]])




```python
torch.add(x, y)
```




    tensor([[0.7195, 0.6728],
            [1.1433, 1.0765]])




```python
y.add(x)
```




    tensor([[0.7195, 0.6728],
            [1.1433, 1.0765]])




```python
y # y값이 변하지 않음
```




    tensor([[0.1032, 0.1465],
            [0.4994, 0.2463]])




```python
y.add_(x) # _ : inplace 연산
y # y에 연산 결과가 반영
```




    tensor([[0.7195, 0.6728],
            [1.1433, 1.0765]])



### 6. 텐서 색인


```python
# 1차원 tensor
a = torch.ones(3)
```


```python
t = a[1] # t에는 개별원소(스칼라)
t
```




    tensor(1.)




```python
type(t), t.size()
```




    (torch.Tensor, torch.Size([]))




```python
# item() 함수로 tensor 안의 값을 꺼내올 수 있음
t.item()
```




    1.0




```python
# 또는 파이썬 float으로 바꿔저도 같은 결과
float(t)
```




    1.0




```python
a[2] = 2.0
a
```




    tensor([1., 1., 2.])




```python
# 2차원 tensor
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points

```




    tensor([[4., 1.],
            [5., 3.],
            [2., 1.]])




```python
points.shape
```




    torch.Size([3, 2])




```python
points[0] # 2차원 데이터에서 색인을 하면 1차원 데이터
```




    tensor([4., 1.])




```python
points[0, 1] # 2차원 데이터에서 두번 색인을 하면 개별 원소(스칼라)
```




    tensor(1.)



### 7. 텐서 저장소


```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.storage()
```

    <ipython-input-76-8575ff47c9e0>:2: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      points.storage()
    /usr/local/lib/python3.9/dist-packages/IPython/lib/pretty.py:700: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      output = repr(obj)
    /usr/local/lib/python3.9/dist-packages/torch/storage.py:645: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      return str(self)
    /usr/local/lib/python3.9/dist-packages/torch/storage.py:636: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      f'device={self.device}) of size {len(self)}]')
    /usr/local/lib/python3.9/dist-packages/torch/storage.py:637: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      if self.device.type == 'meta':
    /usr/local/lib/python3.9/dist-packages/torch/storage.py:640: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      data_str = ' ' + '\n '.join(str(self[i]) for i in range(self.size()))
    




     4.0
     1.0
     5.0
     3.0
     2.0
     1.0
    [torch.storage.TypedStorage(dtype=torch.float32, device=cpu) of size 6]




```python
points_storage = points.storage()
points_storage[0]
```

    <ipython-input-80-c3c08f2b4561>:1: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      points_storage = points.storage()
    <ipython-input-80-c3c08f2b4561>:2: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      points_storage[0]
    




    4.0




```python
points_storage[0] = 2.0
points
```

    <ipython-input-81-143b7e3568af>:1: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      points_storage[0] = 2.0
    




    tensor([[2., 1.],
            [5., 3.],
            [2., 1.]])



### 8. 텐서 메타데이터


```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1]
second_point.storage_offset()
```




    2




```python
points.stride() # 메모리에 있는 연속된 값을 기준으로 열을 이동할 때 1칸, 행을 이동할 때 2칸칸
```




    (2, 1)




```python
second_point.stride()
```




    (1,)




```python
second_point[0] = 10.0
points # second_point와 points[1]가 같은 메모리를 가리키고 있음
```




    tensor([[ 4.,  1.],
            [10.,  3.],
            [ 2.,  1.]])




```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1].clone()
second_point[0] = 10.0
points # second_point와 points[1]가 별도의 메모리에 있음음
```




    tensor([[4., 1.],
            [5., 3.],
            [2., 1.]])



### 9. 텐서 전치


```python
a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)
a.shape, a_t.shape
```




    (torch.Size([3, 2]), torch.Size([2, 3]))




```python
a_t = a.transpose(0, 1)
a_t.shape
```




    torch.Size([2, 3])




```python
a_t = a.t()
a_t.shape
```




    torch.Size([2, 3])




```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points
```




    tensor([[4., 1.],
            [5., 3.],
            [2., 1.]])




```python
points_t = points.t()
points_t
```




    tensor([[4., 5., 2.],
            [1., 3., 1.]])




```python
points.stride()
```




    (2, 1)




```python
points_t.stride()
```




    (1, 2)




```python
points.is_contiguous()
```




    True




```python
points_t.is_contiguous()
```




    False




```python
# contiguous 하지 않은 tensor로 모양을 바꾸려고 하면 error가 발생생
points_t.view(3, 2)
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-112-e365c0847245> in <cell line: 1>()
    ----> 1 points_t.view(3, 2)
    

    RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.



```python
points_t_cont = points_t.contiguous()
points_t_cont
```




    tensor([[4., 5., 2.],
            [1., 3., 1.]])




```python
points_t_cont.stride()
```




    (3, 1)




```python
points_t_cont.storage()
```

    <ipython-input-116-d486264d48a5>:1: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      points_t_cont.storage()
    /usr/local/lib/python3.9/dist-packages/IPython/lib/pretty.py:700: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      output = repr(obj)
    /usr/local/lib/python3.9/dist-packages/torch/storage.py:645: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      return str(self)
    /usr/local/lib/python3.9/dist-packages/torch/storage.py:636: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      f'device={self.device}) of size {len(self)}]')
    /usr/local/lib/python3.9/dist-packages/torch/storage.py:637: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      if self.device.type == 'meta':
    /usr/local/lib/python3.9/dist-packages/torch/storage.py:640: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      data_str = ' ' + '\n '.join(str(self[i]) for i in range(self.size()))
    




     4.0
     5.0
     2.0
     1.0
     3.0
     1.0
    [torch.storage.TypedStorage(dtype=torch.float32, device=cpu) of size 6]




```python
# contiguouse하게 바꿔준 뒤에는 모양을 바꿔줄 수 있음
points_t_cont.view(3, 2)
```




    tensor([[4., 5.],
            [2., 1.],
            [3., 1.]])




```python
# 더 높은 차원에서 전치 연산
some_t = torch.ones(3, 4, 5)
transposed_t = some_t.transpose(0, 2)
transposed_t.shape
```




    torch.Size([5, 4, 3])




```python
some_t.stride()
```




    (20, 5, 1)




```python
transposed_t.stride()
```




    (1, 5, 20)



### 10. 텐서의 크기 변환(reshaping)


```python
x = torch.rand(8, 8)
x.size()
```




    torch.Size([8, 8])




```python
y = x.reshape(64) # numpy reshape과 같은 기능
y.size()
```




    torch.Size([64])




```python
y = x.view(64) # tensor는 view() 메서드로도 shape을 변경할 수 있음음
y.size()
```




    torch.Size([64])




```python
4*4*4
```




    64




```python
z = x.view(4, 4, 4)
z.size()
```




    torch.Size([4, 4, 4])




```python
z = x.view(-1, 4, 4)
z.size()
```




    torch.Size([4, 4, 4])




```python
x.resize_(4, 4, 4) # _ : inplace
x.size()
```




    torch.Size([4, 4, 4])




```python
i = torch.flatten(x, 0) # 첫번째 dim부터 flatten
i.size()
```




    torch.Size([64])




```python
j = torch.flatten(x, 1) # 두번째 dim부터 flatten
j.size()
```




    torch.Size([4, 16])



- squeeze : 데이터 중에 차원이 1인 경우 해당 차원을 제거


```python
data = torch.randn(1, 2, 3)
squeezed_data = data.squeeze()
squeezed_data.size()
```




    torch.Size([2, 3])




```python
data = torch.randn(2, 1, 3)
squeezed_data = data.squeeze()
squeezed_data.size()
```




    torch.Size([2, 3])



- unsqueeze : 특정 위치에 1인 차원을 추가


```python
data = torch.randn(2, 3)
unsqueezed_data = data.unsqueeze(0) # 첫번째 차원 추가
unsqueezed_data.size()
```




    torch.Size([1, 2, 3])




```python
data = torch.randn(2, 3)
unsqueezed_data = data.unsqueeze(1) # 첫번째 차원 추가
unsqueezed_data.size()
```




    torch.Size([2, 1, 3])



- 단일 텐서에서 값으로 변환하기


```python
x = torch.ones(1)
x
```




    tensor([1.])




```python
x.item()
```




    1.0



### 11. 텐서를 GPU로 옮기기

**option 1**
- tensor 생성시시 dtype외에 device까지 지정해서 tensor가 위치할 곳을 gpu로 지정


```python
points_gpu = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]], device='cuda')
points_gpu
```




    tensor([[4., 1.],
            [5., 3.],
            [2., 1.]], device='cuda:0')



- gpu가 1개인 경우에는 cuda와 cuda:0는 동일함
- 뒤의 index는 multiple gpu인 경우에만 의미가 있음

**option 2**
- 이미 cpu에 있는 데이터를 gpu로 옮김


```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_gpu = points.to(device="cuda:0")
points_gpu
```




    tensor([[4., 1.],
            [5., 3.],
            [2., 1.]], device='cuda:0')




```python
points = 2 * points  # cpu에서 실행하는 곱셈
points_gpu = 2 * points_gpu # gpu에서 실행하는 곱셈
```


```python
points
```




    tensor([[ 8.,  2.],
            [10.,  6.],
            [ 4.,  2.]])




```python
points_gpu
```




    tensor([[ 8.,  2.],
            [10.,  6.],
            [ 4.,  2.]], device='cuda:0')




```python
points_cpu = points_gpu.to(device='cpu') # gpu에서 수행한 결과를 cpu로 옮길때 
points_cpu
```




    tensor([[ 8.,  2.],
            [10.,  6.],
            [ 4.,  2.]])



**option 3**
- 이미 cpu에 있는 데이터를 gpu로 옮김


```python
points_gpu = points.cuda() # to 메서드 대신 cuda(), cpu()로 같은 동작을 할 수 있음
points_gpu
```




    tensor([[ 8.,  2.],
            [10.,  6.],
            [ 4.,  2.]], device='cuda:0')




```python
# multiple gpu인 경우에는 index 설정 가능
points_gpu = points.cuda(0)
points_gpu
```




    tensor([[ 8.,  2.],
            [10.,  6.],
            [ 4.,  2.]], device='cuda:0')




```python
points_cpu = points_gpu.cpu() # cpu로 보내기
points_cpu
```




    tensor([[ 8.,  2.],
            [10.,  6.],
            [ 4.,  2.]])



### 12. 넘파이 호환


```python
# from tensor to numpy
points = torch.ones(3, 4)
points_np = points.numpy()
points_np
```




    array([[1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.]], dtype=float32)




```python
# from numpy to tensor
points_ts = torch.from_numpy(points_np)
points_ts
```




    tensor([[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]])



### 13. 텐서 직렬화


```python
points
```




    tensor([[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]])




```python
torch.save(points, './ourpoints.t')
```


```python
loaded_points = torch.load('./ourpoints.t')
loaded_points
```




    tensor([[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]])




```python
with open('./ourpoints2.t', 'wb') as f:
  torch.save(points, f)
```


```python
with open('./ourpoints2.t', 'rb') as f:
  loaded_points2 = torch.load(f)
loaded_points2  
```




    tensor([[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]])



- 한개 이상의 tensor를 저장할때는 dictionary 형태로 준비

## 2. 신경망 모델 구성하기

### 1. Pytorch 모델 용어 정리
- 계층(layer): 신경망을 구성하는 계층으로 한개 이상의 계층이 모여 모듈을 구성 (예: 선형 계층, Linear Layer)
- 모듈(module): 한 개 이상의 계층이 모여 구성된 것. 모듈이 모여서 새로운 모듈 구성 가능
- 모델(model): 최종적인 네트워크. 한 개의 모듈이 모델이 될 수도 있고, 여러 개의 모듈이 하나의 모델이 될 수도 있음

### 2. torch.nn 과 nn.Module
- torch.nn 네임스페이스는 신경망을 구성하는데 필요한 모든 구성 요소 제공
- 모든 PyTorch 모듈은 nn.Module 의 하위 클래스(subclass) 임
- 모든 PyTorch 신경망 모델은 nn.Module 을 상속받은 하위 클래스로 정의함
   - \_\_init\_\_ 에서 신경망 계층 초기화 필요
   - forward() 메서드에서 입력 데이터에 대한 연산 정의 필요

### 3. simpleNet 구현 예시
- 신경망 모델 클래스를 만들고, nn.Module 을 상속받음
- \_\_init\_\_ 에서 신경망 계층 초기화 선언
- forward() 메서드에서 입력 데이터에 대한 연산 정의


```python
import torch
import torch.nn as nn

class simpleNet(nn.Module):
  def __init__(self):
    # 상속받은 부모 클래스(nn.Module)를 초기화
    super().__init__()

    # simpleNet 초기화 하는데 필요한 코드들
    # 가중치 초기화
    self.W = torch.FloatTensor(4, 3)
    self.b = torch.FloatTensor(3)

  def forward(self, x):
    y = torch.matmul(x, self.W) + self.b
    return y


x = torch.FloatTensor(4)
net = simpleNet()
# net.forward(x)와 같이 호출해야 된다고 생각할 수 있으나
# 그냥 net(x) 호출해주면 내부적으로 forward()도 자동호출해주고 forward 앞뒤로 전처리/후처리
y = net(x)
print(y)
```

    tensor([       inf, 2.9375e-09,        inf])
    


```python
import torch
import torch.nn as nn

class simpleNet(nn.Module):
  def __init__(self, input_dim, output_dim):
    # 상속받은 부모 클래스(nn.Module)를 초기화
    super().__init__()

    # simpleNet 초기화 하는데 필요한 코드들
    # 가중치 초기화
    self.W = torch.FloatTensor(input_dim, output_dim)
    self.b = torch.FloatTensor(output_dim)

  def forward(self, x):
    y = torch.matmul(x, self.W) + self.b
    return y


x = torch.FloatTensor(4)
net = simpleNet(4, 3)
# net.forward(x)와 같이 호출해야 된다고 생각할 수 있으나
# 그냥 net(x) 호출해주면 내부적으로 forward()도 자동호출해주고 forward 앞뒤로 전처리/후처리
y = net(x)
print(y)
```

    tensor([1.0300e-02, 0.0000e+00, 1.6815e-34])
    


```python
for param in net.parameters():
  print(param)
```


```python
net.W
```




    tensor([[6.4052e+31, 4.5862e-41, 1.6461e-34],
            [0.0000e+00, 4.4842e-44, 0.0000e+00],
            [1.5695e-43, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 8.4078e-44, 0.0000e+00]])




```python
net.b
```




    tensor([1.7611e-37, 0.0000e+00, 1.6815e-34])



### 4. nn.Parameter 등록하기
- 학습 대상이 되는 텐서는 해당 모듈에 연결된 Parameter 로 등록해야 함
- 이를 통해 특정 모듈에서 학습 처리시 필요한 작업을 알아서 해주도록 구성되어 있음
- \_\_init\_\_ 함수에서 nn.Parameter(텐서, requires_grad=True)와 같이 등록
- requires_grad 는 디폴트 True 이며, 경사하강법으로 모델 파라미터 업데이트를 위해 미분을 계산해야 된다는것을 의미함


```python
import torch
import torch.nn as nn

class simpleNet(nn.Module):
  def __init__(self, input_dim, output_dim):
    # 상속받은 부모 클래스(nn.Module)를 초기화
    super().__init__()

    # simpleNet 초기화 하는데 필요한 코드들
    # 가중치 초기화
    
    # parameter로 등록하기
    # nn.Parameter(텐서, requires_grad=True)
    self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim), requires_grad=True)
    self.b = nn.Parameter(torch.FloatTensor(output_dim), requires_grad=True)

  def forward(self, x):
    y = torch.matmul(x, self.W) + self.b
    return y


x = torch.FloatTensor(4)
net = simpleNet(4, 3)
# net.forward(x)와 같이 호출해야 된다고 생각할 수 있으나
# 그냥 net(x) 호출해주면 내부적으로 forward()도 자동호출해주고 forward 앞뒤로 전처리/후처리
y = net(x)
print(y)
```

    tensor([6.4051e+31, 4.5862e-41, 1.1213e-05], grad_fn=<AddBackward0>)
    


```python
for param in net.parameters():
  print(param)
```

    Parameter containing:
    tensor([[1.6063e-34, 0.0000e+00, 1.7507e-37],
            [0.0000e+00, 1.1210e-43, 0.0000e+00],
            [8.9683e-44, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 1.3563e-19, 2.7523e+23]], requires_grad=True)
    Parameter containing:
    tensor([6.4051e+31, 4.5862e-41, 1.7507e-37], requires_grad=True)
    

### 5. nn.Linear 클래스
- 3.4번에서 작업한 것을 nn.Linear로 한번에 대체할 수 있음


```python
x = torch.FloatTensor(4)
net = nn.Linear(4, 3)
y = net(x)
y
```




    tensor([-0.1136, -0.1801,  0.3312], grad_fn=<AddBackward0>)




```python
for param in net.parameters():
  print(param)
```

    Parameter containing:
    tensor([[ 0.0013, -0.2762,  0.1680, -0.1018],
            [ 0.0896, -0.0747, -0.0339, -0.3895],
            [ 0.4562,  0.0419, -0.2181,  0.2911]], requires_grad=True)
    Parameter containing:
    tensor([-0.1136, -0.1801,  0.3312], requires_grad=True)
    

### 6. nn.Linear 클래스를 이용한 simpleNet 구현
- nn.Module 을 상속받은 하위 클래스 형태로 작성하기


```python
class simpleNet(nn.Module):
  def __init__(self, input_dim, output_dim):
    super().__init__()

    # 계층, 활성화 함수 정의
    self.linear = nn.Linear(input_dim, output_dim)

  def forward(self, x):
    y = self.linear(x)
    return y
```


```python
net = simpleNet(4, 3)
y = net(x)
y
```




    tensor([-0.0446,  0.0007,  0.3013], grad_fn=<AddBackward0>)




```python
for param in net.parameters():
  print(param)
```

    Parameter containing:
    tensor([[-0.1918,  0.1871,  0.3463, -0.2580],
            [ 0.2769, -0.0268,  0.3051,  0.1009],
            [ 0.4578, -0.0698, -0.2821,  0.3556]], requires_grad=True)
    Parameter containing:
    tensor([-0.0446,  0.0007,  0.3013], requires_grad=True)
    

## 3. 자동미분 (torch.autograd) 
- PyTorch 의 autograd 는 신경망 훈련을 지원하는 자동 미분 기능
- torch.autograd 동작 방법
  - 텐서에 requires_grad 속성을 True 로 설정하면, 이후의 텐서 모든 연산들을 추적함
  - 텐서.backward() 를 호출하면, 연산에 연결된 각 텐서들의 미분 값을 계산하여, 각텐서객체.grad 에 저장


```python
x = torch.rand(1, requires_grad=True)
x
```




    tensor([0.7361], requires_grad=True)




```python
y = torch.rand(1)
y.requires_grad = True
y
```




    tensor([0.1590], requires_grad=True)




```python
loss = y - x
loss
```




    tensor([-0.5771], grad_fn=<SubBackward0>)




```python
loss.backward()
```


```python
print(x.grad, y.grad)
```

    tensor([-1.]) tensor([1.])
    


```python
x = torch.ones(2, 2, requires_grad = True)
x
```




    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)




```python
y = x+1
y
```




    tensor([[2., 2.],
            [2., 2.]], grad_fn=<AddBackward0>)




```python
z = 2*(y**2)
z
```




    tensor([[8., 8.],
            [8., 8.]], grad_fn=<MulBackward0>)




```python
loss = z.mean()
loss 
```




    tensor(8., grad_fn=<MeanBackward0>)




```python
# y는 x에 대한 식, z는 y에 대한 식, loss는 z에 대한 식
# loss는 합성함수의 형태이고, loss는 x로 표현이 가능하고, 미분도 가능
```


```python
loss.backward()
```


```python
x.grad
```




    tensor([[2., 2.],
            [2., 2.]])




```python
# loss에 대해 x1으로의 편미분 값을 구해보기기

# loss = (z1 + z2 + z3 + z4)/4
# dloss/dz1 = 1/4

# z1 = 2*(y1**2)
# dz1/dy1 = 4*y1

# y1 = x1+1
# dy1/dx1 = 1

# dloss/dx1 = (dloss/dz1)*(dz1/dy1)*(dy1/dx1)
#           = (1/4)*(4*y1)*(1)

# x1=1, y1=2 이므로
# dloss/dx1 = 2
```


```python
x
```




    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)



## 4. Optimizer(경사하강법) 적용하기

**(1) optimizer 초기화**
- 모델 파라미터들과 학습률을 넣어줌


```python
for param in net.parameters():
  print(param)
```

    Parameter containing:
    tensor([[-0.1918,  0.1871,  0.3463, -0.2580],
            [ 0.2769, -0.0268,  0.3051,  0.1009],
            [ 0.4578, -0.0698, -0.2821,  0.3556]], requires_grad=True)
    Parameter containing:
    tensor([-0.0446,  0.0007,  0.3013], requires_grad=True)
    


```python
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
```

**(2) gradinet를 초기화**
- optimizer.zero_grad()를 통해 미분값으로 0으로 초기화 해줘야 함
- 위와같이 미분값을 초기화해주지 않으면 이어지는 모델 파라미터 업데이트시, 기존 미분값에 현재 미분값을 더해줌


```python
optimizer.zero_grad()
```

**(3) gradient 구하기**
- 오차역전파를 통해 연산에 연결된 각 텐서들의 미분값을 각 텐서객체.grad에 저장
- loss.backward()

**(4) 경사하강법 적용하기**
- 역전파 단계에서 계산된 미분값을 기반으로 모델 파라미터값 업데이트
- optimizer.step()

## 5. 신경망 학습 과정

- (1) 모델 및 데이터 생성
- (2) 전방향(forward pass)으로 입력 데이터를 모델에 넣어서 예측값 계산
- (3) 예측값과 정답값의 차이(loss)를 손실함수(loss function)로 계산
- (4) 역방향(backward pass)으로 loss에 대한 각 모델 파라미터들의 미분값(gradient)을 계산
- (5) 최적화 함수(optimizer, 예:경사하강법)가 미분값들을 바탕으로 각 모델파라미터를 업데이트
- (6) 손실함수가 최소가 될때까지 (2)~(5)까지를 반복


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class simpleNet(nn.Module):
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.linear = nn.Linear(input_dim, output_dim)

  def forward(self, x):
    y = self.linear(x)
    return y

model = simpleNet(4, 3)
```


```python
x = torch.ones(4) # 입력
y = torch.zeros(3) # 정답

# 파라미터와 옵티마이저 초기화
learning_rate = 0.01
epochs = 100
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
  pred = model(x) # forward 방향으로 예측
  loss = F.mse_loss(pred, y) # 정답과 예측의 차이를 손실함수로 계산

  optimizer.zero_grad() # gradient를 구하기 전에 초기화
  loss.backward() # gradient를 역방향으로 계산
  optimizer.step() # 경사하강법 적용 (W(new) <- W(old) -lr*gradient)
```


## Reference
- [파이토치 딥러닝 마스터 (엘리 스티븐스,루카 안티가,토마스 피이만 저)](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=296883495)