## Numpy(넘파이) 개요
- Numpy는 Numerical Python의 줄임말
- 파이썬에서 산술 계산을 위한 가장 중요한 필수 패키지
- 효율적인 다차원 배열 ndarray는 빠른 배열 계산 지원
- 유연한 브로드캐스팅 기능 제공
- 반복문을 작성할 필요 없이 전체 데이터 배열을 빠르게 계산

## 0. Numpy 성능


```python
import numpy as np
```


```python
np.__version__
```




    '1.21.6'




```python
# 1. 파이썬 리스트
py_list = list(range(1000000))
```


```python
# 2. Numpy Array
np_array = np.arange(1000000)
```


```python
%timeit py_list2 = [a*2 for a in py_list] # 리스트 표현식

```

    84.6 ms ± 3.13 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    


```python
%timeit np_array2 = np_array * 2 # Numpy 브로드캐스팅
```

    903 µs ± 24.9 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    

## 1. Numpy ndarray (다차원 배열 객체)

### 1.1 ndarray 생성하기


```python
from IPython.display import Image
Image('./images/image1.PNG')
```




    
![png](/assets/images/output_10_0.png)
    




```python
data1 = [1.1, 2, 3, 4]
data1
```




    [1.1, 2, 3, 4]




```python
arr1 = np.array(data1)
arr1
```




    array([1.1, 2. , 3. , 4. ])




```python
type(arr1)
```




    numpy.ndarray




```python
data2 = [[1, 2, 3], [4, 5, 6]]
arr2 = np.array(data2)
arr2
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
type(arr2)
```




    numpy.ndarray




```python
np.zeros(10)
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])




```python
np.ones(10)
```




    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])




```python
np.empty((10, 3))
```




    array([[1.38334349e-316, 0.00000000e+000, 6.01347002e-154],
           [1.14302518e+243, 1.69664085e-152, 1.06112833e-153],
           [1.17365078e+214, 6.01347002e-154, 1.11763067e+219],
           [2.45185360e+198, 3.00755680e+161, 2.32161868e-152],
           [4.73544961e+223, 3.81391429e+180, 5.85221749e+199],
           [7.13038752e+247, 2.46088219e-154, 6.01347002e-154],
           [2.87521412e+161, 6.03391665e-154, 8.03704345e-095],
           [5.21971393e+180, 1.21906099e-152, 1.94904214e+227],
           [2.64520780e+185, 1.41589192e-308, 6.94820151e-310],
           [1.63041663e-322, 1.53845007e-316, 6.94827483e-310]])




```python
np.zeros((2, 3))
```




    array([[0., 0., 0.],
           [0., 0., 0.]])




```python
np.arange(10)
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
arr1, arr2
```




    (array([1.1, 2. , 3. , 4. ]), array([[1, 2, 3],
            [4, 5, 6]]))




```python
# 데이터 타입
type(arr1)
```




    numpy.ndarray




```python
# 데이터 형상
arr1.shape
```




    (4,)




```python
arr2.shape
```




    (2, 3)




```python
# 차원 확인
arr1.ndim
```




    1




```python
arr2.ndim
```




    2




```python
# 개별 원소들의 데이터 타입
arr1.dtype
```




    dtype('float64')




```python
arr2.dtype
```




    dtype('int64')



### 1.2 ndarray의 dtype


```python
Image('./images/image2.PNG')
```




    
![png](/assets/images/output_30_0.png)
    




```python
arr1 = np.array([1, 2, 3], dtype=np.float64)
arr1.dtype
```




    dtype('float64')




```python
int_arr = arr1.astype(np.int64)
int_arr.dtype
```




    dtype('int64')




```python
# 파이썬 리스트에 다른 데이터 타입의 원소가 있을때는 dtype을 object로 설정
lst = [1, 2.2, 'ss']
np.array(lst, dtype=object)
```

### 1.3 Numpy 배열의 산술 연산


```python
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr
```




    array([[1., 2., 3.],
           [4., 5., 6.]])




```python
arr.dtype
```




    dtype('float64')




```python
arr * arr
```




    array([[ 1.,  4.,  9.],
           [16., 25., 36.]])




```python
arr - arr
```




    array([[0., 0., 0.],
           [0., 0., 0.]])




```python
arr * 2 # 브로드캐스팅
```




    array([[ 2.,  4.,  6.],
           [ 8., 10., 12.]])




```python
arr2 = np.array([[2., 3., 4.], [5., 6., 7.]])
arr2
```




    array([[2., 3., 4.],
           [5., 6., 7.]])




```python
arr
```




    array([[1., 2., 3.],
           [4., 5., 6.]])




```python
arr < arr2
```




    array([[ True,  True,  True],
           [ True,  True,  True]])




```python
5 < 6
```




    True




```python
import numpy as np
```


```python
from IPython.display import Image
```

### 1.4 색인과 슬라이싱


```python
# 1차원 numpy array  준비
arr = np.arange(10) + 10
arr
```




    array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])




```python
# 1차원 데이터 색인
arr[5]
```




    15




```python
# 1차원 데이터 슬라이싱
arr[5:8] # 5번째에서 7번째까지만 가져오고, 8번째는 포함되지 않음
```




    array([15, 16, 17])




```python
arr[5:8] = 100
arr
```




    array([ 10,  11,  12,  13,  14, 100, 100, 100,  18,  19])




```python
arr = np.arange(10) + 10
arr
```




    array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])




```python
# 슬라이스 조각
brr = arr[5:8]
brr
```




    array([15, 16, 17])




```python
arr[5:8] = 100
arr
```




    array([ 10,  11,  12,  13,  14, 100, 100, 100,  18,  19])




```python
brr # brr가 arr[5:8] 와 같은 메모리를 공유하고 있어서 arr[5:8] 의 수정사항이 brr에도 반영
```




    array([100, 100, 100])




```python
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
# arr2d를 아래와 같이 만들 수도 있음
# np.arange(1, 10).reshape(3, 3)
```


```python
arr2d.ndim
```




    2




```python
arr2d.shape
```




    (3, 3)




```python
# 2차원 데이터의 색인
arr2d[2] # 2차원 array를 색인하면 1차원 데이터가 추출
```




    array([7, 8, 9])




```python
arr2d[2][2] # 2차원 arrary 한번 색인한 결과로 다시 색인하면 개별 원소 추출
```




    9




```python
arr2d[2, 2]
```




    9




```python
Image('./images/image3.PNG')
```




    
![png](/assets/images/output_62_0.png)
    




```python
# 2차원 데이터의 슬라이싱
arr2d[0:2] # arr2d의 0번째부터 1번째까지 슬라이싱 (arr2d[0], arr2d[1])
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
arr2d[0:2][0]
```




    array([1, 2, 3])




```python
arr2d[0:2, 1:3] # arr2d[행슬라이싱, 열슬라이싱]
```




    array([[2, 3],
           [5, 6]])




```python
arr2d[:2, 1:] # arr2d[0:2, 1:3] 와 같은 결과, 비워두면 처음과 마지막으로 인식
```




    array([[2, 3],
           [5, 6]])




```python
arr2d[2] # 2차원 데이터를 색인하면 1차원 데이터
```




    array([7, 8, 9])




```python
arr2d[2:] # 위의 색인 결과와 값은 동일하지만 슬라이싱을 하면 차원이 유지
```




    array([[7, 8, 9]])




```python
arr2d[:2] # 2차원 데이터를 슬라이싱하면 2차원 데이터
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
arr2d[:2, 1] # 슬라이싱과, 색인을 같이 사용하면 1차원 데이터
```




    array([2, 5])




```python
arr2d[:2, 1:2] # 슬라이싱만 사용하면 그대로 2차원 데이터
```




    array([[2],
           [5]])




```python
arr2d[1][1]
```




    5




```python
arr2d[1, 1]
```




    5




```python
arr2d[1:3][0:2]
```




    array([[4, 5, 6],
           [7, 8, 9]])




```python
arr2d[1:3, 0:2]
```




    array([[4, 5],
           [7, 8]])




```python
# 위의 실행 결과 정리
# 색인에서는 아래 문법이 동일한 결과
arr2d[1][1]
arr2d[1, 1]

# 슬라이싱 아래 문법이 동일하지 않은 결과
arr2d[1:3][0:2]
arr2d[1:3, 0:2]
```




    array([[4, 5],
           [7, 8]])




```python
Image('./images/image4.PNG', width = 500)
```




    
![png](/assets/images/output_77_0.png)
    



### 1.5 불리안 색인


```python
arr1 = np.arange(10)
arr1
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
arr1 < 5
```




    array([ True,  True,  True,  True,  True, False, False, False, False,
           False])




```python
arr1[[ True,  True,  True,  True,  True, False, False, False, False,
       False]]
```




    array([0, 1, 2, 3, 4])




```python
arr1[arr1 < 5]
```




    array([0, 1, 2, 3, 4])




```python
data = np.arange(28).reshape((7, 4))
data
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19],
           [20, 21, 22, 23],
           [24, 25, 26, 27]])




```python
data[[ True, False, False,  True, False, False, False]]
```




    array([[ 0,  1,  2,  3],
           [12, 13, 14, 15]])




```python
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
```


```python
names == 'Bob'
```




    array([ True, False, False,  True, False, False, False])




```python
data[names == 'Bob']
```




    array([[ 0,  1,  2,  3],
           [12, 13, 14, 15]])




```python
data[names == 'Bob', 2] # 행은 불리안 색인, 열은 숫자로 색인 (차원이 줄어듬)
```




    array([ 2, 14])




```python
data[names == 'Bob', 2:3] # 행은 불리안 색인, 열은 숫자로 슬라이싱 (차원이 유지)
```




    array([[ 2],
           [14]])




```python
data
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19],
           [20, 21, 22, 23],
           [24, 25, 26, 27]])




```python
data < 10
```




    array([[ True,  True,  True,  True],
           [ True,  True,  True,  True],
           [ True,  True, False, False],
           [False, False, False, False],
           [False, False, False, False],
           [False, False, False, False],
           [False, False, False, False]])




```python
data[data < 10]
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
data[data < 10] = -999
data
```




    array([[-999, -999, -999, -999],
           [-999, -999, -999, -999],
           [-999, -999,   10,   11],
           [  12,   13,   14,   15],
           [  16,   17,   18,   19],
           [  20,   21,   22,   23],
           [  24,   25,   26,   27]])



### 1.6 팬시 색인(정수 색인)


```python
arr = np.empty((8, 4))
for i in range(8):
  arr[i] = i
arr
```




    array([[0., 0., 0., 0.],
           [1., 1., 1., 1.],
           [2., 2., 2., 2.],
           [3., 3., 3., 3.],
           [4., 4., 4., 4.],
           [5., 5., 5., 5.],
           [6., 6., 6., 6.],
           [7., 7., 7., 7.]])




```python
arr[[4, 3, 0, 6]]
```




    array([[4., 4., 4., 4.],
           [3., 3., 3., 3.],
           [0., 0., 0., 0.],
           [6., 6., 6., 6.]])




```python
arr[[-3, -5]]
```




    array([[5., 5., 5., 5.],
           [3., 3., 3., 3.]])




```python
arr[[4, 3, 0, 6], [0, 3, 1, 2]]
```




    array([4., 3., 0., 6.])



## 2. 유니버설 함수 : 배열의 각 원소를 빠르게 처리


```python
arr = np.arange(10)
arr
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
np.sqrt(arr)
```




    array([0.        , 1.        , 1.41421356, 1.73205081, 2.        ,
           2.23606798, 2.44948974, 2.64575131, 2.82842712, 3.        ])




```python
np.exp(arr)
```




    array([1.00000000e+00, 2.71828183e+00, 7.38905610e+00, 2.00855369e+01,
           5.45981500e+01, 1.48413159e+02, 4.03428793e+02, 1.09663316e+03,
           2.98095799e+03, 8.10308393e+03])




```python
arr1 = np.arange(10)
arr2 = np.arange(10) + 2

np.maximum(arr1, arr2)
```




    array([ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11])




```python
arr3 = np.arange(12).reshape((4, 3))
arr3
```




    array([[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8],
           [ 9, 10, 11]])




```python
np.sum(arr3)
```




    66




```python
np.sum(arr3, axis=0) # 0번(행) 축을 따라서 합을 구함
```




    array([18, 22, 26])




```python
np.sum(arr3, axis=1) # 1번(열) 축을 따라서 합을 구함
```




    array([ 3, 12, 21, 30])



**현재까지 살펴본 Numpy 정리**

- 임포트


```python
import numpy as np
```

- 객체 생성


```python
a1 = np.array([10, 20, 30], dtype=np.float64)
type(a1)
```




    numpy.ndarray




```python
a2 = np.array([[10, 20, 30], [40, 50, 60]])
type(a2)
```




    numpy.ndarray



- 모양(형상) 확인


```python
a1.shape
```




    (3,)




```python
a2.shape
```




    (2, 3)



- 개별 원소의 자료형


```python
a1.dtype
```




    dtype('float64')




```python
a2.dtype
```




    dtype('int64')



- 색인(인덱스 번호로 한 지점 조회)


```python
a1[0]
```




    10.0




```python
a2[0]
```




    array([10, 20, 30])




```python
a2[0][0] # a2[0, 0]와 동일한 결과
```




    10



- 슬라이싱(연속된 구간 정보 조회)


```python
a2
```




    array([[10, 20, 30],
           [40, 50, 60]])




```python
a2[:1, 1:] # 슬라이싱을 사용하면 2차원 배열이 유지
```




    array([[20, 30]])




```python
a2[0, 1:] # 색인을 사용하면 한차원 줄어들었음
```




    array([20, 30])



- 불리안 색인


```python
a1
```




    array([10., 20., 30.])




```python
a1 < 30
```




    array([ True,  True, False])




```python
a1[a1 < 30]
```




    array([10., 20.])




```python
a2
```




    array([[10, 20, 30],
           [40, 50, 60]])




```python
(a2 < 30) | (a2 >50)
```




    array([[ True,  True, False],
           [False, False,  True]])




```python
a2[(a2 < 30) | (a2 >50)]
```




    array([10, 20, 60])



- 팬시 색인(불연속적으로 가져오거나, 원하는 순서로 색인)


```python
a1
```




    array([10., 20., 30.])




```python
a1[[2, 0]]
```




    array([30., 10.])



- 브로드캐스팅


```python
a1.shape, a2.shape
```




    ((3,), (2, 3))




```python
a2 + 1
```




    array([[11, 21, 31],
           [41, 51, 61]])




```python
a2 + a1
```




    array([[20., 40., 60.],
           [50., 70., 90.]])




```python
a0 = np.array([10, 20])
a2.shape, a0.shape
```




    ((2, 3), (2,))




```python
a2 + a0 # a0는 브로드캐스팅을 할 수 없는 형상
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-117-97135751b3b4> in <module>
    ----> 1 a2 + a0 # a0는 브로드캐스팅을 할 수 없는 형상
    

    ValueError: operands could not be broadcast together with shapes (2,3) (2,) 


- 배열의 축


```python
a1
```




    array([10., 20., 30.])




```python
np.sum(a1)
```




    60.0




```python
np.sum(a2)
```




    210




```python
np.sum(a2, axis=0) # 행축을 따라서 더하기
```




    array([50, 70, 90])




```python
np.sum(a2, axis=1) # 열축을 따라서 더하기
```




    array([ 60, 150])



## 3. 그 외 기능들

### 3.1 배열 전치


```python
arr = np.arange(15).reshape((3, 5)) # 3행 5열의 행렬
arr
```




    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])




```python
# 전치 행렬은 행과 열의 위치를 바꾼 결과
# 0: (0행, 1열) --> (1행, 0열)
# 13 : (2행, 3열) --> (3행, 2열)
```


```python
arr.T
```




    array([[ 0,  5, 10],
           [ 1,  6, 11],
           [ 2,  7, 12],
           [ 3,  8, 13],
           [ 4,  9, 14]])




```python
arr.transpose()
```




    array([[ 0,  5, 10],
           [ 1,  6, 11],
           [ 2,  7, 12],
           [ 3,  8, 13],
           [ 4,  9, 14]])




```python
arr.swapaxes(1, 0)
```




    array([[ 0,  5, 10],
           [ 1,  6, 11],
           [ 2,  7, 12],
           [ 3,  8, 13],
           [ 4,  9, 14]])




```python
arr
```




    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])




```python
arr * arr # 개별 원소의 곱셈
```




    array([[  0,   1,   4,   9,  16],
           [ 25,  36,  49,  64,  81],
           [100, 121, 144, 169, 196]])




```python
# 행렬의 곱셈
# X @ Y
# np.dot(X, Y)
```


```python
arr @ arr
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-131-5fd3d23230c3> in <module>
    ----> 1 arr @ arr
    

    ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 3 is different from 5)



```python
arr @ arr.T # (3, 5) x (5, 3) = (3, 3)
```




    array([[ 30,  80, 130],
           [ 80, 255, 430],
           [130, 430, 730]])




```python
np.dot(arr, arr.T) # (3, 5) x (5, 3) = (3, 3)
```




    array([[ 30,  80, 130],
           [ 80, 255, 430],
           [130, 430, 730]])



### 3.2 배열 연산과 조건절 표현하기


```python
x_list = [1, 2, 3, 4]
y_list = [5, 6, 7, 8]
cond_list = [True, True, False, False]

result = []
# cond_list에 있는 값이 True이면 x_list의 값을 취하고, False이면 y_list의 값을 취하기
for x, y, c in zip(x_list, y_list, cond_list):
  if c:
    result.append(x)
  else:
    result.append(y)
print(result)
```

    [1, 2, 7, 8]
    


```python
# np.where(조건, 조건이 참일때 취할 값, 조건이 거짓일 때 취할 값)
```


```python
x_arr = np.array([1, 2, 3, 4])
y_arr = np.array([5, 6, 7, 8])
cond_arr = np.array([True, True, False, False])

np.where(cond_arr, x_arr, y_arr)
```




    array([1, 2, 7, 8])




```python
arr = np.arange(16).reshape((4, 4))
arr
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])




```python
arr > 10
```




    array([[False, False, False, False],
           [False, False, False, False],
           [False, False, False,  True],
           [ True,  True,  True,  True]])




```python
cond = arr > 10

result = np.where(cond, 1, -1)
result
```




    array([[-1, -1, -1, -1],
           [-1, -1, -1, -1],
           [-1, -1, -1,  1],
           [ 1,  1,  1,  1]])




```python
result2 = np.where(cond, 1, arr)
result2
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10,  1],
           [ 1,  1,  1,  1]])



### 3.3 수학 통계 메서드


```python
Image('./images/image5.PNG')
```




    
![png](/assets/images/output_172_0.png)
    




```python
arr = np.arange(20).reshape((5, 4))
arr
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19]])




```python
np.sum(arr)
```




    190




```python
arr.sum()
```




    190




```python
arr.sum(0) # 0번축을 따라서
```




    array([40, 45, 50, 55])




```python
np.sum(arr, axis=0)
```




    array([40, 45, 50, 55])




```python
np.mean(arr, 0)
```




    array([ 8.,  9., 10., 11.])




```python
arr.min()
```




    0




```python
arr.min(0)
```




    array([0, 1, 2, 3])




```python
arr.min(1)
```




    array([ 0,  4,  8, 12, 16])




```python
arr.cumsum(0)
```




    array([[ 0,  1,  2,  3],
           [ 4,  6,  8, 10],
           [12, 15, 18, 21],
           [24, 28, 32, 36],
           [40, 45, 50, 55]])




```python
arr.cumsum(1)
```




    array([[ 0,  1,  3,  6],
           [ 4,  9, 15, 22],
           [ 8, 17, 27, 38],
           [12, 25, 39, 54],
           [16, 33, 51, 70]])




```python
arr
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19]])




```python
arr.cumprod(1)
```




    array([[    0,     0,     0,     0],
           [    4,    20,   120,   840],
           [    8,    72,   720,  7920],
           [   12,   156,  2184, 32760],
           [   16,   272,  4896, 93024]])



### 3.4 불리언 배열 메서드


```python
bool_arr = np.array([True, False, True])
bool_arr
```




    array([ True, False,  True])




```python
bool_arr.all()
```




    False




```python
bool_arr.any()
```




    True




```python
arr = np.arange(16).reshape((2, 8))
arr
```




    array([[ 0,  1,  2,  3,  4,  5,  6,  7],
           [ 8,  9, 10, 11, 12, 13, 14, 15]])




```python
bool_arr2 = arr < 8
bool_arr2
```




    array([[ True,  True,  True,  True,  True,  True,  True,  True],
           [False, False, False, False, False, False, False, False]])




```python
bool_arr2.all()
```




    False




```python
bool_arr2.all(1)
```




    array([ True, False])




```python
bool_arr2.any(0)
```




    array([ True,  True,  True,  True,  True,  True,  True,  True])



### 3.5 정렬


```python
arr = np.random.randn(100, 100) # numpy에서 제공하는 난수 생성 함수
                            # 표준 정규분포를 따르는 표본을 지정된 사이즈만큼 생성
```


```python
arr.mean(), arr.std()
```




    (0.00024717053106496, 1.0044765053464086)




```python
arr = np.random.randn(3, 4)
arr
```




    array([[ 0.48390323, -1.4603562 ,  0.82174863,  0.51525352],
           [ 0.39243845,  0.7713343 , -0.06983777, -0.81490397],
           [-0.13819069,  0.10405236, -0.62816576,  1.38749351]])




```python
np.sort(arr, 1) # 정렬 결과가 보여짐
```




    array([[-1.4603562 ,  0.48390323,  0.51525352,  0.82174863],
           [-0.81490397, -0.06983777,  0.39243845,  0.7713343 ],
           [-0.62816576, -0.13819069,  0.10405236,  1.38749351]])




```python
arr.sort(1) # 정렬결과가 arr에 반영
```


```python
arr
```




    array([[-1.4603562 ,  0.48390323,  0.51525352,  0.82174863],
           [-0.81490397, -0.06983777,  0.39243845,  0.7713343 ],
           [-0.62816576, -0.13819069,  0.10405236,  1.38749351]])



### 3.6 집합 관련 함수


```python
fruits_list = ["strawberry", "strawberry", "pear", "apple"]
fruits_list
```




    ['strawberry', 'strawberry', 'pear', 'apple']




```python
set(fruits_list) # 파이썬의 집합
```




    {'apple', 'pear', 'strawberry'}




```python
fruits_array = np.array(fruits_list)
fruits_array
```




    array(['strawberry', 'strawberry', 'pear', 'apple'], dtype='<U10')




```python
np.unique(fruits_array)
```




    array(['apple', 'pear', 'strawberry'], dtype='<U10')



### 3.7 난수 생성


```python
Image('./images/image6.png')
```




    
![png](/assets/images/output_208_0.png)
    



- 정규 분포에서 표본을 추출


```python
samples = np.random.normal(size=1000, loc=10, scale=1) # 정규 분포를 따르고 평균 10, 표준편차가 1인 1000개의 샘플 추출
```


```python
samples.mean()
```




    10.016258994031508




```python
samples.std()
```




    0.9756778997071996




```python
plt.hist(samples, bins=30)
plt.show()
```


    
![png](/assets/images/output_213_0.png)
    


- 표준 정규분포에서 표본을 추출


```python
samples = np.random.randn(1000) # 평균 1, 표준편차 0 인 표준 정규분포에서 1000개의 샘플을 추출
```


```python
samples.mean(), samples.std()
```




    (0.029002969016076918, 0.971699214028156)




```python
plt.hist(samples, bins=20)
plt.show()
```


    
![png](/assets/images/output_217_0.png)
    


- 최대, 최소 범위에서 표본을 추출


```python
# 주사위 던지기 50회
for i in range(50):
  draw = np.random.randint(1, 7)
  print(draw, end=' ')
```

    2 2 6 6 5 4 6 2 1 6 6 3 1 6 1 3 5 1 2 1 1 5 3 3 5 5 2 6 6 3 3 4 5 4 1 4 6 2 4 5 3 2 2 1 6 1 5 1 3 2 


```python
# 반복문 없이 주사위 던지기 50회 시뮬레이션
np.random.randint(1, 7, 50)
```




    array([2, 3, 2, 5, 4, 2, 4, 4, 3, 4, 5, 5, 5, 6, 4, 1, 6, 5, 4, 2, 2, 6,
           3, 5, 4, 5, 1, 4, 1, 4, 3, 1, 6, 1, 4, 4, 6, 4, 3, 6, 3, 5, 5, 1,
           3, 2, 4, 6, 2, 6])




```python
arr = np.arange(16)
p = np.random.permutation(arr)
p.reshape(4, 4) # 16명 4팀 무작위 추출
```




    array([[10, 13,  3, 11],
           [15,  7,  9,  0],
           [ 8,  5,  6,  4],
           [ 1,  2, 14, 12]])


