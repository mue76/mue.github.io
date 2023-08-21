---
tag: [computer vision, opencv]
toc: true
toc_sticky: true
toc_label: 목차
author_profile: false

---

# 영상의 필터링과 기하학적 변환

## 1. 영상의 필터링


```python
import cv2
import numpy as np
```

### 엠보싱 필터


```python
src = cv2.imread('./data/rose.bmp', cv2.IMREAD_GRAYSCALE)

# 필터
emboss = np.array([[-1, -1, 0],
                   [-1, 0, 1],
                   [0, 1, 1]])

# cv2.filter2D(적용할 이미지, 출력채널수, 필터)
dst = cv2.filter2D(src, -1, emboss, delta=128) # -1은 입력 영상과 출력 영상의 채널 깊이 동일
                                               # 필터링 연산의 결과 대각선 방향의 차이가 클 수록 도드라져보이는 효과
                                               # delta=128은 전체 결과에 128을 더해준다는 뜻으로
                                               # 전반적으로 영상을 밝게 표시해주는 효과

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```

### 블러링 (영상 부드럽게 하기)

#### (1) 평균값 필터


```python
src = cv2.imread('./data/rose.bmp', cv2.IMREAD_GRAYSCALE)

# 필터
blur = np.array([[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]], dtype=np.float32) *1/9

# cv2.filter2D(적용할 이미지, 출력채널수, 필터)
dst = cv2.filter2D(src, -1, blur)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```


```python
src = cv2.imread('./data/rose.bmp', cv2.IMREAD_GRAYSCALE)

cv2.imshow('src', src)

for ksize in (3, 5, 7):
    dst = cv2.blur(src, (ksize, ksize))
    desc = "Mean : %d x %d"% (ksize, ksize)    
    #cv2.putText(도화지, 텍스트, 좌표, 폰트종류, 폰트스케일, 색상, 굵기, 보정타입)    
    cv2.putText(dst, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
    cv2.imshow('dst', dst)
    cv2.waitKey()

cv2.destroyAllWindows()
```

#### (2) 가우시안 필터


```python
src = cv2.imread('./data/rose.bmp', cv2.IMREAD_GRAYSCALE)

cv2.imshow('src', src)

for sigma in range(1, 6):
    dst = cv2.GaussianBlur(src, (0, 0), sigma)
    desc = "Gaussian : %d sigma"% (sigma)
    cv2.putText(dst, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
    cv2.imshow('dst', dst)
    cv2.waitKey()

cv2.destroyAllWindows()
```

### 샤프닝 (영상 날카롭게 하기)


```python
from IPython.display import Image
Image('./images/image4.png', width=300)
```




    
![png](/assets/images/2023-03-21-Computer Vision 3 (영상의 필터링과 기하학적 변환)/output_12_0.png)
    



**sigma값을 변화시키면서 관찰**


```python
src = cv2.imread('./data/rose.bmp', cv2.IMREAD_GRAYSCALE)

cv2.imshow('src', src)

alpha = 1.0

for sigma in range(1, 6):
    blurred = cv2.GaussianBlur(src, (0, 0), sigma)    
    dst = cv2.addWeighted(src, (1+alpha), blurred, -alpha, 0) # (1+alph)f(x, y) -(alpha)f'(x, y)
    desc = "Gaussian : %d sigma"% (sigma)
    cv2.putText(dst, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
    cv2.imshow('dst', dst)
    cv2.waitKey()

cv2.destroyAllWindows()
```

**alpha값을 변화시키면서 관찰**


```python
 src = cv2.imread('./data/rose.bmp', cv2.IMREAD_GRAYSCALE)

cv2.imshow('src', src)

sigma = 1.0

for alpha in range(1, 4):
    blurred = cv2.GaussianBlur(src, (0, 0), sigma)    
    dst = cv2.addWeighted(src, (1+alpha), blurred, -alpha, 0) # (1+alph)f(x, y) -(alpha)f'(x, y)
    desc = "Gaussian : %d alpha"% (alpha)
    cv2.putText(dst, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
    cv2.imshow('dst', dst)
    cv2.waitKey()

cv2.destroyAllWindows()
```

### 잡음 제거 필터링

#### (1) 가우시안 잡음 모델


```python
Image('./images/image5.png', width=200)
```




    
![png](/assets/images/2023-03-21-Computer Vision 3 (영상의 필터링과 기하학적 변환)/output_19_0.png)
    




```python
src = cv2.imread('./data/lenna.bmp', cv2.IMREAD_GRAYSCALE)
cv2.imshow('src', src)

for stddev in [10, 20, 30]:
    noise = np.zeros(src.shape, np.int32)
    cv2.randn(noise, mean=0, stddev=stddev)
    print(noise.mean(), noise.std())
    dst = cv2.add(src, noise, dtype=cv2.CV_8UC1) # 8UC1 : 8bit Unsigned 1 Channel
    desc = "stddev : %d"%(stddev)
    cv2.putText(dst, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
    cv2.imshow('dst', dst)
    cv2.waitKey()

cv2.destroyAllWindows()
```

    -0.014430999755859375 9.997347366897875
    0.041057586669921875 19.991874484004644
    0.02005767822265625 29.998018487768054
    

#### (2) 양방향 필터


```python
src = cv2.imread('./data/lenna.bmp', cv2.IMREAD_GRAYSCALE)
noise = np.zeros(src.shape, np.int32)
cv2.randn(noise, mean=0, stddev=5)
dst = cv2.add(src, noise, dtype=cv2.CV_8UC1) # np.uint8

# 가우시안
dst_gaussian = cv2.GaussianBlur(dst, (0, 0), 5)

# 양방향 필터
dst_bilateral = cv2.bilateralFilter(dst, -1, sigmaColor=10, sigmaSpace=5)
# sigmaSpace : 거리의 차이 기준으로만 얼마나 블러링을 강하게 할지를 결정
#            : simaSpace값이 클수록 블러링이 강하게 적용
# sigmaColor : 밝기값의 차이 기준으로만 얼마나 블러링을 강하게 할지를 결정
#            : simaColor 값이 커질수록 에지가 무뎌짐 (작을수록 에지가 더 보전)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('dst_gaussian', dst_gaussian)
cv2.imshow('dst_bilateral', dst_bilateral)
cv2.waitKey()
cv2.destroyAllWindows()
```

**sigmaColor에 따른 에지 보전 관찰**


```python
src = cv2.imread('./data/lenna.bmp', cv2.IMREAD_GRAYSCALE)
noise = np.zeros(src.shape, np.int32)
cv2.randn(noise, mean=0, stddev=5)
dst = cv2.add(src, noise, dtype=cv2.CV_8UC1) # np.uint8

# 양방향 필터
# sigmaColor : 밝기값의 차이 기준으로만 얼마나 블러링을 강하게 할지를 결정
#            : simaColor 값이 커질수록 에지가 무뎌짐 (작을수록 에지가 더 보전)
for sigmaColor in (5, 10, 30, 50):
    dst_bilateral = cv2.bilateralFilter(dst, -1, sigmaColor=sigmaColor, sigmaSpace=5)
    desc = "sigmaColor : %d"%(sigmaColor)
    cv2.putText(dst_bilateral, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
    cv2.imshow('dst_bilateral', dst_bilateral)
    cv2.waitKey()
    
cv2.imshow('src', src)
#cv2.imshow('dst', dst)
cv2.destroyAllWindows()
```

#### (3) 미디언 필터


```python
import random
```


```python
src = cv2.imread('./data/lenna.bmp', cv2.IMREAD_GRAYSCALE)

# salt & pepper noise(소금&후추 잡음) 생성
for i in range(0, int(src.size/10)): # 전체 픽셀수의 10분의 1 정도를 노이즈로 사용
    x = random.randint(0, src.shape[1]-1) # 어느 위치에 노이즈를 넣을지 x좌표(np 기준)
    y = random.randint(0, src.shape[0]-1) # 어느 위치에 노이즈를 넣을지 y좌표(np 기준)
    src[y, x] = (i % 2) * 255 # i가 짝수일때는 0(pepper), 홀수일때는 1(salt)

gaussian_blur = cv2.GaussianBlur(src, (0, 0), 5)
bilateral_blur = cv2.bilateralFilter(src, -1, sigmaColor=50, sigmaSpace=5)
median_blur = cv2.medianBlur(src, 3)

cv2.imshow('src', src)
cv2.imshow('gaussian_blur', gaussian_blur)
cv2.imshow('bilateral_blur', bilateral_blur)
cv2.imshow('median_blur', median_blur)
cv2.waitKey()
cv2.destroyAllWindows()
```

## 2. 영상의 기하학적 변환

### (1) 어파인 변환 (평행관계까 유지되는 변환)
- 세 점의 이동 관계를 통해 정의


```python
src = cv2.imread('./data/tekapo.bmp')
cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()
```


```python
src.shape
```




    (480, 640, 3)




```python
src = cv2.imread('./data/tekapo.bmp')

height = src.shape[0]
width = src.shape[1]

# 3점의 이동관계 정의
src_pts = np.array([[0, 0], [width-1, 0], [width-1, height-1]], dtype=np.float32)
dst_pts = np.array([[50, 50], [width-100, 100], [width-50, height-50]], dtype=np.float32)

M = cv2.getAffineTransform(src_pts, dst_pts) # 2x3 변환행렬을 반환
dst = cv2.warpAffine(src, M, dsize=(0, 0)) # 변환된 결과물, dsize=(0, 0) : 원본영상(src)과 동일한 사이즈


cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```

### (2) 이동변환


```python
src = cv2.imread('./data/tekapo.bmp')

a =  150 # x축으로 이동할 픽셀 수
b =  100 # y축으로 이동할 픽셀 수

M = np.array([[1, 0, a],
              [0, 1, b]], dtype=np.float32)

dst = cv2.warpAffine(src, M, dsize=(0, 0)) # 변환된 결과물, dsize=(0, 0) : 원본영상(src)과 동일한 사이즈

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```

### (3) 전단 변환


```python
(width + Mx*height)
```




    784.0




```python
src = cv2.imread('./data/tekapo.bmp')

height = src.shape[0]
width = src.shape[1]

Mx = 0.3
M = np.array([[1, Mx, 0],
              [0, 1, 0]], dtype=np.float32)

dst = cv2.warpAffine(src, M, dsize=(int(width + Mx*height), height)) # 변환된 결과물, dsize=(0, 0) : 원본영상(src)과 동일한 사이즈

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```

### (4) 크기 변환


```python
src = cv2.imread('./data/tekapo.bmp')

height = src.shape[0]
width = src.shape[1]

Sx = 0.8
Sy = 0.8

M = np.array([[Sx, 0, 0],
              [0, Sy, 0]], dtype=np.float32)

dst = cv2.warpAffine(src, M, dsize=(int(width*Sx), int(height*Sy))) # 변환된 결과물, dsize=(0, 0) : 원본영상(src)과 동일한 사이즈

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```

**위의 변환을 resize()함수로 대체**


```python
src = cv2.imread('./data/tekapo.bmp')

dst = cv2.resize(src, dsize=(0, 0), fx=1.2, fy=1.2)
#dst = cv2.resize(src, (1920, 1280))

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```

### (5) 회전 변환


```python
src = cv2.imread('./data/tekapo.bmp')

height = src.shape[0]
width = src.shape[1]

ceter = width//2, height//2
angle = 20 # 반시계방향 20도 
scale = 1

M = cv2.getRotationMatrix2D(ceter, angle, scale)

dst = cv2.warpAffine(src, M, dsize=(0, 0)) # 변환된 결과물, dsize=(0, 0) : 원본영상(src)과 동일한 사이즈

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```


```python
src = cv2.imread('./data/tekapo.bmp')

dst1 = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)
dst2 = cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE)

cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()
```

### (6) 대칭 변환


```python
src = cv2.imread('./data/tekapo.bmp')

flip_code1 = 1 # 1:좌우반전, 0:상하반전, -1:좌우, 상하 모두 반전
flip_code0 = 0
flip_codem1 = -1

dst1 = cv2.flip(src, flip_code1)
dst2 = cv2.flip(src, flip_code0)
dst3 = cv2.flip(src, flip_codem1)

cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.imshow('dst3', dst3)
cv2.waitKey()
cv2.destroyAllWindows()
```

### (7) 투시 변환


```python
src = cv2.imread('./data/tekapo.bmp')

height = src.shape[0]
width = src.shape[1]

src_pts = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)
dst_pts = np.array([[60, 100], [width-20, 80], [width-10, height-10], [5, height-20]], dtype=np.float32)

# 4점의 이동관계 정의
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
dst = cv2.warpPerspective(src, M, dsize=(0, 0))

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```

### 실습 : 카드 정면으로 보이게 하기


```python
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(src, (x, y), 5, (0, 0, 255), 5)
        cv2.imshow('src', src)
        src_pts_list.append([x, y])
        if len(src_pts_list) == 4:
            src_pts = np.array(src_pts_list, dtype=np.float32)
            # 4점의 이동관계 정의
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            dst = cv2.warpPerspective(src, M, dsize=(width, height))
            cv2.imshow('dst', dst)

src = cv2.imread('./data/card.bmp')

cv2.imshow('src', src)
# 마우스 이벤트 등록
cv2.setMouseCallback('src', on_mouse)

width, height = 200, 300
src_pts_list = []
dst_pts = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)

cv2.waitKey()
cv2.destroyAllWindows()
```
## Reference
- [OpenCV 4로 배우는 컴퓨터 비전과 머신 러닝 (황선규 저)](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=187822936)