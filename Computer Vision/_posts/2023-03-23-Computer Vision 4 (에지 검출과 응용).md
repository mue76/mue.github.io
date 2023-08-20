---
tag: [computer vision, opencv]
toc: true
toc_sticky: true
toc_label: 목차
author_profile: false

---

# 에지 검출과 응용


```python
import cv2
import numpy as np
```

## 마스크 기반 에지 검출 - 소벨 마스크


```python
src = cv2.imread('./data/lenna.bmp', cv2.IMREAD_GRAYSCALE)

# 소벨 마스크
Mx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=np.float32)

My = np.array([[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]], dtype=np.float32)

# 필터 적용
dx = cv2.filter2D(src, -1, Mx, delta=128) # 수평방향으로 미분한 결과 영상
dy = cv2.filter2D(src, -1, My, delta=128) # 수직방향으로 미분한 결과 영상


cv2.imshow('src', src)
cv2.imshow('dx', dx)
cv2.imshow('dy', dy)

cv2.waitKey()
cv2.destroyAllWindows()
```


```python
# 미분필터(소벨 마스크)를 자체적으로 준비해서, 원본 이미지와 마스크 연산까지 해주는 함수
dx = cv2.Sobel(src, cv2.CV_32FC1, dx=1, dy=0) # x축으로만 미분한 결과 영상
dy = cv2.Sobel(src, cv2.CV_32FC1, dx=0, dy=1) # y축으로만 미분한 결과 영상
# cv2.Sobel() 함수는 cv2.filter2D() 함수와 마스크연산까지는 동일한데
# 형변환도 안되어 있고, 포화연산도 안되어 있는 상태라 처리가 필요함

fmag = cv2.magnitude(dx, dy) 
# 그레디언트 : [x축으로의 미분, y축으로의 미분]
# 그레디언트의 크기를 magnitude() 함수가 구해줌
fmag = np.clip(fmag, 0, 255) # 포화연산
mag = fmag.astype(np.uint8) # 시각화를 위해 상한선과 하한선을 0~255로 제한, 데이터타입도 uint8로 바꿔줌

T = 160
ret, dst = cv2.threshold(mag, T, 255, cv2.THRESH_BINARY)

cv2.imshow('src', src)
cv2.imshow('mag', mag)
cv2.imshow('dst', dst)

cv2.waitKey()
cv2.destroyAllWindows()
```

## 캐니 에지 검출기


```python
src = cv2.imread('./data/lenna.bmp', cv2.IMREAD_GRAYSCALE)

low_thresh = 70
high_thresh = 160

dst = cv2.Canny(src, low_thresh, high_thresh) # low_thresh : high_thresh (1:2 or 1:3 Canny Recommended)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```

## 허프 변환 직선 검출


```python
import math

src = cv2.imread('./data/building.jpg', cv2.IMREAD_GRAYSCALE)

edge = cv2.Canny(src, 100, 200)

# rho와 theta는 숫자가 작을수록 더 정밀하게 검출되지만 연산시간이 더 걸림        
rho = 1 # or 2 : 축적배열의 해상도(픽셀단위)
theta = math.pi/180 # theta 값의 해상도(라디안 단위)
    
threshold = 150      # 축적배열의 숫자가 높다는 것은 직선을 이루는 점들이 많다는 뜻.
                     # 얼마나 큰 값을 직선으로 판단할지 threshold에 달려있음
minLineLength = 50   # 검출할 선분의 최소길이
maxLineGap = 5       # 직선으로 간주한 최대 에지 점 간격  
    
lines = cv2.HoughLinesP(edge, rho, theta, threshold,
                        minLineLength = minLineLength,
                        maxLineGap = maxLineGap)    

dst = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR) # 빨간색 직선을 그릴 도화지(3 채널 도화지)

if lines is not None:
    for line in lines:
        line = line[0]
        pt1 = line[0], line[1]
        pt2 = line[2], line[3]
        cv2.line(dst, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)                        

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```
