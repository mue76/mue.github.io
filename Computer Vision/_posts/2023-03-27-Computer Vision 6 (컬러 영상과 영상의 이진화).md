---
tag: [computer vision, opencv]
toc: true
toc_sticky: true
toc_label: 목차
author_profile: false

---

## 1. 컬러 영상 다루기

### 컬러 영상의 픽셀값 참조


```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```


```python
src = cv2.imread('./data/butterfly.jpg', cv2.IMREAD_COLOR)
print(src.shape)

# 좌상단 좌표의 B, G, R 광도값을 출력
# todo
b = src[0, 0, 0] # b channel의 좌상단
g = src[0, 0, 1] # g channel의 좌상단
r = src[0, 0, 2] # r channel의 좌상단
print(b, g, r)
cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()
```

    (356, 493, 3)
    47 88 50
    

**참고** matplotlib의 backend 지정


```python
%matplotlib inline # matplotlib의 default backend가 inline
```


```python
plt.imshow(src)
```




    <matplotlib.image.AxesImage at 0x2b5eca6ee50>




    
![png](/assets/images/2023-03-27-Computer Vision 6 (컬러 영상과 영상의 이진화)/output_6_1.png)
    



```python
# matplotlib의 backend를 qt로 사용하면 픽셀의 좌표와 색상(R, G, B) 값을 표시해줌
%matplotlib qt
plt.imshow(src)
```




    <matplotlib.image.AxesImage at 0x2b5ef5fe9a0>



### 컬러 영상의 픽셀값 반전


```python
src = cv2.imread('./data/butterfly.jpg', cv2.IMREAD_COLOR)

dst = src.copy()

# todo (반전 영상)
# option 1
# dst[:, :, 0] = 255 - src[:, :, 0]
# dst[:, :, 1] = 255 - src[:, :, 1]
# dst[:, :, 2] = 255 - src[:, :, 2]

# option 2
dst = 255 - src

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```

### 색공간 변환

**BGR -> GRAY**


```python
src = cv2.imread('./data/butterfly.jpg', cv2.IMREAD_COLOR) # B->G->R
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

cv2.imshow('src', src)
cv2.imshow('gray', gray)
cv2.waitKey()
cv2.destroyAllWindows()
```

**BGR -> HSV**


```python
src = cv2.imread('./data/butterfly.jpg', cv2.IMREAD_COLOR) # B->G->R
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(hsv)

print(h.min(), h.max()) # 0~179 단계로 색상정보 구분
print(s.min(), s.max()) # 0~255 단계로 채도정보 구분
print(v.min(), v.max()) # 0~255 단계로 명도정보 구분


cv2.imshow('src', src)
cv2.imshow('h', h)
cv2.imshow('s', s)
cv2.imshow('v', v)

cv2.waitKey()
cv2.destroyAllWindows()
```

    0 179
    0 255
    0 255
    


```python
(gray == v).sum() # 회색조 영상과 hsv의 v 채널값이 얼마나 같은지 확인
```




    5646



**BGR -> YCrCb**


```python
src = cv2.imread('./data/butterfly.jpg', cv2.IMREAD_COLOR) # B->G->R
yCrCb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)

y, Cr, Cb = cv2.split(yCrCb)

print(y.min(), y.max()) 
print(Cr.min(), Cr.max()) 
print(Cb.min(), Cb.max()) 


cv2.imshow('src', src)
cv2.imshow('y', y)
cv2.imshow('Cr', Cr)
cv2.imshow('Cb', Cb)

cv2.waitKey()
cv2.destroyAllWindows()
```

    0 255
    90 203
    50 161
    


```python
(gray == y).sum()
```




    175502




```python
y.size # 면적
```




    175508




```python
src = cv2.imread('./data/butterfly.jpg', cv2.IMREAD_COLOR) # B->G->R

b, g, r = cv2.split(src)

print(b.min(), b.max()) 
print(g.min(), g.max()) 
print(r.min(), r.max()) 


cv2.imshow('src', src)
cv2.imshow('b', b)
cv2.imshow('g', g)
cv2.imshow('r', r)

cv2.waitKey()
cv2.destroyAllWindows()
```

    0 255
    0 255
    0 255
    

### 컬러 히스토그램 평활화


```python
src = cv2.imread('./data/pepper.bmp')
# cv2.cvtColor(이미지, 바꿀컬러공간)
src_yCrCb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
y, Cr, Cb = cv2.split(src_yCrCb)
y_equalized = cv2.equalizeHist(y)

dst_yCrCb = cv2.merge([y_equalized, Cr, Cb])
dst = cv2.cvtColor(dst_yCrCb, cv2.COLOR_YCrCb2BGR)

cv2.imshow('src', src)
cv2.imshow('dst', dst) # BGR 채널 순서를 기대
cv2.waitKey()
cv2.destroyAllWindows()
```

### 색상 범위 지정에 의한 영역 분할

**B, G, R 값을 기준으로 구간 조정하여 영역 분할**


```python
def on_level_change(pos):
    lower_b = cv2.getTrackbarPos('lower_b', 'dst')
    upper_b = cv2.getTrackbarPos('upper_b', 'dst')
    lower_g = cv2.getTrackbarPos('lower_g', 'dst')
    upper_g = cv2.getTrackbarPos('upper_g', 'dst')
    lower_r = cv2.getTrackbarPos('lower_r', 'dst')
    upper_r = cv2.getTrackbarPos('upper_r', 'dst')
    
    lower = (lower_b, lower_g, lower_r)
    upper = (upper_b, upper_g, upper_r)

    dst = cv2.inRange(src, lower, upper)
    cv2.imshow('dst', dst)

src = cv2.imread('./data/candies.png')
dst = src.copy()

# blue m&m (rgb) : 10, 125, 246 (by jcpicker)

# lower_b, lower_g, lower_r = 220, 100, 0
# upper_b, upper_g, upper_r = 255, 140, 20

cv2.imshow('src', src)
cv2.imshow('dst', dst)

cv2.createTrackbar('lower_b', 'dst', 0, 255, on_level_change)
cv2.createTrackbar('upper_b', 'dst', 0, 255, on_level_change)
cv2.createTrackbar('lower_g', 'dst', 0, 255, on_level_change)
cv2.createTrackbar('upper_g', 'dst', 0, 255, on_level_change)
cv2.createTrackbar('lower_r', 'dst', 0, 255, on_level_change)
cv2.createTrackbar('upper_r', 'dst', 0, 255, on_level_change)

cv2.waitKey()
cv2.destroyAllWindows()
```

**H, S, V 값을 기준으로 구간 조정하여 영역 분할**


```python
# green m&m (hsv) : 129, 93, 70 (by jcpicker) --> 64, 237, 178(by opencv)
```


```python
129/360*180, 93/100*255, 70/100*255
```




    (64.5, 237.15, 178.5)




```python
def on_level_change(pos):
    lower_h = cv2.getTrackbarPos('lower h', 'dst')
    upper_h = cv2.getTrackbarPos('upper h', 'dst')
    lower_s = cv2.getTrackbarPos('lower s', 'dst')
    upper_s = cv2.getTrackbarPos('upper s', 'dst')
    lower_v = cv2.getTrackbarPos('lower v', 'dst')
    upper_v = cv2.getTrackbarPos('upper v', 'dst')    

   
    lower = (lower_h, lower_s, lower_v)
    upper = (upper_h, upper_s, upper_v)
    
    dst = cv2.inRange(hsv, lower, upper)
    cv2.imshow('dst', dst)

src = cv2.imread('./data/candies.png')
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
# hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV_FULL) # h가 0~360도 구간으로 표시
dst = src.copy()

cv2.imshow("src", src)
cv2.imshow("dst", dst)

cv2.createTrackbar('lower h', 'dst', 0, 255, on_level_change)
cv2.createTrackbar('upper h', 'dst', 0, 255, on_level_change)

cv2.createTrackbar('lower s', 'dst', 0, 255, on_level_change)
cv2.createTrackbar('upper s', 'dst', 0, 255, on_level_change)

cv2.createTrackbar('lower v', 'dst', 0, 255, on_level_change)
cv2.createTrackbar('upper v', 'dst', 0, 255, on_level_change)

cv2.waitKey()
cv2.destroyAllWindows()
```

### 히스토그램 역투영


```python
src = np.array([[0, 0, 0, 0],
                [1, 1, 3, 5],
                [6, 1, 1, 3],
                [4, 3, 1, 7]], dtype=np.uint8)

hist = cv2.calcHist(images=[src], channels=[0], mask=None, histSize=[4], ranges=[0, 8])
hist
```




    array([[9.],
           [3.],
           [2.],
           [2.]], dtype=float32)




```python
backP = cv2.calcBackProject(images=[src], channels=[0], hist=hist, ranges=[0, 8], scale=1)
backP
```




    array([[9, 9, 9, 9],
           [9, 9, 3, 2],
           [2, 9, 9, 3],
           [2, 3, 9, 2]], dtype=uint8)




```python
ref = cv2.imread('./data/ref.png')
mask = cv2.imread('./data/mask.bmp', cv2.IMREAD_GRAYSCALE)
src = cv2.imread('./data/kids.png')

ref_yCrCb = cv2.cvtColor(ref, cv2.COLOR_BGR2YCrCb)
src_yCrCb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)

# 히스토그램 구하기
hist = cv2.calcHist(images=[ref_yCrCb], channels=[1, 2], mask=mask, histSize=[128, 128], ranges=[0, 256, 0, 256])
# hist.shape (128, 128) : Cr, Cb 의 조합인 2차원 히스토그램

# 구한 히스토그램을 역투영하기
backP= cv2.calcBackProject(images=[src_yCrCb], channels=[1, 2], hist=hist, ranges=[0, 256, 0, 256], scale=1)

cv2.imshow('src', src)
cv2.imshow('mask', mask)
cv2.imshow('ref', ref)
cv2.imshow('backP', backP)
cv2.waitKey()
cv2.destroyAllWindows()
```

## 2. 영상의 이진화

### 전역 이진화


```python
def on_thresh(pos):
    #ret, dst = cv2.threshold(이미지, 임계값, 임계값을 넘게되면 적용할 최대값, 임계값연산방법)
    ret, dst = cv2.threshold(src, pos, 255, cv2.THRESH_BINARY)  
    cv2.imshow('dst', dst)

src = cv2.imread('./data/neutrophils.png', cv2.IMREAD_GRAYSCALE)
dst = src.copy()

cv2.imshow('src', src) 
cv2.imshow('dst', dst) 

cv2.createTrackbar('threshold', 'dst', 0, 255, on_thresh)

cv2.waitKey()
cv2.destroyAllWindows()
```

**Otsu 알고리즘에 의한 임계값 사용**


```python
src = cv2.imread('./data/neutrophils.png', cv2.IMREAD_GRAYSCALE)
dst = src.copy()


thresh, dst = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  
print(thresh)

cv2.imshow('src', src) 
cv2.imshow('dst', dst) 

cv2.waitKey()
cv2.destroyAllWindows()
```

    206.0
    


```python
src = cv2.imread('./data/heart10.jpg', cv2.IMREAD_GRAYSCALE)

thresh, dst = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  
print(thresh)

cv2.imshow('src', src) 
cv2.imshow('dst', dst) 
cv2.waitKey()
cv2.destroyAllWindows()
```

    175.0
    

### 적응형 이진화


```python
src = cv2.imread('./data/sudoku.jpg', cv2.IMREAD_GRAYSCALE)

thresh, dst = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  
print(thresh)

cv2.imshow('src', src) 
cv2.imshow('dst', dst) 
cv2.waitKey()
cv2.destroyAllWindows()
```

    97.0
    


```python
def on_thresh(pos):
    #ret, dst = cv2.threshold(이미지, 임계값, 임계값을 넘게되면 적용할 최대값, 임계값연산방법)
    ret, dst = cv2.threshold(src, pos, 255, cv2.THRESH_BINARY)  
    cv2.imshow('dst', dst)

src = cv2.imread('./data/sudoku.jpg', cv2.IMREAD_GRAYSCALE)
dst = src.copy()

cv2.imshow('src', src) 
cv2.imshow('dst', dst) 

cv2.createTrackbar('threshold', 'dst', 0, 255, on_thresh)

cv2.waitKey()
cv2.destroyAllWindows()
```


```python
def on_trackbar(pos):
    b_size = pos
    if b_size % 2 == 0: # 짝수
        b_size += 1
    if b_size < 3:
        b_size = 3
        
    C = 5
    dst = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, b_size, C)
    
    cv2.imshow('dst', dst)

src = cv2.imread('./data/sudoku.jpg', cv2.IMREAD_GRAYSCALE)
dst = src.copy()

cv2.imshow('src', src) 
cv2.imshow('dst', dst) 

cv2.createTrackbar('block size', 'dst', 3, 200, on_trackbar)
cv2.setTrackbarPos('block size', 'dst', 11)

cv2.waitKey()
cv2.destroyAllWindows()
```

### kMeans Clustering
- https://docs.opencv.org/4.6.0/d1/d5c/tutorial_py_kmeans_opencv.html


```python
img = cv2.imread('./data/home.jpg')
img.shape

Z = img.reshape((-1,3))
print(Z.shape)

# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
print(center)
print(label.shape)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

    (196608, 3)
    [[131 182 195]
     [103 141 151]
     [199 161 124]
     [ 64  91  99]
     [165  95  20]
     [172 199 205]
     [ 27  42  45]
     [181 123  61]]
    (196608, 1)
    


```python

```
