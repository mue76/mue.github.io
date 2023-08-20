---
tag: [computer vision, opencv]
toc: true
toc_sticky: true
toc_label: 목차
author_profile: false

---

# 이벤트 처리와 영상 분석

## 1. 이벤트 처리


```python
import cv2
import numpy as np
```

### (1) 키보드 이벤트


```python
img = cv2.imread('./data/lena.jpg')

cv2.imshow('img', img)

while True:
    keycode = cv2.waitKey()
    print(keycode)
    if keycode == ord('i'): # ord() : ascii code로 변환
        img = 255 - img
        cv2.imshow('img', img)
    elif keycode == ord('q'): # ord() : ascii code로 변환
        break
cv2.destroyAllWindows()
```

    113
    

### (2) 마우스 이벤트

**참고**
https://docs.opencv.org/4.7.0/d0/d90/group__highgui__window__flags.html


```python
# 마우스 이벤트를 처리할 루틴
def on_mouse(event, x, y, flags, param):
    global old_x, old_y
    
    if event == cv2.EVENT_LBUTTONDOWN:
        old_x, old_y = x, y
        
    elif event == cv2.EVENT_LBUTTONUP:
        pass
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_FLAG_LBUTTON:            
            cv2.line(img, (old_x, old_y), (x, y), (0, 255, 255), 2)
            cv2.imshow('img', img)
            old_x, old_y = x, y    

img = cv2.imread('./data/lena.jpg')
cv2.imshow('img', img)
# 마우스 이벤트가 발생했을 때 처리할 루틴(함수) 등록
cv2.setMouseCallback('img', on_mouse)
cv2.waitKey() 
cv2.destroyAllWindows()
```

### (3) 트랙바 이벤트


```python
def on_level_change(pos):
    # 이벤트 발생할 때마다 화면 밝기가 바뀌도록 구현
    img[:] = pos
    cv2.imshow('img', img)
    
    
img = np.full((512, 512, 3), 0, np.uint8)
cv2.imshow('img', img)
# 트랙바 이벤트가 발생했을 때 처리할 루틴(함수) 등록
#cv2.createTrackbar(트랙바이름, 윈도우, 트랙바값의 범위, 이벤트발생시 처리할 함수)
cv2.createTrackbar('level', 'img', 0, 255, on_level_change)
cv2.setTrackbarPos('level', 'img', 128)

cv2.waitKey()
cv2.destroyAllWindows()
```

### 실습 : 트랙바를 이용한 R, G, B 색상 조합 만들기


```python
# def on_level_change(pos):
#     cv2.getTrackbarPos(트랙바이름, 윈도우)
#     R_pos = cv2.getTrackbarPos('R', 'img')
```


```python
def on_level_change(pos):
    # 이벤트 발생할 때마다 현재 R, G, B 밝기값의 조합으로 새로운 색상 생성
    
    # pos = cv2.getTrackbarPos(트랙바이름, 윈도우)
    R_pos = cv2.getTrackbarPos('R', 'img')
    G_pos = cv2.getTrackbarPos('G', 'img')
    B_pos = cv2.getTrackbarPos('B', 'img')

    img[:] = (B_pos, G_pos, R_pos)
    cv2.imshow('img', img)
        
img = np.full((512, 512, 3), 0, np.uint8)
cv2.imshow('img', img)
# 트랙바 이벤트가 발생했을 때 처리할 루틴(함수) 등록
#cv2.createTrackbar(트랙바이름, 윈도우, 트랙바값의 범위, 이벤트발생시 처리할 함수)
cv2.createTrackbar('R', 'img', 0, 255, on_level_change)
cv2.createTrackbar('G', 'img', 0, 255, on_level_change)
cv2.createTrackbar('B', 'img', 0, 255, on_level_change)

cv2.setTrackbarPos('B', 'img', 255)

cv2.waitKey()
cv2.destroyAllWindows()
```

## 2. 유용한 기능들

### 마스크 연산


```python
lena = cv2.imread('./data/lenna.bmp')
mask = cv2.imread('./data/mask_smile.bmp', cv2.IMREAD_GRAYSCALE)

print(lena.shape, mask.shape)

# mask 이미지에서 흰색부분의 위치만 레나 영상에서 노란색으로 바꾸기
lena[mask>0] = (0, 255, 255) # yellow

cv2.imshow('lena', lena)
cv2.imshow('mask', mask)

cv2.waitKey()
cv2.destroyAllWindows()
```

    (512, 512, 3) (512, 512)
    

### 실습 : 빈 들판에 비행기 띄우기


```python
airplane = cv2.imread('./data/airplane.bmp')
mask = cv2.imread('./data/mask_plane.bmp', cv2.IMREAD_GRAYSCALE)
field = cv2.imread('./data/field.bmp')

field[mask>0] = airplane[mask>0]

cv2.imshow('airplane', airplane)
cv2.imshow('mask', mask)
cv2.imshow('field', field)

cv2.waitKey()
cv2.destroyAllWindows()
```

## 3. 영상의 밝기 조절

**영상의 밝기**
- 영상의 밝기는 특징으로 활용이 될 수 있음(예:주간/야간 영상 분류기)
- 어두운 데이터의 밝기 값을 조정해서 품질 높일 때 사용
- 딥러닝 모델을 학습시킬 때 주어진 데이터를 더 증강(원본 데이터 + 변형이 가해진 데이터)하는 목적으로 활용


```python
# option 1
src = cv2.imread('./data/lenna.bmp', cv2.IMREAD_GRAYSCALE)

src = np.array(src, dtype=np.int32) # 음수와 255보다 큰 수까지 표현할 수 있게

dst = src.copy()
dst = src + 100

dst = np.clip(dst, 0, 255) # dst의 상한선과 하한선을 255, 0으로 지정

# 데이터 타입 복구
src = np.array(src, dtype=np.uint8)
dst = np.array(dst, dtype=np.uint8)

cv2.imshow('src', src)
cv2.imshow('dst', dst)

cv2.waitKey()
cv2.destroyAllWindows()
```


```python
# option 2
src = cv2.imread('./data/lenna.bmp', cv2.IMREAD_GRAYSCALE)

dst = cv2.add(src, 100)

cv2.imshow('src', src)
cv2.imshow('dst', dst)

cv2.waitKey()
cv2.destroyAllWindows()
```

### 실습 : Trackbar를 이용해서 레나 영상의 밝기를 조절


```python
def update(pos):
    dst = cv2.add(src, pos)
    cv2.imshow('dst', dst)

src = cv2.imread('./data/lenna.bmp', cv2.IMREAD_GRAYSCALE)
dst = src.copy()
cv2.imshow('dst', dst)

cv2.createTrackbar('brightness', 'dst', 0, 100, update)
cv2.waitKey()
cv2.destroyAllWindows()
```

## 4. 영상의 명암비 조절

- 알파가 0보다 작으면 명암비가 낮아지고
- 알파가 0보다 크면 명암비가 높아짐


```python
from IPython.display import Image
Image('./images/image1.png', width=400)
```




    
![png](/assets/images/2023-03-20-Computer Vision 2 (이벤트 처리와 영상분석)/output_26_0.png)
    




```python
Image('./images/image2.png', width=400)
```




    
![png](/assets/images/2023-03-20-Computer Vision 2 (이벤트 처리와 영상분석)/output_27_0.png)
    




```python
src = cv2.imread('./data/lenna.bmp', cv2.IMREAD_GRAYSCALE)

alpha = 1.0

dst = src + (src - 128.0) * alpha
dst = np.clip(dst, 0, 255)
dst = dst.astype(src.dtype) # np.uint8로 바꿈

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```

## 5. 히스토그램 분석

### cv2.calcHist() 함수


```python
src = np.array([[0, 0, 0, 0],
                [1, 2, 3, 5],
                [6, 1, 2, 3],
                [4, 3, 1, 7]], dtype=np.uint8) # 4x4, 1 채널 영상
```


```python
#hist = cv2.calcHist(image=[이미지배열], channels=[채널], mask=[마스크], histSize=[구간], ranges=[범위])
hist1 = cv2.calcHist(images=[src], channels=[0], mask=None, histSize=[8], ranges=[0, 8])
hist1
```




    array([[4.],
           [3.],
           [2.],
           [3.],
           [1.],
           [1.],
           [1.],
           [1.]], dtype=float32)




```python
hist2 = cv2.calcHist(images=[src], channels=[0], mask=None, histSize=[4], ranges=[0, 8])
hist2
```




    array([[7.],
           [5.],
           [2.],
           [2.]], dtype=float32)




```python
hist2 = cv2.calcHist(images=[src], channels=[0], mask=None, histSize=[4], ranges=[0, 4])
hist2
```




    array([[4.],
           [3.],
           [2.],
           [3.]], dtype=float32)




```python
# np.histogram() 참고
# hist, bins_edges = np.histogram(src, bins=8, range=[0, 8])
# hist
```

### 히스그램 구하기(lena)


```python
import matplotlib.pyplot as plt
```

**그레이 스케일 영상**


```python
src = cv2.imread('./data/lenna.bmp', cv2.IMREAD_GRAYSCALE)
hist = cv2.calcHist(images=[src], channels=[0], mask=None, histSize=[32], ranges=[0, 255])

plt.plot(hist, color='r')
plt.bar(np.arange(32), hist.flatten(), color='b')
```




    <BarContainer object of 32 artists>




    
![png](/assets/images/2023-03-20-Computer Vision 2 (이벤트 처리와 영상분석)/output_39_1.png)
    



```python
hist.sum() # 영상의 모든 픽셀수 (넓이)
```




    262144.0




```python
src.shape[0] * src.shape[1]
```




    262144



**컬러 영상 (채널별 히스토그램)**


```python
src = cv2.imread('./data/lenna.bmp')
colors = ['b', 'g', 'r']

for i in range(3):
    hist = cv2.calcHist(images=[src], channels=[i], mask=None, histSize=[256], ranges=[0, 256])
    plt.plot(hist, color=colors[i])
plt.show()    
```


    
![png](/assets/images/2023-03-20-Computer Vision 2 (이벤트 처리와 영상분석)/output_43_0.png)
    


### 히스토그램 스트레칭


```python
Image('./images/image3.png', width=300)
```




    
![png](/assets/images/2023-03-20-Computer Vision 2 (이벤트 처리와 영상분석)/output_45_0.png)
    




```python
src = cv2.imread('./data/hawkes.bmp', cv2.IMREAD_GRAYSCALE)

dst = (src - src.min())/(src.max()-src.min()) * 255
dst = dst.astype(np.uint8)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```

**히스토그램으로 src, dst의 명암비 확인**


```python
src_hist = cv2.calcHist(images=[src], channels=[0], mask=None, histSize=[256], ranges=[0, 255])
dst_hist = cv2.calcHist(images=[dst], channels=[0], mask=None, histSize=[256], ranges=[0, 255])
```


```python
# 히스토그램 스트레칭 전
plt.bar(np.arange(256), src_hist.flatten(), color='b')
plt.show()
# 히스토그램 스트레칭 후
plt.bar(np.arange(256),  dst_hist.flatten(), color='b')
plt.show()
```


    
![png](/assets/images/2023-03-20-Computer Vision 2 (이벤트 처리와 영상분석)/output_49_0.png)
    



    
![png](/assets/images/2023-03-20-Computer Vision 2 (이벤트 처리와 영상분석)/output_49_1.png)
    


### 히스토그램 평활화 (Grayscale Image)


```python
src = cv2.imread('./data/hawkes.bmp', cv2.IMREAD_GRAYSCALE)

dst = cv2.equalizeHist(src)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```


```python
src_hist = cv2.calcHist(images=[src], channels=[0], mask=None, histSize=[256], ranges=[0, 255])
dst_hist = cv2.calcHist(images=[dst], channels=[0], mask=None, histSize=[256], ranges=[0, 255])
```


```python
# 히스토그램 평활화 전
plt.bar(np.arange(256), src_hist.flatten(), color='b')
plt.show()
# 히스토그램 평활화 후
plt.bar(np.arange(256),  dst_hist.flatten(), color='b')
plt.show()
```


    
![png](/assets/images/2023-03-20-Computer Vision 2 (이벤트 처리와 영상분석)/output_53_0.png)
    



    
![png](/assets/images/2023-03-20-Computer Vision 2 (이벤트 처리와 영상분석)/output_53_1.png)
    


### 히스토그램 평활화 (Color Image)

**컬러 이미지 히스토그램 평활화의 잘못된 예**
- 각 채널별(R, G, B)로 평활화를 수행한 후 합치게 되면
- 입력 영상만의 고유한 색상 조합이 깨짐


```python
src = cv2.imread('./data/pepper.bmp')
b = src[:, :, 0]
g = src[:, :, 1]
r = src[:, :, 2]
# b, g, r = cv2.split(src)


dst_b = cv2.equalizeHist(b)
dst_g = cv2.equalizeHist(g)
dst_r = cv2.equalizeHist(r)

dst = src.copy()
dst[:, :, 0] = dst_b
dst[:, :, 1] = dst_g
dst[:, :, 2] = dst_r
# dst = cv2.merge([dst_b, dst_g, dst_r])


cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```

**컬러 이미지 히스토그램 평활화의 잘된 예**
- 명암비를 조절한다는것은 '밝기'값만 관계가 있으므로
- 컬러공간(Color Space)을 BGR에서 yCrCb(y:밝기값, Cr/Cb:색상)로 바꾼 뒤에
- y(밝기) 채널에 대해서만 히스토그램 평활화를 수행
- 나머지 Cr/Cb와 합치면 색상 정보는 그대로 유지


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

## 6. 영상의 산술 연산


```python
src1 = cv2.imread('./data/lenna256.bmp', cv2.IMREAD_GRAYSCALE)
src2 = cv2.imread('./data/square.bmp', cv2.IMREAD_GRAYSCALE)

dst1 = cv2.add(src1, src2)
dst2 = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)
dst3 = cv2.subtract(src1, src2)
dst4 = cv2.absdiff(src1, src2)

cv2.imshow('src1', src1)
cv2.imshow('src2', src2) 
cv2.imshow('dst1', dst1) 
cv2.imshow('dst2', dst2) 
cv2.imshow('dst3', dst3) 
cv2.imshow('dst4', dst4) 

cv2.waitKey()
cv2.destroyAllWindows()
```

## 7. 영상의 논리연산


```python
src1 = cv2.imread('./data/lenna256.bmp', cv2.IMREAD_GRAYSCALE)
src2 = cv2.imread('./data/square.bmp', cv2.IMREAD_GRAYSCALE)

dst1 = cv2.bitwise_and(src1, src2)
dst2 = cv2.bitwise_or(src1, src2)
dst3 = cv2.bitwise_xor(src1, src2)
dst4 = cv2.bitwise_not(src1)

cv2.imshow('src1', src1)
cv2.imshow('src2', src2) 
cv2.imshow('dst1', dst1) 
cv2.imshow('dst2', dst2) 
cv2.imshow('dst3', dst3) 
cv2.imshow('dst4', dst4) 

cv2.waitKey()
cv2.destroyAllWindows()
```

**참고** 영상의 이진화


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

### 실습 : 빈 들판에 비행기 띄위기


```python
airplane = cv2.imread('./data/airplane.bmp', cv2.IMREAD_COLOR)
field = cv2.imread('./data/field.bmp', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(airplane, cv2.COLOR_BGR2GRAY)

ret, mask = cv2.threshold(gray, 163, 255, cv2.THRESH_BINARY)

mask = cv2.bitwise_not(mask)

field[mask>0] = airplane[mask>0]

cv2.imshow('airplane', airplane) 
cv2.imshow('field', field) 
cv2.imshow('mask', mask) 
cv2.waitKey()
cv2.destroyAllWindows()
```
