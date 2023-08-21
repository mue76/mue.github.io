---
tag: [computer vision, opencv]
toc: true
toc_sticky: true
toc_label: 목차
author_profile: false

---

# 모폴로지 연산과 객체 검출

## 1. 모폴로지 연산

### 침식과 팽창


```python
import cv2
import numpy as np
```


```python
src = cv2.imread('./data/milkdrop.bmp', cv2.IMREAD_GRAYSCALE)
thresh, src_bin = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print(thresh)

erode = cv2.erode(src_bin, None)
dilate = cv2.dilate(src_bin, None)

cv2.imshow('src', src)
cv2.imshow('src_bin', src_bin)
cv2.imshow('erode', erode)
cv2.imshow('dilate', dilate)

cv2.waitKey()
cv2.destroyAllWindows()
```

    154.0
    

### 열기와 닫기


```python
src = cv2.imread('./data/milkdrop.bmp', cv2.IMREAD_GRAYSCALE)
thresh, src_bin = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print(thresh)

# cv2.morphologyEx(원본영상, 연산방법, 구조물, 반복횟수)

# 열기 : 침식 -> 팽창
opening = cv2.morphologyEx(src_bin, cv2.MORPH_OPEN, None, iterations=1)

# 닫기 : 팽창 -> 침식
closing = cv2.morphologyEx(src_bin, cv2.MORPH_CLOSE, None, iterations=1)

cv2.imshow('src', src)
cv2.imshow('src_bin', src_bin)
cv2.imshow('opening', opening)
cv2.imshow('closing', closing)

cv2.waitKey()
cv2.destroyAllWindows()
```

    154.0
    

## 2. 레이블링과 외곽선 검출

### 레이블링 기본


```python
src = np.array([[0, 0, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 1, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1, 1, 1, 0],
                [0, 0, 0, 1, 0, 0, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]]).astype(np.uint8)

src = src * 255

cnt, labels = cv2.connectedComponents(src)
```


```python
cnt # 4개의 그룹(3개 객체 + 1개 배경)
```




    4




```python
labels
```




    array([[0, 0, 1, 1, 0, 0, 0, 0],
           [1, 1, 1, 1, 0, 0, 2, 0],
           [1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 3, 3, 0],
           [0, 0, 0, 3, 3, 3, 3, 0],
           [0, 0, 0, 3, 0, 0, 3, 0],
           [0, 0, 3, 3, 3, 3, 3, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)




```python
src = cv2.imread('./data/circles.jpg', cv2.IMREAD_GRAYSCALE)
thresh, src_bin = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# todo 객체별로 색깔 다르게 해서 보여주기
cnt, labels = cv2.connectedComponents(src_bin)

dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR) # 객체별로 색깔을 다르게 표시할 3차원 도화지 준비

dst[labels==0] = (0, 255, 255) # 배경을 노란색을 설정
dst[labels==1] = (0, 0, 255) # 1번객체를  빨간색으로 설정
dst[labels==2] = (255, 0, 255) # 2번객체를 보라색으로 설정
dst[labels==3] = (0, 255, 0) #3번 객체를 초록색으로 설정

cv2.imshow('src', src)
cv2.imshow('src_bin', src_bin)
cv2.imshow('dst', dst)

cv2.waitKey()
cv2.destroyAllWindows()
```

### 레이블링 응용


```python
src = cv2.imread('./data/circles.jpg', cv2.IMREAD_GRAYSCALE)
thresh, src_bin = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)


cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(src_bin)

dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR) # 객체별로 색깔을 다르게 표시할 3차원 도화지 준비

for i in range(1, cnt): # 객체 1, 2, 3 (배경 제외)
    b = np.random.randint(0, 255) 
    g = np.random.randint(0, 255) 
    r = np.random.randint(0, 255) 
    
    dst[labels==i] = (b, g, r)
    
    # boungding box (from stats) --> rectangle 그리기
    x, y, width, height, area = stats[i]
    cv2.rectangle(dst, (x, y), (x+width, y+height), (0, 0, 255))
    
    # center (from centroids) --> circle 그리기
    cx, cy = centroids[i]
    cv2.circle(dst, (int(cx), int(cy)), 5, (255, 0, 0), -1)

cv2.imshow('src', src)
cv2.imshow('src_bin', src_bin)
cv2.imshow('dst', dst)

cv2.waitKey()
cv2.destroyAllWindows()
```


```python
src = cv2.imread('./data/keyboard.bmp')
dst = src.copy()

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
thresh, src_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(src_bin)

for i in range(1, cnt): # 객체 1~37 (배경 제외)   
    # boungding box (from stats) --> rectangle 그리기
    x, y, width, height, area = stats[i]
    if area > 20:
        cv2.rectangle(dst, (x, y), (x+width, y+height), (0, 255, 255))

cv2.imshow('src', src)
cv2.imshow('src_bin', src_bin)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```

### 외곽선 검출과 그리기


```python
src = cv2.imread('./data/contours.bmp')
dst = src.copy()

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# thresh, src_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
for i in range(len(contours)): # 9 iterations
    b = np.random.randint(0, 255) 
    g = np.random.randint(0, 255) 
    r = np.random.randint(0, 255)
    cv2.drawContours(dst, contours, i, (b, g, r), 2)

cv2.imshow('src', src)
cv2.imshow('gray', gray)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```

### 실습 thumbs_up_down.jpg의 손 외곽선 그리기


```python
src = cv2.imread('./data/thumbs_up_down.jpg')
dst = src.copy()

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
thresh, gray_bin = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
print(thresh)

contours, hierarchy = cv2.findContours(gray_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(dst, contours, -1, (0, 0, 255), 3, cv2.LINE_AA)

cv2.imshow('src', src)
cv2.imshow('gray', gray)
cv2.imshow('gray_bin', gray_bin)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```

    220.0
    


```python
# 오른손 왼손을 둘러싸는 Bounding Box 그리기

right_hand = contours[0] # 오른손을 둘러싸는 점의 좌표들
left_hand = contours[1] # 왼손을 둘러싸는 점의 좌표들
x1, y1, width1, height1 = cv2.boundingRect(right_hand)
x2, y2, width2, height2 = cv2.boundingRect(left_hand)

dst = src.copy()
cv2.rectangle(dst, (x1, y1), (x1+width1, y1+height1), (0, 255, 255), 3)
cv2.rectangle(dst, (x2, y2), (x2+width2, y2+height2), (0, 0, 255), 3)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```

## 3.객체 검출

### 캐스케이드 분류기와 얼굴 검출
- https://docs.opencv.org/4.6.0/db/d28/tutorial_cascade_classifier.html
- https://towardsdatascience.com/viola-jones-algorithm-and-haar-cascade-classifier-ee3bfb19f7d8

**얼굴 검출**


```python
image = cv2.imread('./data/lena.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray)

for face in faces:
    x, y, w, h = face
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
cv2.imshow('image', image)    
cv2.waitKey()
cv2.destroyAllWindows()
```

**눈 검출**


```python
image = cv2.imread('./data/kids2.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./data/haarcascade_eye.xml')

faces = face_cascade.detectMultiScale(gray)

for face in faces:
    x, y, w, h = face
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    face_rect = image[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(face_rect)
    for eye in eyes:
        x2, y2, w2, h2 = eye
        cv2.rectangle(face_rect, (x2, y2), (x2+w2, y2+h2), (255, 0, 0), 2)    
    
cv2.imshow('image', image)    
cv2.waitKey()
cv2.destroyAllWindows()
```


```python
# detecMultiScale() 의 중요 파라미터들
# scaleFactor : 검색 윈도우 확대 비율(default = 1.1)
# minNeighbors : 검출 영역으로 선택하기 위한 최소 검출 횟수 (default=3)
# minSize : 검출할 객체의 최소 크기 (예, (100, 100))
# maxSize : 검출할 객체의 최대 크기

# scaleFactor가 detect시 미치는 영향
# https://answers.opencv.org/question/10654/how-does-the-parameter-scalefactor-in-detectmultiscale-affect-face-detection/
```

### 실습 : multi face 검출


```python
image = cv2.imread('./data/multi_faces.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=8)
#faces = face_cascade.detectMultiScale(gray, minSize=(90, 90), maxSize=(100, 100))
#faces = face_cascade.detectMultiScale(gray, minNeighbors=15)

for face in faces:
    x, y, w, h = face
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imshow('image', image)    
cv2.waitKey()
cv2.destroyAllWindows()
```

### 실습 : Camera Device로부터 얻은 frame에 대해 얼굴(눈) 검출하기


```python
import sys

face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./data/haarcascade_eye.xml')

cap = cv2.VideoCapture(0) # 0: 메인 카메라

if not cap.isOpened():
    print("camera open failed")
    sys.exit()
    
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)    
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)    
fps = cap.get(cv2.CAP_PROP_FPS) # frame per second
delay = round(1000/30)

print(width, height, fps, delay)

while True:    
    ret, frame = cap.read() # frame : 이미지 한장 (shape : height x width x channel)        
    
    if not ret:
        print("frame read error")
        break
        
    # detect face and eyes
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, minNeighbors=10)
    
    for face in faces:
        x1, y1, w1, h1 = face
        cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (0, 0, 255), 2)
        face_rect = frame[y1:y1+h1, x1:x1+w1]
        eyes = eye_cascade.detectMultiScale(face_rect, minNeighbors=10)
        for eye in eyes:
            x2, y2, w2, h2 = eye
            center = int(x2+w2/2), int(y2+h2/2)
            cv2.circle(face_rect, center, int(w2/2), (255, 0, 0), 2)      
            
    cv2.imshow('camera', frame) # 재생    
       
    key = cv2.waitKey(delay) # delay(ms) 기다리기(sleep 효과)
    if key == 27: # 27 : Esc key,  종료조건    
        break

if cap.isOpened(): # True
    print('cap release!!')
    cap.release()
    
cv2.destroyAllWindows()
```

    640.0 480.0 30.0 33
    cap release!!
    

### 실습 : Blurring을 이용하여 얼굴만 가리기


```python
import sys

face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0) # 0: 메인 카메라

if not cap.isOpened():
    print("camera open failed")
    sys.exit()
    
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)    
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)    
fps = cap.get(cv2.CAP_PROP_FPS) # frame per second
delay = round(1000/30)

print(width, height, fps, delay)

while True:    
    ret, frame = cap.read() # frame : 이미지 한장 (shape : height x width x channel)        
    
    if not ret:
        print("frame read error")
        break
        
    # detect face and eyes
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, minNeighbors=10)
    
    for face in faces:
        x1, y1, w1, h1 = face
        cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (0, 0, 255), 2)
        face_rect = frame[y1:y1+h1, x1:x1+w1]
        kernel_size = 50
        frame[y1:y1+h1, x1:x1+w1] = cv2.blur(face_rect, (kernel_size, kernel_size))          
            
    cv2.imshow('camera', frame) # 재생    
       
    key = cv2.waitKey(delay) # delay(ms) 기다리기(sleep 효과)
    if key == 27: # 27 : Esc key,  종료조건    
        break

if cap.isOpened(): # True
    print('cap release!!')
    cap.release()
    
cv2.destroyAllWindows()
```

    640.0 480.0 30.0 33
    cap release!!
    

**참고** numpy stack
- https://numpy.org/devdocs/reference/generated/numpy.dstack.html

**np.dstack()**


```python
a = np.array([[1],[2],[3]])
b = np.array([[2],[3],[4]])
c = np.array([[3],[4],[5]])
print(a.shape, b.shape, c.shape)
d = np.dstack((a,b,c))
print(d.shape)
```

    (3, 1) (3, 1) (3, 1)
    (3, 1, 3)
    

**np.hstack()**


```python
a = np.array([[1, 1],[2, 2],[3, 3]])
b = np.array([[4, 4],[5, 5],[6, 5]])
print(a.shape, b.shape)
c = np.hstack((a,b))
print(c.shape)
```

    (3, 2) (3, 2)
    (3, 4)
    

**np.vstack()**


```python
a = np.array([[1], [2], [3]])
b = np.array([[4], [5], [6]])
print(a.shape, b.shape)
c = np.vstack((a,b))
print(c.shape)
```

    (3, 1) (3, 1)
    (6, 1)
    
## Reference
- [OpenCV 4로 배우는 컴퓨터 비전과 머신 러닝 (황선규 저)](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=187822936)