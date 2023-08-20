---
tag: [computer vision, opencv]
toc: true
toc_sticky: true
toc_label: 목차
author_profile: false

---

# OpenCV 기초 사용법

**opencv-python tutorial**
* https://docs.opencv.org/4.7.0/d6/d00/tutorial_py_root.html


```python
import cv2
```


```python
cv2.__version__
```




    '4.7.0'



## Hello World


```python
img = cv2.imread('./data/lenna.bmp')
print(type(img), img.shape)

cv2.namedWindow("hello world") # 생략 가능
cv2.imshow("hello world", img)

cv2.waitKey()
cv2.destroyAllWindows()
```

    <class 'numpy.ndarray'> (512, 512, 3)
    


```python
img.min(), img.max() # 이미지의 최소광도값, 최대광도값
```




    (3, 255)




```python
img[0][0] # 이미지의 좌상단 R, G, B
```




    array([125, 137, 226], dtype=uint8)



## 영상 파일 읽고 화면에 표시하기


```python
# Color Mode로 이미지 읽기
imgColor = cv2.imread('./data/lenna.bmp', cv2.IMREAD_COLOR)

# Grayscale Mode로 이미지 읽기
imgGray = cv2.imread('./data/lenna.bmp', cv2.IMREAD_GRAYSCALE)

# type, shape 출력
print(type(imgColor), imgColor.shape)
print(type(imgGray), imgGray.shape)

# 윈도우로 보여주기
cv2.imshow('img color', imgColor)
cv2.imshow('img gray', imgGray)

# 창닫기
cv2.waitKey()
cv2.destroyAllWindows()
```

    <class 'numpy.ndarray'> (512, 512, 3)
    <class 'numpy.ndarray'> (512, 512)
    

## 영상 파일 읽고 저장하기


```python
imgColor = cv2.imread('./data/lena.jpg', cv2.IMREAD_COLOR)
print(type(imgColor), imgColor.shape)

cv2.imwrite('./out/lena.bmp', imgColor)
cv2.imwrite('./out/lena.png', imgColor)

cv2.imwrite('./out/lena2.png', imgColor, [cv2.IMWRITE_PNG_COMPRESSION, 9]) # 0 ~ 9
cv2.imwrite('./out/lena2.jpg', imgColor, [cv2.IMWRITE_JPEG_QUALITY, 90]) # 0 ~ 100

cv2.imshow('img color', imgColor)
cv2.waitKey()
cv2.destroyAllWindows()
```

    <class 'numpy.ndarray'> (512, 512, 3)
    




    True



## matplotlib.pyplot으로 컬러 영상 표시하기


```python
import matplotlib.pyplot as plt
```


```python
# matplotlib.pyplot으로 이미지 읽고 보여주기
img_p = plt.imread('./data/lena.jpg')
plt.imshow(img_p)
```




    <matplotlib.image.AxesImage at 0x27b2e5b3e80>




    
![png](/assets/images/2023-03-17-Computer Vision 1 (OpenCV 기초 사용법)/output_14_1.png)
    



```python
# opencv로 읽고 matplotlib.pyplot으로 보여주기
img_c = cv2.imread('./data/lena.jpg')
plt.imshow(img_c)
```




    <matplotlib.image.AxesImage at 0x27b2f6374c0>




    
![png](/assets/images/2023-03-17-Computer Vision 1 (OpenCV 기초 사용법)/output_15_1.png)
    



```python
# matplotlib.pyplot은 R->G->B vs cv2가 이미지를 읽는 순서는 B->G->R
img_p[0][0], img_c[0][0] 
```




    (array([225, 138, 128], dtype=uint8), array([128, 138, 225], dtype=uint8))




```python
# opencv로 읽고 보여주기 (cv2로 읽은것을 cv2로 보여주는데는 문제가 없음)
img_c = cv2.imread('./data/lena.jpg')
cv2.imshow("opencv img", img_c)
cv2.waitKey()
cv2.destroyAllWindows()
```

**(1) numpy ndarray의 색인 문법으로 컬러채널 순서를 변경**


```python
img_bgr = cv2.imread('./data/lena.jpg') # img_bgr array에는 B->G->R 순서로 데이터가 준비

# option 1
img_rgb = img_bgr.copy()
img_rgb[:, :, 0] = img_bgr[:, :, 2]
img_rgb[:, :, 2] = img_bgr[:, :, 0]

# option 2
# img_rgb = img_bgr.copy()
# img_rgb[:, :, [0, 2]] = img_bgr[:, :, [2, 0]]

# option 3
# img_rgb = img_bgr.copy()
# img_rgb[:, :, :] = img_bgr[:, :, -1::-1]

plt.imshow(img_rgb) # R->G->B 순서의 데이터를 표시하도록 해줌
```




    <matplotlib.image.AxesImage at 0x27b2f7d26d0>




    
![png](/assets/images/2023-03-17-Computer Vision 1 (OpenCV 기초 사용법)/output_19_1.png)
    


**(2) cvtColor() 함수로 컬러채널 순서를 변경**


```python
img_bgr = cv2.imread('./data/lena.jpg') # img_bgr array에는 B->G->R 순서로 데이터가 준비

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
```



보여주기
    <matplotlib.image.AxesImage at 0x27b2f82f310>




    
![png](/assets/images/2023-03-17-Computer Vision 1 (OpenCV 기초 사용법)/output_21_1.png)
    


# 동영상 파일 다루기

```
객체 = 비디오객체 생성()

while True:
   배열 = 객체.read() # 배열 : 이미지 한장 (shape : height x width x channel)
   배열 보여주기 # 재생
   배열 저장하기 # 녹화
   
   키입력을 기다리다가 조건에 맞으면:
       break

창닫기
```

**비디오객체 생성**
```
객체 = cv2.VideoCapture(device) # 카메라 디바이스
객체 = cv2.VideoCapture(filepath) # 동영상 파일
객체 = cv2.VideoCapture(url) # 스트리밍 주소
```

## (1) 카메라 입력


```python
cap = cv2.VideoCapture(0) # 0: 메인 카메라

while True:    
    ret, frame = cap.read() # frame : 이미지 한장 (shape : height x width x channel)
    cv2.imshow('camera', frame) # 재생
       
    key = cv2.waitKey(10) # 10ms 기다리기(sleep 효과), 종료조건
    if key == 27: # 27 : Esc key,  종료조건    
        break

cv2.destroyAllWindows()
```


```python
import sys
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
    

## (2) 동영상 파일


```python
cap = cv2.VideoCapture('./data/stopwatch.avi') # 동영상 파일

if not cap.isOpened():
    print("camera open failed")
    sys.exit()
    
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)    
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)    
fps = cap.get(cv2.CAP_PROP_FPS) # frame per second
delay = round(1000/fps)

print(width, height, fps, delay)

while True:    
    ret, frame = cap.read() # frame : 이미지 한장 (shape : height x width x channel)    

    if not ret:
        print("frame read error")
        break
        
    cv2.imshow('avi file', frame) # 재생    
       
    key = cv2.waitKey(delay) # delay(ms) 기다리기(sleep 효과)
    if key == 27: # 27 : Esc key,  종료조건    
        break

if cap.isOpened(): # True
    print('cap release!!')
    cap.release()
    
cv2.destroyAllWindows()
```

    640.0 480.0 30.0 33
    frame read error
    cap release!!
    

## (3) 동영상 저장


```python
cap = cv2.VideoCapture('./data/stopwatch.avi') # 동영상 파일

if not cap.isOpened():
    print("camera open failed")
    sys.exit()
    
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) # frame per second
delay = round(1000/fps)

print(width, height, fps, delay)

fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X') # DivX Mpeg-4 코덱
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')

# 저장을 위한 객체 = cv2.VideoWriter(파일명, 코덱, FPS, 해상도)
outputVideo = cv2.VideoWriter('./out/output.avi', fourcc, fps, (width, height))

while True:    
    ret, frame = cap.read() # frame : 이미지 한장 (shape : height x width x channel)    

    if not ret:
        print("frame read error")
        break
        
    cv2.imshow('avi file', frame) # 재생   
    outputVideo.write(frame) # 녹화
       
    key = cv2.waitKey(delay) # delay(ms) 기다리기(sleep 효과)
    if key == 27: # 27 : Esc key,  종료조건    
        break

if cap.isOpened(): # True
    print('cap release!!')
    cap.release()

if outputVideo.isOpened():
    print('output release!!')
    outputVideo.release()
    
cv2.destroyAllWindows()
```

    640 480 30.0 33
    frame read error
    cap release!!
    output release!!
    

## (4) 드로이드캠 영상

안드로이드 스마트폰 앱 중 DroidCam을 이용하면 스마트폰 카메라에서 촬영한 영상을 소켓 통신을 통해 보내고 받을 수 있다.
- 사용 순서
1. 플레이스토어에서 DroidCam 설치
2. 스마트폰에서 DroidCam 앱을 실행하고 와이파이 IP, 포트 번호, 'mjpegfeed'를 사용해 VideoCapture 객체 cap을 생성(http://IP:port/mjpegfeed')
3. 와이파이 IP, 포트 번호는 스마트폰 및 와이파이 환경에 따라 다르고, 'mjpegfeed' 문자열은 앱에 따라 다를 수 있음.


```python
cap = cv2.VideoCapture('http://192.168.0.197:4747/mjpegfeed') # 드로이드캠 연결 IP
# cap = cv2.VideoCapture('http://192.168.0.197:4747/video') # (아이폰의 경우)

if not cap.isOpened():
    print("camera open failed")
    sys.exit()
    
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) # frame per second
delay = round(1000/fps)

print(width, height, fps, delay)

fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X') # DivX Mpeg-4 코덱
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')

# 저장을 위한 객체 = cv2.VideoWriter(파일명, 코덱, FPS, 해상도)
outputVideo = cv2.VideoWriter('./out/output.avi', fourcc, fps, (width, height))

while True:    
    ret, frame = cap.read() # frame : 이미지 한장 (shape : height x width x channel)    

    if not ret:
        print("frame read error")
        break
        
    cv2.imshow('avi file', frame) # 재생   
    # outputVideo.write(frame) # 녹화
       
    key = cv2.waitKey(delay) # delay(ms) 기다리기(sleep 효과)
    if key == 27: # 27 : Esc key,  종료조건    
        break

if cap.isOpened(): # True
    print('cap release!!')
    cap.release()

if outputVideo.isOpened():
    print('output release!!')
    outputVideo.release()
    
cv2.destroyAllWindows()
```

    640 480 25.0 33
    frame read error
    cap release!!
    output release!!
    

## (5) 유튜브 영상

- pafy : 비디오에서 메타데이터 획득, 비디오/오디오를 다운로드 하는 패키지
- youtube_dl : patfy의 backend에서 유튜브 동영상을 다운로드

**설치방법**
- pip install pafy
- pip install youtube_dl


```python
import pafy
import youtube_dl
```


```python
# pafy로 video 객체 생성시 오류 문제 임시 해결
# https://stackoverflow.com/questions/75495800/error-unable-to-extract-uploader-id-youtube-discord-py
# youtube.py 파일을 아래와 같이 수정
# (before)
#'uploader_id': self._search_regex(r'/(?:channel|user)/([^/?&#]+)', owner_profile_url, 'uploader id') if owner_profile_url else None
# (after)
#'uploader_id': self._search_regex(r'/(?:channel|user)/([^/?&#]+)', owner_profile_url, 'uploader id', fatal=False) if owner_profile_url else None


# backend_youtube_dl.py 파일도 수정
# 아래코드를 주석 처리
# self._likes = self._ydl_info['like_count']
# self._dislikes = self._ydl_info['dislike_count']

# Jupyter notebook 커널서부터 재가동하면 에러가 임시 해결
```


```python
url = 'https://www.youtube.com/watch?v=9SmQOZWNyWE&list=RD9SmQOZWNyWE&index=2'
video = pafy.new(url)

print('title :', video.title)
print('durationi :', video.duration)

best = video.getbest()
print('download url:', best.url)
print('resolution:', best.resolution)
```

    title : BTS - "Permission to Dance" performed at the United Nations General Assembly | SDGs | Official Video
    durationi : 00:03:43
    download url: https://rr7---sn-3u-bh2sy.googlevideo.com/videoplayback?expire=1679380183&ei=d_oYZPz6KJeUgAPLsqygCA&ip=222.112.208.69&id=o-AGz8U74dK6qbNDTIZgSfWkkSEstcVOPEE2zZnivauFC4&itag=18&source=youtube&requiressl=yes&mh=e9&mm=31%2C26&mn=sn-3u-bh2sy%2Csn-oguesndr&ms=au%2Conr&mv=m&mvi=7&pcm2cms=yes&pl=19&initcwndbps=796250&vprv=1&mime=video%2Fmp4&ns=yq2XQTE5tv0uWAVZw17m8zIL&cnr=14&ratebypass=yes&dur=223.445&lmt=1665478137806352&mt=1679357845&fvip=4&fexp=24007246&c=WEB&txp=5538434&n=ayPMcGWeZgSMXo&sparams=expire%2Cei%2Cip%2Cid%2Citag%2Csource%2Crequiressl%2Cvprv%2Cmime%2Cns%2Ccnr%2Cratebypass%2Cdur%2Clmt&sig=AOq0QJ8wRQIgdsd8Zn7EYghrwQQrYUVFgH-CBgvF5SaDbuObVQlmXg4CIQCxMY1unQ7XbUTK3zFXNFs3TGqkZV1uOYRR12EcB9Z0Qg%3D%3D&lsparams=mh%2Cmm%2Cmn%2Cms%2Cmv%2Cmvi%2Cpcm2cms%2Cpl%2Cinitcwndbps&lsig=AG3C_xAwRQIhAL9v5B6S7qjf-qRpB2jhLGWIZ81dRFfLwWlBYPIxSVnkAiAFkhC7A4VjOsiZ0OpNe0PZB9Q91L7aTe7sa3PyoB83iA%3D%3D
    resolution: 640x360
    


```python
cap = cv2.VideoCapture(best.url) # 유튜브에서 다운로드 가능한 URL


if not cap.isOpened():
    print("camera open failed")
    sys.exit()
    
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) # frame per second
delay = round(1000/fps)

print(width, height, fps, delay)

fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X') # DivX Mpeg-4 코덱
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')

# 저장을 위한 객체 = cv2.VideoWriter(파일명, 코덱, FPS, 해상도)
outputVideo = cv2.VideoWriter('./out/output.avi', fourcc, fps, (width, height))

while True:    
    ret, frame = cap.read() # frame : 이미지 한장 (shape : height x width x channel)    

    if not ret:
        print("frame read error")
        break
        
    cv2.imshow('original', frame) # 원본영상 재생   
    cv2.imshow('inverse', 255-frame) # 반전영상 재생
    edge = cv2.Canny(frame, 100, 200)
    cv2.imshow('edge', edge) # 에지 영상 재생
    
    # outputVideo.write(frame) # 녹화
       
    key = cv2.waitKey(delay) # delay(ms) 기다리기(sleep 효과)
    if key == 27: # 27 : Esc key,  종료조건    
        break

if cap.isOpened(): # True
    print('cap release!!')
    cap.release()

if outputVideo.isOpened():
    print('output release!!')
    outputVideo.release()
    
cv2.destroyAllWindows()
```

    640 360 29.97002997002997 33
    cap release!!
    output release!!
    

# 다양한 그리기 함수


```python
# openCV 함수에는 좌표를 찾아갈 때 x좌표(수평방향) -> y좌표(수직방향)
# 예) pt1 = (50, 100) # x, y

# numpy ndarray 색인할 때는 행(수직방향)을 색인 -> 열(수평방향)을 색인
# 예) img[100, 50]
```

## 직선 그리기


```python
import numpy as np
```


```python
img = np.full((400, 400, 3), 255, np.uint8)

pt1 = (50, 100) # x좌표, y좌표
pt2 = (150, 100) # x좌표, y좌표

pt3 = (200, 100)
pt4 = (300, 250)

# cv2.line(도화지, 시작점, 끝점, 색상, 굵기....)

# 수평선
cv2.line(img, pt1, pt2, (0, 0, 255), 2) 

# 대각선
cv2.line(img, pt3, pt4, (255, 0, 0), 2, cv2.LINE_AA)

cv2.imshow('line', img)
cv2.waitKey()
cv2.destroyAllWindows()
```

## 도형 그리기


```python
cv2.FILLED
```




    -1



**사각형 그리기**


```python
img = np.full((400, 400, 3), 255, np.uint8)

pt1 = (50, 50)
pt2 = (150, 100)

pt3 = (50, 150)
pt4 = (150, 200)

# cv2.rectangle(도화지, 시작점, 마주보는끝점, 색상, 굵기....)
cv2.rectangle(img, pt1, pt2, (0, 0, 255), 3)
cv2.rectangle(img, pt3, pt4, (255, 0, 0), cv2.FILLED)

cv2.imshow('rectangle', img)
cv2.waitKey()
cv2.destroyAllWindows()
```

**원 그리기**


```python
img = np.full((400, 400, 3), 255, np.uint8)

center1 = (150, 200)
center2 = (250, 200)

# cv2.circle(도화지, 중심점, 반지름, 색상, 굵기....)
cv2.circle(img, center1, 30, (0, 255, 255), 3)
cv2.circle(img, center2, 30, (255, 255, 0), cv2.FILLED)

cv2.imshow('circle', img)
cv2.waitKey()
cv2.destroyAllWindows()
```

**타원 그리기**


```python
img = np.full((400, 400, 3), 255, np.uint8)

center1 = (150, 200)

# cv2.ellipse(도화지, 중심점, 반지름쌍, 기울기, 시작각도, 끝각도, 색상, 굵기....)
cv2.ellipse(img, center1, (60, 30), 0, 0, 360, (255, 0, 0), 3)

cv2.imshow('ellipse', img)
cv2.waitKey()
cv2.destroyAllWindows()
```

**다각형 그리기**


```python
img = np.full((400, 400, 3), 255, np.uint8)

pts = np.array([[150, 150], [200, 150], [200, 200], [250, 250], [200, 300]])

#cv2.polylines(도화지, [다각형을 이룰 점들], 다각형을 닫을지 여부, 색상, 굵기....)
cv2.polylines(img, [pts], True, (255, 0, 255), 2)

cv2.imshow('polylines', img)
cv2.waitKey()
cv2.destroyAllWindows()
```

## 문자열 출력하기


```python
img = np.full((400, 400, 3), 255, np.uint8)

#cv2.putText(도화지, 텍스트, 텍스트의 좌하단 좌표, 폰트, 스케일, 색상, 굵기...)
cv2.putText(img, "Hello", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

cv2.imshow('text', img)
cv2.waitKey()
cv2.destroyAllWindows()
```

**text를 도화지 중심부에 위치시키기**


```python
img = np.full((200, 640, 3), 255, np.uint8)

text = 'Hello OpenCV'
fontFace = cv2.FONT_HERSHEY_DUPLEX
fontScale = 2
thickness = 1

size_Text, retVal = cv2.getTextSize(text, fontFace, fontScale, thickness)

print(size_Text) # width, height
print(img.shape)

org_x = (img.shape[1] - size_Text[0]) // 2
org_y = (img.shape[0] + size_Text[1]) // 2
print(org_x, org_y)

cv2.putText(img, text, (org_x, org_y), fontFace, fontScale, (255, 0, 0), thickness)
cv2.rectangle(img, (org_x, org_y), (org_x+size_Text[0], org_y-size_Text[1]), (0, 255, 0), 2)
cv2.circle(img, (org_x, org_y), 5, (0, 0, 255))
cv2.imshow('hello opencv', img)
cv2.waitKey()
cv2.destroyAllWindows()
```

    (429, 43)
    (200, 640, 3)
    105 121
    

## 실습 : 카운트 다운 영상 만들기


```python
img = np.full((512, 512, 3), 255, np.uint8)
cx, cy = img.shape[0]//2, img.shape[1]//2
fontFace = cv2.FONT_HERSHEY_TRIPLEX
fontScale = 5
thickness = 2

fourcc = cv2.VideoWriter_fourcc("D", "I", "V", "X")
frame_size = img.shape[0], img.shape[1]
fps = 1
coutdown_writer = cv2.VideoWriter("./out/countdown0.mp4", fourcc, fps, frame_size)

for count in range(5, 0, -1):
    text = str(count)
    sizeText, _ = cv2.getTextSize(text, fontFace, fontScale, thickness)
    org = (img.shape[1] - sizeText[0])//2, (img.shape[0] + sizeText[1])//2
    cv2.putText(img, text, org, fontFace, fontScale, (255, 0, 0), 3)
    cv2.circle(img, (cx, cy), int(np.max(sizeText) * count * 0.5), (255, 255, 0), 4)
    coutdown_writer.write(img)
    
    cv2.imshow("img", img)
    cv2.waitKey(1000)
    img = np.full((512, 512, 3), 255, np.uint8)

coutdown_writer.release()
cv2.destroyAllWindows()
```
