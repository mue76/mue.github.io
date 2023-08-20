---
tag: [computer vision, opencv]
toc: true
toc_sticky: true
toc_label: 목차
author_profile: false

---

# 차선 인식


```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
```

## 1. 차선 인식 (White lane)


```python
image = mpimg.imread('./data/test.jpg') # R->G-B
plt.imshow(image)
```




    <matplotlib.image.AxesImage at 0x1c94b3baf10>




    
![png](/assets/images/2023-03-24-Computer Vision 5 (차선 인식)/output_3_1.png)
    


**Color Selection**

- Color Picker Tool로 해당 이미지 색상의 RGB 코드값 얻어오기
- https://annystudio.com/software/colorpicker/#download


```python
color_select = np.copy(image)

# todo
red_threshold = 230
green_threshold = 230
blue_threshold = 230

color_threshold = ((image[:, :, 0] < red_threshold) |
                   (image[:, :, 1] < green_threshold) |
                   (image[:, :, 2] < blue_threshold))

# 차선 색깔에 해당하지 않는 영역을 검은색으로 바꾸기
color_select[color_threshold] = [0, 0, 0]
plt.imshow(color_select)
```




    <matplotlib.image.AxesImage at 0x1c94be8ca60>




    
![png](/assets/images/2023-03-24-Computer Vision 5 (차선 인식)/output_6_1.png)
    


**Region Selection**


```python
image.shape
```




    (540, 960, 3)




```python
region_select = np.copy(image)

# todo
left_bottom = [0, 539]
right_bottom = [900, 539]
apex = [475, 320]

pts =np.array([left_bottom, right_bottom, apex])
cv2.fillPoly(region_select, [pts], color=[0, 0, 255]) # 삼각형안을 파란색으로 채우기
plt.imshow(region_select)
```




    <matplotlib.image.AxesImage at 0x1c94bef7580>




    
![png](/assets/images/2023-03-24-Computer Vision 5 (차선 인식)/output_9_1.png)
    


**Color and Regions Selection**


```python
# 1. Color Selection
color_select = np.copy(image)

# todo
red_threshold = 230
green_threshold = 230
blue_threshold = 230

color_threshold = ((image[:, :, 0] < red_threshold) |
                   (image[:, :, 1] < green_threshold) |
                   (image[:, :, 2] < blue_threshold))

# 차선 색깔에 해당하지 않는 영역을 검은색으로 바꾸기
color_select[color_threshold] = [0, 0, 0]

# 2. Region Selection
region_select = np.copy(image)

# todo
left_bottom = [0, 539]
right_bottom = [900, 539]
apex = [475, 320]

pts =np.array([left_bottom, right_bottom, apex])
cv2.fillPoly(region_select, [pts], color=[0, 0, 255]) # 삼각형안을 파란색으로 채우기

region_threshold = ((region_select[:, :, 0] == 0) & # R channel
                   (region_select[:, :, 1] == 0) & # G Channel
                   (region_select[:, :, 2] == 255)) # B Channel


# 3. Color Selection + Region Selection
# color_threshold : 차선(흰색)이 아닌 부분 True로 설정
# region_threshold : 관심영역(Region of interes, roi)에만 True 설정

lane_select = np.copy(image)
lane_select[~color_threshold & region_threshold] = [255, 0, 0]

# 관심영역만 점선으로 표시(추가)
x = left_bottom[0], right_bottom[0], apex[0], left_bottom[0]
y = left_bottom[1], right_bottom[1], apex[1], left_bottom[1]

plt.plot(x, y, 'b--', 4)
plt.imshow(lane_select)
```




    <matplotlib.image.AxesImage at 0x1c94c992f10>




    
![png](/assets/images/2023-03-24-Computer Vision 5 (차선 인식)/output_11_1.png)
    


## 2. 차선 인식 (White and Yellow Lane)


```python
image = mpimg.imread('./data/exit-ramp.jpg') # R->G->B
plt.imshow(image)
```




    <matplotlib.image.AxesImage at 0x7fd3a6621d60>




    
![png](/assets/images/2023-03-24-Computer Vision 5 (차선 인식)/output_13_1.png)
    


### Step 1: Gray Scale 변환


```python
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
print(gray.shape)
plt.imshow(gray, cmap="gray")
```

    (540, 960)
    




    <matplotlib.image.AxesImage at 0x7fd3a2dafdf0>




    
![png](/assets/images/2023-03-24-Computer Vision 5 (차선 인식)/output_15_2.png)
    


### Step 2 : Gaussian Blurring 변환 (optional)


```python
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
plt.imshow(blur_gray, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7fd3948f1d90>




    
![png](/assets/images/2023-03-24-Computer Vision 5 (차선 인식)/output_17_1.png)
    


### Step 3 : Edge Detect 


```python
low_threshold = 50
high_threshold = 150 # low:high 의 비율 1:2 또는 1:3

edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
plt.imshow(edges, cmap="gray")
```




    <matplotlib.image.AxesImage at 0x7fd394907970>




    
![png](/assets/images/2023-03-24-Computer Vision 5 (차선 인식)/output_19_1.png)
    


### Step 4 : ROI(Region of interes) Select


```python
height = edges.shape[0]
width = edges.shape[1]

pts = np.array([[0, height-1], [450, 290], [490, 290], [width-1, height-1]])
mask = np.zeros(edges.shape, edges.dtype)
cv2.fillPoly(mask, [pts], 255)
plt.imshow(mask, cmap="gray")
```




    <matplotlib.image.AxesImage at 0x7fd39423ef40>




    
![png](/assets/images/2023-03-24-Computer Vision 5 (차선 인식)/output_21_1.png)
    



```python
masked_edges = cv2.bitwise_and(edges, mask)
plt.imshow(masked_edges, cmap="gray")
```




    <matplotlib.image.AxesImage at 0x7fd3944d7790>




    
![png](/assets/images/2023-03-24-Computer Vision 5 (차선 인식)/output_22_1.png)
    


### Step 5 : Line Detect (with hough transform)


```python
rho = 1 # 1 or 2 : 숫자가 작을수록 정밀하게 검출되지만 연산시간이 더 걸림
theta = np.pi/180 # 라디안 단위
threshold = 30 # 축적배열에 교차하는 점의 숫자가 높다는것은 직선을 이루는 점들이 많다는 뜻
                # 얼마나 큰 값을 직선으로 판단할지는 threshold에 달려있음
minLineLength = 40 # 검출할 선분의 최소 길이
maxLineGap = 20 # 직선으로 간주할 최대 에지 점 간격

lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold,
                minLineLength = minLineLength,
                maxLineGap = maxLineGap)

# 참고 1차원 데이터를 3차원으로 확장하는 방법
# dst = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR) # 직선을 그릴 도화지(3 채널 도화지)
# dst = cv2.merge([masked_edges, masked_edges, masked_edges])
dst = np.dstack([masked_edges, masked_edges, masked_edges])
print(dst.shape)
if lines is not None:
    for line in lines:
        line = line[0]
        pt1 = line[0], line[1]
        pt2 = line[2], line[3]
        cv2.line(dst, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA) 
plt.imshow(dst)        
```

    (540, 960, 3)
    




    <matplotlib.image.AxesImage at 0x7fd393ec3bb0>




    
![png](/assets/images/2023-03-24-Computer Vision 5 (차선 인식)/output_24_2.png)
    


### Pipeline (Step 1 ~ Step 5)


```python
### Step 1 : Grayscale 변환
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

### Step 2 : Gaussian Blurring 변환
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

### Step 3 : Edge Detect
low_threshold = 50
high_threshold = 150 # low:high 의 비율 1:2 또는 1:3
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

### Step 4 : ROI Select
height = edges.shape[0]
width = edges.shape[1]

pts = np.array([[0, height-1], [450, 290], [490, 290], [width-1, height-1]])
mask = np.zeros(edges.shape, edges.dtype)
cv2.fillPoly(mask, [pts], 255)

masked_edges = cv2.bitwise_and(edges, mask)

### Step 5 : Line Detect
rho = 1 # 1 or 2 : 숫자가 작을수록 정밀하게 검출되지만 연산시간이 더 걸림
theta = np.pi/180 # 라디안 단위
threshold = 30 # 축적배열에 교차하는 점의 숫자가 높다는것은 직선을 이루는 점들이 많다는 뜻
                # 얼마나 큰 값을 직선으로 판단할지는 threshold에 달려있음
minLineLength = 40 # 검출할 선분의 최소 길이
maxLineGap = 20 # 직선으로 간주할 최대 에지 점 간격

lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold,
                minLineLength = minLineLength,
                maxLineGap = maxLineGap)

# 참고 1차원 데이터를 3차원으로 확장하는 방법
# dst = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR) # 직선을 그릴 도화지(3 채널 도화지)
# dst = cv2.merge([masked_edges, masked_edges, masked_edges])
dst = np.dstack([masked_edges, masked_edges, masked_edges])

if lines is not None:
    for line in lines:
        line = line[0]
        pt1 = line[0], line[1]
        pt2 = line[2], line[3]
        cv2.line(dst, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA) 
rho = 1 # 1 or 2 : 숫자가 작을수록 정밀하게 검출되지만 연산시간이 더 걸림
theta = np.pi/180 # 라디안 단위
threshold = 30 # 축적배열에 교차하는 점의 숫자가 높다는것은 직선을 이루는 점들이 많다는 뜻
                # 얼마나 큰 값을 직선으로 판단할지는 threshold에 달려있음
minLineLength = 40 # 검출할 선분의 최소 길이
maxLineGap = 20 # 직선으로 간주할 최대 에지 점 간격

lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold,
                minLineLength = minLineLength,
                maxLineGap = maxLineGap)

# 참고 1차원 데이터를 3차원으로 확장하는 방법
# dst = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR) # 직선을 그릴 도화지(3 채널 도화지)
# dst = cv2.merge([masked_edges, masked_edges, masked_edges])
dst = np.dstack([masked_edges, masked_edges, masked_edges])

if lines is not None:
    for line in lines:
        line = line[0]
        pt1 = line[0], line[1]
        pt2 = line[2], line[3]
        cv2.line(dst, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA) 

# 원본 영상 : image, 차선인식 결과 : dst

result = cv2.addWeighted(image, 1.0, dst, 0.8, 0)
plt.imshow(result)
```




    <matplotlib.image.AxesImage at 0x7fd393daf310>




    
![png](/assets/images/2023-03-24-Computer Vision 5 (차선 인식)/output_26_1.png)
    

