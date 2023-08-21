---
tag: [computer vision, opencv]
toc: true
toc_sticky: true
toc_label: 목차
author_profile: false

---

# HOG 알고리즘과 보행자 검출

**HOG Descriptor**
- https://docs.opencv.org/4.6.0/d5/d33/structcv_1_1HOGDescriptor.html#a723b95b709cfd3f95cf9e616de988fc8


```python
import cv2
import matplotlib.pyplot as plt
```

**HOG Feature 계산하기 (default)**


```python
src = cv2.imread('./data/people1.png')

hog = cv2.HOGDescriptor()
hog_feature = hog.compute(src)

hog_feature.shape # (bin의 개수(9) * 셀의 개수(4) * 가로블록(7) * 세로블록(15))
```




    (3780,)



**HOG Feature 계산하기 (파라미터 설정)**


```python
src.shape
win_size = (src.shape[1], src.shape[0])
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
nbins = 9

# cv.HOGDescriptor(원본소스의사이즈, 블럭사이즈, 블럭스트라이드, 셀사이즈, 빈의개수)
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
hog_feature = hog.compute(src)
hog_feature.shape
```




    (3780,)




```python
src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
plt.imshow(src_rgb)
```




    <matplotlib.image.AxesImage at 0x250d1e675b0>




    
![png](/assets/images/2023-03-28-Computer Vision 8 (HOG 알고리즘과 보행자 검출)/output_7_1.png)
    



```python
from skimage.feature import hog # hog feature 생성

src = cv2.imread('./data/people1.png')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# hog(원본이미지, 셀사이즈, 블럭당셀, 블럭정규화, 시각화, 벡터형태표시여부)
hog_feature, hog_image = hog(gray, pixels_per_cell = (8, 8), cells_per_block = (2, 2), block_norm='L2-Hys',
                             visualize=True, feature_vector=True)

print(hog_feature.shape)

fig = plt.figure()
plt.subplot(121)
plt.imshow(src)

plt.subplot(122)
plt.imshow(hog_image, cmap='gray')
```

    (3780,)
    




    <matplotlib.image.AxesImage at 0x250d316f070>




    
![png](/assets/images/2023-03-28-Computer Vision 8 (HOG 알고리즘과 보행자 검출)/output_8_2.png)
    



```python
hog_image.shape
```




    (128, 64)


## Reference
- [OpenCV 4로 배우는 컴퓨터 비전과 머신 러닝 (황선규 저)](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=187822936)