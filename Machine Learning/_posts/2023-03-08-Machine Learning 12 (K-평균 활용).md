---
tag: [python, machine learning, scikit-learn]
---

# 군집 (Clustering)

## K-평균 활용

### 이미지 분할


```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

image = plt.imread('./images/ladybug.png')
plt.imshow(image)
```




    <matplotlib.image.AxesImage at 0x1a67a7ce550>




    
![png](/assets/images/2023-03-08-Machine Learning 12 (K-평균 활용)/output_3_1.png)
    



```python
image.shape
```




    (533, 800, 3)




```python
X = image.reshape(-1, 3)
X.shape
```




    (426400, 3)




```python
kmeans = KMeans(n_clusters=8, random_state=42)
kmeans.fit(X)
```




    KMeans(random_state=42)




```python
kmeans.labels_ # 군집된 레이블블
```




    array([1, 1, 1, ..., 4, 1, 1])




```python
kmeans.labels_.shape
```




    (426400,)




```python
import numpy as np

np.unique(kmeans.labels_)
```




    array([0, 1, 2, 3, 4, 5, 6, 7])




```python
kmeans.cluster_centers_ # 8개 그룹의 센트로이드 : 각각의 그룹을 대표할 수 있는 RGB 조합 (색상정보보)
```




    array([[0.98363745, 0.9359338 , 0.02574807],
           [0.02289337, 0.11064845, 0.00578197],
           [0.21914783, 0.38675755, 0.05800817],
           [0.75775605, 0.21225454, 0.0445884 ],
           [0.09990625, 0.2542204 , 0.01693457],
           [0.61266166, 0.63010883, 0.38751987],
           [0.37212682, 0.5235918 , 0.15730347],
           [0.8845907 , 0.7256049 , 0.03442054]], dtype=float32)




```python
kmeans.labels_
```




    array([1, 1, 1, ..., 4, 1, 1])




```python
# 각그룹의 대표되는 RGB 값으로 426400개의 픽셀값을 대체
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img.shape
```




    (426400, 3)




```python
segmented_img = segmented_img.reshape(image.shape)
plt.imshow(segmented_img)
```




    <matplotlib.image.AxesImage at 0x1a67cbd19d0>




    
![png](/assets/images/2023-03-08-Machine Learning 12 (K-평균 활용)/output_13_1.png)
    



```python
segmented_imgs = []
n_colors = [10, 8, 6, 4, 2]
for clusters in n_colors:
  kmeans = KMeans(n_clusters=clusters, random_state=42)
  kmeans.fit(X)
  segmented_img = kmeans.cluster_centers_[kmeans.labels_]
  segmented_img = segmented_img.reshape(image.shape)
  segmented_imgs.append(segmented_img)
```


```python
plt.figure(figsize=(10, 5))
plt.subplot(231)
plt.imshow(image)
plt.title('original image')
plt.axis('off')

for idx, n_clusters in enumerate(n_colors):
  plt.subplot(232+idx)
  plt.imshow(segmented_imgs[idx])
  plt.title('{} colors'.format(n_clusters))
  plt.axis('off')
```


    
![png](/assets/images/2023-03-08-Machine Learning 12 (K-평균 활용)/output_15_0.png)
    

## Reference
- [핸즈온 머신러닝 (오렐리앙 제롱 저)](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=237677114)