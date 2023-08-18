# Matplotlib

## Matplotlib API 개요

- 정보 시각화는 데이터 분석에서 무척 중요
- 시각화는 특잇값을 찾아내거나, 데이터 변형이 필요한지 알아보거나, 모델에 대한 아이디어를 찾기 위한 과정의 일부
- 파이썬 시각화 도구로 matplotlib 활용

- 공식 사이트
https://matplotlib.org/


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```


```python
plt.plot([1, 2, 3], [4, 5, 6]) # 각각의 리스트가 x축, y축으로 전달되서 선그래프
plt.title('Hello Plot')
plt.show() # 주피터 노트북에서는 plt.show() 안해줘도 그림이 표시가 되지만, .py 같은 스크립트 작성시에는 필요
```


    
![png](/assets/images/output_4_1.png)
    


### Figure와 Axes 


```python
from IPython.display import Image
Image('./images/figure_axes.png')
```




    
![png](/assets/images/output_6_0.png)
    



**Figure**
- 그림을 그리기 위한 도화지의 역할, 그림판의 크기 등을 조절함

**Axes**
- 실제 그림을 그리는 메소드들과 x축, y축, title 등의 속성을 가짐

```
plt.plot([1, 2, 3], [4, 5, 6]) # 기본 설정값의 Figure 객체가 만들어진 후, 내부적으로 Axes.plot() 호출
plt.title('Hello Plot') # Axes.set_title() 호출
plt.show()              # Figure.show() 호출출
```


```python
plt.figure(figsize=(10, 4)) # 가로가 10, 세로가 4인 Figure 객체를 생성해서 반환
plt.plot([1, 2, 3], [4, 5, 6]) # 기본 설정값의 Figure 객체가 만들어진 후, 내부적으로.plot() Axes 호출
plt.title('Hello Plot') # Axes.set_title() 호출
plt.show()              # Figure.show() 호출
```


    
![png](/assets/images/output_9_1.png)
    



```python
plt.figure(facecolor='red', figsize=(10, 4)) # Figure 객체에 색깔과 크기까지 설정정
plt.plot([1, 2, 3], [4, 5, 6]) 
plt.title('Hello Plot') 
plt.show()              
```


    
![png](/assets/images/output_10_1.png)
    



```python
fig = plt.figure() # Figure 객체 생성
```


    <Figure size 432x288 with 0 Axes>



```python
ax = plt.axes() # Axes 객체 생성
ax.hist(np.random.randn(100))
```




    (array([ 5.,  2.,  5.,  7., 17., 24., 16., 11.,  9.,  4.]),
     array([-2.36681532, -1.93192652, -1.49703772, -1.06214893, -0.62726013,
            -0.19237134,  0.24251746,  0.67740625,  1.11229505,  1.54718384,
             1.98207264]),
     <BarContainer object of 10 artists>)




    
![png](/assets/images/output_12_1.png)
    



```python
ax = plt.axes()
ax.scatter(np.arange(10), np.arange(10))
```




    <matplotlib.collections.PathCollection at 0x7f1341c53af0>




    
![png](/assets/images/output_13_1.png)
    



```python
# Figure와 Axes를 함께 가져옴(기본적으로 한개의 Axes를 설정)
fig, axes = plt.subplots()
print(type(fig), type(axes))
```

    <class 'matplotlib.figure.Figure'> <class 'matplotlib.axes._subplots.AxesSubplot'>
    


    
![png](/assets/images/output_14_1.png)
    



```python
# Figure의 사이즈 설정, Axes의 개수 설정
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 4)) # 정해진 figsize 안에 nrows, ncols 만큼의 axes를 생성
```


    
![png](/assets/images/output_15_0.png)
    


**plot() 함수안에 올 수 있는 데이터**


```python
# list 
plt.plot([1, 2, 3], [4, 5, 6])
plt.show()
```


    
![png](/assets/images/output_17_0.png)
    



```python
# ndarray
plt.plot(np.array([1, 2, 3]), np.array([4, 5, 6]))
plt.show()
```


    
![png](/assets/images/output_18_0.png)
    



```python
# Series
plt.plot(pd.Series([1, 2, 3]), pd.Series([4, 5, 6]))
plt.show()
```


    
![png](/assets/images/output_19_0.png)
    



```python
# DataFrame
df = pd.DataFrame(np.random.randn(10, 4).cumsum(0),
                  columns=['A', 'B', 'C', 'D'],
                  index=np.arange(0, 100, 10))
df
```





  <div id="df-9a15cca6-c938-41a4-8d8a-d830a598aac8">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.213829</td>
      <td>0.158209</td>
      <td>1.234090</td>
      <td>1.625117</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.548749</td>
      <td>-0.829001</td>
      <td>-0.414488</td>
      <td>3.959293</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2.613033</td>
      <td>-1.590840</td>
      <td>-1.829204</td>
      <td>4.968986</td>
    </tr>
    <tr>
      <th>30</th>
      <td>3.831927</td>
      <td>0.818843</td>
      <td>-1.572401</td>
      <td>3.499105</td>
    </tr>
    <tr>
      <th>40</th>
      <td>5.238877</td>
      <td>0.061736</td>
      <td>-3.180235</td>
      <td>2.124488</td>
    </tr>
    <tr>
      <th>50</th>
      <td>4.495027</td>
      <td>0.665889</td>
      <td>-4.014447</td>
      <td>1.158320</td>
    </tr>
    <tr>
      <th>60</th>
      <td>5.204724</td>
      <td>1.713093</td>
      <td>-4.521490</td>
      <td>1.648307</td>
    </tr>
    <tr>
      <th>70</th>
      <td>4.708933</td>
      <td>1.292645</td>
      <td>-4.284156</td>
      <td>1.124667</td>
    </tr>
    <tr>
      <th>80</th>
      <td>5.261158</td>
      <td>1.270810</td>
      <td>-4.213980</td>
      <td>-0.805378</td>
    </tr>
    <tr>
      <th>90</th>
      <td>4.532295</td>
      <td>2.175720</td>
      <td>-5.124437</td>
      <td>-0.357466</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-9a15cca6-c938-41a4-8d8a-d830a598aac8')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-9a15cca6-c938-41a4-8d8a-d830a598aac8 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9a15cca6-c938-41a4-8d8a-d830a598aac8');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
plt.plot(df) # df의 index가 x축, column 아래 값들이 y축으로 매칭칭
plt.show()
```


    
![png](/assets/images/output_21_0.png)
    



```python
df.plot(kind='line') # plt.plot()이 아니라 df.plot()로 호출하면 범례(컬럼명기준)까지 생성성
plt.show()
```


    
![png](/assets/images/output_22_0.png)
    



```python
df.plot(kind='bar')
```




    <AxesSubplot:>




    
![png](/assets/images/output_23_1.png)
    


**색상, 마커, 선 스타일 등 변경**


```python
Image('./images/plot.png')
```




    
![png](/assets/images/output_25_0.png)
    




```python
plt.plot([1, 2, 3], [4, 5, 6], color='green') # 색상 변경
plt.show()
```


    
![png](/assets/images/output_26_0.png)
    



```python
plt.plot([1, 2, 3], [4, 5, 6], color='green', linestyle='dashed') # 선 스타일 변경경
plt.show()
```


    
![png](/assets/images/output_27_0.png)
    



```python
plt.plot([1, 2, 3], [4, 5, 6], color='green', linestyle='dashed', linewidth=3, marker='o', markersize=10) # 마커 변경
plt.show()
```


    
![png](/assets/images/output_28_1.png)
    



```python
plt.plot([1, 2, 3], [4, 5, 6], 'go--')
plt.show()
```


    
![png](/assets/images/output_29_0.png)
    


**x축, y축에 축명을 텍스트로 할당**


```python
plt.plot([1, 2, 3], [4, 5, 6], color='green', linestyle='dashed', linewidth=3, marker='o', markersize=10) 
plt.xlabel('x axis', size=20)
plt.ylabel('y axis', size=20)
plt.show()
```


    
![png](/assets/images/output_31_0.png)
    


**x축, y축 틱값 표기**


```python
plt.plot([1, 2, 3], [4, 5, 6], color='green', linestyle='dashed', linewidth=3, marker='o', markersize=10) 
plt.xlabel('x axis', size=20)
plt.ylabel('y axis', size=20)

plt.xticks([1, 2, 3], rotation=30) # x축 눈금, 30도 회전
plt.yticks([4, 5, 6]) # y축 눈금금
plt.show()
```


    
![png](/assets/images/output_33_0.png)
    


**x축, y축값 제한**


```python
plt.plot([1, 2, 3], [4, 5, 6], color='green', linestyle='dashed', linewidth=3, marker='o', markersize=10) 
plt.xlabel('x axis', size=20)
plt.ylabel('y axis', size=20)

plt.xticks([0, 1, 2], rotation=30) # x축 눈금, 30도 회전
plt.yticks([3, 4, 5]) # y축 눈금

plt.xlim(0, 2)
plt.ylim(3, 5)
plt.show()
```


    
![png](/assets/images/output_35_0.png)
    


**범례 설정하기**


```python
plt.plot([1, 2, 3], [4, 5, 6], label='test', color='green', linestyle='dashed', linewidth=3, marker='o', markersize=10) 
plt.xlabel('x axis', size=20)
plt.ylabel('y axis', size=20)

plt.xticks([0, 1, 2], rotation=30) # x축 눈금, 30도 회전
plt.yticks([3, 4, 5]) # y축 눈금

plt.xlim(0, 2)
plt.ylim(3, 5)

plt.legend() # 범례 설정
plt.show()
```


    
![png](/assets/images/output_37_0.png)
    


**여러개의 plot을 하나의 axes에서 그리기**


```python
x1 = np.arange(100)
y1 = x1 * 2

x2 = np.arange(100)
y2 = x1 * 3

plt.plot(x1, y1, label='y=2x')
plt.plot(x2, y2, label='y=3x')

plt.legend()
plt.show()
```


    
![png](/assets/images/output_39_0.png)
    


**Axes 객체에서 직접 작업하기**


```python
x1 = np.arange(10)
y1 = x1 * 2

fig = plt.figure()
axes = plt.axes()

axes.plot(x1, y1, color='red', marker='+', label='line') # 선그래프
axes.bar(x1, y1, color='green', label='bar') # 막대그래프

axes.set_xlabel('x axis', size=20)
axes.set_ylabel('y axis', size=20)

axes.set_xticks(x1)

axes.legend()
plt.show()
```


    
![png](/assets/images/output_41_0.png)
    


## 실습


```python
from IPython.display import Image
Image('./images/axes.png')
```




    
![png](/assets/images/output_43_0.png)
    



- 위의 그래프와 같이 여러개의 subplot을 한 figure 안에 생성하고 개별 그래프 시각화


```python
# todo
x1 = np.arange(1, 10)
y1 = x1 * 2

x2 = np.arange(1, 20)
y2 = x2 * 2

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# 좌상단 
axes[0][0].plot(x1, y1, color='red', marker='o', linestyle='dashed', label='red line')
axes[0][0].legend()
axes[0][0].set_xlabel('axes[0][0] x axis')
axes[0][0].set_ylabel('axes[0][0] y axis')

# 우상단
axes[0][1].bar(x2, y2, color='green', label='green bar')
axes[0][1].legend()
axes[0][1].set_xlabel('axes[0][1] x axis')
axes[0][1].set_ylabel('axes[0][1] y axis')

# 좌하단
axes[1][0].plot(x1, y1, color='green', marker='o', linestyle='dashed', label='green line')
axes[1][0].legend()
axes[1][0].set_xlabel('axes[1][0] x axis')
axes[1][0].set_ylabel('axes[1][0] y axis')

# 우하단
axes[1][1].bar(x2, y2, color='red', label='red bar')
axes[1][1].legend()
axes[1][1].set_xlabel('axes[1][1] x axis')
axes[1][1].set_ylabel('axes[1][1] y axis')

plt.show()





```


    
![png](/assets/images/output_45_0.png)
    


# Workshop


```python
df = pd.read_csv('./examples/서울경기이동.csv')
sr = pd.Series(df['경기도'])
sr.index = df['Unnamed: 0'].values
```

- 선 그래프 그리기


```python
from IPython.display import Image
Image('./images/1.png')
```




    
![png](/assets/images/output_49_0.png)
    




```python
# todo
plt.plot(sr.index, sr.values)
plt.show()
```


    
![png](/assets/images/output_50_0.png)
    


- 차트 제목 추가


```python
Image('./images/2.png')
```




    
![png](/assets/images/output_52_0.png)
    




```python
# 코랩에서 한글 안보이는 현상
# !sudo apt-get install -y fonts-nanum
# !sudo fc-cache -fv
# !rm ~/.cache/matplotlib -rf
```


```python
# 폰트 설치후 커널 재시작
# plt.rc('font', family='NanumBarunGothic') 
```


```python
# todo
plt.rc('font', family='Malgun Gothic') # for Windows

plt.plot(sr.index, sr.values)
plt.title('서울 -> 경기 인구 이동') 
plt.show()
```


    
![png](/assets/images/output_55_0.png)
    


- 축이름 추가


```python
Image('./images/3.png')
```




    
![png](/assets/images/output_57_0.png)
    




```python
# todo
plt.plot(sr.index, sr.values)
plt.title('서울 -> 경기 인구 이동') # 제목
plt.xlabel('기간간') # x축 이름
plt.ylabel('이동 인구수수') # y축 이름
plt.show()
```


    
![png](/assets/images/output_58_0.png)
    


- Figure 사이즈 지정(가로 14, 세로 5)


```python
Image('./images/4.png')
```




    
![png](/assets/images/output_60_0.png)
    




```python
# todo
plt.figure(figsize=(14, 5))
plt.plot(sr.index, sr.values)
plt.title('서울 -> 경기 인구 이동') # 제목
plt.xlabel('기간') # x축 이름
plt.ylabel('이동 인구수') # y축 이름
plt.show()
```


    
![png](/assets/images/output_61_0.png)
    


- x축 눈금 라벨 회전하기 (90도)


```python
Image('./images/5.png')
```




    
![png](/assets/images/output_63_0.png)
    




```python
# todo
plt.figure(figsize=(14, 5))
plt.plot(sr.index, sr.values)
plt.title('서울 -> 경기 인구 이동') # 제목
plt.xlabel('기간') # x축 이름
plt.ylabel('이동 인구수수') # y축 이름

plt.xticks(rotation=90)
plt.show()
```


    
![png](/assets/images/output_64_0.png)
    


- 마커 표시 추가


```python
Image('./images/6.png')
```




    
![png](/assets/images/output_66_0.png)
    




```python
# todo
plt.figure(figsize=(14, 5))
plt.plot(sr.index, sr.values, marker='o', markersize=10)
plt.title('서울 -> 경기 인구 이동') # 제목
plt.xlabel('기간') # x축 이름
plt.ylabel('이동 인구수수') # y축 이름

plt.xticks(sr.index, rotation=90)
plt.show()
```


    
![png](/assets/images/output_67_0.png)
    


- 범례 표시


```python
Image('./images/7.png')
```




    
![png](/assets/images/output_69_0.png)
    




```python
# todo 
plt.figure(figsize=(14, 5))
plt.plot(sr.index, sr.values, marker='o', markersize=10)
plt.title('서울 -> 경기 인구 이동') # 제목
plt.xlabel('기간') # x축 이름
plt.ylabel('이동 인구수수') # y축 이름

plt.xticks(sr.index, rotation=90)
plt.legend(labels=['서울->경기'], loc='best', fontsize=15)
plt.show()
```


    
![png](/assets/images/output_70_0.png)
    


 - y축 범위 지정 (최소값, 최대값) : (5000, 800000)


```python
Image('./images/8.png')
```




    
![png](/assets/images/output_72_0.png)
    




```python
# todo
plt.figure(figsize=(14, 5))
plt.plot(sr.index, sr.values, marker='o', markersize=10)
plt.title('서울 -> 경기 인구 이동') # 제목
plt.xlabel('기간') # x축 이름
plt.ylabel('이동 인구수수') # y축 이름

plt.xticks(sr.index, rotation=90)
plt.ylim(50000, 800000)

plt.legend(labels=['서울->경기'], loc='best', fontsize=15)
plt.show()
```


    
![png](/assets/images/output_73_0.png)
    


- 스타일 서식 지정


```python
Image('./images/9.png')
```




    
![png](/assets/images/output_75_0.png)
    




```python
# 스타일 리스트 출력
print(plt.style.available)
```

    ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']
    


```python
# todo
plt.style.use('ggplot')

plt.figure(figsize=(14, 5))
plt.plot(sr.index, sr.values, marker='o', markersize=10)
plt.title('서울 -> 경기 인구 이동') # 제목
plt.xlabel('기간') # x축 이름
plt.ylabel('이동 인구수수') # y축 이름

plt.xticks(sr.index, rotation=90)
plt.ylim(50000, 800000)

plt.legend(labels=['서울->경기'], loc='best', fontsize=15)
plt.show()
```


    
![png](/assets/images/output_77_1.png)
    


- 주석표시


```python
Image('./images/10.png')
```




    
![png](/assets/images/output_79_0.png)
    




```python
# todo
plt.style.use('ggplot')

plt.figure(figsize=(14, 5))
plt.plot(sr.index, sr.values, marker='o', markersize=10)
plt.title('서울 -> 경기 인구 이동') # 제목
plt.xlabel('기간') # x축 이름
plt.ylabel('이동 인구수수') # y축 이름

plt.xticks(sr.index, rotation=90)
plt.ylim(50000, 800000)

# 주석 표시
plt.annotate('',  # 표시될 텍스트
             xy = (1995, 640000), # 화살표의 머리(뾰족한 부분분)
             xytext= (1970, 290000), # 화살표의 끝점   
             arrowprops= dict(arrowstyle='->', color='skyblue', lw=5) # 화살표 서식
             )

plt.annotate('',  # 표시될 텍스트
             xy = (2016, 450000), # 화살표의 머리
             xytext= (1996, 640000), # 화살표의 끝점
             arrowprops= dict(arrowstyle='->', color='olive', lw=5) # 화살표 서식
             )

plt.annotate('인구이동 증가(1970-1995)',  # 표시될 텍스트
             xy = (1975, 450000), # 텍스트 위치 기준점
             rotation = 18,
             fontsize= 15
             )

plt.annotate('인구이동 감소(1995-2017)',  # 표시될 텍스트
             xy = (2002, 550000), # 텍스트 위치 기준점
             rotation = -11,         
             fontsize= 15
             )

plt.legend(labels=['서울->경기'], loc='best', fontsize=15)
plt.show()
```


    
![png](/assets/images/output_80_0.png)
    

