# Seaborn

## Seaborn을 이용한 주요 시각화 그래프

- Seaborn 공식 사이트 : https://seaborn.pydata.org/tutorial.html
- Matplotlib 기반으로 쉽게 작성됨. Matplotlib의 high level API
- Matplotlib 보다 수려한 디자인을 제공하며 Pandas와 쉽게 연동
- 그러나 Matplotlib을 어느 정도 알고 있어야함


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
from IPython.display import Image
Image('./images/graph1.png')
```




    
![png](/assets/images2/output_4_0.png)
    




```python
Image('./images/graph2.png')
```




    
![png](/assets/images2/output_5_0.png)
    




```python
Image('./images/graph3.png')
```




    
![png](/assets/images2/output_6_0.png)
    




```python
titanic_df = pd.read_csv('./datasets/titanic_train.csv')
titanic_df.head()
```





  <div id="df-1b04ee75-8db7-4d8d-bbdc-b46d3eeb36e4">
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1b04ee75-8db7-4d8d-bbdc-b46d3eeb36e4')"
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
          document.querySelector('#df-1b04ee75-8db7-4d8d-bbdc-b46d3eeb36e4 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1b04ee75-8db7-4d8d-bbdc-b46d3eeb36e4');
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




## 수치형 데이터 시각화


```python
titanic_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
    

### 1. 히스토그램
- 수치형 데이터의 구간별 빈도수를 나타내는 그래프

- matplotlib 지원


```python
titanic_df['Age']
```


```python
result = plt.hist(titanic_df['Age'], bins=20)
print('빈도수 : ', result[0])
print('범위 :', result[1])
plt.show()
```

    빈도수 :  [40. 14. 15. 31. 79. 98. 85. 84. 73. 45. 35. 35. 29. 16. 13. 11.  4.  5.
      1.  1.]
    범위 : [ 0.42   4.399  8.378 12.357 16.336 20.315 24.294 28.273 32.252 36.231
     40.21  44.189 48.168 52.147 56.126 60.105 64.084 68.063 72.042 76.021
     80.   ]
    


    
![png](/assets/images2/output_13_1.png)
    


- pandas에서 직접 호출 가능


```python
titanic_df['Age'].hist(bins=20)
```




    <AxesSubplot:>




    
![png](/assets/images2/output_15_1.png)
    


- seaborn 지원


```python
sns.histplot(data=titanic_df, x='Age', bins=20)
plt.show()
```


    
![png](/assets/images2/output_17_0.png)
    



```python
sns.histplot(data=titanic_df, x='Age', hue='Survived', bins=20) # Survived 인지 아닌지의 정보를 추가
plt.show()
```


    
![png](/assets/images2/output_18_0.png)
    



```python
# 겹치지 않게 표시하려면 multiple = 'stack' 옵션 추가가
sns.histplot(data=titanic_df, x='Age', hue='Survived', bins=20, multiple='stack') # Survived 인지 아닌지의 정보를 추가
plt.show()
```


    
![png](/assets/images2/output_19_0.png)
    



```python
sns.histplot(data=titanic_df, x='Age', bins=20, kde=True)
plt.show()
```


    
![png](/assets/images2/output_20_0.png)
    


### 2. 커널밀도추정 함수

- 히스토그램을 매끄럽게 곡선으로 연결한 그래프


```python
fig = plt.figure(figsize=(20, 10)) # matplotlib에서 만든 Figure 객체 안에서 seaborn의 그래프도 그려짐
sns.kdeplot(data=titanic_df, x='Age', hue='Survived', multiple='stack') # Axes level 함수
plt.show()
```


    
![png](/assets/images2/output_23_0.png)
    


- Figure-level vs. axes-level


```python
from IPython.display import Image
Image('./images/figure vs axes.png', width=500)
```




    
![png](/assets/images2/output_25_0.png)
    




```python
plt.figure(figsize=(20, 10)) # Figure level 함수는 matplotlib에서 지정한 figsize에 영향을 받지 않음
sns.displot(data=titanic_df, x='Age', kde=True) # Figure level 함수
plt.show()
```




    <seaborn.axisgrid.FacetGrid at 0x7fec2edfa610>




    <Figure size 1440x720 with 0 Axes>



    
![png](/assets/images2/output_26_2.png)
    


## 범주형 데이터 시각화

### 1. 막대그래프(barplot)
- 범주형 데이터 값에 따라 수치형 데이터 값이 어떻게 달라지는지 파악할 때 사용
- 범주형 데이터에 따른 수치형 데이터의 평균과 신뢰구간을 그려줌
- 수치형 데이터 평균은 막대높이로, 신뢰구간은 오차 막대로 표현함


```python
sns.barplot(data=titanic_df, x='Pclass', y='Age') # ci=None이면 오차 막대가 표시되지 않음. 기본은 95% 신뢰 구간
plt.show()
```




    <AxesSubplot:xlabel='Pclass', ylabel='Age'>




    
![png](/assets/images2/output_29_1.png)
    



```python
sns.barplot(data=titanic_df, x='Pclass', y='Fare')
plt.show()
```




    <AxesSubplot:xlabel='Pclass', ylabel='Fare'>




    
![png](/assets/images2/output_30_1.png)
    



```python
# 클래스당 나이평균 + 성별까지 추가 정보를 표시 : hue='Sex'
sns.barplot(data=titanic_df, x='Pclass', y='Age', hue='Sex') 
plt.show()
```




    <AxesSubplot:xlabel='Pclass', ylabel='Age'>




    
![png](/assets/images2/output_31_1.png)
    



```python
# 클래스당 생존률
sns.barplot(data=titanic_df, x='Pclass', y='Survived') 
plt.show()
```




    <AxesSubplot:xlabel='Pclass', ylabel='Survived'>




    
![png](/assets/images2/output_32_1.png)
    



```python
# 클래스당 생존률 + 성별까지 추가 정보를 표시 : hue='Sex'
sns.barplot(data=titanic_df, x='Pclass', y='Survived', hue='Sex')
plt.show()
```


    
![png](/assets/images2/output_33_0.png)
    



```python
bins= [0, 18, 25, 35, 60, 80]
group_names = ['Children', 'Youth', 'YoungAdult', 'MiddleAged', 'Senior']
titanic_df['Age_cat'] = pd.cut(titanic_df['Age'], bins, labels=group_names)
```


```python
titanic_df['Age_cat']
```




    0           Youth
    1      MiddleAged
    2      YoungAdult
    3      YoungAdult
    4      YoungAdult
              ...    
    886    YoungAdult
    887         Youth
    888           NaN
    889    YoungAdult
    890    YoungAdult
    Name: Age_cat, Length: 891, dtype: category
    Categories (5, object): ['Children' < 'Youth' < 'YoungAdult' < 'MiddleAged' < 'Senior']




```python
sns.barplot(data=titanic_df, x='Age_cat', y='Survived')
```




    <AxesSubplot:xlabel='Age_cat', ylabel='Survived'>




    
![png](/assets/images2/output_36_1.png)
    



```python
# 성별에 따른 생존률
sns.barplot(data=titanic_df, x='Sex', y='Survived')
```




    <AxesSubplot:xlabel='Sex', ylabel='Survived'>




    
![png](/assets/images2/output_37_1.png)
    



```python
# 성별에 따른 생존률 + 나이대 정보를 추가 : hue='Age_cat'
sns.barplot(data=titanic_df, x='Sex', y='Survived', hue='Age_cat')
```




    <AxesSubplot:xlabel='Sex', ylabel='Survived'>




    
![png](/assets/images2/output_38_1.png)
    



```python
# 성별에 따른 생존률 + 클래스 정보를 추가 : hue='Pclass'
sns.barplot(data=titanic_df, x='Sex', y='Survived', hue='Pclass')
```




    <AxesSubplot:xlabel='Sex', ylabel='Survived'>




    
![png](/assets/images2/output_39_1.png)
    


### 2. 포인트 플롯(pointplot)
- 막대 그래프와 모양만 다를 뿐 동일한 정보 제공
- 막대 그래프와 마찬가지로 범주형 데이터에 따른 수치형 데이터의 평균과 신뢰구간을 나타냄
- 다만 그래프를 점과 선으로 나타냄


```python
sns.barplot(data=titanic_df, x='Pclass', y='Fare', hue='Age_cat')
```




    <AxesSubplot:xlabel='Pclass', ylabel='Fare'>




    
![png](/assets/images2/output_41_1.png)
    



```python
sns.pointplot(data=titanic_df, x='Pclass', y='Fare', hue='Age_cat')
```




    <AxesSubplot:xlabel='Pclass', ylabel='Fare'>




    
![png](/assets/images2/output_42_1.png)
    


### 3. 박스플롯(boxplot)
- 막대그래프나 포인트플롯보다 더 많은 정보를 제공
- 5가지 요약 수치 : 최솟값, 1사분위수(Q1), 2사분위수(Q2), 3사분위수(Q3), 최댓값

- 1사분위수(Q1) : 전체 데이터 중 하위 25%에 해당하는 값
- 2사분위수(Q2): 50%에 해당하는 값 
- 3사분위수(Q3) : 상위 25%에 해당하는 값
- 사분위 범위수(IQR) : Q3 - Q1
- 최댓값(Max) : Q3 + (1.5 * IQR)
- 최솟값(Min) : Q1 - (1.5 * IQR)


```python
Image('./images/boxplot.png')
```




    
![png](/assets/images2/output_44_0.png)
    




```python
sns.boxplot(data=titanic_df, x='Pclass', y='Age')
```




    <AxesSubplot:xlabel='Pclass', ylabel='Age'>




    
![png](/assets/images2/output_45_1.png)
    


### 4. 바이올린플롯(violinplot)
- 박스플롯과 커널밀도추정 함수 그래프를 합쳐 놓은 그래프
- 박스플롯에 제공하는 정보를 모두 포함하며, 모양은 커널밀도추정 함수 그래프 형태임


```python
Image('./images/violinplot.png')
```




    
![png](/assets/images2/output_47_0.png)
    




```python
sns.boxplot(data=titanic_df, x='Pclass', y='Age')
plt.show()
```


    
![png](/assets/images2/output_48_0.png)
    



```python
sns.violinplot(data=titanic_df, x='Pclass', y='Age')
plt.show()
```


    
![png](/assets/images2/output_49_0.png)
    



```python
sns.violinplot(data=titanic_df, x='Pclass', y='Age', hue='Sex')
plt.show()
```


    
![png](/assets/images2/output_50_0.png)
    



```python
sns.violinplot(data=titanic_df, x='Pclass', y='Age', hue='Sex', split=True)
plt.show()
```


    
![png](/assets/images2/output_51_0.png)
    


### 5. 카운트플롯(countplot)
- 카운트플롯은 범주형 데이터의 개수를 확인할 때 사용하는 그래프
- 주로 범주형 피처나 범주형 타깃값의 분포가 어떤지 파악하는 용도로 사용
- 카운트플롯을 사용하면 범주형 데이터의 개수를 파악할 수 있음



```python
sns.countplot(data=titanic_df, x='Pclass')
plt.show()
```


    
![png](/assets/images2/output_53_0.png)
    



```python
titanic_df['Pclass'].value_counts()
```




    3    491
    1    216
    2    184
    Name: Pclass, dtype: int64




```python
titanic_df.groupby('Pclass').size()
```




    Pclass
    1    216
    2    184
    3    491
    dtype: int64




```python
sns.countplot(data=titanic_df, y='Pclass')
plt.show()
```


    
![png](/assets/images2/output_56_0.png)
    


### 6. 파이 그래프(pie)
- 범주형 데이터별 비율을 알아볼 때 사용하기 좋은 그래프
- seaborn에서 파이 그래프를 지원하지 않아 matplotlib을 사용


```python
data = titanic_df['Pclass'].value_counts()
```


```python
data.plot(kind='pie', autopct='%.2f%%')
```




    <AxesSubplot:ylabel='Pclass'>




    
![png](/assets/images2/output_59_1.png)
    


## 데이터 관계 시각화

### 1. 히트맵(heatmap)
- 데이터 간 관계를 색상으로 표현한 그래프
- 비교해야 할 데이터가 많을 때 주로 사용


```python
corr = titanic_df.corr()
```


```python
plt.figure(figsize=(8, 8))
sns.heatmap(data=corr, annot=True, fmt='.2f', cmap='YlGnBu', linewidth=0.5)
```




    <AxesSubplot:>




    
![png](/assets/images2/output_63_1.png)
    


* [컬러맵 정보](https://matplotlib.org/stable/tutorials/colors/colormaps.html)


### 2. 산점도(scatterplot)
- 산점도는 두 데이터의 관계를 점으로 표현하는 그래프


```python
# Age와 Fare의 상관관계계
sns.scatterplot(data=titanic_df, x='Age', y='Fare')
plt.show()
```


    
![png](/assets/images2/output_66_0.png)
    



```python
# Age와 Fare의 상관관계 + 성별 정보 추가 표시 : hue='Sex'
sns.scatterplot(data=titanic_df, x='Age', y='Fare', hue='Sex')
plt.show()
```


    
![png](/assets/images2/output_67_0.png)
    



```python
sns.regplot(data=titanic_df, x='Age', y='Fare')
plt.show()
```


    
![png](/assets/images2/output_68_0.png)
    


## seaborn에서 subplots(axes) 이용하기 


```python
axes
```




    array([[<AxesSubplot:>, <AxesSubplot:>],
           [<AxesSubplot:>, <AxesSubplot:>]], dtype=object)




```python
fig, axes = plt.subplots(nrows=2, ncols=2)
sns.regplot(data=titanic_df, x='Age', y='Fare', ax=axes[1][1]) # 좌상단의 subplot(axes)에 표시
plt.show()
```




    <AxesSubplot:xlabel='Age', ylabel='Fare'>




    
![png](/assets/images2/output_71_1.png)
    


#### (1) subplots(axes)을 이용하여 주요 범주형 데이터의 건수를 시각화 하기


```python
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(22, 4))
axes
```




    array([<AxesSubplot:>, <AxesSubplot:>, <AxesSubplot:>, <AxesSubplot:>,
           <AxesSubplot:>], dtype=object)




    
![png](/assets/images2/output_73_1.png)
    



```python
cat_columns = ['Survived', 'Pclass', 'Sex', 'Embarked', 'Age_cat']

fig, axes = plt.subplots(nrows=1, ncols=len(cat_columns), figsize=(22, 4))
for index, column in enumerate(cat_columns):
  sns.countplot(data=titanic_df, x=column, ax=axes[index])
  if index == 4:
    label = axes[index].get_xticklabels()
    print(label)
    axes[index].set_xticklabels(label, rotation=90)
plt.show()    
```

    [Text(0, 0, 'Children'), Text(1, 0, 'Youth'), Text(2, 0, 'YoungAdult'), Text(3, 0, 'MiddleAged'), Text(4, 0, 'Senior')]
    


    
![png](/assets/images2/output_74_1.png)
    



```python
Image('./images/seaborn1.png')
```




    
![png](/assets/images2/output_75_0.png)
    



#### (2) subplots(axes)을 이용하여 주요 범주형 데이터별 생존율 시각화 하기


```python
cat_columns = ['Pclass', 'Sex', 'Embarked', 'Age_cat']

fig, axes = plt.subplots(nrows=1, ncols=len(cat_columns), figsize=(22, 4))

for index, column in enumerate(cat_columns):
  sns.barplot(data=titanic_df, x=column, y='Survived', ax=axes[index])
```


    
![png](/assets/images2/output_77_0.png)
    



```python
Image('./images/seaborn2.png')
```




    
![png](/assets/images2/output_78_0.png)
    




```python

```
