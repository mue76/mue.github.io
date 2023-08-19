---
tag: [python, 데이터분석, data analysis overview, kaggle bike sharing]
---

# 데이터 분석

# 1. 데이터 분석 개요

## Step 1: 질문하기 (Ask questions)
데이터가 주어진 상태에서 질문을 할 수도 있고, 질문에 답할 수 있는 데이터를 수집할 수도 있다.

## Step 2: 데이터 랭글링 (Wrangle data)
- 데이터 랭글링 : 원자료(raw data)를 보다 쉽게 접근하고 분석할 수 있도록 데이터를 정리하고 통합하는 과정
(참고. 위키피디아)
- 세부적으로는 데이터의 수집(gather), 평가(assess), 정제(clean) 작업으로 나눌 수 있다.

## Step 3: 데이터 탐색 (Exploratory Data Analysis)
데이터의 패턴을 찾고, 관계를 시각화 하는 작업을 통해 데이터에 대한 직관을 극대화 한다.

## Step 4: 결론 도출 또는 예측 (Draw conclusions or make predictions)
- Step 3에서 분석한 내용을 근거로 질문에 대한 답과 결론을 도출 할 수 있다.
- 머신러닝 또는 통계 추정 과정을 거치게 되면 예측을 만들어 낼 수도 있다.

## Step 5: 결과 공유 (Communicate the results)
보고서, 이메일, 블로그 등 다양한 방법을 통해 발견한 통찰들을 공유할 수 있다.

# 2. Case Study

## Bike Sharing Demand

- 도시 자전거 공유 시스템 사용 예측
- [캐글](https://www.kaggle.com)의 [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand)에서 `train.csv`와 `test.csv`를 다운로드
- 두 파일을 각각 datasets 디렉토리에 bike_train.csv bike_test.csv로 저장

**datetime** : hourly date + timestamp  
**season** : 1 = 봄, 2 = 여름, 3 = 가을, 4 = 겨울  
**holiday**: 1 = 토, 일요일의 주말을 제외한 국경일 등의 휴일, 0 = 휴일이 아닌 날  
**workingday**: 1 = 토, 일요일의 주말 및 휴일이 아닌 주중, 0 = 주말 및 휴일  
**weather**:  
- 1 = 맑음, 약간 구름 낀 흐림  
- 2 = 안개, 안개 + 흐림  
- 3 = 가벼운 눈, 가벼운 비 + 천둥  
- 4 = 심한 눈/비, 천둥/번개  

**temp**: 온도(섭씨)   
**atemp**: 체감온도(섭씨)  
**humidity**: 상대습도  
**windspeed**: 풍속  
**casual**: 사전에 등록되지 않는 사용자가 대여한 횟수  
**registered**: 사전에 등록된 사용자가 대여한 횟수  
**count**: 대여 횟수  

## Step 1: 질문하기 (Ask questions)

**예시**
- (질문 1) 어떤 기상정보가 자전거 대여량에 영향을 미칠까?
- (질문 2) 어떤 날짜(요일, 달, 계절)에 대여량이 많을까(혹은 적을까)?
- (질문 3) 언제 프로모션을 하면 좋을까?

## Step 2: 데이터 랭글링 (Wrangle data)

- 데이터 적재


```python
import pandas as pd
```


```python
bike = pd.read_csv('./datasets/bike_train.csv')
```


```python
type(bike)
```




    pandas.core.frame.DataFrame



- 데이터 평가


```python
bike.head() # 데이터 훑어보기
```





  <div id="df-ff638b6a-d873-4aa6-a550-139343689492">
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
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-01 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-01 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-01 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-01 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-01 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ff638b6a-d873-4aa6-a550-139343689492')"
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
          document.querySelector('#df-ff638b6a-d873-4aa6-a550-139343689492 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ff638b6a-d873-4aa6-a550-139343689492');
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
bike.tail()
```





  <div id="df-95d7a3f0-fce7-478a-ac94-0d5bddea5c15">
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
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10881</th>
      <td>2012-12-19 19:00:00</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>15.58</td>
      <td>19.695</td>
      <td>50</td>
      <td>26.0027</td>
      <td>7</td>
      <td>329</td>
      <td>336</td>
    </tr>
    <tr>
      <th>10882</th>
      <td>2012-12-19 20:00:00</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>14.76</td>
      <td>17.425</td>
      <td>57</td>
      <td>15.0013</td>
      <td>10</td>
      <td>231</td>
      <td>241</td>
    </tr>
    <tr>
      <th>10883</th>
      <td>2012-12-19 21:00:00</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>13.94</td>
      <td>15.910</td>
      <td>61</td>
      <td>15.0013</td>
      <td>4</td>
      <td>164</td>
      <td>168</td>
    </tr>
    <tr>
      <th>10884</th>
      <td>2012-12-19 22:00:00</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>13.94</td>
      <td>17.425</td>
      <td>61</td>
      <td>6.0032</td>
      <td>12</td>
      <td>117</td>
      <td>129</td>
    </tr>
    <tr>
      <th>10885</th>
      <td>2012-12-19 23:00:00</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>13.12</td>
      <td>16.665</td>
      <td>66</td>
      <td>8.9981</td>
      <td>4</td>
      <td>84</td>
      <td>88</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-95d7a3f0-fce7-478a-ac94-0d5bddea5c15')"
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
          document.querySelector('#df-95d7a3f0-fce7-478a-ac94-0d5bddea5c15 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-95d7a3f0-fce7-478a-ac94-0d5bddea5c15');
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
bike.info() # 데이터 타입, 데이터 누락건수, 몇개의 컬럼, 몇개의 샘플
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10886 entries, 0 to 10885
    Data columns (total 12 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   datetime    10886 non-null  object 
     1   season      10886 non-null  int64  
     2   holiday     10886 non-null  int64  
     3   workingday  10886 non-null  int64  
     4   weather     10886 non-null  int64  
     5   temp        10886 non-null  float64
     6   atemp       10886 non-null  float64
     7   humidity    10886 non-null  int64  
     8   windspeed   10886 non-null  float64
     9   casual      10886 non-null  int64  
     10  registered  10886 non-null  int64  
     11  count       10886 non-null  int64  
    dtypes: float64(3), int64(8), object(1)
    memory usage: 1020.7+ KB
    

- 데이터 정제 (누락된 값 처리, 잘못된 데이터 타입)


```python
#bike.datetime
bike['datetime'] # bike 데이터프레임에서 datetime이라는 열의 값
```




    0        2011-01-01 00:00:00
    1        2011-01-01 01:00:00
    2        2011-01-01 02:00:00
    3        2011-01-01 03:00:00
    4        2011-01-01 04:00:00
                    ...         
    10881    2012-12-19 19:00:00
    10882    2012-12-19 20:00:00
    10883    2012-12-19 21:00:00
    10884    2012-12-19 22:00:00
    10885    2012-12-19 23:00:00
    Name: datetime, Length: 10886, dtype: object




```python
bike['datetime']= bike['datetime'].apply(pd.to_datetime)
```


```python
bike.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10886 entries, 0 to 10885
    Data columns (total 12 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   datetime    10886 non-null  datetime64[ns]
     1   season      10886 non-null  int64         
     2   holiday     10886 non-null  int64         
     3   workingday  10886 non-null  int64         
     4   weather     10886 non-null  int64         
     5   temp        10886 non-null  float64       
     6   atemp       10886 non-null  float64       
     7   humidity    10886 non-null  int64         
     8   windspeed   10886 non-null  float64       
     9   casual      10886 non-null  int64         
     10  registered  10886 non-null  int64         
     11  count       10886 non-null  int64         
    dtypes: datetime64[ns](1), float64(3), int64(8)
    memory usage: 1020.7 KB
    


```python
bike['year']= bike['datetime'].apply(lambda x: x.year)
bike['month']= bike['datetime'].apply(lambda x: x.month)
bike['hour']= bike['datetime'].apply(lambda x: x.hour)
bike['dayofweek']= bike['datetime'].apply(lambda x: x.dayofweek)
```


```python
bike.head()
```





  <div id="df-0ef7139d-5af4-4f33-941b-661b094e62ce">
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
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
      <th>year</th>
      <th>month</th>
      <th>hour</th>
      <th>dayofweek</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-01 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
      <td>2011</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-01 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-01 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
      <td>2011</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-01 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
      <td>2011</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-01 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2011</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-0ef7139d-5af4-4f33-941b-661b094e62ce')"
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
          document.querySelector('#df-0ef7139d-5af4-4f33-941b-661b094e62ce button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-0ef7139d-5af4-4f33-941b-661b094e62ce');
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




## Step 3: 데이터 탐색 (Exploratory Data Analysis)


**질문 1에 대한 분석** : 기상정보(온도, 체감온도, 풍속, 습도)와 자전거 대여량의 관계

**수치 데이터 특성간의 상관관계를 확인할 때**
- (1) 산점도로 확인
- (2) 상관계수 확인


```python
bike.plot(kind='scatter', x='temp', y='count', alpha=0.3)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff37f1ba820>




    
![png](/assets/images/2023-02-15-Data Analysis 1 (Overview)/output_31_1.png)
    



```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
axes[0][0].scatter(bike['temp'], bike['count'], alpha=0.3) # 온도와 대여량의 산점도
axes[0][1].scatter(bike['atemp'], bike['count'], alpha=0.3) # 체감온도와 대여량의 산점도
axes[1][0].scatter(bike['windspeed'], bike['count'], alpha=0.3) # 풍속과 대여량의 산점도
axes[1][1].scatter(bike['humidity'], bike['count'], alpha=0.3) # 습도와 대여량의 산점도
```




    <matplotlib.collections.PathCollection at 0x7ff37cadac10>




    
![png](/assets/images/2023-02-15-Data Analysis 1 (Overview)/output_32_1.png)
    



```python
bike.corr()
```





  <div id="df-21aa28c4-c90e-442a-8637-a75612bee304">
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
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
      <th>year</th>
      <th>month</th>
      <th>hour</th>
      <th>dayofweek</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>season</th>
      <td>1.000000</td>
      <td>0.029368</td>
      <td>-0.008126</td>
      <td>0.008879</td>
      <td>0.258689</td>
      <td>0.264744</td>
      <td>0.190610</td>
      <td>-0.147121</td>
      <td>0.096758</td>
      <td>0.164011</td>
      <td>0.163439</td>
      <td>-0.004797</td>
      <td>0.971524</td>
      <td>-0.006546</td>
      <td>-0.010553</td>
    </tr>
    <tr>
      <th>holiday</th>
      <td>0.029368</td>
      <td>1.000000</td>
      <td>-0.250491</td>
      <td>-0.007074</td>
      <td>0.000295</td>
      <td>-0.005215</td>
      <td>0.001929</td>
      <td>0.008409</td>
      <td>0.043799</td>
      <td>-0.020956</td>
      <td>-0.005393</td>
      <td>0.012021</td>
      <td>0.001731</td>
      <td>-0.000354</td>
      <td>-0.191832</td>
    </tr>
    <tr>
      <th>workingday</th>
      <td>-0.008126</td>
      <td>-0.250491</td>
      <td>1.000000</td>
      <td>0.033772</td>
      <td>0.029966</td>
      <td>0.024660</td>
      <td>-0.010880</td>
      <td>0.013373</td>
      <td>-0.319111</td>
      <td>0.119460</td>
      <td>0.011594</td>
      <td>-0.002482</td>
      <td>-0.003394</td>
      <td>0.002780</td>
      <td>-0.704267</td>
    </tr>
    <tr>
      <th>weather</th>
      <td>0.008879</td>
      <td>-0.007074</td>
      <td>0.033772</td>
      <td>1.000000</td>
      <td>-0.055035</td>
      <td>-0.055376</td>
      <td>0.406244</td>
      <td>0.007261</td>
      <td>-0.135918</td>
      <td>-0.109340</td>
      <td>-0.128655</td>
      <td>-0.012548</td>
      <td>0.012144</td>
      <td>-0.022740</td>
      <td>-0.047692</td>
    </tr>
    <tr>
      <th>temp</th>
      <td>0.258689</td>
      <td>0.000295</td>
      <td>0.029966</td>
      <td>-0.055035</td>
      <td>1.000000</td>
      <td>0.984948</td>
      <td>-0.064949</td>
      <td>-0.017852</td>
      <td>0.467097</td>
      <td>0.318571</td>
      <td>0.394454</td>
      <td>0.061226</td>
      <td>0.257589</td>
      <td>0.145430</td>
      <td>-0.038466</td>
    </tr>
    <tr>
      <th>atemp</th>
      <td>0.264744</td>
      <td>-0.005215</td>
      <td>0.024660</td>
      <td>-0.055376</td>
      <td>0.984948</td>
      <td>1.000000</td>
      <td>-0.043536</td>
      <td>-0.057473</td>
      <td>0.462067</td>
      <td>0.314635</td>
      <td>0.389784</td>
      <td>0.058540</td>
      <td>0.264173</td>
      <td>0.140343</td>
      <td>-0.040235</td>
    </tr>
    <tr>
      <th>humidity</th>
      <td>0.190610</td>
      <td>0.001929</td>
      <td>-0.010880</td>
      <td>0.406244</td>
      <td>-0.064949</td>
      <td>-0.043536</td>
      <td>1.000000</td>
      <td>-0.318607</td>
      <td>-0.348187</td>
      <td>-0.265458</td>
      <td>-0.317371</td>
      <td>-0.078606</td>
      <td>0.204537</td>
      <td>-0.278011</td>
      <td>-0.026507</td>
    </tr>
    <tr>
      <th>windspeed</th>
      <td>-0.147121</td>
      <td>0.008409</td>
      <td>0.013373</td>
      <td>0.007261</td>
      <td>-0.017852</td>
      <td>-0.057473</td>
      <td>-0.318607</td>
      <td>1.000000</td>
      <td>0.092276</td>
      <td>0.091052</td>
      <td>0.101369</td>
      <td>-0.015221</td>
      <td>-0.150192</td>
      <td>0.146631</td>
      <td>-0.024804</td>
    </tr>
    <tr>
      <th>casual</th>
      <td>0.096758</td>
      <td>0.043799</td>
      <td>-0.319111</td>
      <td>-0.135918</td>
      <td>0.467097</td>
      <td>0.462067</td>
      <td>-0.348187</td>
      <td>0.092276</td>
      <td>1.000000</td>
      <td>0.497250</td>
      <td>0.690414</td>
      <td>0.145241</td>
      <td>0.092722</td>
      <td>0.302045</td>
      <td>0.246959</td>
    </tr>
    <tr>
      <th>registered</th>
      <td>0.164011</td>
      <td>-0.020956</td>
      <td>0.119460</td>
      <td>-0.109340</td>
      <td>0.318571</td>
      <td>0.314635</td>
      <td>-0.265458</td>
      <td>0.091052</td>
      <td>0.497250</td>
      <td>1.000000</td>
      <td>0.970948</td>
      <td>0.264265</td>
      <td>0.169451</td>
      <td>0.380540</td>
      <td>-0.084427</td>
    </tr>
    <tr>
      <th>count</th>
      <td>0.163439</td>
      <td>-0.005393</td>
      <td>0.011594</td>
      <td>-0.128655</td>
      <td>0.394454</td>
      <td>0.389784</td>
      <td>-0.317371</td>
      <td>0.101369</td>
      <td>0.690414</td>
      <td>0.970948</td>
      <td>1.000000</td>
      <td>0.260403</td>
      <td>0.166862</td>
      <td>0.400601</td>
      <td>-0.002283</td>
    </tr>
    <tr>
      <th>year</th>
      <td>-0.004797</td>
      <td>0.012021</td>
      <td>-0.002482</td>
      <td>-0.012548</td>
      <td>0.061226</td>
      <td>0.058540</td>
      <td>-0.078606</td>
      <td>-0.015221</td>
      <td>0.145241</td>
      <td>0.264265</td>
      <td>0.260403</td>
      <td>1.000000</td>
      <td>-0.004932</td>
      <td>-0.004234</td>
      <td>-0.003785</td>
    </tr>
    <tr>
      <th>month</th>
      <td>0.971524</td>
      <td>0.001731</td>
      <td>-0.003394</td>
      <td>0.012144</td>
      <td>0.257589</td>
      <td>0.264173</td>
      <td>0.204537</td>
      <td>-0.150192</td>
      <td>0.092722</td>
      <td>0.169451</td>
      <td>0.166862</td>
      <td>-0.004932</td>
      <td>1.000000</td>
      <td>-0.006818</td>
      <td>-0.002266</td>
    </tr>
    <tr>
      <th>hour</th>
      <td>-0.006546</td>
      <td>-0.000354</td>
      <td>0.002780</td>
      <td>-0.022740</td>
      <td>0.145430</td>
      <td>0.140343</td>
      <td>-0.278011</td>
      <td>0.146631</td>
      <td>0.302045</td>
      <td>0.380540</td>
      <td>0.400601</td>
      <td>-0.004234</td>
      <td>-0.006818</td>
      <td>1.000000</td>
      <td>-0.002925</td>
    </tr>
    <tr>
      <th>dayofweek</th>
      <td>-0.010553</td>
      <td>-0.191832</td>
      <td>-0.704267</td>
      <td>-0.047692</td>
      <td>-0.038466</td>
      <td>-0.040235</td>
      <td>-0.026507</td>
      <td>-0.024804</td>
      <td>0.246959</td>
      <td>-0.084427</td>
      <td>-0.002283</td>
      <td>-0.003785</td>
      <td>-0.002266</td>
      <td>-0.002925</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-21aa28c4-c90e-442a-8637-a75612bee304')"
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
          document.querySelector('#df-21aa28c4-c90e-442a-8637-a75612bee304 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-21aa28c4-c90e-442a-8637-a75612bee304');
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




**분석 결과**
- 기상 정보 중 온도와 체감온도가 자건거 대여 수량에 영향을 미칠것으로 보임

**질문 2에 대한 분석** : 날짜정보(연도, 월, 시간, 요일)와 자전거 대여량의 관계

**참고**
- year, month, hour, dayofweek : 범주형 데이터
- count(자전거 대여량): 수치형 데이터
- 범주형 데이터 값에 따라 수치형 데이터가 어떻게 달라지는 파악할 때 막대그래프(barplot)


```python
import seaborn as sns
```


```python
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))

sns.barplot(data=bike, x='year', y='count', ax=axes[0][0])
sns.barplot(data=bike, x='month', y='count', ax=axes[0][1])
sns.barplot(data=bike, x='hour', y='count', hue='workingday', ax=axes[1][0])
sns.barplot(data=bike, x='dayofweek', y='count', ax=axes[1][1])
plt.show()
```


    
![png](/assets/images/2023-02-15-Data Analysis 1 (Overview)/output_38_0.png)
    


**분석 결과**
- 연도별 평균 대여량은 2011년도보다 2012년도에 더 많음
- 월별 평균 대여량은 6월에 가장 많고, 7~10월에도 많음. 1월에 가장 적음
- 시간대별 평균 대여량은 오전 8시 전후와 오후 5~6시 부근에 많음
- 시간대별 평균 대여량을 workingday로 나누어서 시각화하면 휴일과 근무일의 대여량 추이가 다름을 알 수 있음

## Step 4: 결론 도출 또는 예측 (Draw conclusions or make predictions)


- 질문1, 질문2에 대한 분석결과를 확인
- 온도에 따른 자전거 대여량 변화가 예상이 되므로 이에 맞는 재고 관리 전략 수립
- 시기별(연도, 월, 시간)로 대여량 변화가 예상이 되므로 이제 맞는 프로모션 전략 수립

## Step 5: 결과 공유 (Communicate the results)

자전거 대여량을 예측할 때 고려해야할 중요한 특성(기상정보, 시기)을 설명하는 보고서, PPT등을 준비
