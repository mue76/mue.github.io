---
tag: [pandas, data analysis]
toc: true
toc_sticky: true
toc_label: 목차
author_profile: false
sidebar:
  nav: "counts"
---

# Pandas (4)

## 10 시계열 데이터 다루기


```python
import pandas as pd
```

#### (1) 다른 자료형을 시계열 객체로 변환

- 문자열을 Timestamp로 변환


```python
df = pd.read_csv('./examples/stock-data.csv')
df.head()
```





  <div id="df-bce5e52d-1f2a-4918-be67-072da4b7b3e5">
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
      <th>Date</th>
      <th>Close</th>
      <th>Start</th>
      <th>High</th>
      <th>Low</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-07-02</td>
      <td>10100</td>
      <td>10850</td>
      <td>10900</td>
      <td>10000</td>
      <td>137977</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-06-29</td>
      <td>10700</td>
      <td>10550</td>
      <td>10900</td>
      <td>9990</td>
      <td>170253</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-06-28</td>
      <td>10400</td>
      <td>10900</td>
      <td>10950</td>
      <td>10150</td>
      <td>155769</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-06-27</td>
      <td>10900</td>
      <td>10800</td>
      <td>11050</td>
      <td>10500</td>
      <td>133548</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-06-26</td>
      <td>10800</td>
      <td>10900</td>
      <td>11000</td>
      <td>10700</td>
      <td>63039</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-bce5e52d-1f2a-4918-be67-072da4b7b3e5')"
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
          document.querySelector('#df-bce5e52d-1f2a-4918-be67-072da4b7b3e5 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-bce5e52d-1f2a-4918-be67-072da4b7b3e5');
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
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20 entries, 0 to 19
    Data columns (total 6 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   Date    20 non-null     object
     1   Close   20 non-null     int64 
     2   Start   20 non-null     int64 
     3   High    20 non-null     int64 
     4   Low     20 non-null     int64 
     5   Volume  20 non-null     int64 
    dtypes: int64(5), object(1)
    memory usage: 1.1+ KB
    


```python
# 문자열 데이터를 Timestamp 로 변환환
# df['new_Date']= pd.to_datetime(df['Date'])
df['new_Date']= df['Date'].apply(pd.to_datetime)
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20 entries, 0 to 19
    Data columns (total 7 columns):
     #   Column    Non-Null Count  Dtype         
    ---  ------    --------------  -----         
     0   Date      20 non-null     object        
     1   Close     20 non-null     int64         
     2   Start     20 non-null     int64         
     3   High      20 non-null     int64         
     4   Low       20 non-null     int64         
     5   Volume    20 non-null     int64         
     6   new_Date  20 non-null     datetime64[ns]
    dtypes: datetime64[ns](1), int64(5), object(1)
    memory usage: 1.2+ KB
    


```python
df['new_Date'][0], type(df['new_Date'][0]) # new_Date 컬럼의 행 하나의 값과 타입 확인인
```




    (Timestamp('2018-07-02 00:00:00'), pandas._libs.tslibs.timestamps.Timestamp)




```python
df.head()
```





  <div id="df-105aead3-657c-40f2-a3ef-64145a937f82">
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
      <th>Date</th>
      <th>Close</th>
      <th>Start</th>
      <th>High</th>
      <th>Low</th>
      <th>Volume</th>
      <th>new_Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-07-02</td>
      <td>10100</td>
      <td>10850</td>
      <td>10900</td>
      <td>10000</td>
      <td>137977</td>
      <td>2018-07-02</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-06-29</td>
      <td>10700</td>
      <td>10550</td>
      <td>10900</td>
      <td>9990</td>
      <td>170253</td>
      <td>2018-06-29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-06-28</td>
      <td>10400</td>
      <td>10900</td>
      <td>10950</td>
      <td>10150</td>
      <td>155769</td>
      <td>2018-06-28</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-06-27</td>
      <td>10900</td>
      <td>10800</td>
      <td>11050</td>
      <td>10500</td>
      <td>133548</td>
      <td>2018-06-27</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-06-26</td>
      <td>10800</td>
      <td>10900</td>
      <td>11000</td>
      <td>10700</td>
      <td>63039</td>
      <td>2018-06-26</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-105aead3-657c-40f2-a3ef-64145a937f82')"
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
          document.querySelector('#df-105aead3-657c-40f2-a3ef-64145a937f82 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-105aead3-657c-40f2-a3ef-64145a937f82');
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
df.set_index('new_Date', inplace=True)
df.drop('Date', axis=1, inplace=True)
df.head()
```





  <div id="df-b3a23a21-9af0-4a61-8d51-49c8b639d63e">
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
      <th>Close</th>
      <th>Start</th>
      <th>High</th>
      <th>Low</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>new_Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-07-02</th>
      <td>10100</td>
      <td>10850</td>
      <td>10900</td>
      <td>10000</td>
      <td>137977</td>
    </tr>
    <tr>
      <th>2018-06-29</th>
      <td>10700</td>
      <td>10550</td>
      <td>10900</td>
      <td>9990</td>
      <td>170253</td>
    </tr>
    <tr>
      <th>2018-06-28</th>
      <td>10400</td>
      <td>10900</td>
      <td>10950</td>
      <td>10150</td>
      <td>155769</td>
    </tr>
    <tr>
      <th>2018-06-27</th>
      <td>10900</td>
      <td>10800</td>
      <td>11050</td>
      <td>10500</td>
      <td>133548</td>
    </tr>
    <tr>
      <th>2018-06-26</th>
      <td>10800</td>
      <td>10900</td>
      <td>11000</td>
      <td>10700</td>
      <td>63039</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-b3a23a21-9af0-4a61-8d51-49c8b639d63e')"
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
          document.querySelector('#df-b3a23a21-9af0-4a61-8d51-49c8b639d63e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-b3a23a21-9af0-4a61-8d51-49c8b639d63e');
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
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 20 entries, 2018-07-02 to 2018-06-01
    Data columns (total 5 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   Close   20 non-null     int64
     1   Start   20 non-null     int64
     2   High    20 non-null     int64
     3   Low     20 non-null     int64
     4   Volume  20 non-null     int64
    dtypes: int64(5)
    memory usage: 960.0 bytes
    


```python
df.index
```




    DatetimeIndex(['2018-07-02', '2018-06-29', '2018-06-28', '2018-06-27',
                   '2018-06-26', '2018-06-25', '2018-06-22', '2018-06-21',
                   '2018-06-20', '2018-06-19', '2018-06-18', '2018-06-15',
                   '2018-06-14', '2018-06-12', '2018-06-11', '2018-06-08',
                   '2018-06-07', '2018-06-05', '2018-06-04', '2018-06-01'],
                  dtype='datetime64[ns]', name='new_Date', freq=None)



- TimeStamp를 Period로 변환


```python
df.index.to_period(freq='D') # day (1일)
```




    PeriodIndex(['2018-07-02', '2018-06-29', '2018-06-28', '2018-06-27',
                 '2018-06-26', '2018-06-25', '2018-06-22', '2018-06-21',
                 '2018-06-20', '2018-06-19', '2018-06-18', '2018-06-15',
                 '2018-06-14', '2018-06-12', '2018-06-11', '2018-06-08',
                 '2018-06-07', '2018-06-05', '2018-06-04', '2018-06-01'],
                dtype='period[D]', name='new_Date')




```python
df.index.to_period(freq='M') # month end
```




    PeriodIndex(['2018-07', '2018-06', '2018-06', '2018-06', '2018-06', '2018-06',
                 '2018-06', '2018-06', '2018-06', '2018-06', '2018-06', '2018-06',
                 '2018-06', '2018-06', '2018-06', '2018-06', '2018-06', '2018-06',
                 '2018-06', '2018-06'],
                dtype='period[M]', name='new_Date')




```python
df.index.to_period(freq='A') # year end
```




    PeriodIndex(['2018', '2018', '2018', '2018', '2018', '2018', '2018', '2018',
                 '2018', '2018', '2018', '2018', '2018', '2018', '2018', '2018',
                 '2018', '2018', '2018', '2018'],
                dtype='period[A-DEC]', name='new_Date')



#### (2) 시계열 데이터 만들기

- TimeStamp 배열


```python
ts_ms = pd.date_range(start='2023-01-01', # 날짜 범위의 시작
              end=None,           # 날짜 범위의 끝
              periods = 10,       # 생성할 Timestamp의 개수수
              freq='MS',          # 시간 간격 (MS : 월의 시작일)
              tz='Asia/Seoul')    # 시간대 (timezone)
ts_ms              
```




    DatetimeIndex(['2023-01-01 00:00:00+09:00', '2023-02-01 00:00:00+09:00',
                   '2023-03-01 00:00:00+09:00', '2023-04-01 00:00:00+09:00',
                   '2023-05-01 00:00:00+09:00', '2023-06-01 00:00:00+09:00',
                   '2023-07-01 00:00:00+09:00', '2023-08-01 00:00:00+09:00',
                   '2023-09-01 00:00:00+09:00', '2023-10-01 00:00:00+09:00'],
                  dtype='datetime64[ns, Asia/Seoul]', freq='MS')




```python
ts_me = pd.date_range(start='2023-01-01', # 날짜 범위의 시작
              end=None,           # 날짜 범위의 끝
              periods = 10,       # 생성할 Timestamp의 개수수
              freq='M',          # 시간 간격 (MS : 월의 마지막날날)
              tz='Asia/Seoul')    # 시간대 (timezone)
ts_me  
```




    DatetimeIndex(['2023-01-31 00:00:00+09:00', '2023-02-28 00:00:00+09:00',
                   '2023-03-31 00:00:00+09:00', '2023-04-30 00:00:00+09:00',
                   '2023-05-31 00:00:00+09:00', '2023-06-30 00:00:00+09:00',
                   '2023-07-31 00:00:00+09:00', '2023-08-31 00:00:00+09:00',
                   '2023-09-30 00:00:00+09:00', '2023-10-31 00:00:00+09:00'],
                  dtype='datetime64[ns, Asia/Seoul]', freq='M')




```python
# 분기(3개월) 간격, 월의 마지막 날 기준준
ts_3m = pd.date_range(start='2023-01-01', # 날짜 범위의 시작
              end=None,           # 날짜 범위의 끝
              periods = 10,       # 생성할 Timestamp의 개수수
              freq='3M',          # 시간 간격 (3M : 3개월)
              tz='Asia/Seoul')    # 시간대 (timezone)
ts_3m  
```




    DatetimeIndex(['2023-01-31 00:00:00+09:00', '2023-04-30 00:00:00+09:00',
                   '2023-07-31 00:00:00+09:00', '2023-10-31 00:00:00+09:00',
                   '2024-01-31 00:00:00+09:00', '2024-04-30 00:00:00+09:00',
                   '2024-07-31 00:00:00+09:00', '2024-10-31 00:00:00+09:00',
                   '2025-01-31 00:00:00+09:00', '2025-04-30 00:00:00+09:00'],
                  dtype='datetime64[ns, Asia/Seoul]', freq='3M')



- Period 배열


```python
# 1개월 길이로 Period 배열 만들기기
pr_m = pd.period_range(start='2023-01-01', # 날짜 범위의 시작
                       end = None,         # 날짜 범위의 끝
                       periods=3,           # 생성할 Period 개수
                       freq='M')           # 기간의 길이 (M:월)
pr_m                       
```




    PeriodIndex(['2023-01', '2023-02', '2023-03'], dtype='period[M]')




```python
# 1시간 길이로 Period 배열 만들기기
pr_h = pd.period_range(start='2023-01-01', # 날짜 범위의 시작
                       end = None,         # 날짜 범위의 끝
                       periods=3,           # 생성할 Period 개수
                       freq='H')           # 기간의 길이 (H:시간)
pr_h  
```




    PeriodIndex(['2023-01-01 00:00', '2023-01-01 01:00', '2023-01-01 02:00'], dtype='period[H]')




```python
# 2시간 길이로 Period 배열 만들기기
pr_2h = pd.period_range(start='2023-01-01', # 날짜 범위의 시작
                       end = None,         # 날짜 범위의 끝
                       periods=3,           # 생성할 Period 개수
                       freq='2H')           # 기간의 길이 (H:시간)
pr_2h  
```




    PeriodIndex(['2023-01-01 00:00', '2023-01-01 02:00', '2023-01-01 04:00'], dtype='period[2H]')



#### (3) 시계열 데이터 활용

- 날짜 데이터 분리


```python
df = pd.read_csv('./examples/stock-data.csv')
df.head()
```





  <div id="df-6fd7af78-1b3d-4b9f-b7e7-31da5985559e">
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
      <th>Date</th>
      <th>Close</th>
      <th>Start</th>
      <th>High</th>
      <th>Low</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-07-02</td>
      <td>10100</td>
      <td>10850</td>
      <td>10900</td>
      <td>10000</td>
      <td>137977</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-06-29</td>
      <td>10700</td>
      <td>10550</td>
      <td>10900</td>
      <td>9990</td>
      <td>170253</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-06-28</td>
      <td>10400</td>
      <td>10900</td>
      <td>10950</td>
      <td>10150</td>
      <td>155769</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-06-27</td>
      <td>10900</td>
      <td>10800</td>
      <td>11050</td>
      <td>10500</td>
      <td>133548</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-06-26</td>
      <td>10800</td>
      <td>10900</td>
      <td>11000</td>
      <td>10700</td>
      <td>63039</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-6fd7af78-1b3d-4b9f-b7e7-31da5985559e')"
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
          document.querySelector('#df-6fd7af78-1b3d-4b9f-b7e7-31da5985559e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-6fd7af78-1b3d-4b9f-b7e7-31da5985559e');
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
df['new_Date'] = pd.to_datetime(df['Date'])
```


```python
# dt 속성
df['Year']= df['new_Date'].dt.year
df['Month'] = df['new_Date'].dt.month
df['Day'] = df['new_Date'].dt.day
df.head()
```





  <div id="df-b51b1269-ba54-433f-85cb-8de559b9c71a">
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
      <th>Date</th>
      <th>Close</th>
      <th>Start</th>
      <th>High</th>
      <th>Low</th>
      <th>Volume</th>
      <th>new_Date</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-07-02</td>
      <td>10100</td>
      <td>10850</td>
      <td>10900</td>
      <td>10000</td>
      <td>137977</td>
      <td>2018-07-02</td>
      <td>2018</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-06-29</td>
      <td>10700</td>
      <td>10550</td>
      <td>10900</td>
      <td>9990</td>
      <td>170253</td>
      <td>2018-06-29</td>
      <td>2018</td>
      <td>6</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-06-28</td>
      <td>10400</td>
      <td>10900</td>
      <td>10950</td>
      <td>10150</td>
      <td>155769</td>
      <td>2018-06-28</td>
      <td>2018</td>
      <td>6</td>
      <td>28</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-06-27</td>
      <td>10900</td>
      <td>10800</td>
      <td>11050</td>
      <td>10500</td>
      <td>133548</td>
      <td>2018-06-27</td>
      <td>2018</td>
      <td>6</td>
      <td>27</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-06-26</td>
      <td>10800</td>
      <td>10900</td>
      <td>11000</td>
      <td>10700</td>
      <td>63039</td>
      <td>2018-06-26</td>
      <td>2018</td>
      <td>6</td>
      <td>26</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-b51b1269-ba54-433f-85cb-8de559b9c71a')"
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
          document.querySelector('#df-b51b1269-ba54-433f-85cb-8de559b9c71a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-b51b1269-ba54-433f-85cb-8de559b9c71a');
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
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20 entries, 0 to 19
    Data columns (total 10 columns):
     #   Column    Non-Null Count  Dtype         
    ---  ------    --------------  -----         
     0   Date      20 non-null     object        
     1   Close     20 non-null     int64         
     2   Start     20 non-null     int64         
     3   High      20 non-null     int64         
     4   Low       20 non-null     int64         
     5   Volume    20 non-null     int64         
     6   new_Date  20 non-null     datetime64[ns]
     7   Year      20 non-null     int64         
     8   Month     20 non-null     int64         
     9   Day       20 non-null     int64         
    dtypes: datetime64[ns](1), int64(8), object(1)
    memory usage: 1.7+ KB
    


```python
# Timestamp를 Period로 변환하여 년월일 표기 변경하기
df['Date_yr'] = df['new_Date'].dt.to_period(freq='A')
df['Date_m'] = df['new_Date'].dt.to_period(freq='M')
df.head()
```





  <div id="df-f59aa56d-d4a3-450d-8540-4d321b57bf13">
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
      <th>Date</th>
      <th>Close</th>
      <th>Start</th>
      <th>High</th>
      <th>Low</th>
      <th>Volume</th>
      <th>new_Date</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>Date_yr</th>
      <th>Date_m</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-07-02</td>
      <td>10100</td>
      <td>10850</td>
      <td>10900</td>
      <td>10000</td>
      <td>137977</td>
      <td>2018-07-02</td>
      <td>2018</td>
      <td>7</td>
      <td>2</td>
      <td>2018</td>
      <td>2018-07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-06-29</td>
      <td>10700</td>
      <td>10550</td>
      <td>10900</td>
      <td>9990</td>
      <td>170253</td>
      <td>2018-06-29</td>
      <td>2018</td>
      <td>6</td>
      <td>29</td>
      <td>2018</td>
      <td>2018-06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-06-28</td>
      <td>10400</td>
      <td>10900</td>
      <td>10950</td>
      <td>10150</td>
      <td>155769</td>
      <td>2018-06-28</td>
      <td>2018</td>
      <td>6</td>
      <td>28</td>
      <td>2018</td>
      <td>2018-06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-06-27</td>
      <td>10900</td>
      <td>10800</td>
      <td>11050</td>
      <td>10500</td>
      <td>133548</td>
      <td>2018-06-27</td>
      <td>2018</td>
      <td>6</td>
      <td>27</td>
      <td>2018</td>
      <td>2018-06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-06-26</td>
      <td>10800</td>
      <td>10900</td>
      <td>11000</td>
      <td>10700</td>
      <td>63039</td>
      <td>2018-06-26</td>
      <td>2018</td>
      <td>6</td>
      <td>26</td>
      <td>2018</td>
      <td>2018-06</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-f59aa56d-d4a3-450d-8540-4d321b57bf13')"
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
          document.querySelector('#df-f59aa56d-d4a3-450d-8540-4d321b57bf13 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-f59aa56d-d4a3-450d-8540-4d321b57bf13');
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
# 원하는 열을 새로운 행 인덱스로 지정
df.set_index('Date_m', inplace=True)
df.head()
```





  <div id="df-bf30ec62-2f84-46a7-9c0d-54fb4ffaa57c">
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
      <th>Date</th>
      <th>Close</th>
      <th>Start</th>
      <th>High</th>
      <th>Low</th>
      <th>Volume</th>
      <th>new_Date</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>Date_yr</th>
    </tr>
    <tr>
      <th>Date_m</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-07</th>
      <td>2018-07-02</td>
      <td>10100</td>
      <td>10850</td>
      <td>10900</td>
      <td>10000</td>
      <td>137977</td>
      <td>2018-07-02</td>
      <td>2018</td>
      <td>7</td>
      <td>2</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>2018-06</th>
      <td>2018-06-29</td>
      <td>10700</td>
      <td>10550</td>
      <td>10900</td>
      <td>9990</td>
      <td>170253</td>
      <td>2018-06-29</td>
      <td>2018</td>
      <td>6</td>
      <td>29</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>2018-06</th>
      <td>2018-06-28</td>
      <td>10400</td>
      <td>10900</td>
      <td>10950</td>
      <td>10150</td>
      <td>155769</td>
      <td>2018-06-28</td>
      <td>2018</td>
      <td>6</td>
      <td>28</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>2018-06</th>
      <td>2018-06-27</td>
      <td>10900</td>
      <td>10800</td>
      <td>11050</td>
      <td>10500</td>
      <td>133548</td>
      <td>2018-06-27</td>
      <td>2018</td>
      <td>6</td>
      <td>27</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>2018-06</th>
      <td>2018-06-26</td>
      <td>10800</td>
      <td>10900</td>
      <td>11000</td>
      <td>10700</td>
      <td>63039</td>
      <td>2018-06-26</td>
      <td>2018</td>
      <td>6</td>
      <td>26</td>
      <td>2018</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-bf30ec62-2f84-46a7-9c0d-54fb4ffaa57c')"
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
          document.querySelector('#df-bf30ec62-2f84-46a7-9c0d-54fb4ffaa57c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-bf30ec62-2f84-46a7-9c0d-54fb4ffaa57c');
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




- 날짜 인덱스 활용


```python
df = pd.read_csv('./examples/stock-data.csv')
df['new_Date'] = pd.to_datetime(df['Date'])
df.set_index('new_Date', inplace=True)
df.head()
```





  <div id="df-97a2c80f-c75e-4ef4-abb0-2e045899e8da">
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
      <th>Date</th>
      <th>Close</th>
      <th>Start</th>
      <th>High</th>
      <th>Low</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>new_Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-07-02</th>
      <td>2018-07-02</td>
      <td>10100</td>
      <td>10850</td>
      <td>10900</td>
      <td>10000</td>
      <td>137977</td>
    </tr>
    <tr>
      <th>2018-06-29</th>
      <td>2018-06-29</td>
      <td>10700</td>
      <td>10550</td>
      <td>10900</td>
      <td>9990</td>
      <td>170253</td>
    </tr>
    <tr>
      <th>2018-06-28</th>
      <td>2018-06-28</td>
      <td>10400</td>
      <td>10900</td>
      <td>10950</td>
      <td>10150</td>
      <td>155769</td>
    </tr>
    <tr>
      <th>2018-06-27</th>
      <td>2018-06-27</td>
      <td>10900</td>
      <td>10800</td>
      <td>11050</td>
      <td>10500</td>
      <td>133548</td>
    </tr>
    <tr>
      <th>2018-06-26</th>
      <td>2018-06-26</td>
      <td>10800</td>
      <td>10900</td>
      <td>11000</td>
      <td>10700</td>
      <td>63039</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-97a2c80f-c75e-4ef4-abb0-2e045899e8da')"
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
          document.querySelector('#df-97a2c80f-c75e-4ef4-abb0-2e045899e8da button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-97a2c80f-c75e-4ef4-abb0-2e045899e8da');
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
df.index
```




    DatetimeIndex(['2018-07-02', '2018-06-29', '2018-06-28', '2018-06-27',
                   '2018-06-26', '2018-06-25', '2018-06-22', '2018-06-21',
                   '2018-06-20', '2018-06-19', '2018-06-18', '2018-06-15',
                   '2018-06-14', '2018-06-12', '2018-06-11', '2018-06-08',
                   '2018-06-07', '2018-06-05', '2018-06-04', '2018-06-01'],
                  dtype='datetime64[ns]', name='new_Date', freq=None)




```python
df.loc['2018']
```





  <div id="df-51bb66ef-1f52-4d13-a255-df9f6aca3051">
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
      <th>Date</th>
      <th>Close</th>
      <th>Start</th>
      <th>High</th>
      <th>Low</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>new_Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-07-02</th>
      <td>2018-07-02</td>
      <td>10100</td>
      <td>10850</td>
      <td>10900</td>
      <td>10000</td>
      <td>137977</td>
    </tr>
    <tr>
      <th>2018-06-29</th>
      <td>2018-06-29</td>
      <td>10700</td>
      <td>10550</td>
      <td>10900</td>
      <td>9990</td>
      <td>170253</td>
    </tr>
    <tr>
      <th>2018-06-28</th>
      <td>2018-06-28</td>
      <td>10400</td>
      <td>10900</td>
      <td>10950</td>
      <td>10150</td>
      <td>155769</td>
    </tr>
    <tr>
      <th>2018-06-27</th>
      <td>2018-06-27</td>
      <td>10900</td>
      <td>10800</td>
      <td>11050</td>
      <td>10500</td>
      <td>133548</td>
    </tr>
    <tr>
      <th>2018-06-26</th>
      <td>2018-06-26</td>
      <td>10800</td>
      <td>10900</td>
      <td>11000</td>
      <td>10700</td>
      <td>63039</td>
    </tr>
    <tr>
      <th>2018-06-25</th>
      <td>2018-06-25</td>
      <td>11150</td>
      <td>11400</td>
      <td>11450</td>
      <td>11000</td>
      <td>55519</td>
    </tr>
    <tr>
      <th>2018-06-22</th>
      <td>2018-06-22</td>
      <td>11300</td>
      <td>11250</td>
      <td>11450</td>
      <td>10750</td>
      <td>134805</td>
    </tr>
    <tr>
      <th>2018-06-21</th>
      <td>2018-06-21</td>
      <td>11200</td>
      <td>11350</td>
      <td>11750</td>
      <td>11200</td>
      <td>133002</td>
    </tr>
    <tr>
      <th>2018-06-20</th>
      <td>2018-06-20</td>
      <td>11550</td>
      <td>11200</td>
      <td>11600</td>
      <td>10900</td>
      <td>308596</td>
    </tr>
    <tr>
      <th>2018-06-19</th>
      <td>2018-06-19</td>
      <td>11300</td>
      <td>11850</td>
      <td>11950</td>
      <td>11300</td>
      <td>180656</td>
    </tr>
    <tr>
      <th>2018-06-18</th>
      <td>2018-06-18</td>
      <td>12000</td>
      <td>13400</td>
      <td>13400</td>
      <td>12000</td>
      <td>309787</td>
    </tr>
    <tr>
      <th>2018-06-15</th>
      <td>2018-06-15</td>
      <td>13400</td>
      <td>13600</td>
      <td>13600</td>
      <td>12900</td>
      <td>201376</td>
    </tr>
    <tr>
      <th>2018-06-14</th>
      <td>2018-06-14</td>
      <td>13450</td>
      <td>13200</td>
      <td>13700</td>
      <td>13150</td>
      <td>347451</td>
    </tr>
    <tr>
      <th>2018-06-12</th>
      <td>2018-06-12</td>
      <td>13200</td>
      <td>12200</td>
      <td>13300</td>
      <td>12050</td>
      <td>558148</td>
    </tr>
    <tr>
      <th>2018-06-11</th>
      <td>2018-06-11</td>
      <td>11950</td>
      <td>12000</td>
      <td>12250</td>
      <td>11950</td>
      <td>62293</td>
    </tr>
    <tr>
      <th>2018-06-08</th>
      <td>2018-06-08</td>
      <td>11950</td>
      <td>11950</td>
      <td>12200</td>
      <td>11800</td>
      <td>59258</td>
    </tr>
    <tr>
      <th>2018-06-07</th>
      <td>2018-06-07</td>
      <td>11950</td>
      <td>12200</td>
      <td>12300</td>
      <td>11900</td>
      <td>49088</td>
    </tr>
    <tr>
      <th>2018-06-05</th>
      <td>2018-06-05</td>
      <td>12150</td>
      <td>11800</td>
      <td>12250</td>
      <td>11800</td>
      <td>42485</td>
    </tr>
    <tr>
      <th>2018-06-04</th>
      <td>2018-06-04</td>
      <td>11900</td>
      <td>11900</td>
      <td>12200</td>
      <td>11700</td>
      <td>25171</td>
    </tr>
    <tr>
      <th>2018-06-01</th>
      <td>2018-06-01</td>
      <td>11900</td>
      <td>11800</td>
      <td>12100</td>
      <td>11750</td>
      <td>32062</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-51bb66ef-1f52-4d13-a255-df9f6aca3051')"
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
          document.querySelector('#df-51bb66ef-1f52-4d13-a255-df9f6aca3051 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-51bb66ef-1f52-4d13-a255-df9f6aca3051');
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
df.loc['2018', 'Start':'High'] # 열 범위 추가로 슬라이싱
```





  <div id="df-828da512-c13a-4bea-8959-f0b3d6267dde">
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
      <th>Start</th>
      <th>High</th>
    </tr>
    <tr>
      <th>new_Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-07-02</th>
      <td>10850</td>
      <td>10900</td>
    </tr>
    <tr>
      <th>2018-06-29</th>
      <td>10550</td>
      <td>10900</td>
    </tr>
    <tr>
      <th>2018-06-28</th>
      <td>10900</td>
      <td>10950</td>
    </tr>
    <tr>
      <th>2018-06-27</th>
      <td>10800</td>
      <td>11050</td>
    </tr>
    <tr>
      <th>2018-06-26</th>
      <td>10900</td>
      <td>11000</td>
    </tr>
    <tr>
      <th>2018-06-25</th>
      <td>11400</td>
      <td>11450</td>
    </tr>
    <tr>
      <th>2018-06-22</th>
      <td>11250</td>
      <td>11450</td>
    </tr>
    <tr>
      <th>2018-06-21</th>
      <td>11350</td>
      <td>11750</td>
    </tr>
    <tr>
      <th>2018-06-20</th>
      <td>11200</td>
      <td>11600</td>
    </tr>
    <tr>
      <th>2018-06-19</th>
      <td>11850</td>
      <td>11950</td>
    </tr>
    <tr>
      <th>2018-06-18</th>
      <td>13400</td>
      <td>13400</td>
    </tr>
    <tr>
      <th>2018-06-15</th>
      <td>13600</td>
      <td>13600</td>
    </tr>
    <tr>
      <th>2018-06-14</th>
      <td>13200</td>
      <td>13700</td>
    </tr>
    <tr>
      <th>2018-06-12</th>
      <td>12200</td>
      <td>13300</td>
    </tr>
    <tr>
      <th>2018-06-11</th>
      <td>12000</td>
      <td>12250</td>
    </tr>
    <tr>
      <th>2018-06-08</th>
      <td>11950</td>
      <td>12200</td>
    </tr>
    <tr>
      <th>2018-06-07</th>
      <td>12200</td>
      <td>12300</td>
    </tr>
    <tr>
      <th>2018-06-05</th>
      <td>11800</td>
      <td>12250</td>
    </tr>
    <tr>
      <th>2018-06-04</th>
      <td>11900</td>
      <td>12200</td>
    </tr>
    <tr>
      <th>2018-06-01</th>
      <td>11800</td>
      <td>12100</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-828da512-c13a-4bea-8959-f0b3d6267dde')"
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
          document.querySelector('#df-828da512-c13a-4bea-8959-f0b3d6267dde button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-828da512-c13a-4bea-8959-f0b3d6267dde');
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
df.loc['2018-06-07']
```





  <div id="df-52e670ac-e949-4fbf-8b9f-bac90fcc294e">
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
      <th>Date</th>
      <th>Close</th>
      <th>Start</th>
      <th>High</th>
      <th>Low</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>new_Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-06-07</th>
      <td>2018-06-07</td>
      <td>11950</td>
      <td>12200</td>
      <td>12300</td>
      <td>11900</td>
      <td>49088</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-52e670ac-e949-4fbf-8b9f-bac90fcc294e')"
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
          document.querySelector('#df-52e670ac-e949-4fbf-8b9f-bac90fcc294e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-52e670ac-e949-4fbf-8b9f-bac90fcc294e');
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
df.loc['2018-06-07':'2018-06-19']
```





  <div id="df-7c5b469b-f735-4985-bef4-86b8c3358da5">
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
      <th>Date</th>
      <th>Close</th>
      <th>Start</th>
      <th>High</th>
      <th>Low</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>new_Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-06-19</th>
      <td>2018-06-19</td>
      <td>11300</td>
      <td>11850</td>
      <td>11950</td>
      <td>11300</td>
      <td>180656</td>
    </tr>
    <tr>
      <th>2018-06-18</th>
      <td>2018-06-18</td>
      <td>12000</td>
      <td>13400</td>
      <td>13400</td>
      <td>12000</td>
      <td>309787</td>
    </tr>
    <tr>
      <th>2018-06-15</th>
      <td>2018-06-15</td>
      <td>13400</td>
      <td>13600</td>
      <td>13600</td>
      <td>12900</td>
      <td>201376</td>
    </tr>
    <tr>
      <th>2018-06-14</th>
      <td>2018-06-14</td>
      <td>13450</td>
      <td>13200</td>
      <td>13700</td>
      <td>13150</td>
      <td>347451</td>
    </tr>
    <tr>
      <th>2018-06-12</th>
      <td>2018-06-12</td>
      <td>13200</td>
      <td>12200</td>
      <td>13300</td>
      <td>12050</td>
      <td>558148</td>
    </tr>
    <tr>
      <th>2018-06-11</th>
      <td>2018-06-11</td>
      <td>11950</td>
      <td>12000</td>
      <td>12250</td>
      <td>11950</td>
      <td>62293</td>
    </tr>
    <tr>
      <th>2018-06-08</th>
      <td>2018-06-08</td>
      <td>11950</td>
      <td>11950</td>
      <td>12200</td>
      <td>11800</td>
      <td>59258</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-7c5b469b-f735-4985-bef4-86b8c3358da5')"
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
          document.querySelector('#df-7c5b469b-f735-4985-bef4-86b8c3358da5 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7c5b469b-f735-4985-bef4-86b8c3358da5');
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
today = pd.to_datetime('2018-12-25') # 기준일 생성
```


```python
df.index
```




    DatetimeIndex(['2018-07-02', '2018-06-29', '2018-06-28', '2018-06-27',
                   '2018-06-26', '2018-06-25', '2018-06-22', '2018-06-21',
                   '2018-06-20', '2018-06-19', '2018-06-18', '2018-06-15',
                   '2018-06-14', '2018-06-12', '2018-06-11', '2018-06-08',
                   '2018-06-07', '2018-06-05', '2018-06-04', '2018-06-01'],
                  dtype='datetime64[ns]', name='new_Date', freq=None)




```python
today - df.index # 기준일로부터 날짜 차이 생성
```




    TimedeltaIndex(['176 days', '179 days', '180 days', '181 days', '182 days',
                    '183 days', '186 days', '187 days', '188 days', '189 days',
                    '190 days', '193 days', '194 days', '196 days', '197 days',
                    '200 days', '201 days', '203 days', '204 days', '207 days'],
                   dtype='timedelta64[ns]', name='new_Date', freq=None)




```python
df['time_delta']= today - df.index
df.set_index('time_delta', inplace=True)
df.head()
```





  <div id="df-bbb26cf6-452c-4cf6-b5f3-e8d92f73c4c7">
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
      <th>Date</th>
      <th>Close</th>
      <th>Start</th>
      <th>High</th>
      <th>Low</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>time_delta</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>176 days</th>
      <td>2018-07-02</td>
      <td>10100</td>
      <td>10850</td>
      <td>10900</td>
      <td>10000</td>
      <td>137977</td>
    </tr>
    <tr>
      <th>179 days</th>
      <td>2018-06-29</td>
      <td>10700</td>
      <td>10550</td>
      <td>10900</td>
      <td>9990</td>
      <td>170253</td>
    </tr>
    <tr>
      <th>180 days</th>
      <td>2018-06-28</td>
      <td>10400</td>
      <td>10900</td>
      <td>10950</td>
      <td>10150</td>
      <td>155769</td>
    </tr>
    <tr>
      <th>181 days</th>
      <td>2018-06-27</td>
      <td>10900</td>
      <td>10800</td>
      <td>11050</td>
      <td>10500</td>
      <td>133548</td>
    </tr>
    <tr>
      <th>182 days</th>
      <td>2018-06-26</td>
      <td>10800</td>
      <td>10900</td>
      <td>11000</td>
      <td>10700</td>
      <td>63039</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-bbb26cf6-452c-4cf6-b5f3-e8d92f73c4c7')"
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
          document.querySelector('#df-bbb26cf6-452c-4cf6-b5f3-e8d92f73c4c7 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-bbb26cf6-452c-4cf6-b5f3-e8d92f73c4c7');
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
df.loc['179 days':'181 days']
```





  <div id="df-809b70a6-8384-450f-ac0b-fde2cd41f48e">
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
      <th>Date</th>
      <th>Close</th>
      <th>Start</th>
      <th>High</th>
      <th>Low</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>time_delta</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>179 days</th>
      <td>2018-06-29</td>
      <td>10700</td>
      <td>10550</td>
      <td>10900</td>
      <td>9990</td>
      <td>170253</td>
    </tr>
    <tr>
      <th>180 days</th>
      <td>2018-06-28</td>
      <td>10400</td>
      <td>10900</td>
      <td>10950</td>
      <td>10150</td>
      <td>155769</td>
    </tr>
    <tr>
      <th>181 days</th>
      <td>2018-06-27</td>
      <td>10900</td>
      <td>10800</td>
      <td>11050</td>
      <td>10500</td>
      <td>133548</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-809b70a6-8384-450f-ac0b-fde2cd41f48e')"
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
          document.querySelector('#df-809b70a6-8384-450f-ac0b-fde2cd41f48e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-809b70a6-8384-450f-ac0b-fde2cd41f48e');
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


## Reference
[파이썬 라이브러리를 활용한 데이터 분석 (웨스 맥키니 저)](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=315354750)
