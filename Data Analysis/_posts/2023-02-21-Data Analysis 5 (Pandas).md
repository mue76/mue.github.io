---
tag: [pandas, data analysis]
toc: true
toc_sticky: true
toc_label: 목차
---

# Pandas (3)

## 8. 데이터 정제 및 준비


```python
import pandas as pd
import numpy as np
```

### 8.2 데이터 변형 

- 중복 제거하기


```python
data = pd.DataFrame({'k1' : ['one', 'two'] * 3 + ['two'],
                     'k2' : [1, 1, 2, 3, 3, 4, 4]} )
data
```





  <div id="df-90dcec25-9538-4ce6-9b95-386664c216d1">
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
      <th>k1</th>
      <th>k2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>one</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>two</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>one</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>two</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>two</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-90dcec25-9538-4ce6-9b95-386664c216d1')"
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
          document.querySelector('#df-90dcec25-9538-4ce6-9b95-386664c216d1 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-90dcec25-9538-4ce6-9b95-386664c216d1');
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
data.duplicated(keep='first') # keep='first' (기본값값)
```




    0    False
    1    False
    2    False
    3    False
    4    False
    5    False
    6     True
    dtype: bool




```python
data.duplicated(keep='last')
```




    0    False
    1    False
    2    False
    3    False
    4    False
    5     True
    6    False
    dtype: bool




```python
data.duplicated(['k1'])
```




    0    False
    1    False
    2     True
    3     True
    4     True
    5     True
    6     True
    dtype: bool




```python
# 중복 삭제
data.drop_duplicates()
```





  <div id="df-cda6559e-2bc6-4595-97f7-f1d085a1b59d">
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
      <th>k1</th>
      <th>k2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>one</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>two</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>one</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>two</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-cda6559e-2bc6-4595-97f7-f1d085a1b59d')"
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
          document.querySelector('#df-cda6559e-2bc6-4595-97f7-f1d085a1b59d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-cda6559e-2bc6-4595-97f7-f1d085a1b59d');
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
# k1 열 기준으로 삭제제
data.drop_duplicates(['k1'])
```





  <div id="df-da87e878-8793-4d80-a8f3-499205e7f563">
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
      <th>k1</th>
      <th>k2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-da87e878-8793-4d80-a8f3-499205e7f563')"
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
          document.querySelector('#df-da87e878-8793-4d80-a8f3-499205e7f563 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-da87e878-8793-4d80-a8f3-499205e7f563');
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
data.drop_duplicates(['k1'], keep='last')
```





  <div id="df-58f54d5e-379c-4b42-8714-1affb8010cf7">
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
      <th>k1</th>
      <th>k2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>one</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>two</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-58f54d5e-379c-4b42-8714-1affb8010cf7')"
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
          document.querySelector('#df-58f54d5e-379c-4b42-8714-1affb8010cf7 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-58f54d5e-379c-4b42-8714-1affb8010cf7');
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
data['v1'] = range(7)
data
```





  <div id="df-908603ff-9780-470d-897c-40c0bb5e3c20">
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
      <th>k1</th>
      <th>k2</th>
      <th>v1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>one</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>two</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>one</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>two</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>two</td>
      <td>4</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-908603ff-9780-470d-897c-40c0bb5e3c20')"
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
          document.querySelector('#df-908603ff-9780-470d-897c-40c0bb5e3c20 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-908603ff-9780-470d-897c-40c0bb5e3c20');
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
data.duplicated()
```




    0    False
    1    False
    2    False
    3    False
    4    False
    5    False
    6    False
    dtype: bool




```python
data.drop_duplicates(['k1', 'k2'])
```





  <div id="df-21c488da-5798-44b9-8361-f8a397ecd86f">
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
      <th>k1</th>
      <th>k2</th>
      <th>v1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>one</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>two</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>one</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>two</td>
      <td>4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-21c488da-5798-44b9-8361-f8a397ecd86f')"
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
          document.querySelector('#df-21c488da-5798-44b9-8361-f8a397ecd86f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-21c488da-5798-44b9-8361-f8a397ecd86f');
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




- 함수나 매핑을 이용해서 데이터 변형하기


```python
data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon',
                              'Pastrami', 'corned beef', 'Bacon',
                              'pastrami', 'honey ham', 'nova lox'],
                     'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
data
```





  <div id="df-4b088d8f-f305-4dad-95f8-2f46fe46b55c">
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
      <th>food</th>
      <th>ounces</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bacon</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pulled pork</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bacon</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pastrami</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>corned beef</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Bacon</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>pastrami</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>honey ham</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>nova lox</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-4b088d8f-f305-4dad-95f8-2f46fe46b55c')"
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
          document.querySelector('#df-4b088d8f-f305-4dad-95f8-2f46fe46b55c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-4b088d8f-f305-4dad-95f8-2f46fe46b55c');
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
meat_to_animal = {
  'bacon': 'pig',
  'pulled pork': 'pig',
  'pastrami': 'cow',
  'corned beef': 'cow',
  'honey ham': 'pig',
  'nova lox': 'salmon'
}
```


```python
"ABC".lower()
```




    'abc'




```python
lowercased = data['food'].str.lower()
```


```python
lowercased
```




    0          bacon
    1    pulled pork
    2          bacon
    3       pastrami
    4    corned beef
    5          bacon
    6       pastrami
    7      honey ham
    8       nova lox
    Name: food, dtype: object




```python
data['animal']= lowercased.map(meat_to_animal)
```

- 값 치환하기


```python
data = pd.Series([1, -999, 2, -999, -1000, 3])
data
```




    0       1
    1    -999
    2       2
    3    -999
    4   -1000
    5       3
    dtype: int64




```python
data.replace(-999, np.nan)
```




    0       1.0
    1       NaN
    2       2.0
    3       NaN
    4   -1000.0
    5       3.0
    dtype: float64




```python
data.replace([-999, -1000], np.nan)
```




    0    1.0
    1    NaN
    2    2.0
    3    NaN
    4    NaN
    5    3.0
    dtype: float64




```python
data.replace({-999:np.nan, -1000:0})
```




    0    1.0
    1    NaN
    2    2.0
    3    NaN
    4    0.0
    5    3.0
    dtype: float64



- 축 색인 이름 바꾸기


```python
data = pd.DataFrame(np.arange(12).reshape(3, 4),
                   index=['Ohio', 'Colorado', 'New York'],
                   columns=['one', 'two', 'three', 'four'])
data
```





  <div id="df-b3a4e055-84ca-497f-8d09-427d7712e929">
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-b3a4e055-84ca-497f-8d09-427d7712e929')"
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
          document.querySelector('#df-b3a4e055-84ca-497f-8d09-427d7712e929 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-b3a4e055-84ca-497f-8d09-427d7712e929');
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
data.index
```




    Index(['Ohio', 'Colorado', 'New York'], dtype='object')




```python
def trans_upper(x):
  return x.upper()

data.index.map(trans_upper)  
```




    Index(['OHIO', 'COLORADO', 'NEW YORK'], dtype='object')




```python
data.index.map(lambda x:x.upper())
```




    Index(['OHIO', 'COLORADO', 'NEW YORK'], dtype='object')




```python
data.index.map(lambda x:x.title())
```




    Index(['Ohio', 'Colorado', 'New York'], dtype='object')




```python
data.rename(index = str.lower, columns= str.title) # rename 함수의 index/columns 에 함수를 적용할 수 있음
```





  <div id="df-10db194a-485d-46a1-a121-b603338c3a1c">
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
      <th>One</th>
      <th>Two</th>
      <th>Three</th>
      <th>Four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ohio</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>colorado</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>new york</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-10db194a-485d-46a1-a121-b603338c3a1c')"
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
          document.querySelector('#df-10db194a-485d-46a1-a121-b603338c3a1c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-10db194a-485d-46a1-a121-b603338c3a1c');
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
data
```





  <div id="df-dc759ab8-ae06-4304-85bb-a6eb55c4a44e">
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-dc759ab8-ae06-4304-85bb-a6eb55c4a44e')"
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
          document.querySelector('#df-dc759ab8-ae06-4304-85bb-a6eb55c4a44e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-dc759ab8-ae06-4304-85bb-a6eb55c4a44e');
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




- 벡터화된 문자열 함수


```python
data = {'Dave':'dave@gmail.com',
        'Steve': 'steve@gmail.com',
        'Rob':'rob@gmail.com',
        'Wes':np.nan,
        'Puppy':'p',
        'Number':'123'}
sr_data= pd.Series(data)        
sr_data
```




    Dave       dave@gmail.com
    Steve     steve@gmail.com
    Rob         rob@gmail.com
    Wes                   NaN
    Puppy                   p
    Number                123
    dtype: object




```python
'dave@gmail.com'.upper()
```




    'DAVE@GMAIL.COM'




```python
sr_data.str.upper()
```




    Dave       DAVE@GMAIL.COM
    Steve     STEVE@GMAIL.COM
    Rob         ROB@GMAIL.COM
    Wes                   NaN
    Puppy                   P
    Number                123
    dtype: object




```python
'123'.isnumeric()
```




    True




```python
'dave@gmail.com'.isnumeric()
```




    False




```python
sr_data.str.isnumeric()
```




    Dave      False
    Steve     False
    Rob       False
    Wes         NaN
    Puppy     False
    Number     True
    dtype: object




```python
sr_data.str.isalpha()
```




    Dave      False
    Steve     False
    Rob       False
    Wes         NaN
    Puppy      True
    Number    False
    dtype: object




```python
sr_data.str.contains('gmail')
```




    Dave       True
    Steve      True
    Rob        True
    Wes         NaN
    Puppy     False
    Number    False
    dtype: object



- 데이터 구간 분할


```python
# 수치데이터(양적데이터) -> 범주형데이터(질적데이터터)

ages = [18, 20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
ages
```




    [18, 20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]




```python
cats = pd.cut(ages, 4) # 범위 안에서 4구간으로 나눌 수 있음음
cats
```


```python
bins= [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins) # bins를 통해 원하는 구간을 지정할 수 있음음
cats
```


```python
bins= [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins, include_lowest=True) # 가장 작은수가 포함시키는 옵션션
cats
```




    [(17.999, 25.0], (17.999, 25.0], (17.999, 25.0], (17.999, 25.0], (25.0, 35.0], ..., (25.0, 35.0], (60.0, 100.0], (35.0, 60.0], (35.0, 60.0], (25.0, 35.0]]
    Length: 13
    Categories (4, interval[float64, right]): [(17.999, 25.0] < (25.0, 35.0] < (35.0, 60.0] <
                                               (60.0, 100.0]]




```python
cats = pd.cut(ages, bins, include_lowest=True, labels=['youth', 'youngadult', 'middleages', 'senior'])
cats
```




    ['youth', 'youth', 'youth', 'youth', 'youngadult', ..., 'youngadult', 'senior', 'middleages', 'middleages', 'youngadult']
    Length: 13
    Categories (4, object): ['youth' < 'youngadult' < 'middleages' < 'senior']




```python
cats.value_counts() # 범주형 데이터로 변환되었기 때문에 빈도수도 카운트 할 수 있음음
```




    youth         6
    youngadult    3
    middleages    3
    senior        1
    dtype: int64




```python
cats = pd.cut(ages, 4) # 범위 안에서 4구간으로 나눌 수 있음
cats.value_counts()
```




    (17.957, 28.75]    7
    (28.75, 39.5]      3
    (39.5, 50.25]      2
    (50.25, 61.0]      1
    dtype: int64




```python
cats = pd.qcut(ages, 4) # 4구간안에 동일한 양이 들어가도록 분할
cats.value_counts()
```




    (17.999, 22.0]    4
    (22.0, 27.0]      3
    (27.0, 37.0]      3
    (37.0, 61.0]      3
    dtype: int64



- 특이값(바깥값, outlier) 찾고 제외하기


```python
data = pd.DataFrame(np.random.randn(1000, 4))
data.describe()
```





  <div id="df-c0f7ee1f-91bc-400e-9577-91a8884ef7fb">
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.073027</td>
      <td>-0.039069</td>
      <td>-0.025754</td>
      <td>-0.033337</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.019392</td>
      <td>0.991063</td>
      <td>0.979328</td>
      <td>0.977642</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-4.479930</td>
      <td>-3.135338</td>
      <td>-3.763374</td>
      <td>-3.288864</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.761516</td>
      <td>-0.672036</td>
      <td>-0.667080</td>
      <td>-0.659778</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.077066</td>
      <td>-0.033668</td>
      <td>-0.001968</td>
      <td>-0.003941</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.637678</td>
      <td>0.637472</td>
      <td>0.604178</td>
      <td>0.677430</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.931344</td>
      <td>3.449874</td>
      <td>3.090328</td>
      <td>3.284176</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c0f7ee1f-91bc-400e-9577-91a8884ef7fb')"
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
          document.querySelector('#df-c0f7ee1f-91bc-400e-9577-91a8884ef7fb button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c0f7ee1f-91bc-400e-9577-91a8884ef7fb');
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
# 3보다 큰 값을 특잇값(바깥값, outlier)로 가정하고, 해당되는 값을 3으로 치환환
```


```python
data > 3
```





  <div id="df-9cb11576-bc74-49c9-a90f-591221384994">
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>996</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>997</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>998</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>999</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 4 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-9cb11576-bc74-49c9-a90f-591221384994')"
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
          document.querySelector('#df-9cb11576-bc74-49c9-a90f-591221384994 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9cb11576-bc74-49c9-a90f-591221384994');
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
(data > 3).any(axis=1) # 열축을 따라서 True 값이 하나라도 있는지 확인
```




    0      False
    1      False
    2      False
    3      False
    4      False
           ...  
    995    False
    996    False
    997    False
    998    False
    999    False
    Length: 1000, dtype: bool




```python
(data > 3).any(axis=1).sum()
```




    3




```python
data[(data > 3).any(axis=1)]
```





  <div id="df-e781d12d-a18f-4cfe-9fae-fe230752ec25">
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>84</th>
      <td>-1.034572</td>
      <td>3.449874</td>
      <td>1.025486</td>
      <td>-0.294470</td>
    </tr>
    <tr>
      <th>704</th>
      <td>-2.300875</td>
      <td>-0.342656</td>
      <td>3.090328</td>
      <td>0.381429</td>
    </tr>
    <tr>
      <th>860</th>
      <td>-1.471528</td>
      <td>0.187926</td>
      <td>0.499964</td>
      <td>3.284176</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-e781d12d-a18f-4cfe-9fae-fe230752ec25')"
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
          document.querySelector('#df-e781d12d-a18f-4cfe-9fae-fe230752ec25 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-e781d12d-a18f-4cfe-9fae-fe230752ec25');
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
data[data > 3] = 3
```


```python
data.describe()
```





  <div id="df-9abe993a-9bd0-4ddb-bf9d-68b03322130d">
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.073027</td>
      <td>-0.039519</td>
      <td>-0.025844</td>
      <td>-0.033622</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.019392</td>
      <td>0.989579</td>
      <td>0.979044</td>
      <td>0.976717</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-4.479930</td>
      <td>-3.135338</td>
      <td>-3.763374</td>
      <td>-3.288864</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.761516</td>
      <td>-0.672036</td>
      <td>-0.667080</td>
      <td>-0.659778</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.077066</td>
      <td>-0.033668</td>
      <td>-0.001968</td>
      <td>-0.003941</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.637678</td>
      <td>0.637472</td>
      <td>0.604178</td>
      <td>0.677430</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.931344</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-9abe993a-9bd0-4ddb-bf9d-68b03322130d')"
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
          document.querySelector('#df-9abe993a-9bd0-4ddb-bf9d-68b03322130d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9abe993a-9bd0-4ddb-bf9d-68b03322130d');
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




- 더미변수 계산하기(one-hot encoding)


```python
df = pd.DataFrame({'fruit' : ['apple', 'apple', 'pear', 'peach', 'pear'],
                   'data' : range(5)})
df
```





  <div id="df-d5a8f9f4-8868-4673-9d83-39664bd9068f">
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
      <th>fruit</th>
      <th>data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>apple</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pear</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>peach</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pear</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d5a8f9f4-8868-4673-9d83-39664bd9068f')"
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
          document.querySelector('#df-d5a8f9f4-8868-4673-9d83-39664bd9068f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d5a8f9f4-8868-4673-9d83-39664bd9068f');
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
dummies = pd.get_dummies(df['fruit'], prefix='fruit')
dummies
```





  <div id="df-25444369-b714-469c-96db-857ccd33a58c">
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
      <th>fruit_apple</th>
      <th>fruit_peach</th>
      <th>fruit_pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-25444369-b714-469c-96db-857ccd33a58c')"
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
          document.querySelector('#df-25444369-b714-469c-96db-857ccd33a58c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-25444369-b714-469c-96db-857ccd33a58c');
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
df[['data']]
```





  <div id="df-4e53b11b-d15c-413b-8d7c-d59d3ba9566b">
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
      <th>data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-4e53b11b-d15c-413b-8d7c-d59d3ba9566b')"
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
          document.querySelector('#df-4e53b11b-d15c-413b-8d7c-d59d3ba9566b button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-4e53b11b-d15c-413b-8d7c-d59d3ba9566b');
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
pd.concat([dummies, df[['data']]], axis=1)
```





  <div id="df-02b5772f-679d-48f6-a954-b49e78221ece">
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
      <th>fruit_apple</th>
      <th>fruit_peach</th>
      <th>fruit_pear</th>
      <th>data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-02b5772f-679d-48f6-a954-b49e78221ece')"
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
          document.querySelector('#df-02b5772f-679d-48f6-a954-b49e78221ece button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-02b5772f-679d-48f6-a954-b49e78221ece');
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




## 9. 데이터 재구성

### 9.1 계층적 색인


```python
data = pd.Series(np.random.randn(9),
                 index = [['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
                          [1, 2, 3, 1, 3, 1, 2, 2, 3]])
data
```




    a  1    0.759512
       2    2.266371
       3   -0.547481
    b  1    0.900922
       3   -1.452136
    c  1    0.321089
       2   -0.123710
    d  2    0.133820
       3   -1.512736
    dtype: float64




```python
data.index
```




    MultiIndex([('a', 1),
                ('a', 2),
                ('a', 3),
                ('b', 1),
                ('b', 3),
                ('c', 1),
                ('c', 2),
                ('d', 2),
                ('d', 3)],
               )




```python
data.loc['b']
```




    1    0.900922
    3   -1.452136
    dtype: float64




```python
data.loc['b' : 'c'] # 라벨 색인
```




    b  1    0.900922
       3   -1.452136
    c  1    0.321089
       2   -0.123710
    dtype: float64




```python
data.iloc[3: 7] # 정수 색인
```




    b  1    0.900922
       3   -1.452136
    c  1    0.321089
       2   -0.123710
    dtype: float64




```python
data # index가 쌓여있는 상태(stack)
```




    a  1    0.759512
       2    2.266371
       3   -0.547481
    b  1    0.900922
       3   -1.452136
    c  1    0.321089
       2   -0.123710
    d  2    0.133820
       3   -1.512736
    dtype: float64




```python
unstacked_data = data.unstack()
unstacked_data
```





  <div id="df-d0711bf0-e22d-4907-bbd6-cf9cd5cc4905">
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.759512</td>
      <td>2.266371</td>
      <td>-0.547481</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.900922</td>
      <td>NaN</td>
      <td>-1.452136</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.321089</td>
      <td>-0.123710</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>0.133820</td>
      <td>-1.512736</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d0711bf0-e22d-4907-bbd6-cf9cd5cc4905')"
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
          document.querySelector('#df-d0711bf0-e22d-4907-bbd6-cf9cd5cc4905 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d0711bf0-e22d-4907-bbd6-cf9cd5cc4905');
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
stacked_data = unstacked_data.stack()
stacked_data
```




    a  1    0.759512
       2    2.266371
       3   -0.547481
    b  1    0.900922
       3   -1.452136
    c  1    0.321089
       2   -0.123710
    d  2    0.133820
       3   -1.512736
    dtype: float64




```python
stacked_data.unstack()
```





  <div id="df-b9e2a129-9b57-4709-9007-06cc0410b92f">
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.759512</td>
      <td>2.266371</td>
      <td>-0.547481</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.900922</td>
      <td>NaN</td>
      <td>-1.452136</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.321089</td>
      <td>-0.123710</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>0.133820</td>
      <td>-1.512736</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-b9e2a129-9b57-4709-9007-06cc0410b92f')"
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
          document.querySelector('#df-b9e2a129-9b57-4709-9007-06cc0410b92f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-b9e2a129-9b57-4709-9007-06cc0410b92f');
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
unstacked_data.reset_index()
```





  <div id="df-a079dcf9-3498-4fd8-a14a-61834e1bf1b4">
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
      <th>index</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>0.759512</td>
      <td>2.266371</td>
      <td>-0.547481</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>0.900922</td>
      <td>NaN</td>
      <td>-1.452136</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>0.321089</td>
      <td>-0.123710</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>NaN</td>
      <td>0.133820</td>
      <td>-1.512736</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a079dcf9-3498-4fd8-a14a-61834e1bf1b4')"
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
          document.querySelector('#df-a079dcf9-3498-4fd8-a14a-61834e1bf1b4 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a079dcf9-3498-4fd8-a14a-61834e1bf1b4');
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
stacked_data.reset_index()
```





  <div id="df-2bd545ee-0941-4530-94d6-e9fe575518e0">
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
      <th>level_0</th>
      <th>level_1</th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>1</td>
      <td>0.759512</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>2</td>
      <td>2.266371</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>3</td>
      <td>-0.547481</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>1</td>
      <td>0.900922</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>3</td>
      <td>-1.452136</td>
    </tr>
    <tr>
      <th>5</th>
      <td>c</td>
      <td>1</td>
      <td>0.321089</td>
    </tr>
    <tr>
      <th>6</th>
      <td>c</td>
      <td>2</td>
      <td>-0.123710</td>
    </tr>
    <tr>
      <th>7</th>
      <td>d</td>
      <td>2</td>
      <td>0.133820</td>
    </tr>
    <tr>
      <th>8</th>
      <td>d</td>
      <td>3</td>
      <td>-1.512736</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-2bd545ee-0941-4530-94d6-e9fe575518e0')"
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
          document.querySelector('#df-2bd545ee-0941-4530-94d6-e9fe575518e0 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-2bd545ee-0941-4530-94d6-e9fe575518e0');
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




### 9.2 데이터 합치기

- 데이터베이스 스타일로 DataFrame 합치기

- inner join


```python
df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                    'data1': range(7)})
df2 = pd.DataFrame({'key': ['a', 'b', 'd'],
                    'data2': range(3)})
```


```python
df1
```





  <div id="df-4c044a4f-86b8-4e9f-aa63-6bde197fa63c">
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
      <th>key</th>
      <th>data1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>c</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>a</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>b</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-4c044a4f-86b8-4e9f-aa63-6bde197fa63c')"
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
          document.querySelector('#df-4c044a4f-86b8-4e9f-aa63-6bde197fa63c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-4c044a4f-86b8-4e9f-aa63-6bde197fa63c');
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
df2
```





  <div id="df-3031fba0-caf2-4b79-b148-1a5dcb8c73f2">
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
      <th>key</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>d</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-3031fba0-caf2-4b79-b148-1a5dcb8c73f2')"
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
          document.querySelector('#df-3031fba0-caf2-4b79-b148-1a5dcb8c73f2 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-3031fba0-caf2-4b79-b148-1a5dcb8c73f2');
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
pd.merge(df1, df2, on='key', how='inner')
```





  <div id="df-0d68040e-4bf1-4b74-b163-0bfa956b93c0">
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
      <th>key</th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>a</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-0d68040e-4bf1-4b74-b163-0bfa956b93c0')"
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
          document.querySelector('#df-0d68040e-4bf1-4b74-b163-0bfa956b93c0 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-0d68040e-4bf1-4b74-b163-0bfa956b93c0');
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
# 위의 merge 함수 사용법은 아래 query문과 같음
# SELECT a.key, data1, data2
#   FROM df1 a
#   INNER JOIN df2
#     ON a.key = b.key;
```


```python
pd.merge(df1, df2, on='key') # how='inner'가 기본값값
```





  <div id="df-71067085-6651-4cba-9d0f-62199adddb40">
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
      <th>key</th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>a</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-71067085-6651-4cba-9d0f-62199adddb40')"
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
          document.querySelector('#df-71067085-6651-4cba-9d0f-62199adddb40 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-71067085-6651-4cba-9d0f-62199adddb40');
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
df3 = pd.DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
df4 = pd.DataFrame({'rkey': ['a', 'b', 'd'], 'data2': range(3)})
```


```python
df3
```





  <div id="df-fefdfdf0-3356-4c11-8797-d98927b2e147">
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
      <th>lkey</th>
      <th>data1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>c</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>a</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>b</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-fefdfdf0-3356-4c11-8797-d98927b2e147')"
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
          document.querySelector('#df-fefdfdf0-3356-4c11-8797-d98927b2e147 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-fefdfdf0-3356-4c11-8797-d98927b2e147');
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
df4
```





  <div id="df-7bdc60de-d6f8-412f-b6e8-8e32d16a7b59">
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
      <th>rkey</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>d</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-7bdc60de-d6f8-412f-b6e8-8e32d16a7b59')"
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
          document.querySelector('#df-7bdc60de-d6f8-412f-b6e8-8e32d16a7b59 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7bdc60de-d6f8-412f-b6e8-8e32d16a7b59');
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
pd.merge(df3, df4, left_on= 'lkey', right_on='rkey')
```





  <div id="df-1ec4bbed-7275-482b-b65e-cc31fd131c25">
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
      <th>lkey</th>
      <th>data1</th>
      <th>rkey</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>0</td>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>6</td>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>2</td>
      <td>a</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>4</td>
      <td>a</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>a</td>
      <td>5</td>
      <td>a</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1ec4bbed-7275-482b-b65e-cc31fd131c25')"
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
          document.querySelector('#df-1ec4bbed-7275-482b-b65e-cc31fd131c25 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1ec4bbed-7275-482b-b65e-cc31fd131c25');
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
# 위의 merge 함수 사용법은 아래 query문과 같음
# SELECT a.lkey, data1, b.rkey, data2
#   FROM df3 a
#   INNER JOIN df4
#     ON a.lkey = b.rkey;
```

- outer join


```python
df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                    'data1': range(7)})
df2 = pd.DataFrame({'key': ['a', 'b', 'd'],
                    'data2': range(3)})
```


```python
df1
```





  <div id="df-5f626c67-8549-4992-aea1-d676b2c263c3">
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
      <th>key</th>
      <th>data1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>c</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>a</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>b</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-5f626c67-8549-4992-aea1-d676b2c263c3')"
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
          document.querySelector('#df-5f626c67-8549-4992-aea1-d676b2c263c3 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-5f626c67-8549-4992-aea1-d676b2c263c3');
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
df2
```





  <div id="df-ff0dbdf0-c014-4908-a8f2-31e466978592">
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
      <th>key</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>d</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ff0dbdf0-c014-4908-a8f2-31e466978592')"
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
          document.querySelector('#df-ff0dbdf0-c014-4908-a8f2-31e466978592 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ff0dbdf0-c014-4908-a8f2-31e466978592');
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
pd.merge(df1, df2, on='key', how='left')
```





  <div id="df-170dcd0b-13a0-4a14-bd43-81f93ef82403">
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
      <th>key</th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>c</td>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>4</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>a</td>
      <td>5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>b</td>
      <td>6</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-170dcd0b-13a0-4a14-bd43-81f93ef82403')"
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
          document.querySelector('#df-170dcd0b-13a0-4a14-bd43-81f93ef82403 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-170dcd0b-13a0-4a14-bd43-81f93ef82403');
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
pd.merge(df1, df2, how='left') # join할 대상의 컬럼명이 같으면 on 파라미터 생략 가능 (natural join)
```





  <div id="df-cb8df6d6-65a6-4264-a7e2-c5fa4c390008">
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
      <th>key</th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>c</td>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>4</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>a</td>
      <td>5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>b</td>
      <td>6</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-cb8df6d6-65a6-4264-a7e2-c5fa4c390008')"
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
          document.querySelector('#df-cb8df6d6-65a6-4264-a7e2-c5fa4c390008 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-cb8df6d6-65a6-4264-a7e2-c5fa4c390008');
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
pd.merge(df1, df2, how='right')
```





  <div id="df-a95134af-687d-49a0-8e0e-17a6f8111f5e">
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
      <th>key</th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>4.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>5.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>b</td>
      <td>6.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>d</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a95134af-687d-49a0-8e0e-17a6f8111f5e')"
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
          document.querySelector('#df-a95134af-687d-49a0-8e0e-17a6f8111f5e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a95134af-687d-49a0-8e0e-17a6f8111f5e');
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
# df의 위치만 변경시키면 위의 right join과 동일
pd.merge(df2, df1, how='left')
```





  <div id="df-779c5adf-41b0-463b-ad6a-65b9424f621d">
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
      <th>key</th>
      <th>data2</th>
      <th>data1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>b</td>
      <td>1</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>d</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-779c5adf-41b0-463b-ad6a-65b9424f621d')"
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
          document.querySelector('#df-779c5adf-41b0-463b-ad6a-65b9424f621d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-779c5adf-41b0-463b-ad6a-65b9424f621d');
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
pd.merge(df1, df2, how='outer') # left, right를 모두 포함한 join (한쪽에만 있는 데이터도 모두 취함)
```





  <div id="df-d161ad27-0eeb-4ca0-b8ea-aa31ffbf5916">
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
      <th>key</th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>6.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>4.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>a</td>
      <td>5.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>c</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>d</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d161ad27-0eeb-4ca0-b8ea-aa31ffbf5916')"
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
          document.querySelector('#df-d161ad27-0eeb-4ca0-b8ea-aa31ffbf5916 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d161ad27-0eeb-4ca0-b8ea-aa31ffbf5916');
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




- 색인 병합하기


```python
left1 = pd.DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'], 'value': range(6)})
right1 = pd.DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])
```


```python
left1
```





  <div id="df-9bac5fe6-0459-4705-8e61-99ce50337a84">
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
      <th>key</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>c</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-9bac5fe6-0459-4705-8e61-99ce50337a84')"
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
          document.querySelector('#df-9bac5fe6-0459-4705-8e61-99ce50337a84 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9bac5fe6-0459-4705-8e61-99ce50337a84');
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
right1
```





  <div id="df-a225bdeb-d2bd-441f-ac0c-56d088f4d3d4">
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
      <th>group_val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>3.5</td>
    </tr>
    <tr>
      <th>b</th>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a225bdeb-d2bd-441f-ac0c-56d088f4d3d4')"
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
          document.querySelector('#df-a225bdeb-d2bd-441f-ac0c-56d088f4d3d4 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a225bdeb-d2bd-441f-ac0c-56d088f4d3d4');
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
pd.merge(left1, right1, left_on='key', right_index=True) # inner join
```





  <div id="df-8042188f-94ec-486c-8b10-6b0ae61edacc">
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
      <th>key</th>
      <th>value</th>
      <th>group_val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>0</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>2</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>3</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>4</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-8042188f-94ec-486c-8b10-6b0ae61edacc')"
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
          document.querySelector('#df-8042188f-94ec-486c-8b10-6b0ae61edacc button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-8042188f-94ec-486c-8b10-6b0ae61edacc');
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
pd.merge(left1, right1, left_on='key', right_index=True, how='outer') # outer join
```





  <div id="df-6a4dcd03-2f83-4582-99e8-75ddf51d5748">
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
      <th>key</th>
      <th>value</th>
      <th>group_val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>0</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>2</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>3</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>4</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>c</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-6a4dcd03-2f83-4582-99e8-75ddf51d5748')"
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
          document.querySelector('#df-6a4dcd03-2f83-4582-99e8-75ddf51d5748 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-6a4dcd03-2f83-4582-99e8-75ddf51d5748');
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
left2 = pd.DataFrame([[1., 2.], [3., 4.], [5., 6.]],
                     index=['a', 'c', 'e'],
                     columns=['Ohio', 'Nevada'])
right2 = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],
                      index=['b', 'c', 'd', 'e'],
                      columns=['Missouri', 'Alabama'])
```


```python
left2
```





  <div id="df-05ff09be-8cf8-448a-b94f-217f0a81272a">
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
      <th>Ohio</th>
      <th>Nevada</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>e</th>
      <td>5.0</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-05ff09be-8cf8-448a-b94f-217f0a81272a')"
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
          document.querySelector('#df-05ff09be-8cf8-448a-b94f-217f0a81272a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-05ff09be-8cf8-448a-b94f-217f0a81272a');
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
right2
```





  <div id="df-12f9b99e-22b1-42b2-8197-4e1948a596d0">
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
      <th>Missouri</th>
      <th>Alabama</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>b</th>
      <td>7.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>9.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>d</th>
      <td>11.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>e</th>
      <td>13.0</td>
      <td>14.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-12f9b99e-22b1-42b2-8197-4e1948a596d0')"
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
          document.querySelector('#df-12f9b99e-22b1-42b2-8197-4e1948a596d0 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-12f9b99e-22b1-42b2-8197-4e1948a596d0');
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
pd.merge(left2, right2, left_index=True, right_index=True) # inner join
```





  <div id="df-b6105c3a-8dc6-4bcc-9333-31f1ab834ce1">
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
      <th>Ohio</th>
      <th>Nevada</th>
      <th>Missouri</th>
      <th>Alabama</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>c</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>9.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>e</th>
      <td>5.0</td>
      <td>6.0</td>
      <td>13.0</td>
      <td>14.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-b6105c3a-8dc6-4bcc-9333-31f1ab834ce1')"
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
          document.querySelector('#df-b6105c3a-8dc6-4bcc-9333-31f1ab834ce1 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-b6105c3a-8dc6-4bcc-9333-31f1ab834ce1');
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
# join 함수는 인덱스 값을 기준으로 조인을 함
# 위의 pd.merge 결과와 동일
left2.join(right2, how='inner') # join 함수의 how='left'가 기본값이므로 inner로 변경경

```





  <div id="df-28411537-3af5-4c48-a703-dd962993c985">
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
      <th>Ohio</th>
      <th>Nevada</th>
      <th>Missouri</th>
      <th>Alabama</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>c</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>9.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>e</th>
      <td>5.0</td>
      <td>6.0</td>
      <td>13.0</td>
      <td>14.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-28411537-3af5-4c48-a703-dd962993c985')"
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
          document.querySelector('#df-28411537-3af5-4c48-a703-dd962993c985 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-28411537-3af5-4c48-a703-dd962993c985');
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
left1
```





  <div id="df-1b581628-2adb-4761-82b8-a918da4e37d8">
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
      <th>key</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>c</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1b581628-2adb-4761-82b8-a918da4e37d8')"
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
          document.querySelector('#df-1b581628-2adb-4761-82b8-a918da4e37d8 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1b581628-2adb-4761-82b8-a918da4e37d8');
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
right1
```





  <div id="df-dd4ac889-b973-4602-a75d-484ee24d977a">
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
      <th>group_val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>3.5</td>
    </tr>
    <tr>
      <th>b</th>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-dd4ac889-b973-4602-a75d-484ee24d977a')"
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
          document.querySelector('#df-dd4ac889-b973-4602-a75d-484ee24d977a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-dd4ac889-b973-4602-a75d-484ee24d977a');
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
pd.merge(left1, right1, left_on='key', right_index=True, how='inner')
```





  <div id="df-48f5940f-4a63-4109-a53f-2b721c8bfd01">
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
      <th>key</th>
      <th>value</th>
      <th>group_val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>0</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>2</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>3</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>4</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-48f5940f-4a63-4109-a53f-2b721c8bfd01')"
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
          document.querySelector('#df-48f5940f-4a63-4109-a53f-2b721c8bfd01 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-48f5940f-4a63-4109-a53f-2b721c8bfd01');
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
# join 함수를 사용해서 위와 동일하게 합치려면면
left1.join(right1, on='key', how='inner')
```





  <div id="df-6c3ba5e8-d51b-464b-8eaf-a6c9ebb4f8f7">
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
      <th>key</th>
      <th>value</th>
      <th>group_val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>0</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>2</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>3</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>4</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-6c3ba5e8-d51b-464b-8eaf-a6c9ebb4f8f7')"
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
          document.querySelector('#df-6c3ba5e8-d51b-464b-8eaf-a6c9ebb4f8f7 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-6c3ba5e8-d51b-464b-8eaf-a6c9ebb4f8f7');
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




- 축따라 이어붙이기

**참고** np.concatenate


```python
arr = np.arange(12).reshape(3, 4)
arr
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])




```python
np.concatenate([arr, arr], axis=0) # axis=0 기본값, 0번축을 따라서 이어붙임
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])




```python
np.concatenate([arr, arr], axis=1)
```




    array([[ 0,  1,  2,  3,  0,  1,  2,  3],
           [ 4,  5,  6,  7,  4,  5,  6,  7],
           [ 8,  9, 10, 11,  8,  9, 10, 11]])




```python
s1 = pd.Series([0, 1], index=['a', 'b'])
s2 = pd.Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = pd.Series([5, 6], index=['f', 'g'])
```


```python
s1
```




    a    0
    b    1
    dtype: int64




```python
s2
```




    c    2
    d    3
    e    4
    dtype: int64




```python
s3
```




    f    5
    g    6
    dtype: int64




```python
pd.concat([s1, s2, s3], axis=0)
```




    a    0
    b    1
    c    2
    d    3
    e    4
    f    5
    g    6
    dtype: int64




```python
pd.concat([s1, s2, s3], axis=1)
```





  <div id="df-d2a5985e-b4a4-49fa-9b4e-59f89dd8c373">
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>c</th>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>e</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>f</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>g</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d2a5985e-b4a4-49fa-9b4e-59f89dd8c373')"
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
          document.querySelector('#df-d2a5985e-b4a4-49fa-9b4e-59f89dd8c373 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d2a5985e-b4a4-49fa-9b4e-59f89dd8c373');
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
df1 = pd.DataFrame(np.random.randn(3, 4), columns=['a', 'b', 'c', 'd'])
df2 = pd.DataFrame(np.random.randn(2, 3), columns=['b', 'd', 'a'])
```


```python
df1
```





  <div id="df-ccc8eb3c-2cae-4cc1-aec8-ebf0ac08c48e">
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.127909</td>
      <td>1.862198</td>
      <td>-1.044944</td>
      <td>1.537435</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.243453</td>
      <td>-0.300529</td>
      <td>2.783077</td>
      <td>0.080733</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.917728</td>
      <td>-1.234895</td>
      <td>-0.285104</td>
      <td>3.605365</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ccc8eb3c-2cae-4cc1-aec8-ebf0ac08c48e')"
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
          document.querySelector('#df-ccc8eb3c-2cae-4cc1-aec8-ebf0ac08c48e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ccc8eb3c-2cae-4cc1-aec8-ebf0ac08c48e');
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
df2
```





  <div id="df-a2a4eb4b-4b2a-41ad-9b23-1e945daa9f6f">
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
      <th>b</th>
      <th>d</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.967257</td>
      <td>0.747703</td>
      <td>1.636920</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.947134</td>
      <td>-0.868835</td>
      <td>0.943723</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a2a4eb4b-4b2a-41ad-9b23-1e945daa9f6f')"
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
          document.querySelector('#df-a2a4eb4b-4b2a-41ad-9b23-1e945daa9f6f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a2a4eb4b-4b2a-41ad-9b23-1e945daa9f6f');
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
pd.concat([df1, df2], axis=0)
```





  <div id="df-6eaa993c-0065-44e7-a25c-335df4605b25">
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.127909</td>
      <td>1.862198</td>
      <td>-1.044944</td>
      <td>1.537435</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.243453</td>
      <td>-0.300529</td>
      <td>2.783077</td>
      <td>0.080733</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.917728</td>
      <td>-1.234895</td>
      <td>-0.285104</td>
      <td>3.605365</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1.636920</td>
      <td>-0.967257</td>
      <td>NaN</td>
      <td>0.747703</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.943723</td>
      <td>0.947134</td>
      <td>NaN</td>
      <td>-0.868835</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-6eaa993c-0065-44e7-a25c-335df4605b25')"
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
          document.querySelector('#df-6eaa993c-0065-44e7-a25c-335df4605b25 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-6eaa993c-0065-44e7-a25c-335df4605b25');
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
pd.concat([df1, df2], axis=0, ignore_index=True)
```





  <div id="df-1bf825d4-e310-4d7a-b0a7-3234603fb285">
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.127909</td>
      <td>1.862198</td>
      <td>-1.044944</td>
      <td>1.537435</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.243453</td>
      <td>-0.300529</td>
      <td>2.783077</td>
      <td>0.080733</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.917728</td>
      <td>-1.234895</td>
      <td>-0.285104</td>
      <td>3.605365</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.636920</td>
      <td>-0.967257</td>
      <td>NaN</td>
      <td>0.747703</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.943723</td>
      <td>0.947134</td>
      <td>NaN</td>
      <td>-0.868835</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1bf825d4-e310-4d7a-b0a7-3234603fb285')"
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
          document.querySelector('#df-1bf825d4-e310-4d7a-b0a7-3234603fb285 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1bf825d4-e310-4d7a-b0a7-3234603fb285');
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




### Workshop

- 축따라 이어붙이기


```python
df1 = pd.DataFrame({'a': ['a0', 'a1', 'a2', 'a3'],
                    'b': ['b0', 'b1', 'b2', 'b3'],
                    'c': ['c0', 'c1', 'c2', 'c3']},
                    index=[0, 1, 2, 3])
 
df2 = pd.DataFrame({'a': ['a2', 'a3', 'a4', 'a5'],
                    'b': ['b2', 'b3', 'b4', 'b5'],
                    'c': ['c2', 'c3', 'c4', 'c5'],
                    'd': ['d2', 'd3', 'd4', 'd5']},
                    index=[2, 3, 4, 5])
```


```python
df1
```





  <div id="df-58e60435-292e-4804-afe7-0f47b06ed2fc">
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a0</td>
      <td>b0</td>
      <td>c0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a1</td>
      <td>b1</td>
      <td>c1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a2</td>
      <td>b2</td>
      <td>c2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a3</td>
      <td>b3</td>
      <td>c3</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-58e60435-292e-4804-afe7-0f47b06ed2fc')"
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
          document.querySelector('#df-58e60435-292e-4804-afe7-0f47b06ed2fc button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-58e60435-292e-4804-afe7-0f47b06ed2fc');
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
df2
```





  <div id="df-57a28815-1d50-4851-949d-a344a7665a32">
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>a2</td>
      <td>b2</td>
      <td>c2</td>
      <td>d2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a3</td>
      <td>b3</td>
      <td>c3</td>
      <td>d3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a4</td>
      <td>b4</td>
      <td>c4</td>
      <td>d4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>a5</td>
      <td>b5</td>
      <td>c5</td>
      <td>d5</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-57a28815-1d50-4851-949d-a344a7665a32')"
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
          document.querySelector('#df-57a28815-1d50-4851-949d-a344a7665a32 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-57a28815-1d50-4851-949d-a344a7665a32');
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
# 2개의 데이터프레임을 위, 아래 (행축으로) 이어붙이듯 연결하기
```


```python
pd.concat([df1, df2], axis=0)
```





  <div id="df-eef60384-d03a-4b97-96e6-6ef311b336b6">
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a0</td>
      <td>b0</td>
      <td>c0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a1</td>
      <td>b1</td>
      <td>c1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a2</td>
      <td>b2</td>
      <td>c2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a3</td>
      <td>b3</td>
      <td>c3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a2</td>
      <td>b2</td>
      <td>c2</td>
      <td>d2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a3</td>
      <td>b3</td>
      <td>c3</td>
      <td>d3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a4</td>
      <td>b4</td>
      <td>c4</td>
      <td>d4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>a5</td>
      <td>b5</td>
      <td>c5</td>
      <td>d5</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-eef60384-d03a-4b97-96e6-6ef311b336b6')"
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
          document.querySelector('#df-eef60384-d03a-4b97-96e6-6ef311b336b6 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-eef60384-d03a-4b97-96e6-6ef311b336b6');
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
# 인덱스를 재 설정 (ignore_index 사용용)
```


```python
pd.concat([df1, df2], axis=0, ignore_index=True)
```





  <div id="df-46cc51b7-2d43-4c68-95b2-6fbfa7d3ca39">
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a0</td>
      <td>b0</td>
      <td>c0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a1</td>
      <td>b1</td>
      <td>c1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a2</td>
      <td>b2</td>
      <td>c2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a3</td>
      <td>b3</td>
      <td>c3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a2</td>
      <td>b2</td>
      <td>c2</td>
      <td>d2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>a3</td>
      <td>b3</td>
      <td>c3</td>
      <td>d3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>a4</td>
      <td>b4</td>
      <td>c4</td>
      <td>d4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>a5</td>
      <td>b5</td>
      <td>c5</td>
      <td>d5</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-46cc51b7-2d43-4c68-95b2-6fbfa7d3ca39')"
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
          document.querySelector('#df-46cc51b7-2d43-4c68-95b2-6fbfa7d3ca39 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-46cc51b7-2d43-4c68-95b2-6fbfa7d3ca39');
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
# 2개의 데이터프레임을 좌, 우 (열축으로) 이어붙이듯 연결하기
```


```python
pd.concat([df1, df2], axis=1)
```





  <div id="df-0ee26653-6874-4e81-8719-4263d0fcf058">
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a0</td>
      <td>b0</td>
      <td>c0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a1</td>
      <td>b1</td>
      <td>c1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a2</td>
      <td>b2</td>
      <td>c2</td>
      <td>a2</td>
      <td>b2</td>
      <td>c2</td>
      <td>d2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a3</td>
      <td>b3</td>
      <td>c3</td>
      <td>a3</td>
      <td>b3</td>
      <td>c3</td>
      <td>d3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>a4</td>
      <td>b4</td>
      <td>c4</td>
      <td>d4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>a5</td>
      <td>b5</td>
      <td>c5</td>
      <td>d5</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-0ee26653-6874-4e81-8719-4263d0fcf058')"
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
          document.querySelector('#df-0ee26653-6874-4e81-8719-4263d0fcf058 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-0ee26653-6874-4e81-8719-4263d0fcf058');
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
sr = pd.Series(['e0', 'e1', 'e2', 'e3'], name='e')
```


```python
df1
```





  <div id="df-09b17e04-1a3a-40e3-abcf-c7d24b04dd5f">
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a0</td>
      <td>b0</td>
      <td>c0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a1</td>
      <td>b1</td>
      <td>c1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a2</td>
      <td>b2</td>
      <td>c2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a3</td>
      <td>b3</td>
      <td>c3</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-09b17e04-1a3a-40e3-abcf-c7d24b04dd5f')"
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
          document.querySelector('#df-09b17e04-1a3a-40e3-abcf-c7d24b04dd5f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-09b17e04-1a3a-40e3-abcf-c7d24b04dd5f');
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
sr
```




    0    e0
    1    e1
    2    e2
    3    e3
    Name: e, dtype: object




```python
# df1과 sr을 좌, 우(열축으로) 이어붙이듯 연결하기
```


```python
pd.concat([df1, sr], axis=1)
```





  <div id="df-039de77c-0fb6-4d45-9cef-89a20373c259">
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a0</td>
      <td>b0</td>
      <td>c0</td>
      <td>e0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a1</td>
      <td>b1</td>
      <td>c1</td>
      <td>e1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a2</td>
      <td>b2</td>
      <td>c2</td>
      <td>e2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a3</td>
      <td>b3</td>
      <td>c3</td>
      <td>e3</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-039de77c-0fb6-4d45-9cef-89a20373c259')"
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
          document.querySelector('#df-039de77c-0fb6-4d45-9cef-89a20373c259 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-039de77c-0fb6-4d45-9cef-89a20373c259');
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




- 데이터베이스 스타일로 DataFrame 합치기


```python
df1 = pd.read_excel('./examples/stock price.xlsx')
df2 = pd.read_excel('./examples/stock valuation.xlsx')
```


```python
df1
```





  <div id="df-466b594c-7d3c-4cb8-b21a-f6a54cdfd458">
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
      <th>id</th>
      <th>stock_name</th>
      <th>value</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>128940</td>
      <td>한미약품</td>
      <td>59385.666667</td>
      <td>421000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>130960</td>
      <td>CJ E&amp;M</td>
      <td>58540.666667</td>
      <td>98900</td>
    </tr>
    <tr>
      <th>2</th>
      <td>138250</td>
      <td>엔에스쇼핑</td>
      <td>14558.666667</td>
      <td>13200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>139480</td>
      <td>이마트</td>
      <td>239230.833333</td>
      <td>254500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>142280</td>
      <td>녹십자엠에스</td>
      <td>468.833333</td>
      <td>10200</td>
    </tr>
    <tr>
      <th>5</th>
      <td>145990</td>
      <td>삼양사</td>
      <td>82750.000000</td>
      <td>82000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>185750</td>
      <td>종근당</td>
      <td>40293.666667</td>
      <td>100500</td>
    </tr>
    <tr>
      <th>7</th>
      <td>192400</td>
      <td>쿠쿠홀딩스</td>
      <td>179204.666667</td>
      <td>177500</td>
    </tr>
    <tr>
      <th>8</th>
      <td>199800</td>
      <td>툴젠</td>
      <td>-2514.333333</td>
      <td>115400</td>
    </tr>
    <tr>
      <th>9</th>
      <td>204210</td>
      <td>모두투어리츠</td>
      <td>3093.333333</td>
      <td>3475</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-466b594c-7d3c-4cb8-b21a-f6a54cdfd458')"
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
          document.querySelector('#df-466b594c-7d3c-4cb8-b21a-f6a54cdfd458 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-466b594c-7d3c-4cb8-b21a-f6a54cdfd458');
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
df2
```





  <div id="df-23365228-4596-4c6b-ad97-a668c744f1b8">
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
      <th>id</th>
      <th>name</th>
      <th>eps</th>
      <th>bps</th>
      <th>per</th>
      <th>pbr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>130960</td>
      <td>CJ E&amp;M</td>
      <td>6301.333333</td>
      <td>54068</td>
      <td>15.695091</td>
      <td>1.829178</td>
    </tr>
    <tr>
      <th>1</th>
      <td>136480</td>
      <td>하림</td>
      <td>274.166667</td>
      <td>3551</td>
      <td>11.489362</td>
      <td>0.887074</td>
    </tr>
    <tr>
      <th>2</th>
      <td>138040</td>
      <td>메리츠금융지주</td>
      <td>2122.333333</td>
      <td>14894</td>
      <td>6.313806</td>
      <td>0.899691</td>
    </tr>
    <tr>
      <th>3</th>
      <td>139480</td>
      <td>이마트</td>
      <td>18268.166667</td>
      <td>295780</td>
      <td>13.931338</td>
      <td>0.860437</td>
    </tr>
    <tr>
      <th>4</th>
      <td>145990</td>
      <td>삼양사</td>
      <td>5741.000000</td>
      <td>108090</td>
      <td>14.283226</td>
      <td>0.758627</td>
    </tr>
    <tr>
      <th>5</th>
      <td>161390</td>
      <td>한국타이어</td>
      <td>5648.500000</td>
      <td>51341</td>
      <td>7.453306</td>
      <td>0.820007</td>
    </tr>
    <tr>
      <th>6</th>
      <td>181710</td>
      <td>NHN엔터테인먼트</td>
      <td>2110.166667</td>
      <td>78434</td>
      <td>30.755864</td>
      <td>0.827447</td>
    </tr>
    <tr>
      <th>7</th>
      <td>185750</td>
      <td>종근당</td>
      <td>3990.333333</td>
      <td>40684</td>
      <td>25.185866</td>
      <td>2.470259</td>
    </tr>
    <tr>
      <th>8</th>
      <td>204210</td>
      <td>모두투어리츠</td>
      <td>85.166667</td>
      <td>5335</td>
      <td>40.802348</td>
      <td>0.651359</td>
    </tr>
    <tr>
      <th>9</th>
      <td>207940</td>
      <td>삼성바이오로직스</td>
      <td>4644.166667</td>
      <td>60099</td>
      <td>89.790059</td>
      <td>6.938551</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-23365228-4596-4c6b-ad97-a668c744f1b8')"
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
          document.querySelector('#df-23365228-4596-4c6b-ad97-a668c744f1b8 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-23365228-4596-4c6b-ad97-a668c744f1b8');
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
# id 를 조인 조건으로 해서 inner join
```


```python
pd.merge(df1, df2, on='id', how='inner')
```





  <div id="df-8b2ac2cf-8f6c-44c8-babd-35f2810e8ba1">
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
      <th>id</th>
      <th>stock_name</th>
      <th>value</th>
      <th>price</th>
      <th>name</th>
      <th>eps</th>
      <th>bps</th>
      <th>per</th>
      <th>pbr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>130960</td>
      <td>CJ E&amp;M</td>
      <td>58540.666667</td>
      <td>98900</td>
      <td>CJ E&amp;M</td>
      <td>6301.333333</td>
      <td>54068</td>
      <td>15.695091</td>
      <td>1.829178</td>
    </tr>
    <tr>
      <th>1</th>
      <td>139480</td>
      <td>이마트</td>
      <td>239230.833333</td>
      <td>254500</td>
      <td>이마트</td>
      <td>18268.166667</td>
      <td>295780</td>
      <td>13.931338</td>
      <td>0.860437</td>
    </tr>
    <tr>
      <th>2</th>
      <td>145990</td>
      <td>삼양사</td>
      <td>82750.000000</td>
      <td>82000</td>
      <td>삼양사</td>
      <td>5741.000000</td>
      <td>108090</td>
      <td>14.283226</td>
      <td>0.758627</td>
    </tr>
    <tr>
      <th>3</th>
      <td>185750</td>
      <td>종근당</td>
      <td>40293.666667</td>
      <td>100500</td>
      <td>종근당</td>
      <td>3990.333333</td>
      <td>40684</td>
      <td>25.185866</td>
      <td>2.470259</td>
    </tr>
    <tr>
      <th>4</th>
      <td>204210</td>
      <td>모두투어리츠</td>
      <td>3093.333333</td>
      <td>3475</td>
      <td>모두투어리츠</td>
      <td>85.166667</td>
      <td>5335</td>
      <td>40.802348</td>
      <td>0.651359</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-8b2ac2cf-8f6c-44c8-babd-35f2810e8ba1')"
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
          document.querySelector('#df-8b2ac2cf-8f6c-44c8-babd-35f2810e8ba1 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-8b2ac2cf-8f6c-44c8-babd-35f2810e8ba1');
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
pd.merge(df1, df2) # 양쪽 데이터프레임의 컬럼명이 id로 동일하므로 생략 가능
                   # merge 함수의 how='inner' 가 기본값이므로 생략 가능
```





  <div id="df-eb794a12-9afe-4fa1-9b75-acfb00dd9b54">
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
      <th>id</th>
      <th>stock_name</th>
      <th>value</th>
      <th>price</th>
      <th>name</th>
      <th>eps</th>
      <th>bps</th>
      <th>per</th>
      <th>pbr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>130960</td>
      <td>CJ E&amp;M</td>
      <td>58540.666667</td>
      <td>98900</td>
      <td>CJ E&amp;M</td>
      <td>6301.333333</td>
      <td>54068</td>
      <td>15.695091</td>
      <td>1.829178</td>
    </tr>
    <tr>
      <th>1</th>
      <td>139480</td>
      <td>이마트</td>
      <td>239230.833333</td>
      <td>254500</td>
      <td>이마트</td>
      <td>18268.166667</td>
      <td>295780</td>
      <td>13.931338</td>
      <td>0.860437</td>
    </tr>
    <tr>
      <th>2</th>
      <td>145990</td>
      <td>삼양사</td>
      <td>82750.000000</td>
      <td>82000</td>
      <td>삼양사</td>
      <td>5741.000000</td>
      <td>108090</td>
      <td>14.283226</td>
      <td>0.758627</td>
    </tr>
    <tr>
      <th>3</th>
      <td>185750</td>
      <td>종근당</td>
      <td>40293.666667</td>
      <td>100500</td>
      <td>종근당</td>
      <td>3990.333333</td>
      <td>40684</td>
      <td>25.185866</td>
      <td>2.470259</td>
    </tr>
    <tr>
      <th>4</th>
      <td>204210</td>
      <td>모두투어리츠</td>
      <td>3093.333333</td>
      <td>3475</td>
      <td>모두투어리츠</td>
      <td>85.166667</td>
      <td>5335</td>
      <td>40.802348</td>
      <td>0.651359</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-eb794a12-9afe-4fa1-9b75-acfb00dd9b54')"
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
          document.querySelector('#df-eb794a12-9afe-4fa1-9b75-acfb00dd9b54 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-eb794a12-9afe-4fa1-9b75-acfb00dd9b54');
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
# id 를 조인 조건으로 해서 outer join
```


```python
pd.merge(df1, df2, on='id', how='outer')
```





  <div id="df-a7a8993c-b0a5-40f3-81b2-ad2bdb81c5f2">
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
      <th>id</th>
      <th>stock_name</th>
      <th>value</th>
      <th>price</th>
      <th>name</th>
      <th>eps</th>
      <th>bps</th>
      <th>per</th>
      <th>pbr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>128940</td>
      <td>한미약품</td>
      <td>59385.666667</td>
      <td>421000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>130960</td>
      <td>CJ E&amp;M</td>
      <td>58540.666667</td>
      <td>98900.0</td>
      <td>CJ E&amp;M</td>
      <td>6301.333333</td>
      <td>54068.0</td>
      <td>15.695091</td>
      <td>1.829178</td>
    </tr>
    <tr>
      <th>2</th>
      <td>138250</td>
      <td>엔에스쇼핑</td>
      <td>14558.666667</td>
      <td>13200.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>139480</td>
      <td>이마트</td>
      <td>239230.833333</td>
      <td>254500.0</td>
      <td>이마트</td>
      <td>18268.166667</td>
      <td>295780.0</td>
      <td>13.931338</td>
      <td>0.860437</td>
    </tr>
    <tr>
      <th>4</th>
      <td>142280</td>
      <td>녹십자엠에스</td>
      <td>468.833333</td>
      <td>10200.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>145990</td>
      <td>삼양사</td>
      <td>82750.000000</td>
      <td>82000.0</td>
      <td>삼양사</td>
      <td>5741.000000</td>
      <td>108090.0</td>
      <td>14.283226</td>
      <td>0.758627</td>
    </tr>
    <tr>
      <th>6</th>
      <td>185750</td>
      <td>종근당</td>
      <td>40293.666667</td>
      <td>100500.0</td>
      <td>종근당</td>
      <td>3990.333333</td>
      <td>40684.0</td>
      <td>25.185866</td>
      <td>2.470259</td>
    </tr>
    <tr>
      <th>7</th>
      <td>192400</td>
      <td>쿠쿠홀딩스</td>
      <td>179204.666667</td>
      <td>177500.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>199800</td>
      <td>툴젠</td>
      <td>-2514.333333</td>
      <td>115400.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>204210</td>
      <td>모두투어리츠</td>
      <td>3093.333333</td>
      <td>3475.0</td>
      <td>모두투어리츠</td>
      <td>85.166667</td>
      <td>5335.0</td>
      <td>40.802348</td>
      <td>0.651359</td>
    </tr>
    <tr>
      <th>10</th>
      <td>136480</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>하림</td>
      <td>274.166667</td>
      <td>3551.0</td>
      <td>11.489362</td>
      <td>0.887074</td>
    </tr>
    <tr>
      <th>11</th>
      <td>138040</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>메리츠금융지주</td>
      <td>2122.333333</td>
      <td>14894.0</td>
      <td>6.313806</td>
      <td>0.899691</td>
    </tr>
    <tr>
      <th>12</th>
      <td>161390</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>한국타이어</td>
      <td>5648.500000</td>
      <td>51341.0</td>
      <td>7.453306</td>
      <td>0.820007</td>
    </tr>
    <tr>
      <th>13</th>
      <td>181710</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NHN엔터테인먼트</td>
      <td>2110.166667</td>
      <td>78434.0</td>
      <td>30.755864</td>
      <td>0.827447</td>
    </tr>
    <tr>
      <th>14</th>
      <td>207940</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>삼성바이오로직스</td>
      <td>4644.166667</td>
      <td>60099.0</td>
      <td>89.790059</td>
      <td>6.938551</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a7a8993c-b0a5-40f3-81b2-ad2bdb81c5f2')"
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
          document.querySelector('#df-a7a8993c-b0a5-40f3-81b2-ad2bdb81c5f2 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a7a8993c-b0a5-40f3-81b2-ad2bdb81c5f2');
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
pd.merge(df1, df2, how='outer') # 양쪽 데이터프레임의 컬럼명이 id로 동일하므로 생략 가능
```





  <div id="df-6887b279-f7a2-4ced-aa8a-f64e442a2fab">
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
      <th>id</th>
      <th>stock_name</th>
      <th>value</th>
      <th>price</th>
      <th>name</th>
      <th>eps</th>
      <th>bps</th>
      <th>per</th>
      <th>pbr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>128940</td>
      <td>한미약품</td>
      <td>59385.666667</td>
      <td>421000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>130960</td>
      <td>CJ E&amp;M</td>
      <td>58540.666667</td>
      <td>98900.0</td>
      <td>CJ E&amp;M</td>
      <td>6301.333333</td>
      <td>54068.0</td>
      <td>15.695091</td>
      <td>1.829178</td>
    </tr>
    <tr>
      <th>2</th>
      <td>138250</td>
      <td>엔에스쇼핑</td>
      <td>14558.666667</td>
      <td>13200.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>139480</td>
      <td>이마트</td>
      <td>239230.833333</td>
      <td>254500.0</td>
      <td>이마트</td>
      <td>18268.166667</td>
      <td>295780.0</td>
      <td>13.931338</td>
      <td>0.860437</td>
    </tr>
    <tr>
      <th>4</th>
      <td>142280</td>
      <td>녹십자엠에스</td>
      <td>468.833333</td>
      <td>10200.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>145990</td>
      <td>삼양사</td>
      <td>82750.000000</td>
      <td>82000.0</td>
      <td>삼양사</td>
      <td>5741.000000</td>
      <td>108090.0</td>
      <td>14.283226</td>
      <td>0.758627</td>
    </tr>
    <tr>
      <th>6</th>
      <td>185750</td>
      <td>종근당</td>
      <td>40293.666667</td>
      <td>100500.0</td>
      <td>종근당</td>
      <td>3990.333333</td>
      <td>40684.0</td>
      <td>25.185866</td>
      <td>2.470259</td>
    </tr>
    <tr>
      <th>7</th>
      <td>192400</td>
      <td>쿠쿠홀딩스</td>
      <td>179204.666667</td>
      <td>177500.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>199800</td>
      <td>툴젠</td>
      <td>-2514.333333</td>
      <td>115400.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>204210</td>
      <td>모두투어리츠</td>
      <td>3093.333333</td>
      <td>3475.0</td>
      <td>모두투어리츠</td>
      <td>85.166667</td>
      <td>5335.0</td>
      <td>40.802348</td>
      <td>0.651359</td>
    </tr>
    <tr>
      <th>10</th>
      <td>136480</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>하림</td>
      <td>274.166667</td>
      <td>3551.0</td>
      <td>11.489362</td>
      <td>0.887074</td>
    </tr>
    <tr>
      <th>11</th>
      <td>138040</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>메리츠금융지주</td>
      <td>2122.333333</td>
      <td>14894.0</td>
      <td>6.313806</td>
      <td>0.899691</td>
    </tr>
    <tr>
      <th>12</th>
      <td>161390</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>한국타이어</td>
      <td>5648.500000</td>
      <td>51341.0</td>
      <td>7.453306</td>
      <td>0.820007</td>
    </tr>
    <tr>
      <th>13</th>
      <td>181710</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NHN엔터테인먼트</td>
      <td>2110.166667</td>
      <td>78434.0</td>
      <td>30.755864</td>
      <td>0.827447</td>
    </tr>
    <tr>
      <th>14</th>
      <td>207940</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>삼성바이오로직스</td>
      <td>4644.166667</td>
      <td>60099.0</td>
      <td>89.790059</td>
      <td>6.938551</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-6887b279-f7a2-4ced-aa8a-f64e442a2fab')"
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
          document.querySelector('#df-6887b279-f7a2-4ced-aa8a-f64e442a2fab button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-6887b279-f7a2-4ced-aa8a-f64e442a2fab');
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
# 왼쪽 데이터프레임(df1)에서는 stock_name, 오른쪽 데이터프레임(df2)에서는 name을 조인조건으로 하되
# left join
```


```python
pd.merge(df1, df2, left_on='stock_name', right_on='name', how='left')
```





  <div id="df-fa85a8c5-0c61-4f2b-971b-2cfb919fdcdc">
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
      <th>id_x</th>
      <th>stock_name</th>
      <th>value</th>
      <th>price</th>
      <th>id_y</th>
      <th>name</th>
      <th>eps</th>
      <th>bps</th>
      <th>per</th>
      <th>pbr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>128940</td>
      <td>한미약품</td>
      <td>59385.666667</td>
      <td>421000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>130960</td>
      <td>CJ E&amp;M</td>
      <td>58540.666667</td>
      <td>98900</td>
      <td>130960.0</td>
      <td>CJ E&amp;M</td>
      <td>6301.333333</td>
      <td>54068.0</td>
      <td>15.695091</td>
      <td>1.829178</td>
    </tr>
    <tr>
      <th>2</th>
      <td>138250</td>
      <td>엔에스쇼핑</td>
      <td>14558.666667</td>
      <td>13200</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>139480</td>
      <td>이마트</td>
      <td>239230.833333</td>
      <td>254500</td>
      <td>139480.0</td>
      <td>이마트</td>
      <td>18268.166667</td>
      <td>295780.0</td>
      <td>13.931338</td>
      <td>0.860437</td>
    </tr>
    <tr>
      <th>4</th>
      <td>142280</td>
      <td>녹십자엠에스</td>
      <td>468.833333</td>
      <td>10200</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>145990</td>
      <td>삼양사</td>
      <td>82750.000000</td>
      <td>82000</td>
      <td>145990.0</td>
      <td>삼양사</td>
      <td>5741.000000</td>
      <td>108090.0</td>
      <td>14.283226</td>
      <td>0.758627</td>
    </tr>
    <tr>
      <th>6</th>
      <td>185750</td>
      <td>종근당</td>
      <td>40293.666667</td>
      <td>100500</td>
      <td>185750.0</td>
      <td>종근당</td>
      <td>3990.333333</td>
      <td>40684.0</td>
      <td>25.185866</td>
      <td>2.470259</td>
    </tr>
    <tr>
      <th>7</th>
      <td>192400</td>
      <td>쿠쿠홀딩스</td>
      <td>179204.666667</td>
      <td>177500</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>199800</td>
      <td>툴젠</td>
      <td>-2514.333333</td>
      <td>115400</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>204210</td>
      <td>모두투어리츠</td>
      <td>3093.333333</td>
      <td>3475</td>
      <td>204210.0</td>
      <td>모두투어리츠</td>
      <td>85.166667</td>
      <td>5335.0</td>
      <td>40.802348</td>
      <td>0.651359</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-fa85a8c5-0c61-4f2b-971b-2cfb919fdcdc')"
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
          document.querySelector('#df-fa85a8c5-0c61-4f2b-971b-2cfb919fdcdc button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-fa85a8c5-0c61-4f2b-971b-2cfb919fdcdc');
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
# 왼쪽 데이터프레임(df1)에서는 stock_name, 오른쪽 데이터프레임(df2)에서는 name을 조인조건으로 하되
# right join
```


```python
pd.merge(df1, df2, left_on='stock_name', right_on='name', how='right')
```





  <div id="df-ec3b27dd-6289-4884-af00-6e6dfaf41bb7">
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
      <th>id_x</th>
      <th>stock_name</th>
      <th>value</th>
      <th>price</th>
      <th>id_y</th>
      <th>name</th>
      <th>eps</th>
      <th>bps</th>
      <th>per</th>
      <th>pbr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>130960.0</td>
      <td>CJ E&amp;M</td>
      <td>58540.666667</td>
      <td>98900.0</td>
      <td>130960</td>
      <td>CJ E&amp;M</td>
      <td>6301.333333</td>
      <td>54068</td>
      <td>15.695091</td>
      <td>1.829178</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>136480</td>
      <td>하림</td>
      <td>274.166667</td>
      <td>3551</td>
      <td>11.489362</td>
      <td>0.887074</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>138040</td>
      <td>메리츠금융지주</td>
      <td>2122.333333</td>
      <td>14894</td>
      <td>6.313806</td>
      <td>0.899691</td>
    </tr>
    <tr>
      <th>3</th>
      <td>139480.0</td>
      <td>이마트</td>
      <td>239230.833333</td>
      <td>254500.0</td>
      <td>139480</td>
      <td>이마트</td>
      <td>18268.166667</td>
      <td>295780</td>
      <td>13.931338</td>
      <td>0.860437</td>
    </tr>
    <tr>
      <th>4</th>
      <td>145990.0</td>
      <td>삼양사</td>
      <td>82750.000000</td>
      <td>82000.0</td>
      <td>145990</td>
      <td>삼양사</td>
      <td>5741.000000</td>
      <td>108090</td>
      <td>14.283226</td>
      <td>0.758627</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>161390</td>
      <td>한국타이어</td>
      <td>5648.500000</td>
      <td>51341</td>
      <td>7.453306</td>
      <td>0.820007</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>181710</td>
      <td>NHN엔터테인먼트</td>
      <td>2110.166667</td>
      <td>78434</td>
      <td>30.755864</td>
      <td>0.827447</td>
    </tr>
    <tr>
      <th>7</th>
      <td>185750.0</td>
      <td>종근당</td>
      <td>40293.666667</td>
      <td>100500.0</td>
      <td>185750</td>
      <td>종근당</td>
      <td>3990.333333</td>
      <td>40684</td>
      <td>25.185866</td>
      <td>2.470259</td>
    </tr>
    <tr>
      <th>8</th>
      <td>204210.0</td>
      <td>모두투어리츠</td>
      <td>3093.333333</td>
      <td>3475.0</td>
      <td>204210</td>
      <td>모두투어리츠</td>
      <td>85.166667</td>
      <td>5335</td>
      <td>40.802348</td>
      <td>0.651359</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>207940</td>
      <td>삼성바이오로직스</td>
      <td>4644.166667</td>
      <td>60099</td>
      <td>89.790059</td>
      <td>6.938551</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ec3b27dd-6289-4884-af00-6e6dfaf41bb7')"
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
          document.querySelector('#df-ec3b27dd-6289-4884-af00-6e6dfaf41bb7 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ec3b27dd-6289-4884-af00-6e6dfaf41bb7');
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




- 색인 병합으로 데이터프레임 합치기


```python
df1 = pd.read_excel('./examples/stock price.xlsx', index_col = 'id')
df2 = pd.read_excel('./examples/stock valuation.xlsx', index_col = 'id')
```


```python
df1
```





  <div id="df-c5aa1a04-763d-406f-a238-4c3f1d333dfc">
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
      <th>stock_name</th>
      <th>value</th>
      <th>price</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>128940</th>
      <td>한미약품</td>
      <td>59385.666667</td>
      <td>421000</td>
    </tr>
    <tr>
      <th>130960</th>
      <td>CJ E&amp;M</td>
      <td>58540.666667</td>
      <td>98900</td>
    </tr>
    <tr>
      <th>138250</th>
      <td>엔에스쇼핑</td>
      <td>14558.666667</td>
      <td>13200</td>
    </tr>
    <tr>
      <th>139480</th>
      <td>이마트</td>
      <td>239230.833333</td>
      <td>254500</td>
    </tr>
    <tr>
      <th>142280</th>
      <td>녹십자엠에스</td>
      <td>468.833333</td>
      <td>10200</td>
    </tr>
    <tr>
      <th>145990</th>
      <td>삼양사</td>
      <td>82750.000000</td>
      <td>82000</td>
    </tr>
    <tr>
      <th>185750</th>
      <td>종근당</td>
      <td>40293.666667</td>
      <td>100500</td>
    </tr>
    <tr>
      <th>192400</th>
      <td>쿠쿠홀딩스</td>
      <td>179204.666667</td>
      <td>177500</td>
    </tr>
    <tr>
      <th>199800</th>
      <td>툴젠</td>
      <td>-2514.333333</td>
      <td>115400</td>
    </tr>
    <tr>
      <th>204210</th>
      <td>모두투어리츠</td>
      <td>3093.333333</td>
      <td>3475</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c5aa1a04-763d-406f-a238-4c3f1d333dfc')"
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
          document.querySelector('#df-c5aa1a04-763d-406f-a238-4c3f1d333dfc button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c5aa1a04-763d-406f-a238-4c3f1d333dfc');
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
df2
```





  <div id="df-b98b99cf-a1b9-491c-9919-014fe4c0a46b">
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
      <th>name</th>
      <th>eps</th>
      <th>bps</th>
      <th>per</th>
      <th>pbr</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>130960</th>
      <td>CJ E&amp;M</td>
      <td>6301.333333</td>
      <td>54068</td>
      <td>15.695091</td>
      <td>1.829178</td>
    </tr>
    <tr>
      <th>136480</th>
      <td>하림</td>
      <td>274.166667</td>
      <td>3551</td>
      <td>11.489362</td>
      <td>0.887074</td>
    </tr>
    <tr>
      <th>138040</th>
      <td>메리츠금융지주</td>
      <td>2122.333333</td>
      <td>14894</td>
      <td>6.313806</td>
      <td>0.899691</td>
    </tr>
    <tr>
      <th>139480</th>
      <td>이마트</td>
      <td>18268.166667</td>
      <td>295780</td>
      <td>13.931338</td>
      <td>0.860437</td>
    </tr>
    <tr>
      <th>145990</th>
      <td>삼양사</td>
      <td>5741.000000</td>
      <td>108090</td>
      <td>14.283226</td>
      <td>0.758627</td>
    </tr>
    <tr>
      <th>161390</th>
      <td>한국타이어</td>
      <td>5648.500000</td>
      <td>51341</td>
      <td>7.453306</td>
      <td>0.820007</td>
    </tr>
    <tr>
      <th>181710</th>
      <td>NHN엔터테인먼트</td>
      <td>2110.166667</td>
      <td>78434</td>
      <td>30.755864</td>
      <td>0.827447</td>
    </tr>
    <tr>
      <th>185750</th>
      <td>종근당</td>
      <td>3990.333333</td>
      <td>40684</td>
      <td>25.185866</td>
      <td>2.470259</td>
    </tr>
    <tr>
      <th>204210</th>
      <td>모두투어리츠</td>
      <td>85.166667</td>
      <td>5335</td>
      <td>40.802348</td>
      <td>0.651359</td>
    </tr>
    <tr>
      <th>207940</th>
      <td>삼성바이오로직스</td>
      <td>4644.166667</td>
      <td>60099</td>
      <td>89.790059</td>
      <td>6.938551</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-b98b99cf-a1b9-491c-9919-014fe4c0a46b')"
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
          document.querySelector('#df-b98b99cf-a1b9-491c-9919-014fe4c0a46b button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-b98b99cf-a1b9-491c-9919-014fe4c0a46b');
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
# 데이터프레임 인덱스를 기준으로 병합 (왼쪽 데이터프레임(df1) 기준)
```


```python
df1.join(df2) # how='left' 가 기본값
```





  <div id="df-785ad4b8-3d21-45da-95b1-dbbb957593a6">
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
      <th>stock_name</th>
      <th>value</th>
      <th>price</th>
      <th>name</th>
      <th>eps</th>
      <th>bps</th>
      <th>per</th>
      <th>pbr</th>
    </tr>
    <tr>
      <th>id</th>
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
      <th>128940</th>
      <td>한미약품</td>
      <td>59385.666667</td>
      <td>421000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>130960</th>
      <td>CJ E&amp;M</td>
      <td>58540.666667</td>
      <td>98900</td>
      <td>CJ E&amp;M</td>
      <td>6301.333333</td>
      <td>54068.0</td>
      <td>15.695091</td>
      <td>1.829178</td>
    </tr>
    <tr>
      <th>138250</th>
      <td>엔에스쇼핑</td>
      <td>14558.666667</td>
      <td>13200</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>139480</th>
      <td>이마트</td>
      <td>239230.833333</td>
      <td>254500</td>
      <td>이마트</td>
      <td>18268.166667</td>
      <td>295780.0</td>
      <td>13.931338</td>
      <td>0.860437</td>
    </tr>
    <tr>
      <th>142280</th>
      <td>녹십자엠에스</td>
      <td>468.833333</td>
      <td>10200</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>145990</th>
      <td>삼양사</td>
      <td>82750.000000</td>
      <td>82000</td>
      <td>삼양사</td>
      <td>5741.000000</td>
      <td>108090.0</td>
      <td>14.283226</td>
      <td>0.758627</td>
    </tr>
    <tr>
      <th>185750</th>
      <td>종근당</td>
      <td>40293.666667</td>
      <td>100500</td>
      <td>종근당</td>
      <td>3990.333333</td>
      <td>40684.0</td>
      <td>25.185866</td>
      <td>2.470259</td>
    </tr>
    <tr>
      <th>192400</th>
      <td>쿠쿠홀딩스</td>
      <td>179204.666667</td>
      <td>177500</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>199800</th>
      <td>툴젠</td>
      <td>-2514.333333</td>
      <td>115400</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>204210</th>
      <td>모두투어리츠</td>
      <td>3093.333333</td>
      <td>3475</td>
      <td>모두투어리츠</td>
      <td>85.166667</td>
      <td>5335.0</td>
      <td>40.802348</td>
      <td>0.651359</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-785ad4b8-3d21-45da-95b1-dbbb957593a6')"
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
          document.querySelector('#df-785ad4b8-3d21-45da-95b1-dbbb957593a6 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-785ad4b8-3d21-45da-95b1-dbbb957593a6');
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
# 데이터프레임 인덱스를 기준으로 병합 (공통된 인덱스만)
```


```python
df1.join(df2, how='inner')
```





  <div id="df-98aef9e0-528c-4d02-a8a2-92ca7d912b4a">
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
      <th>stock_name</th>
      <th>value</th>
      <th>price</th>
      <th>name</th>
      <th>eps</th>
      <th>bps</th>
      <th>per</th>
      <th>pbr</th>
    </tr>
    <tr>
      <th>id</th>
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
      <th>130960</th>
      <td>CJ E&amp;M</td>
      <td>58540.666667</td>
      <td>98900</td>
      <td>CJ E&amp;M</td>
      <td>6301.333333</td>
      <td>54068</td>
      <td>15.695091</td>
      <td>1.829178</td>
    </tr>
    <tr>
      <th>139480</th>
      <td>이마트</td>
      <td>239230.833333</td>
      <td>254500</td>
      <td>이마트</td>
      <td>18268.166667</td>
      <td>295780</td>
      <td>13.931338</td>
      <td>0.860437</td>
    </tr>
    <tr>
      <th>145990</th>
      <td>삼양사</td>
      <td>82750.000000</td>
      <td>82000</td>
      <td>삼양사</td>
      <td>5741.000000</td>
      <td>108090</td>
      <td>14.283226</td>
      <td>0.758627</td>
    </tr>
    <tr>
      <th>185750</th>
      <td>종근당</td>
      <td>40293.666667</td>
      <td>100500</td>
      <td>종근당</td>
      <td>3990.333333</td>
      <td>40684</td>
      <td>25.185866</td>
      <td>2.470259</td>
    </tr>
    <tr>
      <th>204210</th>
      <td>모두투어리츠</td>
      <td>3093.333333</td>
      <td>3475</td>
      <td>모두투어리츠</td>
      <td>85.166667</td>
      <td>5335</td>
      <td>40.802348</td>
      <td>0.651359</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-98aef9e0-528c-4d02-a8a2-92ca7d912b4a')"
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
          document.querySelector('#df-98aef9e0-528c-4d02-a8a2-92ca7d912b4a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-98aef9e0-528c-4d02-a8a2-92ca7d912b4a');
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




### 9.3 그룹연산

#### (1) 그룹 객체 만들기


```python
titanic = pd.read_csv('./datasets/titanic_train.csv')
```


```python
titanic.head()
```


```python
titanic['Pclass'].unique()
```




    array([3, 1, 2])




```python
titanic['Pclass'].value_counts()
```




    3    491
    1    216
    2    184
    Name: Pclass, dtype: int64



- 한 컬럼을 기준으로 그룹화


```python
grouped = titanic.groupby('Pclass')
grouped
```




    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x7f040107bfa0>




```python
# grouped 객체를 순회
for group_key, group in grouped:
  print('group key :', group_key)
  print('length of group : ', len(group))
  
last_group = group.head(3)
last_group  
```

    group key : 1
    length of group :  216
    group key : 2
    length of group :  184
    group key : 3
    length of group :  491
    





  <div id="df-dcd0122c-336f-4af0-bc91-0425e4461a28">
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
      <td>7.250</td>
      <td>NaN</td>
      <td>S</td>
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
      <td>7.925</td>
      <td>NaN</td>
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
      <td>8.050</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-dcd0122c-336f-4af0-bc91-0425e4461a28')"
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
          document.querySelector('#df-dcd0122c-336f-4af0-bc91-0425e4461a28 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-dcd0122c-336f-4af0-bc91-0425e4461a28');
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
# 각 그룹들의 평균
grouped.mean() 
```





  <div id="df-23a13456-d22c-46fc-a268-356abf852628">
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
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
    <tr>
      <th>Pclass</th>
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
      <th>1</th>
      <td>461.597222</td>
      <td>0.629630</td>
      <td>38.233441</td>
      <td>0.416667</td>
      <td>0.356481</td>
      <td>84.154687</td>
    </tr>
    <tr>
      <th>2</th>
      <td>445.956522</td>
      <td>0.472826</td>
      <td>29.877630</td>
      <td>0.402174</td>
      <td>0.380435</td>
      <td>20.662183</td>
    </tr>
    <tr>
      <th>3</th>
      <td>439.154786</td>
      <td>0.242363</td>
      <td>25.140620</td>
      <td>0.615071</td>
      <td>0.393075</td>
      <td>13.675550</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-23a13456-d22c-46fc-a268-356abf852628')"
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
          document.querySelector('#df-23a13456-d22c-46fc-a268-356abf852628 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-23a13456-d22c-46fc-a268-356abf852628');
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
grouped.mean()['Survived'] # 그룹별 생존률
```




    Pclass
    1    0.629630
    2    0.472826
    3    0.242363
    Name: Survived, dtype: float64




```python
grouped.mean()['Age'] # 그룹별 나이 평균
```




    Pclass
    1    38.233441
    2    29.877630
    3    25.140620
    Name: Age, dtype: float64




```python
grouped.mean()['Fare'] # 그룹별 요금 평균균
```




    Pclass
    1    84.154687
    2    20.662183
    3    13.675550
    Name: Fare, dtype: float64




```python
# 한 그룹만 추출
first_group = grouped.get_group(1) # Pclass =1 인 그룹만 DF 으로 반환
first_group.head(3)
```





  <div id="df-d6ea8c4c-2ada-4408-9840-78c560a9fe19">
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
      <th>6</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d6ea8c4c-2ada-4408-9840-78c560a9fe19')"
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
          document.querySelector('#df-d6ea8c4c-2ada-4408-9840-78c560a9fe19 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d6ea8c4c-2ada-4408-9840-78c560a9fe19');
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




 - 두 컬럼을 기준으로 그룹화


```python
grouped2 = titanic.groupby(['Pclass', 'Sex'])
grouped2
```




    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x7f0401056400>




```python
# grouped2 객체를 순회
for group_key, group in grouped2:
  print('group key :', group_key)
  print('length of group :', len(group))
```

    group key : (1, 'female')
    length of group : 94
    group key : (1, 'male')
    length of group : 122
    group key : (2, 'female')
    length of group : 76
    group key : (2, 'male')
    length of group : 108
    group key : (3, 'female')
    length of group : 144
    group key : (3, 'male')
    length of group : 347
    


```python
# 각 그룹들의 평균
grouped2.mean()
```





  <div id="df-4787bfb2-bbdb-4553-a123-3012191b5295">
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
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th>Sex</th>
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
      <th rowspan="2" valign="top">1</th>
      <th>female</th>
      <td>469.212766</td>
      <td>0.968085</td>
      <td>34.611765</td>
      <td>0.553191</td>
      <td>0.457447</td>
      <td>106.125798</td>
    </tr>
    <tr>
      <th>male</th>
      <td>455.729508</td>
      <td>0.368852</td>
      <td>41.281386</td>
      <td>0.311475</td>
      <td>0.278689</td>
      <td>67.226127</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2</th>
      <th>female</th>
      <td>443.105263</td>
      <td>0.921053</td>
      <td>28.722973</td>
      <td>0.486842</td>
      <td>0.605263</td>
      <td>21.970121</td>
    </tr>
    <tr>
      <th>male</th>
      <td>447.962963</td>
      <td>0.157407</td>
      <td>30.740707</td>
      <td>0.342593</td>
      <td>0.222222</td>
      <td>19.741782</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">3</th>
      <th>female</th>
      <td>399.729167</td>
      <td>0.500000</td>
      <td>21.750000</td>
      <td>0.895833</td>
      <td>0.798611</td>
      <td>16.118810</td>
    </tr>
    <tr>
      <th>male</th>
      <td>455.515850</td>
      <td>0.135447</td>
      <td>26.507589</td>
      <td>0.498559</td>
      <td>0.224784</td>
      <td>12.661633</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-4787bfb2-bbdb-4553-a123-3012191b5295')"
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
          document.querySelector('#df-4787bfb2-bbdb-4553-a123-3012191b5295 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-4787bfb2-bbdb-4553-a123-3012191b5295');
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
grouped2.mean()['Survived'] # 각 그룹의 생존률률
```




    Pclass  Sex   
    1       female    0.968085
            male      0.368852
    2       female    0.921053
            male      0.157407
    3       female    0.500000
            male      0.135447
    Name: Survived, dtype: float64




```python
grouped2.mean()['Age']
```




    Pclass  Sex   
    1       female    34.611765
            male      41.281386
    2       female    28.722973
            male      30.740707
    3       female    21.750000
            male      26.507589
    Name: Age, dtype: float64




```python
grouped2.mean()['Fare']
```




    Pclass  Sex   
    1       female    106.125798
            male       67.226127
    2       female     21.970121
            male       19.741782
    3       female     16.118810
            male       12.661633
    Name: Fare, dtype: float64




```python
first_female_group = grouped2.get_group((1, 'female'))
first_female_group.head()
```





  <div id="df-216d7d49-d69d-4cae-b368-79af8ae0916c">
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
      <th>11</th>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>Bonnell, Miss. Elizabeth</td>
      <td>female</td>
      <td>58.0</td>
      <td>0</td>
      <td>0</td>
      <td>113783</td>
      <td>26.5500</td>
      <td>C103</td>
      <td>S</td>
    </tr>
    <tr>
      <th>31</th>
      <td>32</td>
      <td>1</td>
      <td>1</td>
      <td>Spencer, Mrs. William Augustus (Marie Eugenie)</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17569</td>
      <td>146.5208</td>
      <td>B78</td>
      <td>C</td>
    </tr>
    <tr>
      <th>52</th>
      <td>53</td>
      <td>1</td>
      <td>1</td>
      <td>Harper, Mrs. Henry Sleeper (Myna Haxtun)</td>
      <td>female</td>
      <td>49.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17572</td>
      <td>76.7292</td>
      <td>D33</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-216d7d49-d69d-4cae-b368-79af8ae0916c')"
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
          document.querySelector('#df-216d7d49-d69d-4cae-b368-79af8ae0916c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-216d7d49-d69d-4cae-b368-79af8ae0916c');
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




#### (2) 그룹 연산 메소드


```python
grouped = titanic.groupby('Pclass')
grouped.mean() # 그룹별 평균
```





  <div id="df-3bad9bbe-930b-4520-88d4-f7f4e3d25328">
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
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
    <tr>
      <th>Pclass</th>
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
      <th>1</th>
      <td>461.597222</td>
      <td>0.629630</td>
      <td>38.233441</td>
      <td>0.416667</td>
      <td>0.356481</td>
      <td>84.154687</td>
    </tr>
    <tr>
      <th>2</th>
      <td>445.956522</td>
      <td>0.472826</td>
      <td>29.877630</td>
      <td>0.402174</td>
      <td>0.380435</td>
      <td>20.662183</td>
    </tr>
    <tr>
      <th>3</th>
      <td>439.154786</td>
      <td>0.242363</td>
      <td>25.140620</td>
      <td>0.615071</td>
      <td>0.393075</td>
      <td>13.675550</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-3bad9bbe-930b-4520-88d4-f7f4e3d25328')"
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
          document.querySelector('#df-3bad9bbe-930b-4520-88d4-f7f4e3d25328 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-3bad9bbe-930b-4520-88d4-f7f4e3d25328');
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
grouped.agg('mean') # agg(적용하고자 하는 함수), mean은 이미 제공되는 함수라서 문자열로 배치
```





  <div id="df-fba38b50-9697-4a7f-95e8-666ba0893103">
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
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
    <tr>
      <th>Pclass</th>
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
      <th>1</th>
      <td>461.597222</td>
      <td>0.629630</td>
      <td>38.233441</td>
      <td>0.416667</td>
      <td>0.356481</td>
      <td>84.154687</td>
    </tr>
    <tr>
      <th>2</th>
      <td>445.956522</td>
      <td>0.472826</td>
      <td>29.877630</td>
      <td>0.402174</td>
      <td>0.380435</td>
      <td>20.662183</td>
    </tr>
    <tr>
      <th>3</th>
      <td>439.154786</td>
      <td>0.242363</td>
      <td>25.140620</td>
      <td>0.615071</td>
      <td>0.393075</td>
      <td>13.675550</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-fba38b50-9697-4a7f-95e8-666ba0893103')"
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
          document.querySelector('#df-fba38b50-9697-4a7f-95e8-666ba0893103 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-fba38b50-9697-4a7f-95e8-666ba0893103');
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
def min_max(x):
  return x.max() - x.min()
```


```python
grouped.agg(min_max) # agg(적용하고자 하는 함수), min_max를 적용함으로 각 열의 최댓값과 최솟값의 차이를 조회해 볼 수 있음음
# grouped[['PassengerId', 'Survived', 'Age', 'SibSp', 'Parch', 'Fare']].agg(min_max)
```

    /usr/local/lib/python3.8/dist-packages/pandas/core/groupby/generic.py:303: FutureWarning: Dropping invalid columns in SeriesGroupBy.agg is deprecated. In a future version, a TypeError will be raised. Before calling .agg, select only columns which should be valid for the aggregating function.
      results[key] = self.aggregate(func)
    





  <div id="df-74c32d3f-b6af-4f2f-bb42-759ab2ba59a9">
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
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
    <tr>
      <th>Pclass</th>
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
      <th>1</th>
      <td>888</td>
      <td>1</td>
      <td>79.08</td>
      <td>3</td>
      <td>4</td>
      <td>512.3292</td>
    </tr>
    <tr>
      <th>2</th>
      <td>877</td>
      <td>1</td>
      <td>69.33</td>
      <td>3</td>
      <td>3</td>
      <td>73.5000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>890</td>
      <td>1</td>
      <td>73.58</td>
      <td>8</td>
      <td>6</td>
      <td>69.5500</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-74c32d3f-b6af-4f2f-bb42-759ab2ba59a9')"
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
          document.querySelector('#df-74c32d3f-b6af-4f2f-bb42-759ab2ba59a9 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-74c32d3f-b6af-4f2f-bb42-759ab2ba59a9');
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
grouped.agg(['mean', 'min', 'max']) # 여러 함수를 매핑할 수도 있음
```





  <div id="df-09c3c6ca-b7be-49df-8867-19fd997e37ed">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">PassengerId</th>
      <th colspan="3" halign="left">Survived</th>
      <th colspan="3" halign="left">Age</th>
      <th colspan="3" halign="left">SibSp</th>
      <th colspan="3" halign="left">Parch</th>
      <th colspan="3" halign="left">Fare</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>min</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Pclass</th>
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
      <th>1</th>
      <td>461.597222</td>
      <td>2</td>
      <td>890</td>
      <td>0.629630</td>
      <td>0</td>
      <td>1</td>
      <td>38.233441</td>
      <td>0.92</td>
      <td>80.0</td>
      <td>0.416667</td>
      <td>0</td>
      <td>3</td>
      <td>0.356481</td>
      <td>0</td>
      <td>4</td>
      <td>84.154687</td>
      <td>0.0</td>
      <td>512.3292</td>
    </tr>
    <tr>
      <th>2</th>
      <td>445.956522</td>
      <td>10</td>
      <td>887</td>
      <td>0.472826</td>
      <td>0</td>
      <td>1</td>
      <td>29.877630</td>
      <td>0.67</td>
      <td>70.0</td>
      <td>0.402174</td>
      <td>0</td>
      <td>3</td>
      <td>0.380435</td>
      <td>0</td>
      <td>3</td>
      <td>20.662183</td>
      <td>0.0</td>
      <td>73.5000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>439.154786</td>
      <td>1</td>
      <td>891</td>
      <td>0.242363</td>
      <td>0</td>
      <td>1</td>
      <td>25.140620</td>
      <td>0.42</td>
      <td>74.0</td>
      <td>0.615071</td>
      <td>0</td>
      <td>8</td>
      <td>0.393075</td>
      <td>0</td>
      <td>6</td>
      <td>13.675550</td>
      <td>0.0</td>
      <td>69.5500</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-09c3c6ca-b7be-49df-8867-19fd997e37ed')"
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
          document.querySelector('#df-09c3c6ca-b7be-49df-8867-19fd997e37ed button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-09c3c6ca-b7be-49df-8867-19fd997e37ed');
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
grouped.agg({'Survived' :'mean', 'Age':['min', 'max']}) 
```





  <div id="df-21a322cb-f782-4947-b250-b8bffb6641db">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>Survived</th>
      <th colspan="2" halign="left">Age</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>min</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.629630</td>
      <td>0.92</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.472826</td>
      <td>0.67</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.242363</td>
      <td>0.42</td>
      <td>74.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-21a322cb-f782-4947-b250-b8bffb6641db')"
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
          document.querySelector('#df-21a322cb-f782-4947-b250-b8bffb6641db button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-21a322cb-f782-4947-b250-b8bffb6641db');
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
# 데이터 변환
grouped['Fare'].transform(lambda x: x*1.2) # 파운드를 달러로 변환
```




    0       8.70000
    1      85.53996
    2       9.51000
    3      63.72000
    4       9.66000
             ...   
    886    15.60000
    887    36.00000
    888    28.14000
    889    36.00000
    890     9.30000
    Name: Fare, Length: 891, dtype: float64




```python
# 데이터 필터링
grouped.filter(lambda x : len(x) >= 200) # 그룹의 갯수가 200개 이상인 그룹만 가져옴
```





  <div id="df-0f1715b2-96f1-4b57-96d2-62aec0d3e83f">
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
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>885</th>
      <td>886</td>
      <td>0</td>
      <td>3</td>
      <td>Rice, Mrs. William (Margaret Norton)</td>
      <td>female</td>
      <td>39.0</td>
      <td>0</td>
      <td>5</td>
      <td>382652</td>
      <td>29.1250</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.0000</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.0000</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>707 rows × 12 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-0f1715b2-96f1-4b57-96d2-62aec0d3e83f')"
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
          document.querySelector('#df-0f1715b2-96f1-4b57-96d2-62aec0d3e83f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-0f1715b2-96f1-4b57-96d2-62aec0d3e83f');
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
grouped.filter(lambda x: x.Age.mean() < 30) # Age열의 그룹 평균이 30보다 작은 그룹만을 선택(2, 3 Class)
```





  <div id="df-3fa3b6ce-b0c2-4dd2-af27-7310b889ece6">
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
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>884</th>
      <td>885</td>
      <td>0</td>
      <td>3</td>
      <td>Sutehall, Mr. Henry Jr</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>SOTON/OQ 392076</td>
      <td>7.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>885</th>
      <td>886</td>
      <td>0</td>
      <td>3</td>
      <td>Rice, Mrs. William (Margaret Norton)</td>
      <td>female</td>
      <td>39.0</td>
      <td>0</td>
      <td>5</td>
      <td>382652</td>
      <td>29.1250</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>675 rows × 12 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-3fa3b6ce-b0c2-4dd2-af27-7310b889ece6')"
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
          document.querySelector('#df-3fa3b6ce-b0c2-4dd2-af27-7310b889ece6 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-3fa3b6ce-b0c2-4dd2-af27-7310b889ece6');
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
# 함수 매핑
grouped.apply(lambda x: x.describe())
```


```python
# grouped.filter(lambda x: x.Age.mean() < 30) # filter : 조건에 해당하는 그룹까지 가져오기
grouped.apply(lambda x: x.Age.mean() < 30) # apply : 해당식의 판별 결과 가져오기
```




    Pclass
    1    False
    2     True
    3     True
    dtype: bool




```python
t = titanic.groupby(['Pclass', 'Survived']).size()
t
```




    Pclass  Survived
    1       0            80
            1           136
    2       0            97
            1            87
    3       0           372
            1           119
    dtype: int64




```python
t.index
```




    MultiIndex([(1, 0),
                (1, 1),
                (2, 0),
                (2, 1),
                (3, 0),
                (3, 1)],
               names=['Pclass', 'Survived'])




```python
t.unstack()
```





  <div id="df-fd8f632a-45d6-4b7b-9337-1bbfacd28225">
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
      <th>Survived</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>80</td>
      <td>136</td>
    </tr>
    <tr>
      <th>2</th>
      <td>97</td>
      <td>87</td>
    </tr>
    <tr>
      <th>3</th>
      <td>372</td>
      <td>119</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-fd8f632a-45d6-4b7b-9337-1bbfacd28225')"
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
          document.querySelector('#df-fd8f632a-45d6-4b7b-9337-1bbfacd28225 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-fd8f632a-45d6-4b7b-9337-1bbfacd28225');
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
t.unstack().stack()
```




    Pclass  Survived
    1       0            80
            1           136
    2       0            97
            1            87
    3       0           372
            1           119
    dtype: int64



### 9.4 Pivot, Melt


```python
from IPython.display import Image
Image('./images/reshaping_pivot.png', width=600)
```




    
![png](/assets/images/2023-02-21-Data Analysis 5 (Pandas)/output_211_0.png)
    




```python
# Pandas example

df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two',
                           'two'],
                   'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'baz': [1, 2, 3, 4, 5, 6],
                   'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
df
```





  <div id="df-46884b85-c9b6-4b1a-a7d7-6788b6f5e724">
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
      <th>foo</th>
      <th>bar</th>
      <th>baz</th>
      <th>zoo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>A</td>
      <td>1</td>
      <td>x</td>
    </tr>
    <tr>
      <th>1</th>
      <td>one</td>
      <td>B</td>
      <td>2</td>
      <td>y</td>
    </tr>
    <tr>
      <th>2</th>
      <td>one</td>
      <td>C</td>
      <td>3</td>
      <td>z</td>
    </tr>
    <tr>
      <th>3</th>
      <td>two</td>
      <td>A</td>
      <td>4</td>
      <td>q</td>
    </tr>
    <tr>
      <th>4</th>
      <td>two</td>
      <td>B</td>
      <td>5</td>
      <td>w</td>
    </tr>
    <tr>
      <th>5</th>
      <td>two</td>
      <td>C</td>
      <td>6</td>
      <td>t</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-46884b85-c9b6-4b1a-a7d7-6788b6f5e724')"
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
          document.querySelector('#df-46884b85-c9b6-4b1a-a7d7-6788b6f5e724 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-46884b85-c9b6-4b1a-a7d7-6788b6f5e724');
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
df.pivot(index='foo', columns='bar', values='baz')
```





  <div id="df-92d1b309-9c09-4434-b3bc-1059e49caa6e">
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
      <th>bar</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
    <tr>
      <th>foo</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>two</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-92d1b309-9c09-4434-b3bc-1059e49caa6e')"
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
          document.querySelector('#df-92d1b309-9c09-4434-b3bc-1059e49caa6e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-92d1b309-9c09-4434-b3bc-1059e49caa6e');
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
Image('./images/reshaping_melt.png', width=600)
```




    
![png](/assets/images/2023-02-21-Data Analysis 5 (Pandas)/output_214_0.png)
    




```python
cheese = pd.DataFrame(
    {
        "first": ["John", "Mary"],
        "last": ["Doe", "Bo"],
        "height": [5.5, 6.0],
        "weight": [130, 150],
    }
)
cheese
```





  <div id="df-23944726-4a58-4713-8769-ab8a27930753">
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
      <th>first</th>
      <th>last</th>
      <th>height</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>Doe</td>
      <td>5.5</td>
      <td>130</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mary</td>
      <td>Bo</td>
      <td>6.0</td>
      <td>150</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-23944726-4a58-4713-8769-ab8a27930753')"
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
          document.querySelector('#df-23944726-4a58-4713-8769-ab8a27930753 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-23944726-4a58-4713-8769-ab8a27930753');
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
cheese.melt(id_vars=['first', 'last'])
```





  <div id="df-abb66366-99f5-42ef-aa85-d4df35d70314">
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
      <th>first</th>
      <th>last</th>
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>Doe</td>
      <td>height</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mary</td>
      <td>Bo</td>
      <td>height</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>John</td>
      <td>Doe</td>
      <td>weight</td>
      <td>130.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mary</td>
      <td>Bo</td>
      <td>weight</td>
      <td>150.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-abb66366-99f5-42ef-aa85-d4df35d70314')"
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
          document.querySelector('#df-abb66366-99f5-42ef-aa85-d4df35d70314 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-abb66366-99f5-42ef-aa85-d4df35d70314');
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




**추가 예제**


```python
Image('./images/long_wide_format.png', width=600)
```




    
![png](/assets/images/2023-02-21-Data Analysis 5 (Pandas)/output_218_0.png)
    




```python
df = pd.DataFrame(
    {
        'Item': ['Cereals', 'Dairy', 'Frozen', 'Meat'],
        'Price': [100, 50, 200, 250],
        'Hour_1': [5, 5, 3, 8],
        'Hour_2': [8, 8, 2, 1],
        'Hour_3': [7, 7, 8, 2]
    }
)
df
```





  <div id="df-09dd18b2-43a1-41ba-b265-981a417c5722">
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
      <th>Item</th>
      <th>Price</th>
      <th>Hour_1</th>
      <th>Hour_2</th>
      <th>Hour_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cereals</td>
      <td>100</td>
      <td>5</td>
      <td>8</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dairy</td>
      <td>50</td>
      <td>5</td>
      <td>8</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Frozen</td>
      <td>200</td>
      <td>3</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Meat</td>
      <td>250</td>
      <td>8</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-09dd18b2-43a1-41ba-b265-981a417c5722')"
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
          document.querySelector('#df-09dd18b2-43a1-41ba-b265-981a417c5722 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-09dd18b2-43a1-41ba-b265-981a417c5722');
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




**from wide format to long format**


```python
melted_df = pd.melt(df, id_vars=['Item'], value_vars=('Hour_1', 'Hour_2', 'Hour_3'),
        var_name='Hour', value_name='Sales')
melted_df
```





  <div id="df-c2e691fa-00be-4c04-9ea9-de7e7fb4eac3">
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
      <th>Item</th>
      <th>Hour</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cereals</td>
      <td>Hour_1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dairy</td>
      <td>Hour_1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Frozen</td>
      <td>Hour_1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Meat</td>
      <td>Hour_1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cereals</td>
      <td>Hour_2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Dairy</td>
      <td>Hour_2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Frozen</td>
      <td>Hour_2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Meat</td>
      <td>Hour_2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Cereals</td>
      <td>Hour_3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Dairy</td>
      <td>Hour_3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Frozen</td>
      <td>Hour_3</td>
      <td>8</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Meat</td>
      <td>Hour_3</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c2e691fa-00be-4c04-9ea9-de7e7fb4eac3')"
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
          document.querySelector('#df-c2e691fa-00be-4c04-9ea9-de7e7fb4eac3 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c2e691fa-00be-4c04-9ea9-de7e7fb4eac3');
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
melted_df.groupby('Item').sum()
```





  <div id="df-5156f7e7-4218-4bc1-be8f-e3cf6a55b432">
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
      <th>Sales</th>
    </tr>
    <tr>
      <th>Item</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cereals</th>
      <td>20</td>
    </tr>
    <tr>
      <th>Dairy</th>
      <td>20</td>
    </tr>
    <tr>
      <th>Frozen</th>
      <td>13</td>
    </tr>
    <tr>
      <th>Meat</th>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-5156f7e7-4218-4bc1-be8f-e3cf6a55b432')"
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
          document.querySelector('#df-5156f7e7-4218-4bc1-be8f-e3cf6a55b432 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-5156f7e7-4218-4bc1-be8f-e3cf6a55b432');
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
melted_df.groupby('Hour').sum()
```





  <div id="df-adf894f7-3f4d-4126-8b2a-4c1e281b97ce">
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
      <th>Sales</th>
    </tr>
    <tr>
      <th>Hour</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Hour_1</th>
      <td>21</td>
    </tr>
    <tr>
      <th>Hour_2</th>
      <td>19</td>
    </tr>
    <tr>
      <th>Hour_3</th>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-adf894f7-3f4d-4126-8b2a-4c1e281b97ce')"
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
          document.querySelector('#df-adf894f7-3f4d-4126-8b2a-4c1e281b97ce button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-adf894f7-3f4d-4126-8b2a-4c1e281b97ce');
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
# price 포함
melted_df = pd.melt(df, id_vars=['Item', 'Price'],
                    value_vars = ('Hour_1', 'Hour_2', 'Hour_3'),
                    var_name = 'Hour',
                    value_name = 'Sales')
melted_df
```





  <div id="df-49550e3a-06cd-4f85-bdb3-240968c604ab">
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
      <th>Item</th>
      <th>Price</th>
      <th>Hour</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cereals</td>
      <td>100</td>
      <td>Hour_1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dairy</td>
      <td>50</td>
      <td>Hour_1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Frozen</td>
      <td>200</td>
      <td>Hour_1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Meat</td>
      <td>250</td>
      <td>Hour_1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cereals</td>
      <td>100</td>
      <td>Hour_2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Dairy</td>
      <td>50</td>
      <td>Hour_2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Frozen</td>
      <td>200</td>
      <td>Hour_2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Meat</td>
      <td>250</td>
      <td>Hour_2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Cereals</td>
      <td>100</td>
      <td>Hour_3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Dairy</td>
      <td>50</td>
      <td>Hour_3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Frozen</td>
      <td>200</td>
      <td>Hour_3</td>
      <td>8</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Meat</td>
      <td>250</td>
      <td>Hour_3</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-49550e3a-06cd-4f85-bdb3-240968c604ab')"
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
          document.querySelector('#df-49550e3a-06cd-4f85-bdb3-240968c604ab button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-49550e3a-06cd-4f85-bdb3-240968c604ab');
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
melted_df['Sale_amt'] = melted_df['Price'] * melted_df['Sales']
melted_df
```





  <div id="df-c38228b3-d50f-4d04-9ed1-4ab5acf76003">
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
      <th>Item</th>
      <th>Price</th>
      <th>Hour</th>
      <th>Sales</th>
      <th>Sale_amt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cereals</td>
      <td>100</td>
      <td>Hour_1</td>
      <td>5</td>
      <td>500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dairy</td>
      <td>50</td>
      <td>Hour_1</td>
      <td>5</td>
      <td>250</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Frozen</td>
      <td>200</td>
      <td>Hour_1</td>
      <td>3</td>
      <td>600</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Meat</td>
      <td>250</td>
      <td>Hour_1</td>
      <td>8</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cereals</td>
      <td>100</td>
      <td>Hour_2</td>
      <td>8</td>
      <td>800</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Dairy</td>
      <td>50</td>
      <td>Hour_2</td>
      <td>8</td>
      <td>400</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Frozen</td>
      <td>200</td>
      <td>Hour_2</td>
      <td>2</td>
      <td>400</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Meat</td>
      <td>250</td>
      <td>Hour_2</td>
      <td>1</td>
      <td>250</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Cereals</td>
      <td>100</td>
      <td>Hour_3</td>
      <td>7</td>
      <td>700</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Dairy</td>
      <td>50</td>
      <td>Hour_3</td>
      <td>7</td>
      <td>350</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Frozen</td>
      <td>200</td>
      <td>Hour_3</td>
      <td>8</td>
      <td>1600</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Meat</td>
      <td>250</td>
      <td>Hour_3</td>
      <td>2</td>
      <td>500</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c38228b3-d50f-4d04-9ed1-4ab5acf76003')"
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
          document.querySelector('#df-c38228b3-d50f-4d04-9ed1-4ab5acf76003 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c38228b3-d50f-4d04-9ed1-4ab5acf76003');
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
melted_df.groupby('Hour').sum()['Sale_amt'] # 시간대별 매출량 집계계
```




    Hour
    Hour_1    3350
    Hour_2    1850
    Hour_3    3150
    Name: Sale_amt, dtype: int64




```python
melted_df.groupby('Item').sum()['Sale_amt']
```




    Item
    Cereals    2000
    Dairy      1000
    Frozen     2600
    Meat       2750
    Name: Sale_amt, dtype: int64



**from long format to wide format**


```python
pivoted_df = melted_df.pivot(index=['Item', 'Price'], columns='Hour', values='Sales')
pivoted_df
```





  <div id="df-382d483d-dac7-45f5-a489-2d0cb097c3d9">
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
      <th>Hour</th>
      <th>Hour_1</th>
      <th>Hour_2</th>
      <th>Hour_3</th>
    </tr>
    <tr>
      <th>Item</th>
      <th>Price</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cereals</th>
      <th>100</th>
      <td>5</td>
      <td>8</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Dairy</th>
      <th>50</th>
      <td>5</td>
      <td>8</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Frozen</th>
      <th>200</th>
      <td>3</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Meat</th>
      <th>250</th>
      <td>8</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-382d483d-dac7-45f5-a489-2d0cb097c3d9')"
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
          document.querySelector('#df-382d483d-dac7-45f5-a489-2d0cb097c3d9 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-382d483d-dac7-45f5-a489-2d0cb097c3d9');
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




### Workshop


```python
cpi = pd.read_excel('./datasets/CPI2.xlsx')
```


```python
cpi
```





  <div id="df-8ac74795-c672-4337-b477-55fedf8b1f4c">
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
      <th>Year</th>
      <th>Jan</th>
      <th>Feb</th>
      <th>Mar</th>
      <th>Apr</th>
      <th>May</th>
      <th>Jun</th>
      <th>Jul</th>
      <th>Aug</th>
      <th>Sep</th>
      <th>Oct</th>
      <th>Nov</th>
      <th>Dec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001</td>
      <td>175.100</td>
      <td>175.800</td>
      <td>176.200</td>
      <td>176.900</td>
      <td>177.700</td>
      <td>178.000</td>
      <td>177.500</td>
      <td>177.500</td>
      <td>178.300</td>
      <td>177.700</td>
      <td>177.400</td>
      <td>176.700</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2002</td>
      <td>177.100</td>
      <td>177.800</td>
      <td>178.800</td>
      <td>179.800</td>
      <td>179.800</td>
      <td>179.900</td>
      <td>180.100</td>
      <td>180.700</td>
      <td>181.000</td>
      <td>181.300</td>
      <td>181.300</td>
      <td>180.900</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>181.700</td>
      <td>183.100</td>
      <td>184.200</td>
      <td>183.800</td>
      <td>183.500</td>
      <td>183.700</td>
      <td>183.900</td>
      <td>184.600</td>
      <td>185.200</td>
      <td>185.000</td>
      <td>184.500</td>
      <td>184.300</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2004</td>
      <td>185.200</td>
      <td>186.200</td>
      <td>187.400</td>
      <td>188.000</td>
      <td>189.100</td>
      <td>189.700</td>
      <td>189.400</td>
      <td>189.500</td>
      <td>189.900</td>
      <td>190.900</td>
      <td>191.000</td>
      <td>190.300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005</td>
      <td>190.700</td>
      <td>191.800</td>
      <td>193.300</td>
      <td>194.600</td>
      <td>194.400</td>
      <td>194.500</td>
      <td>195.400</td>
      <td>196.400</td>
      <td>198.800</td>
      <td>199.200</td>
      <td>197.600</td>
      <td>196.800</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2006</td>
      <td>198.300</td>
      <td>198.700</td>
      <td>199.800</td>
      <td>201.500</td>
      <td>202.500</td>
      <td>202.900</td>
      <td>203.500</td>
      <td>203.900</td>
      <td>202.900</td>
      <td>201.800</td>
      <td>201.500</td>
      <td>201.800</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2007</td>
      <td>202.416</td>
      <td>203.499</td>
      <td>205.352</td>
      <td>206.686</td>
      <td>207.949</td>
      <td>208.352</td>
      <td>208.299</td>
      <td>207.917</td>
      <td>208.490</td>
      <td>208.936</td>
      <td>210.177</td>
      <td>210.036</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2008</td>
      <td>211.080</td>
      <td>211.693</td>
      <td>213.528</td>
      <td>214.823</td>
      <td>216.632</td>
      <td>218.815</td>
      <td>219.964</td>
      <td>219.086</td>
      <td>218.783</td>
      <td>216.573</td>
      <td>212.425</td>
      <td>210.228</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2009</td>
      <td>211.143</td>
      <td>212.193</td>
      <td>212.709</td>
      <td>213.240</td>
      <td>213.856</td>
      <td>215.693</td>
      <td>215.351</td>
      <td>215.834</td>
      <td>215.969</td>
      <td>216.177</td>
      <td>216.330</td>
      <td>215.949</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2010</td>
      <td>216.687</td>
      <td>216.741</td>
      <td>217.631</td>
      <td>218.009</td>
      <td>218.178</td>
      <td>217.965</td>
      <td>218.011</td>
      <td>218.312</td>
      <td>218.439</td>
      <td>218.711</td>
      <td>218.803</td>
      <td>219.179</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2011</td>
      <td>220.223</td>
      <td>221.309</td>
      <td>223.467</td>
      <td>224.906</td>
      <td>225.964</td>
      <td>225.722</td>
      <td>225.922</td>
      <td>226.545</td>
      <td>226.889</td>
      <td>226.421</td>
      <td>226.230</td>
      <td>225.672</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2012</td>
      <td>226.665</td>
      <td>227.663</td>
      <td>229.392</td>
      <td>230.085</td>
      <td>229.815</td>
      <td>229.478</td>
      <td>229.104</td>
      <td>230.379</td>
      <td>231.407</td>
      <td>231.317</td>
      <td>230.221</td>
      <td>229.601</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2013</td>
      <td>230.280</td>
      <td>232.166</td>
      <td>232.773</td>
      <td>232.531</td>
      <td>232.945</td>
      <td>233.504</td>
      <td>233.596</td>
      <td>233.877</td>
      <td>234.149</td>
      <td>233.546</td>
      <td>233.069</td>
      <td>233.049</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2014</td>
      <td>233.916</td>
      <td>234.781</td>
      <td>236.293</td>
      <td>237.072</td>
      <td>237.900</td>
      <td>238.343</td>
      <td>238.250</td>
      <td>237.852</td>
      <td>238.031</td>
      <td>237.433</td>
      <td>236.151</td>
      <td>234.812</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2015</td>
      <td>233.707</td>
      <td>234.722</td>
      <td>236.119</td>
      <td>236.599</td>
      <td>237.805</td>
      <td>238.638</td>
      <td>238.654</td>
      <td>238.316</td>
      <td>237.945</td>
      <td>237.838</td>
      <td>237.336</td>
      <td>236.525</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2016</td>
      <td>236.916</td>
      <td>237.111</td>
      <td>238.132</td>
      <td>239.261</td>
      <td>240.229</td>
      <td>241.018</td>
      <td>240.628</td>
      <td>240.849</td>
      <td>241.428</td>
      <td>241.729</td>
      <td>241.353</td>
      <td>241.432</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2017</td>
      <td>242.839</td>
      <td>243.603</td>
      <td>243.801</td>
      <td>244.524</td>
      <td>244.733</td>
      <td>244.955</td>
      <td>244.786</td>
      <td>245.519</td>
      <td>246.819</td>
      <td>246.663</td>
      <td>246.669</td>
      <td>246.524</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2018</td>
      <td>247.867</td>
      <td>248.991</td>
      <td>249.554</td>
      <td>250.546</td>
      <td>251.588</td>
      <td>251.989</td>
      <td>252.006</td>
      <td>252.146</td>
      <td>252.439</td>
      <td>252.885</td>
      <td>252.038</td>
      <td>251.233</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2019</td>
      <td>251.712</td>
      <td>252.776</td>
      <td>254.202</td>
      <td>255.548</td>
      <td>256.092</td>
      <td>256.143</td>
      <td>256.571</td>
      <td>256.558</td>
      <td>256.759</td>
      <td>257.346</td>
      <td>257.208</td>
      <td>256.974</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2020</td>
      <td>257.971</td>
      <td>258.678</td>
      <td>258.115</td>
      <td>256.389</td>
      <td>256.394</td>
      <td>257.797</td>
      <td>259.101</td>
      <td>259.918</td>
      <td>260.280</td>
      <td>260.388</td>
      <td>260.229</td>
      <td>260.474</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2021</td>
      <td>261.582</td>
      <td>263.014</td>
      <td>264.877</td>
      <td>267.054</td>
      <td>269.195</td>
      <td>271.696</td>
      <td>273.003</td>
      <td>273.567</td>
      <td>274.310</td>
      <td>276.589</td>
      <td>277.948</td>
      <td>278.802</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2022</td>
      <td>281.148</td>
      <td>283.716</td>
      <td>287.504</td>
      <td>289.109</td>
      <td>292.296</td>
      <td>296.311</td>
      <td>296.276</td>
      <td>296.171</td>
      <td>296.808</td>
      <td>298.012</td>
      <td>297.711</td>
      <td>296.797</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2023</td>
      <td>299.170</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-8ac74795-c672-4337-b477-55fedf8b1f4c')"
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
          document.querySelector('#df-8ac74795-c672-4337-b477-55fedf8b1f4c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-8ac74795-c672-4337-b477-55fedf8b1f4c');
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




- wide to long format


```python
melted_cpi = cpi.melt(id_vars=['Year'], var_name='Month', value_name='cpi')
melted_cpi
```





  <div id="df-28d3439d-8fa7-446f-a368-9f5f9e172d0e">
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
      <th>Year</th>
      <th>Month</th>
      <th>cpi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001</td>
      <td>Jan</td>
      <td>175.100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2002</td>
      <td>Jan</td>
      <td>177.100</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>Jan</td>
      <td>181.700</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2004</td>
      <td>Jan</td>
      <td>185.200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005</td>
      <td>Jan</td>
      <td>190.700</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>271</th>
      <td>2019</td>
      <td>Dec</td>
      <td>256.974</td>
    </tr>
    <tr>
      <th>272</th>
      <td>2020</td>
      <td>Dec</td>
      <td>260.474</td>
    </tr>
    <tr>
      <th>273</th>
      <td>2021</td>
      <td>Dec</td>
      <td>278.802</td>
    </tr>
    <tr>
      <th>274</th>
      <td>2022</td>
      <td>Dec</td>
      <td>296.797</td>
    </tr>
    <tr>
      <th>275</th>
      <td>2023</td>
      <td>Dec</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>276 rows × 3 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-28d3439d-8fa7-446f-a368-9f5f9e172d0e')"
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
          document.querySelector('#df-28d3439d-8fa7-446f-a368-9f5f9e172d0e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-28d3439d-8fa7-446f-a368-9f5f9e172d0e');
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
melted_cpi.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 276 entries, 0 to 275
    Data columns (total 3 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   Year    276 non-null    int64  
     1   Month   276 non-null    object 
     2   cpi     265 non-null    float64
    dtypes: float64(1), int64(1), object(1)
    memory usage: 6.6+ KB
    

![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABB8AAAKXCAYAAADKC+InAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAAP+lSURBVHhe7N0PXNRVvj/+l7/s2+h2A+5qwKYSYiG0YHBXdLjFKl6xMFjF8BZE/4D2JliboLvK5HevjdoqaKtQWzD9m6Cuo+hCYuJX8Isto9iFxAIpkVA3htWu4DUdv7m33/nMfMCZYYABGf7o6/l4fJj5nPkw85lBz5zz/pzzPqNyc3N/BBERERERERGRk/x/8i0RERERERERkVMw+EBERERERERETsXgAxERERERERE5FYMPRERERERERORUDD4QERERERERkVMx+EBERERERERETsXgAxERERERERE5FYMPRERERERERORUDD4QERERERERkVMx+EBERERERERETsXgAxERERERERE5FYMPRERERERERORUDD4QERERERERkVPdEh0d/Xv5Pt2Q2nH4zaX49zeL8bVbKEK9xsrlDvrhEi5d+QE//ADceustcmHvLp38FOVVx9HYCnhO+kfcKpeb/YD2b77GN4ZzuHSLC1zGOP68na6I8/p/4rz+51bcOlouu07njpbi06ONaPzeBT4effychFNlWcjbpYf+gjtCp/yjXOqgS434tOwzHG9sAX42Cf9o/YEREQ2566rjhEsG8Z1w5hzOXR2LcbezkiMiul7X1Xb92zGUfnoUjY2X4DLFHX1v+RL1HYMPN4RTKN+ch8JDelx0D4WPm1xscgV//c9SVH8LjJv2L30OPrT/Zx6WrX0Hn5wah1DlJIcrpktf78ZrH+xH3f/zwS+7/N5FHP2Pf0fODj3+5vlAN+d0Dsf2foqjjY24dIcP3H8iF8saP1kC1Wuf4JO/34uH/cehH+GLLlo+exVv7axDndv9156z/Rh0b3yAT8Rnq+9m+/qWe3H/XWNx7st3sOtT0bCeOKPv53Txa+z+4wfYX3cFPg+GYpLN+yUicq5LOPfFYewu0eGTclG3NZzHLXe4Y4KbQn5c1MrXU8cJpw6sxMZ3xHP/r/sGrN4mIroRtVe9gaW/fwvFxcVdNsuLiXbbro5q/QyvvlWIujoX3P/wfRjHSpkGwajc3Nwf5fs0YjVid/Kr2CXuRbz0OmL9La8oSSMf0pH3GTD1yfVIe3CcufjkbiSvl37DvoCn1+OFfx5nqvzSc6sB/6ew/qUHIP+2Hddep3sRSHs9FlNv7eacrPT0nsSjHyfj1T+LO/PS8PqjU21GVlg4fxhvrMiDeAfdk99bu73ndOD3O96D3XNy5PWl4+e0I8903FQ8tTYND9wpP0ZE5GxXRH2b/QZ2HW+XC65xCXoKv1tirvvt17u91P0WxzpcbxMR3eTaj+qQV3ZK3pP8gHN1jTgn3f35bMRONbedzx3XofwLcceqXr3WhrZPbmte7OgLdLTPzY8SORNzPtys/r+xmOo/1Xqb6gkX04MumPRT872+GOtm83xdtrGD39gU73OcxTl4ym/L5S6L83Lr4azcZuD53FzkmrbfYb5cPPvF1+Wy3G6CJzJHXn8sa3siGjqnDhSYAg8uyjj8boO5bnt9wwuYP1k0gGveg05vau72atxki3rtrr5/hxARkZnLtFikvZR2bUuOwFT5MXxRDt12nWkzBR66uBVjO+piq82nh4uIRIODwYeb1d2zrSs1aVv8gDn44BIMnwl97RC7IGCx/DzJcVgwawYC/AMQEDIbCx5/Xn6NB3Du7WQkJ/c2QkL4WwuOy3eP/bVFvtcPLgGI7XyPSZh/j7nYc85TcpnYnp7hWGUszqlRvlve2BGNlqa8ZCFLbDq9XGTJkdeP9OEVQCIaIqdw/LBUn01FxNwH4CMHY291C8D8X0WY7lfXNqLrmIiugn/1Qme9liTqNSIiGgiX0HigFJ+Key7KJKzvvCiWi9/9ynyEtUmY3dHGtNpi8YB8BNFQYfDhRnD+nGg+mjX+zbErVF2dw+ESnel5Jj04A1NvN5d2qnsPK5OlwIEOx3+Qy+w4d7QAr760Gq++/p45KvvuG3j15ReR/vZh01CxjtERHSMA7PsBjVXlncGHlm3lqL4o71wXceJ/l+/9vYc3YZf1OeHjUhw+L935AZfqjuO42Br/ZnqkBz9I+TtN2o3yHSKiIfbDBdNP3HqrdRj01v8l7xtF3WW+R0REg+niKXz67it49c+NgMsMLJjlguN7S1Eqb4c7rooRjRAMPtwAfmg51ZlToLHhlENXqKxcaREVW5Z5NMLk+YidY+dKvIunPGSrh6kTPxzHp9pyNCIAC9IyzVHZP76M2J+LzrY+D6VHb5VHR1wbAWDPpbrd0P1ZCoMEY4ZSilJ8ijc0pTh1xfTwdWhHS435XuOpFvSl+3/pRGnnOc2eNUncViPvT7vReMkH83uMPlv44Rxa6sx3W75uMc/b6+I4dmvMIyneq+pvIImIyFGe8PlnqZ5txO5Pj12rF//nHA5XSNfZgIAAH3lKHhEROd3FFhyrKocudzXSX3oF7/1FtAcnz0ZS+lN4wK0dx+QpF91Pu+jQMTr3PRz+Tux2XqychHFWyemJBg+DDyPeJVFBlcr3hc+qcVyqYBzxP5fQ8kUp3liz2lyx3fkAnnpmftdRD5K7IvCUacjWfPh0F324dAnnTJEPF3hOkJuqYz0x6S7z3XMXj8mVYB52f20usyJVtnvfwCubRadePMeMJxfgqfjnsWCyeOwLHV7JyILuL41o/x/z4X12uhHH5Ls4eByNjkQfxGd06i/v4ZU/7DJNuQhYPB+xj8aZz+nkLrz64kq8d9CxQMYPDdXo/EvVfIrj3YyUOHfSPJLi3HUHW4iIenMrpobFYoaostv3bsGL6atNwc/VK1YiTy8q9MnzERHiyalhRESD5VIjynMLUFrVgnYXT8x49AWsfzEOMzxETWyVi6y3C18do3PP4Qdp5O8PYl8qdrGt0UuRtUQa3ZyFT3sdxUt0fRh8GOlOf4rSv0h3HsDsWVKHvxp5u6t76Qyfw7Gd75mmQ6z+ow7VoqIZ94tYpK14Cg9IFVt/uUzFDNM5fIo31r8B3d5S7Hp/C/L2So/NwIyp4zqnKLTYGZ5x7tgubNleLc7OBQELn0fcg6LBe5sP5i9Nw/ypUstYmtrwA27t17/aH3D8cKkp4uviIp1jOUr/s/eRBef+kodX3v3UNErBJzwJsWGTrM9JPHJJNMt7/9TO4fD+cnE7CTOUAeL2ON77xN7fScpAbP5C6TGRJRHRQPnpDCStWYPnf/UAfG5rMdXRl24PwOzHpQbvAvsBaSIico47Rf37aCySXnwZmevXIGleAMZ1s9a95y/SkJYmtl94yiU9+K9z5unDd43DOKuG6zj4mEY3j2OgmZyOwYeRTFoe7QOd6Yr8pEceQOzcBZC6tTj4BvL2nephju44TLrrkujIu8AzZAGSVmZiza8jYOpLX5exCIhfg989PhvBt59C6XYddjf8AJ9ZcXhh5VOY8dOOKQqZSPqF/CsWxiljReM3Ds+vfhkvRPqIZ5PdPhUL0tYjc+ULiJs79Vp5H/xwshS79rYDLrMR93QEpIkTx98v6DXCO+7BBYj9eQAinv4dnv/XGfC8TX7AdE5rsF5U+AtCequsf0Djx3l4Txoa9/MHMH9xBCKkz1r6Ox24jmSaREQDZawngh95Cr+Tg5+Zvxf1bXj3DV4iInIWFwTMi8CMn7vguClRe/fbiy9nIStLbJ850J6cusA8YqLL0vnBWJAqjW4WbXUu9U5OxuDDCHXpr4dRsOlV7DopdkSHer6Up+HOBxCXPMM0N/fYtlewMmsXjneTqFFauz1zcyZ+F+UDl//XgkaDIxMHHDEWPuFxeH7lenMFt1Z02uNnI+Cn1t3zjsST40bLBSbjRON3NoIn2ouC3AqXyQGYJF+B60uk9wfDp3jvdXnaxLwHECCtjzzP9CnhvYxXsft4T+99EiJefAGx/yw+J9P/lh9wydCI4zWfonTvp6huPoVjfzmMY8dPwWWuudGea7l+/f+049ift+ANKVEQfDB/3gx43j4VEY/Jf6f81Vj57qdo4RQLIhqOrlzCpUvncO48U04SEQ223pax9+lPsMBUr7NOp6ExSnSWfpTv04hxDoffftU8H9clAAv+LQnzp3RcnvoBp/a9gS3bjgHKp5AWH4BT75qXtpz65PouQ/kbP07Gq38Wd+al4XXLTrOs/agOeWWnALcH8JSjS1JKRKe78bNqVNdV49Q3LTj+V3mehZS48i4fTAoJxuygbq6qndyN5PW75J1edHPeHVoOis/ifWkqh3hp0+fxgHn0wpVTKM3dAt1R6bx8MP+3LyDg+Is9fhY/nC5H3p8KTNNUujPun59H2tPBnZ9TS9kWrP5QyjQh5bBIw1PSVBLTIz+g5S8F2CJN6fh5LF7+VxfsfjkP1aZpF2l4gJFnIhoEpw6IerDGphFqPIfjJ62npUnfHwvaV9qpI9tx+M0elk+2OLa37xsiIuqbrvVqO45ty0PpX6VHf8C5ukZzG9jFBe3tlnOeF+B3K4FXTe3tCKS9HouprJRpEDD4MFJdPAbdu8cxaeF8zLiraw/+0l+P4dStUzH1zkudDcP+BB/65YdT2J35ijwqwxMBQcGY2hFl+O8WHPvimByMCBAd7Re6drS/KUfWzo71O+z74Zy8tGVv5/3dYeStz0PjPbFIio+Aj+XcZWlUQsl7KP8hAkkLp6Klp8/i0jEUrN6C8nYpH8VTiP1nabnQjiOk0RDHUPrOG9gt3rPLwpeRGSlN7JCcw+H3C3AuIA6zg8Z1mTIi/Z0ab5mKgNuq8cYKBh+IaHC1f5aH9DcPy3sSF3hO9TSP9BrrCZ+7PeF5l9h3m4Rba+wFaC0bunb4LjBNo5OOZfCBiKgvegnuWrKoVxv/LOraj83FknGTp2Kcwnx/7J0+8Bk3DuO8JsHnf44hfTODDzS4GHy44V2ruAYr+HDpaAFezJaSKz6A5zc/heAuycpaUL55NQqkZSd/9TvkPuJjLu6Dvpz3pb+1iJrXE2N7mWTU43N2jsbovoJur3oD6bnVgP9TWN9lPl0vzh9m8IGIhrXr/b5g8IGIqC+uteEnKe1fbOzkNhWzQyb1rV51oG1LNNCY84HMDhVgi2kZzO63UmkkgwPG/uM4U0JHKafC8WMtsJpW9j8/oP10Ixrlq2TBP7WX32Fgjb2z98BDrzwmYbbpVEtRuvc4zlm9KWnkQzXK95tHa/j4TjLlcyAiIiIiul6nGo/jWN2x7re/HMep/i5FTzSIGHwgs3bz8mo9be1GB5PTTJyNpx6X8h60o/zt1XjRtHawvP16CdLXvIfD7S6YOjcJC4L7ND5g6IwNQOyLcQi+Ezj25yysfHGJRbbhJXjxZWnKhfk9PTWnj5FnIiIiIqLu/K3Rbtv82taOH/4uH0s0jHHaxU3gh0uXYAob3DIWYzuWihwMP1zCOcMptLS0oKUjU/rt4zDpp+MwbpInxo3tfxf9VFkWdEfFHYv5xNfrkuE4TrWJOwppnrNLN88pjXI4hVPiPZ36m7xKxq0u8PyZeE8TJsHzdoYdiOgGJWVINzVub8XYftTf546Wotog7vRneDAREQ2s/xFtWvnC4q1jx7JOpkHB4AMRERERERERORWnXRARERERERGRUzH4QEREREREREROxeADERERERERETkVgw9ERERERERE5FQMPhARERERERGRUzH4QEREREREREROxeADERERERERETkVgw9ERERERERE5FQMPhARERERERGRUzH4QEREREREREROxeADERERERERETkVgw9ERERERERE5FQMPhARERERERGRUzH4QEREREREREROxeADERERERERETnVqBONjT/K94mIyMnGjhkDT09PeY+IiIZaS0sL62UiokHAkQ9ERERERERE5FQMPhARERERERGRUzH4QEREREREREROxeADERERERERETkVgw9ERERE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WxLhx5UpyAKDD0REg4jBByKi4YXBB+o3YxuMo4fRaDBHXDWi7aoC8gA3pzC2GaFw5gvQiMWEk0RERERERH2lGGGBB8lo5wYeJAw8UHcYfCAiIiIiIiIip2LwgYiIiIiIiIicisEHIiIiIiIiInIqBh+IiIiIiIiIyKkYfCAiIiIiIiIip2LwgYiIiIiIiIicisEHIiIiIiIiInIqBh+IiIiIiIiIyKkYfCAiIiIiIiIip2LwgYiIiIiIiIicisEHIiIiIiIiInKqUd9+++2P8n0iIiIiIiIiogE36kRjI4MPRESDZOyYMRg7dqy8R0REQ+3SpUusl4mIBgGnXRARERERERGRUzH4QEREREREREROxeADERERERERETkVgw9ERERERERE5FQMPhARERERERGRUzH4QEREREREREROxeADERERERERETkVgw9ERERERERE5FQMPhARERERERGRUzH4QEREREREREROxeADERERERERETkVgw9ERERERERE5FQMPhARERERERGRUzH4QEREREREREROxeADERERdTIezMTSLVVol/dvON9VITtFhdLv5H0iIiIaFAw+EBER3ajaq6Ddko3sT5rlgt7UIm+NGkehgItc4jBjK2oPVqDiqwEMWzjjOX+qEG8zG+r8WrmAiKh/jO3taG83ynvUI6P0WbXDeFXep5sSgw9ERETDwffmhpmjDdnWg7lY/tgM+Lq6wtV1GiKeXo7cQzad9O+bUbpaBVVtq1zQs/aPc6E6koDl8YFyiSA3GO1vFuf6XxXYGBWN6F0NcoFjzI130SCV96308JxVG6T37ciWAN238i+ZBCLhtwmoXZ2LYo5+IKJ+a0Xxi17w8spGlVxC3Wv9OFV8Vl7IrpEL6KbE4AMNkQuoe/8lzJvpgyk+s5D06l6c+bv8UKfT0D3rg5Sdp+V9+y6UrhTPIZ4nKg91chkR0YjQ3oDiDUsRESw6yXdJjVhp8xAd5mmISMlEcb39K/6tu5IRFpWJhsmp0NY2o7k2Fwk/a0DmQyFI3uVYoKGrVpRu1wKJUQj7qVwkdDQY7W/X0ej+rgKZsdPgIT+XR/BiqBweoQGEpIj33dzzdviPUfLR1lwejEIytND24fWIaIjJ9WW00tcUWPRVRmPphmI03LBzxBxVhUw52JqwvZv6/1g2wuwGY4kGF4MPNCTOFv8O0a8DKdsbcaJmM4I/W4Ilb9Tgivy45MJ+DTZdWImUhRPlEnsuQL9/GzB+PMbX/Rl6Rh+IaKQ4pUNyyAykH3RBwptH0WxoQ1ubtBnMwQSXCqQrQ5C83baDXAXtb3Vojc+Edl0CQia5wGVSCBLWaZEZ3wpdlg79mlDwbQWKdgHJc2dYTblwf1Qrn5flVgF1gHjwES+4mw/rG2MtMh+LhrotFtsaxPOda8C2BW3IfiwZmUcGYQizywzMTQRK91Whv6EaIhpE34sO9qMzRD1XAZeFamwr3gb1475o/SgBM0JEh/qUfNxNrvjjCrt1Wm35tv59LxANMAYfaAichb50L8Y/8SSiJ4jdO4IQ+9Q81H1QiUbzAcCVGnzw2ieY9/xi+MtFdp3X48B2YN5vliEadfjzXxh9IKKRoBnatGToJqmg/UiNhOlecFHID0n5FkzBhG3QrvKCLmk5tCflhyTfNuOoaF1GzQ6xycvggpDZUcCxw2jox5Wt9toqFCMKIf4OZHs4VoFtx4CIh0PgJRf1RevHG6E+EgL1JhUipOjFaHdErM6EenoV1DnFDgUEqnJsR2F03Wa8WCwfbcsFvtPFZ7WrCkdv+qumRMNf8y5RPxwR9d6fiqBdEYuIByMQ+8JGbDt4GNqs5YiaJB94s9ulQ6nl94VJLSp0DD3Q8MDgAw2BMzhTAgR7SZEHs/ET7gPOfokTBvP+mZ3Z2DR+GZLm3GEu6MaFIweggxKzlA9BGQ/UFVVy6gURDX+tojG4TzSkE2MR8hO5rAsFQh5LRhRKUWGZs+Ef3U0d/qq6Bps8CUY01EmTILzg/o/mkr5orisSP33h1Wsjvhm6nK2odU9G0kP9Cj2g6v8UA7MTECWNnugUiKgnwkTjuRRVDg9HUKG0y6gM202L2J/Jh1vwuttX/CxCbZeGOhENN61/LTXd+t5tU+f8xBdRjwSK2lL2rQ4J0vSCDTYTwuyVH8k0TVWQRls1fJyJpXOnydM5kpFb0zUq2X4ou/MY1+AILN1SgVZ7yRNbq6DbshwJHcf6zsDi1cVo7jhWngLha3uOQvN70eJ3fMU5yQV9EZ+ABPF9Uay3GS1nChYHIiHe/jS09hotVJb5g1KyUXrKZgRaHz8r86pCEZhm8ZwV9oLivX1W3xVjqVT+tK5LUNpYrjL9TrdTTWhYYvCBhp9Lemhf+xuWLf0VroUn7JGnXPjPwn1ed+C+6fMATr0gopFAcRtcxU275Vwze64aTUteuio6m9bid8OweF0IWjep8fL2WrRLjbSr7ajd/jLUm1oRsm4xwiwOd4xRdNJFA252b9MojKjakIzkj4DYtamIsMgN4bhWNJSLm+m+XUZNePmGiJ8VfRi5oUaEqXHb82a3cXqnF8LEubS1DcI0DyK6Lr5Byaa6SfuudsBzPJSuisaM1Ufh9euN2JaXjmn/pcPyx9QotXid5u3JCHlIBa2oNTZ+UISiNbHAPhVUH8sHdGqGNjUC6oO3IWL1NjQ0H0XpskAc3ZKAhE1V5oBxgKjDA0RNqClFhVX104yKjysA92SETZeL+uJ7X/jGi/ezvUI80zVVe7aiNmAxfO/qOhLM9L5mL0XxT5NM+YMaDqoQcmorFgdG250C58hnZZpSGBoBVT4Q9gctiorViB1dCtVq29d34LP6aRiiEsVtlxEd4rvooE7cRiE6tF+T/2iIMPhAQ2ACfBYCe7880Znj4UyDHhh/H6Z4AHUfZaEoIg1PBIn7HUkpZ87DsvfrrHJCdEy5GD/vn0xTM8YHzsIsTr0gopHAlHPAHRXr1D3MVW6GboMaFaIhOne69VSIwCWiIbbGCxUZYfAaJzrY40RHOqMCXn+ogG6JxUoVDmtHq3QeLoprVxBttTdA99toRKxrRuwfi7D10f6MepCIJqUUCxht3rNyi/Tq4kEHlmJzJOFkx5b9iJ3GqcK8nGjzdwPckyGiAecyV4XcFWHAR0sxI2QGkjfoUPVt185xfzT7JODwQS3SH41AxKMqbFwlXqc1F1X18gHfFSMzSYdW91QU7dqK5EfCEPZIMrYWl2LrI/IxnbyQ8K4BR3VqJDzoC3cXL4T8Wg31AqB2XamcdyEQUf8WIV5DiwrLDv63VSjdB7gnhkEKw/adO8IeTgDKdajo6KgbK1CqaUVgrHhO2zq3431NV0Obk2zKH+QeEAv1u+I9uldBvSyvS56IXj8r8V1S/Idk6FrdkbpTfE/8OgphUoJf8Z1RmmM78sKRz8oFYQvSxTuzHdFRi6p88V2xIBphdka20fDF4AMNgfFQispxfK4WRWfE7tkD0ObrEfTcbPgbipCzFkh6bBawfz0SO5JSbk8BXn8W6/dfMD+FcPbTP0Mnnivhn4PMBV5KzHsAqNOUg6v4ENHw5oKINVqo7q1A8txoLH2zGFVftZqXr2xtQNXHuVgaFYHkcl+oPnjFzggDF4S8kIvDDQa5g21AW8Nh5P460NSh7vQTL/E6ojEXeB1Xhr5vRsWbSxEtNfj3uUP9SRVyn/LtPkjRKwUU0nSLi0bzlS0Lxott4megFBfoXsfSnw4EKDpdsVkWlIhGGNEJXVWEKtHxVSmN0K1LRoT/NMxIykaFnYFNfRHyLxHwtZj+5uUvOtRCwxnzE7cfqYBW3IatSkaY1TQ5UZfZC6L+xLoCM7Yr4DJZuteAZnlUl9dD0hSJVmTuk6/wC+3VFSgW3ezk2f0LPUjcTSv5VCD3E3PX3Side2sgFs/uGpRuPVhkel9RiVGi1rUgjTb4V3F7bBsqjpmLOvT2WaH9MCryxe1slXgf1p+D4jb5jiUHPivFg+I7UnxnWI3oOFaFIvGSUY+E9TJaj4YbBh9oSNwxZyU0//s25D0qjWr4PRr/WYPNz/pA/+46VCen4Qn/s9Dv0eLsv8pJKSdEI+Ffz+KDPXqcNT2DOWkl8BB8PC/gwgVpc8GE6VPEQ1roGX0gouHuJyFIz1EjqlU0DvUVKMqIMCdK9F0M1R+zoT3Yiqi1uUif3l1PvBW6pz3k5IrS8pxdpxu4ekVg6WoVVJY5I+wSDT4pwCE69FZd9FM6JNw1DdFvtsJ3WREaqrRInelAQsoe3YvA2eKmvBZfmQs6fVVfIX6GIfAe8749PS/92dNmsyyoHLzwuL2nSAcRDScuAVFIf/coDHWl2LrEF23bVYgOS3bqahfmfDii4+3r+GivVilIIuc88Jgdi8xd8gMdRAc/Il7cbilGhWnwVTsOl2v7P+Wig0sYol4Aaj+sQK00NUE8Z+vsZJv8OmatJ6X6FpjmY/u+FPANlEYp1OJoYx8jOydrYfq07Eyr606vnxUCERYbKL4zclEsB0NqD0qrd3DKxUjE4AMNkdvg/+Rm7D3UiBONB5D3u1mYUKfFptxgrHxaKR49g8adwLwp17I+TJgyD9h5QjwiGPTYWyLd0SLln4MQHGTeEjafEGVnof0Low9ENAJ0XDXzj4Vad1ROkHgUpesS5Ad64o6oP3adYmC9bUO6fHTPXOAqteE+bjbNiOg0KQrqqgYYqrdh46/D4G7vKl+fKTDtwWS4H9uKbeUWoY7vK7Dtj7VwF68zrYd4gP2lPx3Z0q2HMrc2Q5qB7P7T6w2mENFgU/xMWhGoCKV/Ep3kVh1UH3VN3jhQjFfNtaLiFtNNL0S9kjINvlFatAemY1ujAW3VpVA/Jj/cyQVzH5OmE+Rin77dNGJgn0bUR/2ectFBgZC54nlNoxbMUxMiHg2zGwjoeF89MfZlhJlEHG96Voe+Kxz9rIDAR5IRgVpsOyiN6KhFxYfillMuRiQGH2iYuIADH+UBGamI9pCLenD2yCeQxj0kvleHE41SAKNjK8ZKf/H4B5WcekFENzyFiwtcetxcRVPUMV73xoqfRzuHupopRLm7+FmL3IXRiF5Vah2c6CeXuammBmb2vz2JzIPNaD9VgczEZGQjFuqUCOupI92o2mBnpIfdLQE6OwksW5uPip+x8OUSfUQjlldYNKRr9K11NoHTAeQ+yZyr4GizA69wRIf0/GZE/SkXW38dAd+fdl8DKx6MQLI7kLvvMFqrK5B7nVMuOiimhyHBXXTUl6mQ2xqBKKX9MQjmOl+8r0ab1TGE5q9MoVlM8+njyAJ3L9PfA478PfrwWWFyBGKlXBDSiI6TR1F6jFMuRioGH2hYuFKjxabSh5CyUEodKTEnpaxuNo1zMDnTXA0snCIe6ZhykQBloO0EMn8EzxvPqRdEdBOQpl3Y62xbbhFQy0f3xj0wDGEoxtEGe7kRjGgvr0DFtz0kZ/xZLLTSCIMVjjSevRCbXYptiS7QJU2DV2AydLcnQ1u8FbF9CgakY5vdER/m7fAfzZ2GroxoqJWW+wzD9aTDIKLBIK2yE43k9xpMq/9YMjYcNU2nCgz2NXdE5aWIcaThWn4A6fc/yDWNdOoPr/tmmHIiFH9svYoEror6zuaEpKCm1Om2XhK0HW3fyXethCAiRTyzphiZH+tEJWxnyoUpx00f89UowhCR6I7aI1VoXRCLCFMOha7cH4wWLWnxvsTrWyWW/K4UuvfEbcBSRMhp1Rw2yRczpCkeu4pQYTMVps0muW/fPit3RDwqzvbYNmhzSlFhZ8qF8VQVKo40i7/2NfbK2uvFd1l9D99l5FQMPtAwcBpFWzfhzt8kYpabXITxCP7lPJz94H1zUsozRdB+cBbzfhmM8R1TLuJnIfgO08FWgv45Qfw2p14Q0fDUuj3hWnDAP9ncIF4XYREwENtcc8igOMnXojzTOmdBh0e24rCdjrfVluJAQGDyNESIRqPu4LUEaE412gsRK6SkmdKUiAYczktH1L19zb+ggKvdER/mzdUqOZwFo2iQ/ofosMyd5vC8ZCIaIldbTfWY7sUZCFEmQ5VfjIqDooO8IQEzFmajdXoq1PFyykTFNISJjjf2qbF8jQ6lB4uR/fQMROyTuvr9FJAAlfScu5KR/Fstqk61o/mIFqqF0VguLRtswf0+87QJ7WuZovPdjtavSsXrh2Hxm+bHbZmnE2iRq2ntOuWivRTLp0n5amZAfahvtXLIw0tNAZOERyO6Hx3w0yik58XC/YgKCSm5pvfV+lUx1E8vRW5rCFSbkqwTUTokEAkZ0rKoxUhOXA6t6Pi3n6qCdlU0on9rzjHRoa+flcu/xCLdvVZ8VuJb03bKhfisXp4bgWixvbxPDizYK6vPRaxSnIsyFrmdK3TQYGLwgYbchf0abDq7DCkLJ8olZuOjXoUuEciRklI+mgMkbsf6qPGdUy5iHwiGndgD4B+MWGnwA6deENEw5P5IdtfggENbqv3G8+ieO+CmrbtOuBXz8m+t+RWo6q6de9WINmmliR63QQldXBdzBvgIJD/U96Y1EQ2y0V6IzTmKhn25SA5uxb41CYiOWgzVznaErdLi8HY1wjpXBHJBhChTP+qKo5uSsThJjaP+ahzepRad/P4Sz5lVhdI/JMO1QjxPoBeiV1XB9ddF2LZCPqSDXzJyP0jFtOZcRIvjIlKKYXwkFw3FKvkAG5PDEDVXumNnysVtHqZpb1JuhMwKq7EJvQtIRUVbG7Y+0vMkNq9Hc1H1yUZEfJeHBHG+viEqVExaim21RT0kO+6Zy0MbTc+Z/NMKqKUA7wIVqlyTUfSRTQaivn5WihCExZtDKV2mXLh44F7ps3L3xb0e8nu2V3anF6ZJI0Emi/O601xEg2vUicbGH+X7RETkZGPHjMHYsWPlPaLrIU278EVyl8zg9kQht06L2N6Sc31fAVVwNL5afRTb4i3HBFQh0+EpHCqU2iZ37K9vdUiQRoesKu0ynUPK+RCxTt7pke17b4Y2dhrUfkU4uibM4ZwYdOO6dOkS62UaInI9rhf1ZoO9elPUVwunQbfgKIqe4jitqg2+ot4Pcez7jIYlBh+IiAYRgw80kIzt7bgi3+/NbS4uDnW0mz9KQPT/iUZRXuzQT0lor4L2vSq03xuF1Idszub7drQ7mInd6r2f0iF5gQ5h27choZu50HRzYfCBhsxJLRYHL4VhTQUqXug6Eqt9nwpRsc1Yys52Z3A8+1+0aM6JcigxMQ0/DD4QEQ0iBh+IiIYXBh9ocBnRerIVuFqL3JQEZJ5KRVG1GmE20+PaP1mKkBebkfAnLVSzb96udvupBvGJtaNoXQKWfzQNW6sZOB7JGHwgIhpEDD4QEQ0vDD7Q4KpFtjIMqnp3+D6ajFdWpyPC3io/V0WX+6oCLjf13LB2lL4YgsXvtcJrbjLSV6mQEMQxDyMZgw9ERIOIwQciouGFwQciosHB1S6IiIiIiIiIyKkYfCAiIiIiIiIip2LwgYiIiIiIiIicisEHIiIiIiIiInIqBh+IiIiIiIiIyKkYfCAiIiIiIiIip2LwgYiIiIiIiIicisEHIiIiIiIiInIqBh+IiIiIiIiIyKkYfCAiIiIiIiIip2LwgYiIiIiIiIicisEHIiIiIiIiInKqUd9+++2P8n0iIiIiIiIiogE36kRjI4MPRESDZOyYMfD09JT3iIhoqLW0tLBeJiIaBJx2QUREREREREROxeADERERERERETkVgw9ERERERERE5FQMPhARERERERGRUzH4QEREREREREROxeADERERERERETkVgw9ERERERERE5FQMPhARERERERGRUzH4QEREREREREROxeADERERERERETkVgw9ERERERERE5FQMPhARERERERGRUzH4QEREREREREROxeADERERERERETkVgw9ERESDwFi+Fkkb9WiT94l60nYoC0krSvjvhYiIbhgMPhAREfVVmx6ajVnIKm6SC3pTg5xVKlRDAVe5xD4j2tra0HZR3r1uTSiRzvOdGyjoca4eZeVlqDljlAsGwIkSZInPSXNoiD4lO/+eXBVA9UYVNJ/LBURERCMcgw9ERHRjuyg681KHvs2xzqqhPAep0f7wHDUKo0ZNRujiVOR8atMpvSg69SvSkf65QS7oWdvOHKQfSoTqmSC5pBtnCpHo5ga3zXq54HoZUCOd554mDGBXvRsGFCwWn9niAnGvO/Ixo9ai3+/wRCHmhM+B+qAjgYI21O9ciyTlZPGa4nU9/THn2bUoPG7zaZyrQbr4nEqaHPiUBuvf0/3SvxdxXlsKOfqBiIhuCAw+0KA7U7oeSeE+mOIzE/N+o0XdBfmBTlegf3UmojfXiHtERP3QVo/CV5IQeo/o8P2D6MxLHXq3MebOn9T5/MJ+d86gi0dQuBr1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tFDNHqTJZd+VYmnoUjQ/lQvtKk5po75h8IGIaBAx+EBENLww+EADwtgO42gXKEbL+4PhqhHtVxWDlwNIZmw3QjHYL0o3BCacJCIiIiIiuh6KQQ48SEYPfuBBwsAD9ReDD0RERERERETkVAw+EBEREREREZFTMfhARERERERERE7F4AMRERERERERORWDD0RERERERETkVAw+EBEREREREZFTMfhARERERERERE7F4AMRERERERERORWDD0RERERERETkVAw+EBEREREREZFTMfhARERERERERE7F4AMREREREREROdWob7/99kf5PhERERERERHRgBt1orGRwQciokEydswYeHp6yntERDTUWlpaWC8TEQ0CTrsgIiIiIiIiIqdi8IGIiIiIiIiInIrBByIiIiIiIiJyKgYfiIiIiIiIiMipGHwgIiIiIiIiIqdi8IGIiIiIiIiInIrBByIiIiIiIiJyKgYfiIiIiIiIiMipGHwgIiIiIiIiIqdi8IGIiIiIiIiInIrBByIiIiIiIiJyKgYfiIiIiIiIiMipGHwgIiIiIiIiIqdi8IGIiIiIiIiInIrBByIiohuIsXwtkjbq0Sbvk41zemQ9m46Sc/I+ERERDQoGH2jEunLhAi5cuCLvERHdiIxoa2tD20V5t1c1yFmlQjUUcJVLRjSjATXlZSg7PoChlHEK8TFlQfVOjVxARDSCXJW/F4zyPtEIwuADjVA10AQFIThjL87KJUREI4qhDDlLFiH0nlEYNUraJiN0cSpyyg3yAcKZQiS6ucFts14u6FnbzhykH0qE6pkgc0FHI9ViMxq7lkmNWP0r0jksQsEZ86920daEknfSkRQxB/6eHefsCf/wOUhaoUHJif4FCIzyOdhtR58rg1o8/xxdvVxwjfl8Hdls31MQElcnomZFDgo5+oGIRprPsuAmvhcSd1p8VxCNEAw+0NC4UIcPfjMPSh8fTAlPxPrS0/IDFpq3IclnCYq6awgTEY1UxjKoguZAfTESOfrL+PHHH/Hj5WrkzD0vOttBUJX355KWASUfaoDnYxA+Ti6SG6mWW9ZOOaBhsfXWiDUeWos5fpORqG1D0PMZyNefx/nz0laJ/JdTENRWgMR7/DBnvd5+EMGec2VYGzkZY+RzGHPPfKQXN8kP9k75Usc5dL/VvRUjH23NdXYMUqCBpg+vR0QjS0eA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9nujkPqQzdl8345203KZvbvNxaXz6lzz9mRE/0cYinRsuJNjGHygQSMHgLP/RYvmnKiuyXe/K4VqwWI0v9gA7aNW1/LJVi+fZfs+FaJim7G0myAPDQ0GH4iIBhGDD0REwwuDD+Rs7acapHAyitYlYPlH07C1ehsSpKH9lr4rxdLQpWh+KhfaVWEOrQp0M3Lks2z/ZClCXmxGwp+0UM3mJzmcMPhARDSIGHwgIhpeGHwg52pH6YshWPxeK7zmJiN9lQoJQfY7xMZ2IxQOjmK7OTn4WV41ov2qAvwohx8GH4iIBhGDD0REwwuDD0REg4OrXRARERERERGRUzH4QEREREREREROxeADERERERERETkVgw9ERERERERE5FQMPhARERERERGRUzH4QEREREREREROxeADERERERERETkVgw9ERERERERE5FQMPhARERERERGRUzH4QEREREREREROxeADERERERERETkVgw9ERERERERE5FSjvv322x/l+0REREREREREA27UicZGBh+IiAbJ2DFj4OnpKe8REdFQa2lpYb1MRDQIOO2CiIiIiIiIiJyKwQciIiIiIiIicioGH4iIiIiIiIjIqRh8ICIiIiIiIiKnYvCBiIiIiIiIiJyKwQciIiIiIiIicioGH4iIiIiIiIjIqRh8ICIiIiIiIiKnYvCBiIiIiIiIiJyKwQciIiIiIiIicioGH4iIiIiIiIjIqRh8ICIiIiIiIiKnYvCBiIiIiIiIiJyKwQciIiIiIiIicioGH4iIiMiOJhQsSUfhCaO8P4SMTShckYScz+V9IiIiGnEYfCAiIroZXGxDW1sbHA0ltBVnIe2NNsBVIZd0o4/P25u2QxpkbcxCyQm5QKJwA85pkLqlEOKMiIiIaARi8IGIiGhEa0P9zrVICveH56hRGCU2z4A5SHqlEPUWPXX9Zje4uSWi8Ixc0KMaaFbnwHNDCmLGyUXd6Nvz9s7YVIL0FemoOScXmLgi5oVMBL3z/7d3P1BRnXf++N+euieDm43w21qGrZaAJBS6YuAXicNWqnjEiIVVzLAJhDTJQL6bAPkjaKtO3KwZtFUwqULSBqbZZoSkophCxIo/0cX8HIP9MRG/hZCqk6k0zBTzdXCTODm1x99zZ+7AAMNfGQTyfp1zvXOfe5m5dzznmed+7vN8Hh307P1AREQ0JTH4QBPo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7Zo0WJ5rh+GhYcO/4+OLdpTlPABDZD3qfdR7oPuYFslqC/LGHEimrzsGH2hq+qoLHV1iJSq/nzxdio7sfTj5ExXukHcTTVYMPhARTS4MPtC4+qweeXF5sPyoDIbNEzUkrAUlqniUxezF/p3e8vDcuu7f5SH2OQsyf2GAdhkHutHYMPhAU1NrOVYm78DFOWFIeXQT1j+9FHO/Ie8jmsQYfCAimlwYfKDx5uh2QDGSjL/jyOefecOB7huKESUyJhoMgw9ERBOIwQciosmFwQcioonB2S6IiIiIiIiIyKcYfCAiIiIiIiIin2LwgYiIiIiIiIh8isEHIiIiIiIiIvIpBh+IiIiIiIiIyKcYfCAiIiIiIiIin2LwgYiIiIiIiIh8isEHIiIiIiIiIvIpBh+IiIiIiIiIyKcYfCAiIiIiIiIin2LwgYiIiIiIiIh8isEHIiIiIiIiIvKpGZ9++ulN+TURERERERER0bibceHiRQYfiIgmyCw/PwQFBclbRER0u3V2drJeJiKaABx2QUREREREREQ+xeADEREREREREfkUgw9ERERERERE5FMMPhARERERERGRTzH4QEREREREREQ+xeADEREREREREfkUgw9ERERERERE5FMMPhARERERERGRTzH4QEREREREREQ+xeADEREREREREfkUgw9ERERERERE5FMMPhARERERERGRTzH4QEREREREREQ+xeADEREREREREfkUgw9EREQEsyEXBVVmOOTtSc9hRvXGLJR+KG8TERHRpMbgAxER0XRzpQ0NJxpg6hhhKOFKHYo3lsI+MwAKuWg8mWuLUbxLD6NdLhg1B6y/bxDX1Iaet1AEiPPWI3dPdW8ZERERTVoMPhAREU01Nxyw2+2wfy5v93ehGssTlkN3amS35aY3tShVFiFnrb9cMr6sHxagYGMdzIOd77DsaNi5XFxTNdrkEsAfqc8WIfpNHfTs/UBERDTpMfhAt0/HUex4cinC5s+HatUL2Hf+mryj11fv74AqeTdMX8oFTtdwcuti8XfJ0LfKRR463tGIfYvx0vGB70dENKXZ21C5fjmC/s4PAQEBCPiHGQhakIHi92/l2b8RdbtNiH4sAdFySX/mstWYMWMGZqRVwiqXQbyqTBNlUvlgS5/jvThT6P3v5KXwjHzcYO5LQMZ9JhTtb5g6w0WIaHgOO8wfVqP0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SxEn9Y67Zx0q3UFzL9r2JiEyYTkik0t7er2MtIx8h8EHmgRase/VcsRsyYJqllwkuSMamldfQUzzC86klC80/9/Y/WoWZ7MgInK6jqvuBvqgy6BNy17+CUjdpERdVcMQjWnv/MNCBnmyNkGkqd9Ew3SLOoFJy4imi45qaNxT77qnDd7hnkpYI8+mo0B0djG0321D8dpQBARFIrcmCLnVrWgY8FQ9BOm/MuH4+iAcfjYSQSGpKP4kBuVn6zyGQoj321SH5l8loXN7DIKCYsQNdJLrmPt73y1grmuaSeuhgzB6ndVHOFOK5Ttcg+La3i7AanFju7zPUt1zk6tUV+B0XS6UhzQIDQiF5pASuXWnUaH27LU2gnMT35n2KXFzL71+vxgZAz5Thwbnk30FguZFQnp388sNgybbNL6+HIVnpFdtqFy/ut97iaXKfQXjeW4j/D78g1zTlVqrcfCs8129CgqLcf5fhSwKgTsj0EjLyHdmXLh48ab8moiIfGyWn5/4cebPG90KKedDEDKq5M2hiIZc5/50Z0NzSBf0WH2PFtENZuiW9W22jwfjyzMQtzUVFZcPIr3fGOCRcV+zDqdvbpGnqnOgYXMIln+ow6U6jbPxSDQWnZ2drJdpxBzHCuCXCBy/XtT7NH6KMZctR2hVOi7VT9G609GAAr/lQP11FK2Yov8JX1MMPhARTSAGH2hycsC4NQHavyvG4UHHCY+d/aMGNHcGIFIVDeV4vbmUCC0lA1c3tbLxSbeEwQcasc/bUPpkJPQLTuO0D+rKCXGlDgUrVsO8sRMHHxk2ND0JOdD2egYi34jBaeMWqFj9TykMPhARTSAGH4iIJhcGH2hkTCheEIPSReU4vMdbrpsp4EodshZoYM6uwMFtU3O4mumVSMS8Fofy2hJovsvIw1TD4AMR0QRi8IGIaHJh8IFGymF3QDGSTL6T2JS/BocdjpmjTU5MkwUTThIREREREQ1jqgceJFP+GhQMPExlDD4QERERERERkU8x+EBEREREREREPsXgAxERERERERH5FIMPRERERERERORTDD4QERERERERkU8x+EBEREREREREPsXgAxERERERERH5FIMPRERERERERORTDD4QERERERERkU8x+EBEREREREREPsXgAxERERERERH51IxPP/30pvyaiIiIiIiIiGjczbhw8SKDD0REE2SWnx9mzZolbxER0e325Zdfsl4mIpoAHHZBRERERERERD7F4AMRERERERER+RSDD0RERERERETkUww+EBEREREREZFPMfhARERERERERD7F4AMRERERERER+RSDD0RERERERETkUww+EBEREREREZFPMfhARERERERERD7F4AMRERERERER+RSDD0RERERERETkUww+EBEREREREZFPMfhARERERERERD7F4AMRERERERER+RSDD0RERNSH5Z0N0L5rgUPenjY+a0JJjhb1n8nbRERENGEYfCAiIprMTGVIWZsC7e9scsFodKP9VCMaTbaRBxI+q0fJf5She6Y/FHLR7WFD/eYUce1laJFLRsPxaQsaxbW3ewYa/lFcUUsJdBVjeUciIiK6FQw+EBER3WauG+UW2LxFCG50o/FEIyyfy9syR3c3ugdbvpAPQjtqk8UN/M8b0S2XDKelQoeyb+mQ/cPZcokvycGRj72fXfenYt+J7r6BE4eX6/VY3Md2n96FFHHttZfkAqcoZP44Ey1by1DL3g9EREQTisEHmnjXWrHv+ZVQzZ+PsAQNdtRflnd4sOxH1vxnUNMhb/dhwmvS384vFa+8MJWKfWJ/ide9RESTjutGeRca/49cMKwmlAQHI3iQJbfWIh83Wk2oL21B1CPx4ja9V/cZA7Q5iVjo7w9//4VIfFyLqo+H7kth+XWaOFYc/3gVBu+zIQdH3m2Xt4dney/X6zW7lr3D9pKYvSQZ2TDA8LuxfkdEND11o+mXG5AWI9Vz/lio3oCyM14Co581oSw/rac+TMsvQ9OYg5kOtL9XhGxVuPMzw1XZKHqvfWBPNUc3LOdrUfZcCsKlz93ZJO+YnEb0m3GjGy3i2vOSH3BdU/gDSMkpQv2f5P00LTH4QBOsCzVbklE6M0dUQhfRvDMGzU8/g9dMX8n7JddwUr8b3VtykTJXLiIiIg+xKLDbYe+3WA/lOvcGBwY616N2thFltiikLXGHHkRjfGcKYh/UoWV2JspaLDh3rAAL/1yC7NgUlJyXD+vvkgEbnquXN8ZX4EOGAddtt5/D3mVi57JgDHvlsx/ACg1Qf6xpiKAIEX29OERdp0bij+sRmN+I9vZGaEPPYcODahSd9bhp/qIJRQ8nYsOJQBScakf7KS2CWzYg8eEiNPX0OBs5y4E8pDxaBlvyXpyznEOZuhtlj6Yg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lfeNhNe/9+aCHqvv0SK6wQzdstvZCdaKyrQgZFTpcPrmFqjkUhf3PnlzSJ5/b4Y+KRTafz4O884EdvElp87OTtbLNO7MZcsRWpWOS/WaCetl5ThWAL9E4Pj1op6eE+NL1KGJoahUX8LxbPYdo9Fj8IGIaAIx+ECTnwPGrQnQ/l0xDk+3MbjSuOkVFUg4chiaMLmMvvYYfKBxd6UOBStWw7xxhEHf8fB5G0qfjIR+wWmc9lHdbT9SgIQkMwouH0Q6U6/RGDD4QEQ0gRh8ICKaXBh8oHF1pQ5ZCzQwZ1fg4LYE3+aw6WFC8YIYlC4qx+E9msHzCd0Ce20WIp4yQ/PWQehWTMxV0fTD4AMR0QRi8IGIaHJh8IHGm8PugMJ/YvuN+fwzbzhgv6HABF8WTTNMOElERERERDROJjrwIPH5Z85k4IFuHYMPRERERERERORTDD4QERERERERkU8x+EBEREREREREPsXgAxERERERERH5FIMPRERERERERORTDD4QERERERERkU8x+EBEREREREREPsXgAxERERERERH5FIMPRERERERERORTDD4QERERERERkU8x+EBEREREREREPjXj008/vSm/JiIiIiIiIiIadzMuXLzI4AMR0QSZ5eeHWbNmyVtERHS7ffnll6yXiYgmAIddEBEREREREZFPMfhARERERERERD7F4AMRERERERER+RSDD0RERERERETkUww+EBEREREREZFPMfhARERERERERD7F4AMRERERERER+RSDD0RERERERETkUww+EBEREREREZFPMfhARERERERERD7F4AMRERERERER+RSDD0RERERERETkUww+EBEREREREZFPMfhARERERERERD7F4AMRERHB8s4GaN+1wCFvT3oOC2q35qHsvLxNREREkxqDD0RERF93n9Wj5D/K0D3THwq5aDxZfleCkj0GNHXLBaPmgM3UiMZT7eh5C4W/OG8DNvyitreMiIiIJi0GH4iIiKYMG6oe94f/41Xi1WBGckxfLRU6lH1Lh+wfzpZLxpetRQvt1npYvpALRq0bjT9PQUpyLdrlEmA2kv9dh6iKXTCw9wMREdGkx+ADTbBraH3rBaxcPB9h85ci66dH0fE3eVePy6h6cj5yDl2Wtz11oSZP+tv5UJWY5LJ+rp3ES2K/dMxrgxxCRERuTagvbUHUI/GI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adxZfp2Che+qce7QxNVPjhNaKNcCNVadR8+J8WSBYe1CVK05h5ofsdal0WPwgYhoAjH4QJOTA03bU6CbqcN+Hzx16v64Eeds/ghfFIXA8XrzL5pQ9HA27Os/gG4Zn5PR2DH4QOPus3po16TB8lz7mIfEjdoX7SjLeQCGyHrU+6j3QPcxLZLVFuSNOZBMX3cMPhARTSAGH4iIJhcGH2hcfVaPvLg8WH5UBsPmiRoS1oISVTzKYvZi/05veXhuXffv8hD7nAWZvzBAu4wD3WhsGHwgIppADD4QEU0uDD7QeHN0O6CYPbE9snz+mTcc6L6hwARfFk0zTDhJREREREQ0TiY68CDx+WdKsxMx8EC3iMEHIiIiIiIiIvIpBh+IiIiIiIiIyKcYfCAiIiIiIiIin2LwgYiIiIiIiIh8isEHIiIiIiIiIvIpBh+IiIiIiIiIyKcYfCAiIiIiIiIin2LwgYiIiIiIiIh8isEHIiIiIiIiIvIpBh+IiIiIiIiIyKcYfCAiIiIiIiIin2LwgYiIiIiIiIh8asann356U35NRERERERERDTuZly4eJHBByKiCTLLzw9BQUHyFhER3W6dnZ2sl4mIJgCHXRARERERERGRTzH4QEREREREREQ+xeADEREREREREfkUgw9ERERERERE5FMMPhARERERERGRTzH4QEREREREREQ+xeADEREREREREfkUgw9ERERERERE5FMMPhARERERERGRTzH4QEREREREREQ+xeADEREREREREfkUgw9ERERERERE5FMMPhARERERERGRTzH4QEREREREREQ+xeADERERwWzIRUGVGQ55e9JzmFG9MQulH8rbRERENKkx+EBERPR1d6UOxRtLYZ8ZAIVcNJ7MtcUo3qWH0S4XjJoD1t83oOFEG3reQhEgzluP3D3VvWVEREQ0aTH4QERENJWdKcSMGTOw7m2rXDB6pje1KFUWIWetv1wyvqwfFqBgYx3Mn8sFo2ZHw87lWJ5QjTa5BPBH6rNFiH5TBz17PxAREU16DD7QbXINrW+9gJWL5yNs/lJk/fQoOv4m7+pxGVVPzkfOocvy9kBdZwx4Ke8hLJ3veh913kvYd6ZL3ktERMMzom63CdGPJSBaLpFu9o1vFiBLFeoMbMy4Jw7rNlairU/wwIrKNLFP2j/YklYpjhqCHDgZbCk8Ix83mPsS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)


```python
# 참고
x = pd.to_datetime(melted_cpi['Month'], format='%b')
x
```




    0     1900-01-01
    1     1900-01-01
    2     1900-01-01
    3     1900-01-01
    4     1900-01-01
             ...    
    271   1900-12-01
    272   1900-12-01
    273   1900-12-01
    274   1900-12-01
    275   1900-12-01
    Name: Month, Length: 276, dtype: datetime64[ns]




```python
melted_cpi['Month'] = melted_cpi['Month'].map(lambda x: pd.to_datetime(x, format='%b')).dt.month
```


```python
melted_cpi
```





  <div id="df-638e2b18-8d9c-4dd0-87c7-6d48daf63319">
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
      <th>Year</th>
      <th>Month</th>
      <th>cpi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001</td>
      <td>1</td>
      <td>175.100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2002</td>
      <td>1</td>
      <td>177.100</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>1</td>
      <td>181.700</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2004</td>
      <td>1</td>
      <td>185.200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005</td>
      <td>1</td>
      <td>190.700</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>271</th>
      <td>2019</td>
      <td>12</td>
      <td>256.974</td>
    </tr>
    <tr>
      <th>272</th>
      <td>2020</td>
      <td>12</td>
      <td>260.474</td>
    </tr>
    <tr>
      <th>273</th>
      <td>2021</td>
      <td>12</td>
      <td>278.802</td>
    </tr>
    <tr>
      <th>274</th>
      <td>2022</td>
      <td>12</td>
      <td>296.797</td>
    </tr>
    <tr>
      <th>275</th>
      <td>2023</td>
      <td>12</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>276 rows × 3 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-638e2b18-8d9c-4dd0-87c7-6d48daf63319')"
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
          document.querySelector('#df-638e2b18-8d9c-4dd0-87c7-6d48daf63319 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-638e2b18-8d9c-4dd0-87c7-6d48daf63319');
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




- 월별 cpi 평균 구하기


```python
melted_cpi.groupby('Month').mean()['cpi']
```




    Month
    1     224.931391
    2     222.548000
    3     223.779500
    4     224.590091
    5     225.389591
    6     226.141773
    7     226.332818
    8     226.611182
    9     227.047500
    10    227.112000
    11    226.690818
    12    226.276682
    Name: cpi, dtype: float64



- 연도별 cpi 평균 구하기


```python
melted_cpi.groupby('Year').mean()['cpi']
```




    Year
    2001    177.066667
    2002    179.875000
    2003    183.958333
    2004    188.883333
    2005    195.291667
    2006    201.591667
    2007    207.342417
    2008    215.302500
    2009    214.537000
    2010    218.055500
    2011    224.939167
    2012    229.593917
    2013    232.957083
    2014    236.736167
    2015    237.017000
    2016    240.007167
    2017    245.119583
    2018    251.106833
    2019    255.657417
    2020    258.811167
    2021    270.969750
    2022    292.654917
    2023    299.170000
    Name: cpi, dtype: float64



- long to wide format


```python
pivoted_cpi = melted_cpi.pivot(index='Year', columns='Month', values='cpi')
pivoted_cpi
```





  <div id="df-3903191b-6683-4cea-9ae8-fa433626e7dc">
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
      <th>Month</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
    </tr>
    <tr>
      <th>Year</th>
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
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001</th>
      <td>175.100</td>
      <td>175.800</td>
      <td>176.200</td>
      <td>176.900</td>
      <td>177.700</td>
      <td>178.000</td>
      <td>177.500</td>
      <td>177.500</td>
      <td>178.300</td>
      <td>177.700</td>
      <td>177.400</td>
      <td>176.700</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>177.100</td>
      <td>177.800</td>
      <td>178.800</td>
      <td>179.800</td>
      <td>179.800</td>
      <td>179.900</td>
      <td>180.100</td>
      <td>180.700</td>
      <td>181.000</td>
      <td>181.300</td>
      <td>181.300</td>
      <td>180.900</td>
    </tr>
    <tr>
      <th>2003</th>
      <td>181.700</td>
      <td>183.100</td>
      <td>184.200</td>
      <td>183.800</td>
      <td>183.500</td>
      <td>183.700</td>
      <td>183.900</td>
      <td>184.600</td>
      <td>185.200</td>
      <td>185.000</td>
      <td>184.500</td>
      <td>184.300</td>
    </tr>
    <tr>
      <th>2004</th>
      <td>185.200</td>
      <td>186.200</td>
      <td>187.400</td>
      <td>188.000</td>
      <td>189.100</td>
      <td>189.700</td>
      <td>189.400</td>
      <td>189.500</td>
      <td>189.900</td>
      <td>190.900</td>
      <td>191.000</td>
      <td>190.300</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>190.700</td>
      <td>191.800</td>
      <td>193.300</td>
      <td>194.600</td>
      <td>194.400</td>
      <td>194.500</td>
      <td>195.400</td>
      <td>196.400</td>
      <td>198.800</td>
      <td>199.200</td>
      <td>197.600</td>
      <td>196.800</td>
    </tr>
    <tr>
      <th>2006</th>
      <td>198.300</td>
      <td>198.700</td>
      <td>199.800</td>
      <td>201.500</td>
      <td>202.500</td>
      <td>202.900</td>
      <td>203.500</td>
      <td>203.900</td>
      <td>202.900</td>
      <td>201.800</td>
      <td>201.500</td>
      <td>201.800</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>202.416</td>
      <td>203.499</td>
      <td>205.352</td>
      <td>206.686</td>
      <td>207.949</td>
      <td>208.352</td>
      <td>208.299</td>
      <td>207.917</td>
      <td>208.490</td>
      <td>208.936</td>
      <td>210.177</td>
      <td>210.036</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>211.080</td>
      <td>211.693</td>
      <td>213.528</td>
      <td>214.823</td>
      <td>216.632</td>
      <td>218.815</td>
      <td>219.964</td>
      <td>219.086</td>
      <td>218.783</td>
      <td>216.573</td>
      <td>212.425</td>
      <td>210.228</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>211.143</td>
      <td>212.193</td>
      <td>212.709</td>
      <td>213.240</td>
      <td>213.856</td>
      <td>215.693</td>
      <td>215.351</td>
      <td>215.834</td>
      <td>215.969</td>
      <td>216.177</td>
      <td>216.330</td>
      <td>215.949</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>216.687</td>
      <td>216.741</td>
      <td>217.631</td>
      <td>218.009</td>
      <td>218.178</td>
      <td>217.965</td>
      <td>218.011</td>
      <td>218.312</td>
      <td>218.439</td>
      <td>218.711</td>
      <td>218.803</td>
      <td>219.179</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>220.223</td>
      <td>221.309</td>
      <td>223.467</td>
      <td>224.906</td>
      <td>225.964</td>
      <td>225.722</td>
      <td>225.922</td>
      <td>226.545</td>
      <td>226.889</td>
      <td>226.421</td>
      <td>226.230</td>
      <td>225.672</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>226.665</td>
      <td>227.663</td>
      <td>229.392</td>
      <td>230.085</td>
      <td>229.815</td>
      <td>229.478</td>
      <td>229.104</td>
      <td>230.379</td>
      <td>231.407</td>
      <td>231.317</td>
      <td>230.221</td>
      <td>229.601</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>230.280</td>
      <td>232.166</td>
      <td>232.773</td>
      <td>232.531</td>
      <td>232.945</td>
      <td>233.504</td>
      <td>233.596</td>
      <td>233.877</td>
      <td>234.149</td>
      <td>233.546</td>
      <td>233.069</td>
      <td>233.049</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>233.916</td>
      <td>234.781</td>
      <td>236.293</td>
      <td>237.072</td>
      <td>237.900</td>
      <td>238.343</td>
      <td>238.250</td>
      <td>237.852</td>
      <td>238.031</td>
      <td>237.433</td>
      <td>236.151</td>
      <td>234.812</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>233.707</td>
      <td>234.722</td>
      <td>236.119</td>
      <td>236.599</td>
      <td>237.805</td>
      <td>238.638</td>
      <td>238.654</td>
      <td>238.316</td>
      <td>237.945</td>
      <td>237.838</td>
      <td>237.336</td>
      <td>236.525</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>236.916</td>
      <td>237.111</td>
      <td>238.132</td>
      <td>239.261</td>
      <td>240.229</td>
      <td>241.018</td>
      <td>240.628</td>
      <td>240.849</td>
      <td>241.428</td>
      <td>241.729</td>
      <td>241.353</td>
      <td>241.432</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>242.839</td>
      <td>243.603</td>
      <td>243.801</td>
      <td>244.524</td>
      <td>244.733</td>
      <td>244.955</td>
      <td>244.786</td>
      <td>245.519</td>
      <td>246.819</td>
      <td>246.663</td>
      <td>246.669</td>
      <td>246.524</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>247.867</td>
      <td>248.991</td>
      <td>249.554</td>
      <td>250.546</td>
      <td>251.588</td>
      <td>251.989</td>
      <td>252.006</td>
      <td>252.146</td>
      <td>252.439</td>
      <td>252.885</td>
      <td>252.038</td>
      <td>251.233</td>
    </tr>
    <tr>
      <th>2019</th>
      <td>251.712</td>
      <td>252.776</td>
      <td>254.202</td>
      <td>255.548</td>
      <td>256.092</td>
      <td>256.143</td>
      <td>256.571</td>
      <td>256.558</td>
      <td>256.759</td>
      <td>257.346</td>
      <td>257.208</td>
      <td>256.974</td>
    </tr>
    <tr>
      <th>2020</th>
      <td>257.971</td>
      <td>258.678</td>
      <td>258.115</td>
      <td>256.389</td>
      <td>256.394</td>
      <td>257.797</td>
      <td>259.101</td>
      <td>259.918</td>
      <td>260.280</td>
      <td>260.388</td>
      <td>260.229</td>
      <td>260.474</td>
    </tr>
    <tr>
      <th>2021</th>
      <td>261.582</td>
      <td>263.014</td>
      <td>264.877</td>
      <td>267.054</td>
      <td>269.195</td>
      <td>271.696</td>
      <td>273.003</td>
      <td>273.567</td>
      <td>274.310</td>
      <td>276.589</td>
      <td>277.948</td>
      <td>278.802</td>
    </tr>
    <tr>
      <th>2022</th>
      <td>281.148</td>
      <td>283.716</td>
      <td>287.504</td>
      <td>289.109</td>
      <td>292.296</td>
      <td>296.311</td>
      <td>296.276</td>
      <td>296.171</td>
      <td>296.808</td>
      <td>298.012</td>
      <td>297.711</td>
      <td>296.797</td>
    </tr>
    <tr>
      <th>2023</th>
      <td>299.170</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-3903191b-6683-4cea-9ae8-fa433626e7dc')"
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
          document.querySelector('#df-3903191b-6683-4cea-9ae8-fa433626e7dc button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-3903191b-6683-4cea-9ae8-fa433626e7dc');
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
pivoted_cpi.columns
```




    Int64Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype='int64', name='Month')


## Reference
[파이썬 라이브러리를 활용한 데이터 분석 (웨스 맥키니 저)](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=315354750)