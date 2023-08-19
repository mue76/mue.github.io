---
tag: [machine learning, scikit-learn]
toc: true
toc_sticky: true
toc_label: 목차
---

# PCA 활용

## default of credit card clients dataset

- [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php)의 [
default of credit card clients dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)에서 `default of credit card clients.xls`를 다운로드

- 신용카드 데이터 세트 PCA 변환하여 분류 예측 성능 확인하기


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
df = pd.read_excel('./datasets/pca_credit_card.xls', header=1)
df.head(3).T
```




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
      <th>ID</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>LIMIT_BAL</th>
      <td>20000</td>
      <td>120000</td>
      <td>90000</td>
    </tr>
    <tr>
      <th>SEX</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>EDUCATION</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>MARRIAGE</th>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>AGE</th>
      <td>24</td>
      <td>26</td>
      <td>34</td>
    </tr>
    <tr>
      <th>PAY_0</th>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>PAY_2</th>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>PAY_3</th>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>PAY_4</th>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>PAY_5</th>
      <td>-2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>PAY_6</th>
      <td>-2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BILL_AMT1</th>
      <td>3913</td>
      <td>2682</td>
      <td>29239</td>
    </tr>
    <tr>
      <th>BILL_AMT2</th>
      <td>3102</td>
      <td>1725</td>
      <td>14027</td>
    </tr>
    <tr>
      <th>BILL_AMT3</th>
      <td>689</td>
      <td>2682</td>
      <td>13559</td>
    </tr>
    <tr>
      <th>BILL_AMT4</th>
      <td>0</td>
      <td>3272</td>
      <td>14331</td>
    </tr>
    <tr>
      <th>BILL_AMT5</th>
      <td>0</td>
      <td>3455</td>
      <td>14948</td>
    </tr>
    <tr>
      <th>BILL_AMT6</th>
      <td>0</td>
      <td>3261</td>
      <td>15549</td>
    </tr>
    <tr>
      <th>PAY_AMT1</th>
      <td>0</td>
      <td>0</td>
      <td>1518</td>
    </tr>
    <tr>
      <th>PAY_AMT2</th>
      <td>689</td>
      <td>1000</td>
      <td>1500</td>
    </tr>
    <tr>
      <th>PAY_AMT3</th>
      <td>0</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>PAY_AMT4</th>
      <td>0</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>PAY_AMT5</th>
      <td>0</td>
      <td>0</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>PAY_AMT6</th>
      <td>0</td>
      <td>2000</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>default payment next month</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



- ID: ID of each client
- LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
- SEX: Gender (1=male, 2=female)
- EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
- MARRIAGE: Marital status (1=married, 2=single, 3=others)
- AGE: Age in years
- PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)
- PAY_2: Repayment status in August, 2005 (scale same as above)
- PAY_3: Repayment status in July, 2005 (scale same as above)
- PAY_4: Repayment status in June, 2005 (scale same as above)
- PAY_5: Repayment status in May, 2005 (scale same as above)
- PAY_6: Repayment status in April, 2005 (scale same as above)
- BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
- BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
- BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
- BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
- BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
- BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
- PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
- PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
- PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
- PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
- PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
- PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
- default.payment.next.month: Default payment (1=yes, 0=no)


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 30000 entries, 0 to 29999
    Data columns (total 25 columns):
     #   Column                      Non-Null Count  Dtype
    ---  ------                      --------------  -----
     0   ID                          30000 non-null  int64
     1   LIMIT_BAL                   30000 non-null  int64
     2   SEX                         30000 non-null  int64
     3   EDUCATION                   30000 non-null  int64
     4   MARRIAGE                    30000 non-null  int64
     5   AGE                         30000 non-null  int64
     6   PAY_0                       30000 non-null  int64
     7   PAY_2                       30000 non-null  int64
     8   PAY_3                       30000 non-null  int64
     9   PAY_4                       30000 non-null  int64
     10  PAY_5                       30000 non-null  int64
     11  PAY_6                       30000 non-null  int64
     12  BILL_AMT1                   30000 non-null  int64
     13  BILL_AMT2                   30000 non-null  int64
     14  BILL_AMT3                   30000 non-null  int64
     15  BILL_AMT4                   30000 non-null  int64
     16  BILL_AMT5                   30000 non-null  int64
     17  BILL_AMT6                   30000 non-null  int64
     18  PAY_AMT1                    30000 non-null  int64
     19  PAY_AMT2                    30000 non-null  int64
     20  PAY_AMT3                    30000 non-null  int64
     21  PAY_AMT4                    30000 non-null  int64
     22  PAY_AMT5                    30000 non-null  int64
     23  PAY_AMT6                    30000 non-null  int64
     24  default payment next month  30000 non-null  int64
    dtypes: int64(25)
    memory usage: 5.7 MB
    


```python
corr_matrix = df.corr()
corr_matrix
```




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
      <th>ID</th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>...</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default payment next month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ID</th>
      <td>1.000000</td>
      <td>0.026179</td>
      <td>0.018497</td>
      <td>0.039177</td>
      <td>-0.029079</td>
      <td>0.018678</td>
      <td>-0.030575</td>
      <td>-0.011215</td>
      <td>-0.018494</td>
      <td>-0.002735</td>
      <td>...</td>
      <td>0.040351</td>
      <td>0.016705</td>
      <td>0.016730</td>
      <td>0.009742</td>
      <td>0.008406</td>
      <td>0.039151</td>
      <td>0.007793</td>
      <td>0.000652</td>
      <td>0.003000</td>
      <td>-0.013952</td>
    </tr>
    <tr>
      <th>LIMIT_BAL</th>
      <td>0.026179</td>
      <td>1.000000</td>
      <td>0.024755</td>
      <td>-0.219161</td>
      <td>-0.108139</td>
      <td>0.144713</td>
      <td>-0.271214</td>
      <td>-0.296382</td>
      <td>-0.286123</td>
      <td>-0.267460</td>
      <td>...</td>
      <td>0.293988</td>
      <td>0.295562</td>
      <td>0.290389</td>
      <td>0.195236</td>
      <td>0.178408</td>
      <td>0.210167</td>
      <td>0.203242</td>
      <td>0.217202</td>
      <td>0.219595</td>
      <td>-0.153520</td>
    </tr>
    <tr>
      <th>SEX</th>
      <td>0.018497</td>
      <td>0.024755</td>
      <td>1.000000</td>
      <td>0.014232</td>
      <td>-0.031389</td>
      <td>-0.090874</td>
      <td>-0.057643</td>
      <td>-0.070771</td>
      <td>-0.066096</td>
      <td>-0.060173</td>
      <td>...</td>
      <td>-0.021880</td>
      <td>-0.017005</td>
      <td>-0.016733</td>
      <td>-0.000242</td>
      <td>-0.001391</td>
      <td>-0.008597</td>
      <td>-0.002229</td>
      <td>-0.001667</td>
      <td>-0.002766</td>
      <td>-0.039961</td>
    </tr>
    <tr>
      <th>EDUCATION</th>
      <td>0.039177</td>
      <td>-0.219161</td>
      <td>0.014232</td>
      <td>1.000000</td>
      <td>-0.143464</td>
      <td>0.175061</td>
      <td>0.105364</td>
      <td>0.121566</td>
      <td>0.114025</td>
      <td>0.108793</td>
      <td>...</td>
      <td>-0.000451</td>
      <td>-0.007567</td>
      <td>-0.009099</td>
      <td>-0.037456</td>
      <td>-0.030038</td>
      <td>-0.039943</td>
      <td>-0.038218</td>
      <td>-0.040358</td>
      <td>-0.037200</td>
      <td>0.028006</td>
    </tr>
    <tr>
      <th>MARRIAGE</th>
      <td>-0.029079</td>
      <td>-0.108139</td>
      <td>-0.031389</td>
      <td>-0.143464</td>
      <td>1.000000</td>
      <td>-0.414170</td>
      <td>0.019917</td>
      <td>0.024199</td>
      <td>0.032688</td>
      <td>0.033122</td>
      <td>...</td>
      <td>-0.023344</td>
      <td>-0.025393</td>
      <td>-0.021207</td>
      <td>-0.005979</td>
      <td>-0.008093</td>
      <td>-0.003541</td>
      <td>-0.012659</td>
      <td>-0.001205</td>
      <td>-0.006641</td>
      <td>-0.024339</td>
    </tr>
    <tr>
      <th>AGE</th>
      <td>0.018678</td>
      <td>0.144713</td>
      <td>-0.090874</td>
      <td>0.175061</td>
      <td>-0.414170</td>
      <td>1.000000</td>
      <td>-0.039447</td>
      <td>-0.050148</td>
      <td>-0.053048</td>
      <td>-0.049722</td>
      <td>...</td>
      <td>0.051353</td>
      <td>0.049345</td>
      <td>0.047613</td>
      <td>0.026147</td>
      <td>0.021785</td>
      <td>0.029247</td>
      <td>0.021379</td>
      <td>0.022850</td>
      <td>0.019478</td>
      <td>0.013890</td>
    </tr>
    <tr>
      <th>PAY_0</th>
      <td>-0.030575</td>
      <td>-0.271214</td>
      <td>-0.057643</td>
      <td>0.105364</td>
      <td>0.019917</td>
      <td>-0.039447</td>
      <td>1.000000</td>
      <td>0.672164</td>
      <td>0.574245</td>
      <td>0.538841</td>
      <td>...</td>
      <td>0.179125</td>
      <td>0.180635</td>
      <td>0.176980</td>
      <td>-0.079269</td>
      <td>-0.070101</td>
      <td>-0.070561</td>
      <td>-0.064005</td>
      <td>-0.058190</td>
      <td>-0.058673</td>
      <td>0.324794</td>
    </tr>
    <tr>
      <th>PAY_2</th>
      <td>-0.011215</td>
      <td>-0.296382</td>
      <td>-0.070771</td>
      <td>0.121566</td>
      <td>0.024199</td>
      <td>-0.050148</td>
      <td>0.672164</td>
      <td>1.000000</td>
      <td>0.766552</td>
      <td>0.662067</td>
      <td>...</td>
      <td>0.222237</td>
      <td>0.221348</td>
      <td>0.219403</td>
      <td>-0.080701</td>
      <td>-0.058990</td>
      <td>-0.055901</td>
      <td>-0.046858</td>
      <td>-0.037093</td>
      <td>-0.036500</td>
      <td>0.263551</td>
    </tr>
    <tr>
      <th>PAY_3</th>
      <td>-0.018494</td>
      <td>-0.286123</td>
      <td>-0.066096</td>
      <td>0.114025</td>
      <td>0.032688</td>
      <td>-0.053048</td>
      <td>0.574245</td>
      <td>0.766552</td>
      <td>1.000000</td>
      <td>0.777359</td>
      <td>...</td>
      <td>0.227202</td>
      <td>0.225145</td>
      <td>0.222327</td>
      <td>0.001295</td>
      <td>-0.066793</td>
      <td>-0.053311</td>
      <td>-0.046067</td>
      <td>-0.035863</td>
      <td>-0.035861</td>
      <td>0.235253</td>
    </tr>
    <tr>
      <th>PAY_4</th>
      <td>-0.002735</td>
      <td>-0.267460</td>
      <td>-0.060173</td>
      <td>0.108793</td>
      <td>0.033122</td>
      <td>-0.049722</td>
      <td>0.538841</td>
      <td>0.662067</td>
      <td>0.777359</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.245917</td>
      <td>0.242902</td>
      <td>0.239154</td>
      <td>-0.009362</td>
      <td>-0.001944</td>
      <td>-0.069235</td>
      <td>-0.043461</td>
      <td>-0.033590</td>
      <td>-0.026565</td>
      <td>0.216614</td>
    </tr>
    <tr>
      <th>PAY_5</th>
      <td>-0.022199</td>
      <td>-0.249411</td>
      <td>-0.055064</td>
      <td>0.097520</td>
      <td>0.035629</td>
      <td>-0.053826</td>
      <td>0.509426</td>
      <td>0.622780</td>
      <td>0.686775</td>
      <td>0.819835</td>
      <td>...</td>
      <td>0.271915</td>
      <td>0.269783</td>
      <td>0.262509</td>
      <td>-0.006089</td>
      <td>-0.003191</td>
      <td>0.009062</td>
      <td>-0.058299</td>
      <td>-0.033337</td>
      <td>-0.023027</td>
      <td>0.204149</td>
    </tr>
    <tr>
      <th>PAY_6</th>
      <td>-0.020270</td>
      <td>-0.235195</td>
      <td>-0.044008</td>
      <td>0.082316</td>
      <td>0.034345</td>
      <td>-0.048773</td>
      <td>0.474553</td>
      <td>0.575501</td>
      <td>0.632684</td>
      <td>0.716449</td>
      <td>...</td>
      <td>0.266356</td>
      <td>0.290894</td>
      <td>0.285091</td>
      <td>-0.001496</td>
      <td>-0.005223</td>
      <td>0.005834</td>
      <td>0.019018</td>
      <td>-0.046434</td>
      <td>-0.025299</td>
      <td>0.186866</td>
    </tr>
    <tr>
      <th>BILL_AMT1</th>
      <td>0.019389</td>
      <td>0.285430</td>
      <td>-0.033642</td>
      <td>0.023581</td>
      <td>-0.023472</td>
      <td>0.056239</td>
      <td>0.187068</td>
      <td>0.234887</td>
      <td>0.208473</td>
      <td>0.202812</td>
      <td>...</td>
      <td>0.860272</td>
      <td>0.829779</td>
      <td>0.802650</td>
      <td>0.140277</td>
      <td>0.099355</td>
      <td>0.156887</td>
      <td>0.158303</td>
      <td>0.167026</td>
      <td>0.179341</td>
      <td>-0.019644</td>
    </tr>
    <tr>
      <th>BILL_AMT2</th>
      <td>0.017982</td>
      <td>0.278314</td>
      <td>-0.031183</td>
      <td>0.018749</td>
      <td>-0.021602</td>
      <td>0.054283</td>
      <td>0.189859</td>
      <td>0.235257</td>
      <td>0.237295</td>
      <td>0.225816</td>
      <td>...</td>
      <td>0.892482</td>
      <td>0.859778</td>
      <td>0.831594</td>
      <td>0.280365</td>
      <td>0.100851</td>
      <td>0.150718</td>
      <td>0.147398</td>
      <td>0.157957</td>
      <td>0.174256</td>
      <td>-0.014193</td>
    </tr>
    <tr>
      <th>BILL_AMT3</th>
      <td>0.024354</td>
      <td>0.283236</td>
      <td>-0.024563</td>
      <td>0.013002</td>
      <td>-0.024909</td>
      <td>0.053710</td>
      <td>0.179785</td>
      <td>0.224146</td>
      <td>0.227494</td>
      <td>0.244983</td>
      <td>...</td>
      <td>0.923969</td>
      <td>0.883910</td>
      <td>0.853320</td>
      <td>0.244335</td>
      <td>0.316936</td>
      <td>0.130011</td>
      <td>0.143405</td>
      <td>0.179712</td>
      <td>0.182326</td>
      <td>-0.014076</td>
    </tr>
    <tr>
      <th>BILL_AMT4</th>
      <td>0.040351</td>
      <td>0.293988</td>
      <td>-0.021880</td>
      <td>-0.000451</td>
      <td>-0.023344</td>
      <td>0.051353</td>
      <td>0.179125</td>
      <td>0.222237</td>
      <td>0.227202</td>
      <td>0.245917</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.940134</td>
      <td>0.900941</td>
      <td>0.233012</td>
      <td>0.207564</td>
      <td>0.300023</td>
      <td>0.130191</td>
      <td>0.160433</td>
      <td>0.177637</td>
      <td>-0.010156</td>
    </tr>
    <tr>
      <th>BILL_AMT5</th>
      <td>0.016705</td>
      <td>0.295562</td>
      <td>-0.017005</td>
      <td>-0.007567</td>
      <td>-0.025393</td>
      <td>0.049345</td>
      <td>0.180635</td>
      <td>0.221348</td>
      <td>0.225145</td>
      <td>0.242902</td>
      <td>...</td>
      <td>0.940134</td>
      <td>1.000000</td>
      <td>0.946197</td>
      <td>0.217031</td>
      <td>0.181246</td>
      <td>0.252305</td>
      <td>0.293118</td>
      <td>0.141574</td>
      <td>0.164184</td>
      <td>-0.006760</td>
    </tr>
    <tr>
      <th>BILL_AMT6</th>
      <td>0.016730</td>
      <td>0.290389</td>
      <td>-0.016733</td>
      <td>-0.009099</td>
      <td>-0.021207</td>
      <td>0.047613</td>
      <td>0.176980</td>
      <td>0.219403</td>
      <td>0.222327</td>
      <td>0.239154</td>
      <td>...</td>
      <td>0.900941</td>
      <td>0.946197</td>
      <td>1.000000</td>
      <td>0.199965</td>
      <td>0.172663</td>
      <td>0.233770</td>
      <td>0.250237</td>
      <td>0.307729</td>
      <td>0.115494</td>
      <td>-0.005372</td>
    </tr>
    <tr>
      <th>PAY_AMT1</th>
      <td>0.009742</td>
      <td>0.195236</td>
      <td>-0.000242</td>
      <td>-0.037456</td>
      <td>-0.005979</td>
      <td>0.026147</td>
      <td>-0.079269</td>
      <td>-0.080701</td>
      <td>0.001295</td>
      <td>-0.009362</td>
      <td>...</td>
      <td>0.233012</td>
      <td>0.217031</td>
      <td>0.199965</td>
      <td>1.000000</td>
      <td>0.285576</td>
      <td>0.252191</td>
      <td>0.199558</td>
      <td>0.148459</td>
      <td>0.185735</td>
      <td>-0.072929</td>
    </tr>
    <tr>
      <th>PAY_AMT2</th>
      <td>0.008406</td>
      <td>0.178408</td>
      <td>-0.001391</td>
      <td>-0.030038</td>
      <td>-0.008093</td>
      <td>0.021785</td>
      <td>-0.070101</td>
      <td>-0.058990</td>
      <td>-0.066793</td>
      <td>-0.001944</td>
      <td>...</td>
      <td>0.207564</td>
      <td>0.181246</td>
      <td>0.172663</td>
      <td>0.285576</td>
      <td>1.000000</td>
      <td>0.244770</td>
      <td>0.180107</td>
      <td>0.180908</td>
      <td>0.157634</td>
      <td>-0.058579</td>
    </tr>
    <tr>
      <th>PAY_AMT3</th>
      <td>0.039151</td>
      <td>0.210167</td>
      <td>-0.008597</td>
      <td>-0.039943</td>
      <td>-0.003541</td>
      <td>0.029247</td>
      <td>-0.070561</td>
      <td>-0.055901</td>
      <td>-0.053311</td>
      <td>-0.069235</td>
      <td>...</td>
      <td>0.300023</td>
      <td>0.252305</td>
      <td>0.233770</td>
      <td>0.252191</td>
      <td>0.244770</td>
      <td>1.000000</td>
      <td>0.216325</td>
      <td>0.159214</td>
      <td>0.162740</td>
      <td>-0.056250</td>
    </tr>
    <tr>
      <th>PAY_AMT4</th>
      <td>0.007793</td>
      <td>0.203242</td>
      <td>-0.002229</td>
      <td>-0.038218</td>
      <td>-0.012659</td>
      <td>0.021379</td>
      <td>-0.064005</td>
      <td>-0.046858</td>
      <td>-0.046067</td>
      <td>-0.043461</td>
      <td>...</td>
      <td>0.130191</td>
      <td>0.293118</td>
      <td>0.250237</td>
      <td>0.199558</td>
      <td>0.180107</td>
      <td>0.216325</td>
      <td>1.000000</td>
      <td>0.151830</td>
      <td>0.157834</td>
      <td>-0.056827</td>
    </tr>
    <tr>
      <th>PAY_AMT5</th>
      <td>0.000652</td>
      <td>0.217202</td>
      <td>-0.001667</td>
      <td>-0.040358</td>
      <td>-0.001205</td>
      <td>0.022850</td>
      <td>-0.058190</td>
      <td>-0.037093</td>
      <td>-0.035863</td>
      <td>-0.033590</td>
      <td>...</td>
      <td>0.160433</td>
      <td>0.141574</td>
      <td>0.307729</td>
      <td>0.148459</td>
      <td>0.180908</td>
      <td>0.159214</td>
      <td>0.151830</td>
      <td>1.000000</td>
      <td>0.154896</td>
      <td>-0.055124</td>
    </tr>
    <tr>
      <th>PAY_AMT6</th>
      <td>0.003000</td>
      <td>0.219595</td>
      <td>-0.002766</td>
      <td>-0.037200</td>
      <td>-0.006641</td>
      <td>0.019478</td>
      <td>-0.058673</td>
      <td>-0.036500</td>
      <td>-0.035861</td>
      <td>-0.026565</td>
      <td>...</td>
      <td>0.177637</td>
      <td>0.164184</td>
      <td>0.115494</td>
      <td>0.185735</td>
      <td>0.157634</td>
      <td>0.162740</td>
      <td>0.157834</td>
      <td>0.154896</td>
      <td>1.000000</td>
      <td>-0.053183</td>
    </tr>
    <tr>
      <th>default payment next month</th>
      <td>-0.013952</td>
      <td>-0.153520</td>
      <td>-0.039961</td>
      <td>0.028006</td>
      <td>-0.024339</td>
      <td>0.013890</td>
      <td>0.324794</td>
      <td>0.263551</td>
      <td>0.235253</td>
      <td>0.216614</td>
      <td>...</td>
      <td>-0.010156</td>
      <td>-0.006760</td>
      <td>-0.005372</td>
      <td>-0.072929</td>
      <td>-0.058579</td>
      <td>-0.056250</td>
      <td>-0.056827</td>
      <td>-0.055124</td>
      <td>-0.053183</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>25 rows × 25 columns</p>
</div>




```python
df.columns
```




    Index(['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
           'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
           'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
           'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
           'default payment next month'],
          dtype='object')




```python
df.corrwith(df['default payment next month']).sort_values(ascending=False)
```




    default payment next month    1.000000
    PAY_0                         0.324794
    PAY_2                         0.263551
    PAY_3                         0.235253
    PAY_4                         0.216614
    PAY_5                         0.204149
    PAY_6                         0.186866
    EDUCATION                     0.028006
    AGE                           0.013890
    BILL_AMT6                    -0.005372
    BILL_AMT5                    -0.006760
    BILL_AMT4                    -0.010156
    ID                           -0.013952
    BILL_AMT3                    -0.014076
    BILL_AMT2                    -0.014193
    BILL_AMT1                    -0.019644
    MARRIAGE                     -0.024339
    SEX                          -0.039961
    PAY_AMT6                     -0.053183
    PAY_AMT5                     -0.055124
    PAY_AMT3                     -0.056250
    PAY_AMT4                     -0.056827
    PAY_AMT2                     -0.058579
    PAY_AMT1                     -0.072929
    LIMIT_BAL                    -0.153520
    dtype: float64




```python
plt.figure(figsize=(16, 16))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='Blues')
plt.show()
```


    
![png](/assets/images/2023-03-09-Machine Learning 14 (PCA 활용)/output_10_0.png)
    



```python
df.columns
```




    Index(['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
           'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
           'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
           'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
           'default payment next month'],
          dtype='object')




```python
from sklearn.model_selection import train_test_split
X = df.drop(['ID', 'default payment next month'], axis=1)
y = df['default payment next month']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
```

    (24000, 23) (24000,)
    

- 상관관계가 높은 BILL_AMT1 ~ BILL_AMT6 6개 열의 변동 비율


```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

bill_columns = ['BILL_AMT' + str(i) for i in range(1, 7)]


scaler = StandardScaler()
bill_columns_sclaed = scaler.fit_transform(X_train[bill_columns])

pca = PCA(n_components = 2)
bill_columns_scaled_pca = pca.fit_transform(bill_columns_sclaed)
```


```python
pca.explained_variance_ratio_
```




    array([0.90358461, 0.05178874])



- 상관관계가 높은 PAY_0, PAY_2 ~ PAY_6 6개 열의 변동 비율


```python
pay_columns = ['PAY_' + str(i) for i in range(0, 7) if i != 1]


scaler = StandardScaler()
pay_columns_sclaed = scaler.fit_transform(X_train[pay_columns])

pca = PCA(n_components = 2)
pay_columns_scaled_pca = pca.fit_transform(pay_columns_sclaed)
print(pca.explained_variance_ratio_)
```

    [0.71700378 0.11658426]
    

- 원본 데이터로 모델 성능 측정하기


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rf_clf = RandomForestClassifier(random_state=42)
scores = cross_val_score(rf_clf, X_train, y_train, scoring='accuracy', cv=3)
scores
```




    array([0.8155  , 0.818125, 0.810125])



- PCA로 압축된 데이터로 모델 성능 측정하기


```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=8)
X_train_scaled_pca = pca.fit_transform(X_train_scaled)
X_train_scaled_pca.shape
```




    (24000, 8)




```python
rf_clf = RandomForestClassifier(random_state=42)
scores = cross_val_score(rf_clf, X_train_scaled_pca, y_train, scoring='accuracy', cv=3)
scores
```




    array([0.796875, 0.795125, 0.797125])




```python
pca.explained_variance_ratio_ # 각 component의 분산 비율
```




    array([0.28465193, 0.17799685, 0.06810077, 0.06421023, 0.04460868,
           0.04197626, 0.03963272, 0.0382991 ])




```python
pca.explained_variance_ratio_.sum()
```




    0.7594765578111686




```python
X_train_scaled_pca.shape
```




    (24000, 8)

## Reference
- [파이썬 머신러닝 완벽 가이드 (권철민 저)](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=292601583)
