---
tag: [Deep Learning, 딥러닝, RNN, 자연어처리, word2vec, pytorch, 파이토치, AG News Dataset]
toc: true
toc_sticky: true
toc_label: 목차
author_profile: false

---

# 문서 분류를 위해 임베딩 사용하기
**(Predict News Category with Pretrained word2vec)**


```python
import os
import random
from collections import Counter
import string
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import nltk
nltk.download('punkt')

import string
from collections import Counter
```

    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt.zip.
    


```python
# 시드값 고정
seed = 50
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)                # 파이썬 난수 생성기 시드 고정
np.random.seed(seed)             # 넘파이 난수 생성기 시드 고정
torch.manual_seed(seed)          # 파이토치 난수 생성기 시드 고정 (CPU 사용 시)
torch.cuda.manual_seed(seed)     # 파이토치 난수 생성기 시드 고정 (GPU 사용 시)
torch.cuda.manual_seed_all(seed) # 파이토치 난수 생성기 시드 고정 (멀티GPU 사용 시)
torch.backends.cudnn.deterministic = True # 확정적 연산 사용
torch.backends.cudnn.benchmark = False    # 벤치마크 기능 해제
torch.backends.cudnn.enabled = False      # cudnn 사용 해제
```


```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
```




    device(type='cuda')



## 1. 데이터 다운로드


```python
# kaggle api를 사용할 수 있는 패키지 설치
!pip install kaggle

# kaggle.json upload
from google.colab import files
files.upload()

# permmision warning 방지
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# download
!kaggle datasets download -d amananandrai/ag-news-classification-dataset

!mkdir dataset
# unzip(압축풀기)
!unzip -q ag-news-classification-dataset.zip -d dataset/
```


    Saving kaggle.json to kaggle.json
    Downloading ag-news-classification-dataset.zip to /content
      0% 0.00/11.4M [00:00<?, ?B/s]
    100% 11.4M/11.4M [00:00<00:00, 167MB/s]
    

## 2. 데이터 불러오기

### Vocabulary


```python
class Vocabulary():    
    def __init__(self, vocab_threshold, vocab_file,
                 mask_word="<mask>",
                 start_word="<start>",
                 end_word="<end>",
                 unk_word="<unk>",
                 news_df=None, vocab_from_file=False):
        
        self.vocab_threshold = vocab_threshold
        # train과 valid로 나귀기전 전체 데이터(train_news_df)
        self.news_df = news_df

        # dictionary 초기화
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
                
        if vocab_from_file:
            # 파일로부터 읽기
            with open(vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print('Vocabulary succesfully loaded from vocab.pkl file!')
        else:
            self.build_vocab()
            with open(vocab_file, 'wb') as f:
                pickle.dump(self, f)

    def build_vocab(self):
        # mask_word (0), start_word(1), end_word (2), unk_word (3)
        self.mask_index = self.add_word(mask_word) # 0
        self.begin_seq_index = self.add_word(start_word) # 1
        self.end_seq_index = self.add_word(end_word) # 2
        self.unk_index = self.add_word(unk_word) # 3
        self.add_titles()

    def add_word(self, word):
        if not word in self.word2idx:
            idx = self.idx
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
        return idx

    def add_titles(self):
        counter = Counter()
        for title in self.news_df.Title:
            for token in nltk.tokenize.word_tokenize(title.lower()):
                if token not in string.punctuation:
                    counter[token] += 1 # 단어별 빈도수를 계산

        for word, cnt in counter.items():
            if cnt >= self.vocab_threshold:
                self.add_word(word)

        print("title_vocab 길이 :", len(self.word2idx))      

    def __call__(self, word): # lookup word
        return self.word2idx.get(word, self.unk_index)

    def __len__(self):
        return len(self.word2idx)
```


```python
train_news_csv="./dataset/train.csv"
test_news_csv="./dataset/test.csv"
train_news_df = pd.read_csv(train_news_csv)
test_news_df = pd.read_csv(test_news_csv)
```


```python
len(train_news_df)
```




    120000




```python
from sklearn.model_selection import train_test_split

train_indices, valid_indices = train_test_split(range(len(train_news_df)), 
                                                stratify= train_news_df['Class Index'], 
                                                test_size=0.2)
len(train_indices), len(valid_indices)
```




    (96000, 24000)




```python
train_df = train_news_df.iloc[train_indices]
valid_df = train_news_df.iloc[valid_indices]
```

**클래스별 분포**


```python
train_news_df['Class Index'].value_counts()/len(train_news_df)
```




    3    0.25
    4    0.25
    2    0.25
    1    0.25
    Name: Class Index, dtype: float64




```python
valid_df['Class Index'].value_counts()/len(valid_df)
```




    3    0.25
    2    0.25
    4    0.25
    1    0.25
    Name: Class Index, dtype: float64




```python
# Consists of class ids 1-4 where 1-World, 2-Sports, 3-Business, 4-Sci/Tech
category_map = {1:"World", 2:"Sports", 3:"Business", 4:"Sci/Tech"}
```

### Dataset


```python
vocab_threshold = 25
mask_word = "<mask>"
start_word = "<start>"
end_word = "<end>"
unk_word = "<unk>"
vocab_file = './vocab.pkl'
vocab_from_file = False

vocab = Vocabulary(vocab_threshold, vocab_file, 
                    mask_word, start_word, end_word, unk_word, 
                    train_news_df, vocab_from_file)
```

    title_vocab 길이 : 4470
    


```python
def vectorize(text, vector_length = -1):
    # 입력 : 'Clijsters Unsure About Latest Injury, Says Hewitt'
    # 출력 : [1, 2, 4, 9, 10, 9, 2, 0, 0, 0, 0]
    # vocabulary 에서 text의 각 단어들의 id를 가져올 수 있도록

    indices = [vocab.begin_seq_index]
    word_list = nltk.tokenize.word_tokenize(text.lower())
    for word in word_list:
        indices.append(vocab(word))
    indices.append(vocab.end_seq_index)

    if vector_length < 0:
        vector_length = len(indices)

    out_vector = np.zeros(vector_length, dtype=np.int64)
    out_vector[:len(indices)] = indices
    out_vector[len(indices):] = vocab.mask_index

    return out_vector    
```


```python
vectorize("I am a boy", 21)
```




    array([   1, 1319,    3,   78, 2296,    2,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0])




```python
class NewsDataset(Dataset):
  def __init__(self, mode, vocab_threshold, vocab_file,
               mask_word, start_word, end_word, unk_word,
               news_df, vocab_from_file):
      self.news_df = news_df
      self.title_vocab = Vocabulary(vocab_threshold, vocab_file,
                                    mask_word, start_word, end_word, unk_word,
                                    train_news_df, vocab_from_file)
      measure_len = lambda context : len(nltk.tokenize.word_tokenize(context))
      # measure_len = lambda context : len(nltk.tokenize.word_tokenize(context.lower()))
      self.max_seq_length = max(map(measure_len, train_news_df.Title)) + 2

  def __getitem__(self, index):
      row = self.news_df.iloc[index]
      title_vector = vectorize(row.Title, self.max_seq_length)
      category_index = row['Class Index'] - 1
      return {'x_data' : title_vector, 
              'y_target' : category_index }

  def __len__(self):
      return len(self.news_df)
```


```python
vocab_threshold = 25
mask_word = "<mask>"
start_word = "<start>"
end_word = "<end>"
unk_word = "<unk>"
vocab_file = './vocab.pkl'
vocab_from_file = False

trainset = NewsDataset("train", vocab_threshold, vocab_file,
                       mask_word, start_word, end_word, unk_word,
                       train_df, vocab_from_file)
```

    title_vocab 길이 : 4470
    


```python
vocab_from_file = True
validset = NewsDataset("valid", vocab_threshold, vocab_file,
                       mask_word, start_word, end_word, unk_word,
                       valid_df, vocab_from_file)
testset = NewsDataset("test", vocab_threshold, vocab_file,
                       mask_word, start_word, end_word, unk_word,
                       test_news_df, vocab_from_file)
```

    Vocabulary succesfully loaded from vocab.pkl file!
    Vocabulary succesfully loaded from vocab.pkl file!
    


```python
len(trainset), len(validset), len(testset)
```




    (96000, 24000, 7600)



## 3. 데이터 적재 : DataLoader


```python
batch_size = 128
trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)

validloader = DataLoader(dataset=validset, batch_size=batch_size, shuffle=True)

testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)
```


```python
batch = next(iter(trainloader))
batch['x_data'].size(), batch['y_target'].size()
```




    (torch.Size([128, 43]), torch.Size([128]))




```python
batch['x_data']
```




    tensor([[   1,  166,  306,  ...,    0,    0,    0],
            [   1, 3870, 2013,  ...,    0,    0,    0],
            [   1, 2847,  150,  ...,    0,    0,    0],
            ...,
            [   1, 4083,   30,  ...,    0,    0,    0],
            [   1, 3931,  882,  ...,    0,    0,    0],
            [   1, 2863, 2942,  ...,    0,    0,    0]])



## 4. Embedding Matrix


```python
# https://nlp.stanford.edu/projects/glove/
```


```python
# GloVe 데이터를 다운로드
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip
!mkdir -p data/glove
!mv glove.6B.100d.txt data/glove
```

    --2023-04-26 01:00:53--  http://nlp.stanford.edu/data/glove.6B.zip
    Resolving nlp.stanford.edu (nlp.stanford.edu)... 171.64.67.140
    Connecting to nlp.stanford.edu (nlp.stanford.edu)|171.64.67.140|:80... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://nlp.stanford.edu/data/glove.6B.zip [following]
    --2023-04-26 01:00:53--  https://nlp.stanford.edu/data/glove.6B.zip
    Connecting to nlp.stanford.edu (nlp.stanford.edu)|171.64.67.140|:443... connected.
    HTTP request sent, awaiting response... 301 Moved Permanently
    Location: https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip [following]
    --2023-04-26 01:00:53--  https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
    Resolving downloads.cs.stanford.edu (downloads.cs.stanford.edu)... 171.64.64.22
    Connecting to downloads.cs.stanford.edu (downloads.cs.stanford.edu)|171.64.64.22|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 862182613 (822M) [application/zip]
    Saving to: ‘glove.6B.zip’
    
    glove.6B.zip        100%[===================>] 822.24M  5.02MB/s    in 2m 39s  
    
    2023-04-26 01:03:33 (5.17 MB/s) - ‘glove.6B.zip’ saved [862182613/862182613]
    
    Archive:  glove.6B.zip
      inflating: glove.6B.50d.txt        
      inflating: glove.6B.100d.txt       
      inflating: glove.6B.200d.txt       
      inflating: glove.6B.300d.txt       
    


```python
!head -5 data/glove/glove.6B.100d.txt
```

    the -0.038194 -0.24487 0.72812 -0.39961 0.083172 0.043953 -0.39141 0.3344 -0.57545 0.087459 0.28787 -0.06731 0.30906 -0.26384 -0.13231 -0.20757 0.33395 -0.33848 -0.31743 -0.48336 0.1464 -0.37304 0.34577 0.052041 0.44946 -0.46971 0.02628 -0.54155 -0.15518 -0.14107 -0.039722 0.28277 0.14393 0.23464 -0.31021 0.086173 0.20397 0.52624 0.17164 -0.082378 -0.71787 -0.41531 0.20335 -0.12763 0.41367 0.55187 0.57908 -0.33477 -0.36559 -0.54857 -0.062892 0.26584 0.30205 0.99775 -0.80481 -3.0243 0.01254 -0.36942 2.2167 0.72201 -0.24978 0.92136 0.034514 0.46745 1.1079 -0.19358 -0.074575 0.23353 -0.052062 -0.22044 0.057162 -0.15806 -0.30798 -0.41625 0.37972 0.15006 -0.53212 -0.2055 -1.2526 0.071624 0.70565 0.49744 -0.42063 0.26148 -1.538 -0.30223 -0.073438 -0.28312 0.37104 -0.25217 0.016215 -0.017099 -0.38984 0.87424 -0.72569 -0.51058 -0.52028 -0.1459 0.8278 0.27062
    , -0.10767 0.11053 0.59812 -0.54361 0.67396 0.10663 0.038867 0.35481 0.06351 -0.094189 0.15786 -0.81665 0.14172 0.21939 0.58505 -0.52158 0.22783 -0.16642 -0.68228 0.3587 0.42568 0.19021 0.91963 0.57555 0.46185 0.42363 -0.095399 -0.42749 -0.16567 -0.056842 -0.29595 0.26037 -0.26606 -0.070404 -0.27662 0.15821 0.69825 0.43081 0.27952 -0.45437 -0.33801 -0.58184 0.22364 -0.5778 -0.26862 -0.20425 0.56394 -0.58524 -0.14365 -0.64218 0.0054697 -0.35248 0.16162 1.1796 -0.47674 -2.7553 -0.1321 -0.047729 1.0655 1.1034 -0.2208 0.18669 0.13177 0.15117 0.7131 -0.35215 0.91348 0.61783 0.70992 0.23955 -0.14571 -0.37859 -0.045959 -0.47368 0.2385 0.20536 -0.18996 0.32507 -1.1112 -0.36341 0.98679 -0.084776 -0.54008 0.11726 -1.0194 -0.24424 0.12771 0.013884 0.080374 -0.35414 0.34951 -0.7226 0.37549 0.4441 -0.99059 0.61214 -0.35111 -0.83155 0.45293 0.082577
    . -0.33979 0.20941 0.46348 -0.64792 -0.38377 0.038034 0.17127 0.15978 0.46619 -0.019169 0.41479 -0.34349 0.26872 0.04464 0.42131 -0.41032 0.15459 0.022239 -0.64653 0.25256 0.043136 -0.19445 0.46516 0.45651 0.68588 0.091295 0.21875 -0.70351 0.16785 -0.35079 -0.12634 0.66384 -0.2582 0.036542 -0.13605 0.40253 0.14289 0.38132 -0.12283 -0.45886 -0.25282 -0.30432 -0.11215 -0.26182 -0.22482 -0.44554 0.2991 -0.85612 -0.14503 -0.49086 0.0082973 -0.17491 0.27524 1.4401 -0.21239 -2.8435 -0.27958 -0.45722 1.6386 0.78808 -0.55262 0.65 0.086426 0.39012 1.0632 -0.35379 0.48328 0.346 0.84174 0.098707 -0.24213 -0.27053 0.045287 -0.40147 0.11395 0.0062226 0.036673 0.018518 -1.0213 -0.20806 0.64072 -0.068763 -0.58635 0.33476 -1.1432 -0.1148 -0.25091 -0.45907 -0.096819 -0.17946 -0.063351 -0.67412 -0.068895 0.53604 -0.87773 0.31802 -0.39242 -0.23394 0.47298 -0.028803
    of -0.1529 -0.24279 0.89837 0.16996 0.53516 0.48784 -0.58826 -0.17982 -1.3581 0.42541 0.15377 0.24215 0.13474 0.41193 0.67043 -0.56418 0.42985 -0.012183 -0.11677 0.31781 0.054177 -0.054273 0.35516 -0.30241 0.31434 -0.33846 0.71715 -0.26855 -0.15837 -0.47467 0.051581 -0.33252 0.15003 -0.1299 -0.54617 -0.37843 0.64261 0.82187 -0.080006 0.078479 -0.96976 -0.57741 0.56491 -0.39873 -0.057099 0.19743 0.065706 -0.48092 -0.20125 -0.40834 0.39456 -0.02642 -0.11838 1.012 -0.53171 -2.7474 -0.042981 -0.74849 1.7574 0.59085 0.04885 0.78267 0.38497 0.42097 0.67882 0.10337 0.6328 -0.026595 0.58647 -0.44332 0.33057 -0.12022 -0.55645 0.073611 0.20915 0.43395 -0.012761 0.089874 -1.7991 0.084808 0.77112 0.63105 -0.90685 0.60326 -1.7515 0.18596 -0.50687 -0.70203 0.66578 -0.81304 0.18712 -0.018488 -0.26757 0.727 -0.59363 -0.34839 -0.56094 -0.591 1.0039 0.20664
    to -0.1897 0.050024 0.19084 -0.049184 -0.089737 0.21006 -0.54952 0.098377 -0.20135 0.34241 -0.092677 0.161 -0.13268 -0.2816 0.18737 -0.42959 0.96039 0.13972 -1.0781 0.40518 0.50539 -0.55064 0.4844 0.38044 -0.0029055 -0.34942 -0.099696 -0.78368 1.0363 -0.2314 -0.47121 0.57126 -0.21454 0.35958 -0.48319 1.0875 0.28524 0.12447 -0.039248 -0.076732 -0.76343 -0.32409 -0.5749 -1.0893 -0.41811 0.4512 0.12112 -0.51367 -0.13349 -1.1378 -0.28768 0.16774 0.55804 1.5387 0.018859 -2.9721 -0.24216 -0.92495 2.1992 0.28234 -0.3478 0.51621 -0.43387 0.36852 0.74573 0.072102 0.27931 0.92569 -0.050336 -0.85856 -0.1358 -0.92551 -0.33991 -1.0394 -0.067203 -0.21379 -0.4769 0.21377 -0.84008 0.052536 0.59298 0.29604 -0.67644 0.13916 -1.5504 -0.20765 0.7222 0.52056 -0.076221 -0.15194 -0.13134 0.058617 -0.31869 -0.61419 -0.62393 -0.41548 -0.038175 -0.39804 0.47647 -0.15983
    


```python
!tail -5 data/glove/glove.6B.100d.txt
```

    chanty -0.15577 -0.049188 -0.064377 0.2236 -0.20146 -0.038963 0.12971 -0.29451 0.0035897 -0.098377 -0.30939 0.050878 0.24574 -0.25382 -0.048145 0.15506 -0.39446 0.086549 0.22096 0.010005 -0.029974 -0.16488 -0.51622 -0.00016491 -0.11421 0.26008 -0.19419 0.36296 0.0099712 0.1731 -0.1477 -0.78212 0.19243 -0.14533 0.41308 0.0048941 -0.33375 -0.20914 0.26039 0.10949 0.49339 0.089623 -0.020955 0.15683 0.3137 -0.11759 -0.31317 0.69917 -0.13166 0.64363 -0.23016 0.20046 0.14912 -0.075741 -0.0029152 0.52797 0.18252 0.091756 -0.34982 -0.082495 0.0635 -0.30766 -0.13771 -0.55364 -3.6811e-05 0.11055 -0.42719 -0.21429 0.12669 -0.08624 0.23952 -0.037189 0.273 0.26172 -0.44486 0.10141 0.23421 0.0083713 0.71138 0.31115 -0.39297 -0.26296 0.40601 0.20615 0.20524 -0.11601 0.0101 0.15099 0.13692 0.18864 0.093324 0.094486 -0.023469 -0.48099 0.62332 0.024318 -0.27587 0.075044 -0.5638 0.14501
    kronik -0.094426 0.14725 -0.15739 0.071966 -0.29845 0.039432 0.02187 0.0080409 -0.18682 -0.31101 0.043422 0.16147 -0.012647 0.050696 -0.050954 0.013533 -0.20035 0.3019 -0.010799 -0.19664 -0.26712 -0.38311 -0.08666 -0.10954 0.0042728 -0.15433 0.15416 0.22333 0.13355 0.076866 0.045246 -0.00021332 0.0067774 0.047134 0.32453 -0.31853 -0.35445 -0.21979 -0.12723 0.10492 0.26715 0.058452 0.16751 0.31884 0.046914 -0.16315 -0.078414 0.50551 0.32689 0.046858 0.041268 0.49351 -0.44075 -0.15669 0.62512 0.59474 0.084773 0.017492 -0.89279 -0.28656 0.39685 -0.35591 -0.15007 -0.12261 -0.2569 0.08311 0.013643 0.16162 -0.006617 0.015909 -0.16797 0.11139 0.09692 -0.006589 -0.26508 -0.11282 0.034702 -0.022573 0.44549 0.30833 -0.20067 -0.033317 0.10966 -0.18257 0.54201 -0.11415 -0.19819 0.28277 -0.34232 0.3163 -0.30545 -0.011082 0.11855 -0.11312 0.33951 -0.22449 0.25743 0.63143 -0.2009 -0.10542
    rolonda 0.36088 -0.16919 -0.32704 0.098332 -0.4297 -0.18874 0.45556 0.28529 0.3034 -0.36683 -0.13923 0.10053 -0.52026 -0.30629 -0.18236 0.23908 -0.45987 0.35561 0.067856 0.069954 -0.044425 -0.19452 -0.30248 -0.31011 0.43554 0.24623 0.05153 0.31476 0.094553 0.32482 0.38447 -0.099659 0.414 0.25902 0.08238 0.096832 0.22163 -0.47655 0.13628 0.12927 0.26019 0.45182 -0.079522 0.72982 -0.56586 -0.18653 -0.6082 0.16635 -0.52492 0.14167 -0.26089 -0.046457 -0.060964 -0.48269 0.32158 0.7501 0.52608 0.2913 -0.46036 -0.39314 0.17445 -0.25116 -0.2416 -0.33391 0.086837 0.1027 0.074325 -0.29071 0.3768 0.0079988 0.42131 -0.30349 0.11643 0.38284 0.030575 0.11889 0.42949 -0.0054543 0.83973 0.16628 0.087226 -0.2906 0.16843 -0.19309 0.35477 0.24789 0.14577 0.31387 -0.084938 0.21647 -0.044082 0.14003 0.30007 -0.12731 -0.14304 -0.069396 0.2816 0.27139 -0.29188 0.16109
    zsombor -0.10461 -0.5047 -0.49331 0.13516 -0.36371 -0.4475 0.18429 -0.05651 0.40474 -0.72583 0.31079 -0.31763 -0.019824 -0.29765 0.16847 -0.029003 -0.42048 0.039778 0.10003 0.14749 -0.24683 0.040093 -0.0938 0.32488 -0.22667 -0.039094 0.21616 0.4959 0.069714 0.16686 -0.026112 0.096436 0.18843 -0.3967 0.082282 -0.38073 0.086211 -0.40775 -0.1726 -0.29836 0.29015 -0.0774 0.017294 0.32361 -0.22261 -0.72733 -0.070333 0.17454 0.021926 0.37076 0.37268 0.037672 -0.29863 -0.9022 0.28775 0.60194 0.028256 -0.15408 -0.39262 -0.22826 0.10673 -0.3631 0.35778 0.034102 -0.29885 0.42406 -0.57664 -0.40484 -0.15435 0.23217 -0.014499 0.2932 -0.030599 0.62079 -0.02442 -0.22534 0.13813 -0.21491 0.61883 0.047665 -0.2661 -0.35747 0.32165 -0.53815 0.63114 0.10025 0.22458 0.28004 -0.048782 0.72537 0.15153 -0.10842 0.34064 -0.40916 -0.081263 0.095315 0.15018 0.42527 -0.5125 -0.17054
    sandberger 0.28365 -0.6263 -0.44351 0.2177 -0.087421 -0.17062 0.29266 -0.024899 0.26414 -0.17023 0.25817 0.097484 -0.33103 -0.43859 0.0095799 0.095624 -0.17777 0.38886 0.27151 0.14742 -0.43973 -0.26588 -0.024271 0.27186 -0.36761 -0.24827 -0.20815 0.22128 -0.044409 0.021373 0.24594 0.26143 0.29303 0.13281 0.082232 -0.12869 0.1622 -0.22567 -0.060348 0.28703 0.11381 0.34839 0.3419 0.36996 -0.13592 0.0062694 0.080317 0.0036251 0.43093 0.01882 0.31008 0.16722 0.074112 -0.37745 0.47363 0.41284 0.24471 0.075965 -0.51725 -0.49481 0.526 -0.074645 0.41434 -0.1956 -0.16544 -0.045649 -0.40153 -0.13136 -0.4672 0.18825 0.2612 0.16854 0.22615 0.62992 -0.1288 0.055841 0.01928 0.024572 0.46875 0.2582 -0.31672 0.048591 0.3277 -0.50141 0.30855 0.11997 -0.25768 -0.039867 -0.059672 0.5525 0.13885 -0.22862 0.071792 -0.43208 0.5398 -0.085806 0.032651 0.43678 -0.82607 -0.15701
    


```python
# glove 파일에 40만개의 단어에 대해 100차원의 학습된 임베딩이 들어가있음
glove_filepath='data/glove/glove.6B.100d.txt'
```


```python
def load_glove_from_file(glove_filepath):
    # GloVe 임베딩 로드 
    word_to_index = {}
    embeddings = []
    with open(glove_filepath, "r") as fp:
        for index, line in enumerate(fp):
            line = line.split(" ") # each line: word num1 num2 ...
            word_to_index[line[0]] = index # word = line[0] 
            embedding_i = np.array([float(val) for val in line[1:]])
            embeddings.append(embedding_i)
    return word_to_index, np.stack(embeddings)
    
def make_embedding_matrix(glove_filepath, words):
    # 특정 단어 집합에 대한 임베딩 행렬을 만들기
    word_to_idx, glove_embeddings = load_glove_from_file(glove_filepath)
    embedding_size = glove_embeddings.shape[1]
    
    final_embeddings = np.zeros((len(words), embedding_size))

    for i, word in enumerate(words):
        if word in word_to_idx:
            final_embeddings[i, :] = glove_embeddings[word_to_idx[word]]
        else:
            embedding_i = torch.ones(1, embedding_size)
            torch.nn.init.xavier_uniform_(embedding_i)
            final_embeddings[i, :] = embedding_i

    return final_embeddings
```


```python
len(vocab)
```




    4470




```python
words = vocab.word2idx.keys()
embeddings = make_embedding_matrix(glove_filepath, words) 
```


```python
embeddings.shape # (len(vocab) , embedding_size)
```




    (4470, 100)



## 5. 모델 생성: `NewsClassifier`


```python
class NewsClassifier(nn.Module):
    def __init__(self, embedding_size, num_embeddings,                            
                 hidden_dim, num_classes, dropout_p,
                 pretrained_embeddings=None):
        super().__init__()
        if pretrained_embeddings is None:
            self.emb = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_size)
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            print(pretrained_embeddings.shape)
            self.emb = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_size,
                                    _weight = pretrained_embeddings)
            
        self.classifier = nn.Sequential(
                            nn.Linear(in_features=embedding_size, out_features=hidden_dim),
                            nn.BatchNorm1d(hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(dropout_p),
                            nn.Linear(in_features=hidden_dim, out_features=num_classes)

                        )
            
    def forward(self, inputs, apply_softmax=False): # input (batch_size, context_size) = (128, 43)
        h = self.emb(inputs)  # h (batch_size, context_size, embedding_size) = (128, 43, 100)
        h_mean = h.mean(axis=1) # h_mean (batch_size, embedding_size) = (128, 100)
        output = self.classifier(h_mean) # ouput (batch_size, num_classes) = (128, 4)

        if apply_softmax:
            output = F.softmax(output, dim=1)

        return output   
```

**하이퍼 파라미터 설정**


```python
embedding_size=100
hidden_dim=50
dropout_p=0.1
learning_rate=0.001
num_epochs=10
```


```python
classifier = NewsClassifier(embedding_size=embedding_size, 
                            num_embeddings=len(vocab),                            
                            hidden_dim=hidden_dim, 
                            num_classes=4, 
                            dropout_p=dropout_p,
                            pretrained_embeddings=embeddings
                            #pretrained_embeddings=None
                            )
```

    torch.Size([4470, 100])
    


```python
classifier = classifier.to(device)
classifier
```




    NewsClassifier(
      (emb): Embedding(4470, 100)
      (classifier): Sequential(
        (0): Linear(in_features=100, out_features=50, bias=True)
        (1): BatchNorm1d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Dropout(p=0.1, inplace=False)
        (4): Linear(in_features=50, out_features=4, bias=True)
      )
    )




```python
out = classifier(batch['x_data'].to(device))
out.shape
```




    torch.Size([128, 4])



## 6. 모델 설정 (손실함수, 옵티마이저 선택)


```python
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                           mode='min', factor=0.01,
                                           patience=1, verbose=True)
```

## 7. 모델 훈련


```python
def validate(model, validloader, loss_fn):
    model.eval()
    total = 0   
    correct = 0
    valid_loss = []
    valid_epoch_loss=0
    valid_accuracy = 0

    with torch.no_grad():
        for batch_data in validloader:
            titles = batch_data['x_data'].to(device)
            labels = batch_data['y_target'].to(device)

            # 전방향 예측과 손실
            logits = model(titles)
            loss = loss_fn(logits, labels)
            valid_loss.append(loss.item())
            
            # 정확도
            _, preds = torch.max(logits, 1) # 배치에 대한 최종 예측
            # preds = logit.max(dim=1)[1] 
            correct += int((preds == labels).sum()) # 배치 중 맞은 것의 개수가 correct에 누적
            total += labels.shape[0] # 배치 사이즈만큼씩 total에 누적

    valid_epoch_loss = np.mean(valid_loss)
    total_loss["val"].append(valid_epoch_loss)
    valid_accuracy = correct / total

    return valid_epoch_loss, valid_accuracy
```


```python
def train_loop(model, trainloader, loss_fn, epochs, optimizer):  
    min_loss = 1000000  
    trigger = 0
    patience = 3     

    for epoch in range(epochs):
        model.train()
        train_loss = []

        for batch_data in trainloader:
            titles = batch_data['x_data'].to(device)
            labels = batch_data['y_target'].to(device)

            optimizer.zero_grad()
            outputs = model(titles)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        train_epoch_loss = np.mean(train_loss)
        total_loss["train"].append(train_epoch_loss)
 
        valid_epoch_loss, valid_accuracy = validate(model, validloader, loss_fn)

        print("Epoch: {}/{}, Train Loss={:.4f}, Val Loss={:.4f}, Val Accyracy={:.4f}".format(
                epoch + 1, epochs,
                total_loss["train"][-1],
                total_loss["val"][-1],
                valid_accuracy))  
        
        # Early Stopping (조기 종료)
        if valid_epoch_loss > min_loss: # valid_loss가 min_loss를 갱신하지 못하면
          trigger += 1
          print('trigger : ', trigger)
          if trigger > patience:
            print('Early Stopping !!!')
            print('Training loop is finished !!')
            return
        else:
          trigger = 0
          min_loss = valid_epoch_loss
        # -------------------------------------------

        # Learning Rate Scheduler
        scheduler.step(valid_epoch_loss)
        # -------------------------------------------                      
```


```python
# without pretrained weight
epochs = 15
total_loss = {"train": [], "val": []}
%time train_loop(classifier, trainloader, loss_fn, epochs, optimizer)
```

    Epoch: 1/15, Train Loss=0.9641, Val Loss=0.7056, Val Accyracy=0.7392
    Epoch: 2/15, Train Loss=0.5927, Val Loss=1.1107, Val Accyracy=0.6199
    trigger :  1
    Epoch: 3/15, Train Loss=0.4938, Val Loss=0.6064, Val Accyracy=0.7741
    Epoch: 4/15, Train Loss=0.4487, Val Loss=0.5597, Val Accyracy=0.7895
    Epoch: 5/15, Train Loss=0.4234, Val Loss=0.7212, Val Accyracy=0.7613
    trigger :  1
    Epoch: 6/15, Train Loss=0.4041, Val Loss=0.5561, Val Accyracy=0.7881
    Epoch: 7/15, Train Loss=0.3933, Val Loss=0.7822, Val Accyracy=0.7169
    trigger :  1
    Epoch: 8/15, Train Loss=0.3818, Val Loss=0.5118, Val Accyracy=0.8197
    Epoch: 9/15, Train Loss=0.3741, Val Loss=0.6004, Val Accyracy=0.8008
    trigger :  1
    Epoch: 10/15, Train Loss=0.3681, Val Loss=0.5059, Val Accyracy=0.8157
    Epoch: 11/15, Train Loss=0.3625, Val Loss=0.6378, Val Accyracy=0.7687
    trigger :  1
    Epoch: 12/15, Train Loss=0.3600, Val Loss=0.8636, Val Accyracy=0.7327
    trigger :  2
    Epoch 00012: reducing learning rate of group 0 to 1.0000e-05.
    Epoch: 13/15, Train Loss=0.3372, Val Loss=0.4387, Val Accyracy=0.8448
    Epoch: 14/15, Train Loss=0.3339, Val Loss=0.4367, Val Accyracy=0.8440
    Epoch: 15/15, Train Loss=0.3335, Val Loss=0.4379, Val Accyracy=0.8449
    trigger :  1
    CPU times: user 8min 11s, sys: 2.29 s, total: 8min 13s
    Wall time: 8min 17s
    


```python
# with pretrained weight
epochs = 15
total_loss = {"train": [], "val": []}
%time train_loop(classifier, trainloader, loss_fn, epochs, optimizer)
```

    Epoch: 1/15, Train Loss=0.5550, Val Loss=0.4535, Val Accyracy=0.8364
    Epoch: 2/15, Train Loss=0.4330, Val Loss=0.4560, Val Accyracy=0.8330
    trigger :  1
    Epoch: 3/15, Train Loss=0.3992, Val Loss=0.4257, Val Accyracy=0.8440
    Epoch: 4/15, Train Loss=0.3787, Val Loss=0.4188, Val Accyracy=0.8483
    Epoch: 5/15, Train Loss=0.3646, Val Loss=0.4180, Val Accyracy=0.8491
    Epoch: 6/15, Train Loss=0.3538, Val Loss=0.4238, Val Accyracy=0.8475
    trigger :  1
    Epoch: 7/15, Train Loss=0.3423, Val Loss=0.4262, Val Accyracy=0.8466
    trigger :  2
    Epoch 00007: reducing learning rate of group 0 to 1.0000e-05.
    Epoch: 8/15, Train Loss=0.3154, Val Loss=0.4165, Val Accyracy=0.8509
    Epoch: 9/15, Train Loss=0.3142, Val Loss=0.4166, Val Accyracy=0.8508
    trigger :  1
    Epoch: 10/15, Train Loss=0.3137, Val Loss=0.4167, Val Accyracy=0.8504
    trigger :  2
    Epoch 00010: reducing learning rate of group 0 to 1.0000e-07.
    Epoch: 11/15, Train Loss=0.3129, Val Loss=0.4159, Val Accyracy=0.8518
    Epoch: 12/15, Train Loss=0.3123, Val Loss=0.4164, Val Accyracy=0.8516
    trigger :  1
    Epoch: 13/15, Train Loss=0.3132, Val Loss=0.4165, Val Accyracy=0.8515
    trigger :  2
    Epoch 00013: reducing learning rate of group 0 to 1.0000e-09.
    Epoch: 14/15, Train Loss=0.3131, Val Loss=0.4151, Val Accyracy=0.8511
    Epoch: 15/15, Train Loss=0.3137, Val Loss=0.4150, Val Accyracy=0.8506
    CPU times: user 8min 51s, sys: 2.57 s, total: 8min 54s
    Wall time: 10min 2s
    


```python
import matplotlib.pyplot as plt

plt.plot(total_loss['train'], label="train_loss")
plt.plot(total_loss['val'], label="vallid_loss")
plt.legend()
plt.show()
```


    
![png](/assets/images/2023-04-20-RNN 7 (word2vec application in Pytorch)/output_53_0.png)
    


## 8. 모델 평가


```python
def evaluate(model, testloader, loss_fn):
    model.eval()
    total = 0   
    correct = 0
    test_loss = []
    test_epoch_loss=0
    test_accuracy = 0

    with torch.no_grad():
        for batch_data in testloader:
            titles = batch_data['x_data'].to(device)
            labels = batch_data['y_target'].to(device)

            # 전방향 예측과 손실
            logits = model(titles)
            loss = loss_fn(logits, labels)
            test_loss.append(loss.item())
            
            # 정확도
            _, preds = torch.max(logits, 1) # 배치에 대한 최종 예측
            # preds = logit.max(dim=1)[1] 
            correct += int((preds == labels).sum()) # 배치 중 맞은 것의 개수가 correct에 누적
            total += labels.shape[0] # 배치 사이즈만큼씩 total에 누적

    test_epoch_loss = np.mean(test_loss)
    # total_loss["val"].append(test_epoch_loss)
    test_accuracy = correct / total

    print('Test Loss : {:.5f}'.format(test_epoch_loss), 
        'Test Accuracy : {:.5f}'.format(test_accuracy))


evaluate(classifier, testloader, loss_fn)  
```

    Test Loss : 0.41626 Test Accuracy : 0.85539
    

## 9. 모델 예측


```python
def predict_category(title, classifier, max_length):
    # 뉴스 제목을 기반으로 카테고리를 예측
    
    # 1. vetororize
    vectorized_title = vectorize(title, vector_length=max_length)
    vectorized_title = torch.tensor(vectorized_title).unsqueeze(0) # tensor로 바꾸고, 배치처리를 위해 차원 늘림

    # 2. model의 예측
    result = classifier(vectorized_title, apply_softmax=True) # result : 예측 확률
    probability, index= result.max(dim=1)
    predict = index.item() + 1 # 0번 클래스 예측은 실제 데이터 에서는 1번 클래스와 같다.
    probability = probability.item()
    preidct_category = category_map[predict]
    
    return {'category':preidct_category, 'probability':probability}    
    
```


```python
def get_samples():
    # True Category 기반 샘플 얻어오기
    # 클래스 별로 5개씩 샘플을 준비
    samples = {}

    for category in testset.news_df['Class Index'].unique(): # 1=>2=>3=>4
        samples[category]= testset.news_df.Title[testset.news_df['Class Index'] == category].tolist()[-5:]
    
    return samples

test_samples = get_samples() 
```


```python
# Consists of class ids 1-4 where 1-World, 2-Sports, 3-Business, 4-Sci/Tech
category_map = {1:"World", 2:"Sports", 3:"Business", 4:"Sci/Tech"}
```


```python
classifier = classifier.to('cpu')
```


```python
for truth, sample_group in test_samples.items():
    print(f"True Category: {category_map[truth]}")
    print('='*50)
    for sample in sample_group:
        prediction = predict_category(sample, classifier, testset.max_seq_length)
        print("예측: {} (p={:0.2f})".format(prediction['category'], prediction['probability']))
        print("샘플: {}".format(sample))
        print('-'*30)
    print()
```

    True Category: Business
    ==================================================
    예측: Business (p=0.82)
    샘플: Russia shrugs off US court freeze on oil giant Yukos auction
    ------------------------------
    예측: Business (p=1.00)
    샘플: Airbus chief wins fight to take controls at Eads
    ------------------------------
    예측: Sci/Tech (p=0.55)
    샘플: EBay #39;s Buy Of Rent.com May Lack Strategic Sense
    ------------------------------
    예측: Business (p=1.00)
    샘플: 5 of arthritis patients in Singapore take Bextra or Celebrex &lt;b&gt;...&lt;/b&gt;
    ------------------------------
    예측: Sci/Tech (p=0.58)
    샘플: EBay gets into rentals
    ------------------------------
    
    True Category: Sci/Tech
    ==================================================
    예측: Sci/Tech (p=0.76)
    샘플: Microsoft buy comes with strings attached
    ------------------------------
    예측: Sci/Tech (p=0.94)
    샘플: U.S. Army aims to halt paperwork with IBM system
    ------------------------------
    예측: Sci/Tech (p=0.93)
    샘플: Analysis: PeopleSoft users speak out about Oracle takeover (InfoWorld)
    ------------------------------
    예측: Sci/Tech (p=0.96)
    샘플: Hobbit-finding Boffins in science top 10
    ------------------------------
    예측: Sci/Tech (p=0.86)
    샘플: Search providers seek video, find challenges
    ------------------------------
    
    True Category: Sports
    ==================================================
    예측: World (p=0.71)
    샘플: The Newest Hope ; Marriage of Necessity Just Might Work Out
    ------------------------------
    예측: Business (p=0.79)
    샘플: Saban hiring on hold
    ------------------------------
    예측: Sports (p=0.72)
    샘플: Mortaza strikes to lead superb Bangladesh rally
    ------------------------------
    예측: Sports (p=0.53)
    샘플: Void is filled with Clement
    ------------------------------
    예측: Sports (p=0.99)
    샘플: Martinez leaves bitter
    ------------------------------
    
    True Category: World
    ==================================================
    예측: Business (p=0.50)
    샘플: Pricey Drug Trials Turn Up Few New Blockbusters
    ------------------------------
    예측: World (p=1.00)
    샘플: Bosnian-Serb prime minister resigns in protest against U.S. sanctions (Canadian Press)
    ------------------------------
    예측: Business (p=0.68)
    샘플: Historic Turkey-EU deal welcomed
    ------------------------------
    예측: World (p=1.00)
    샘플: Powell pushes diplomacy for N. Korea
    ------------------------------
    예측: World (p=0.41)
    샘플: Around the world
    ------------------------------
    
    


## Reference
- [Kaggle AG News Classification Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
