---
tag: [Deep Learning, 딥러닝, RNN, 자연어처리, 순환신경망, pytorch, 파이토치, AG News Dataset, LSTM]
toc: true
toc_sticky: true
toc_label: 목차
author_profile: false

---

# LSTM 모델을 사용하여 AG News Category 예측하기


```python
import os
import random
import string
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import pickle
import nltk
nltk.download('punkt')

import string
from collections import Counter
from copy import deepcopy
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

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: kaggle in /usr/local/lib/python3.10/dist-packages (1.5.13)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from kaggle) (2.27.1)
    Requirement already satisfied: urllib3 in /usr/local/lib/python3.10/dist-packages (from kaggle) (1.26.15)
    Requirement already satisfied: python-dateutil in /usr/local/lib/python3.10/dist-packages (from kaggle) (2.8.2)
    Requirement already satisfied: certifi in /usr/local/lib/python3.10/dist-packages (from kaggle) (2022.12.7)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from kaggle) (4.65.0)
    Requirement already satisfied: six>=1.10 in /usr/local/lib/python3.10/dist-packages (from kaggle) (1.16.0)
    Requirement already satisfied: python-slugify in /usr/local/lib/python3.10/dist-packages (from kaggle) (8.0.1)
    Requirement already satisfied: text-unidecode>=1.3 in /usr/local/lib/python3.10/dist-packages (from python-slugify->kaggle) (1.3)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.10/dist-packages (from requests->kaggle) (2.0.12)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->kaggle) (3.4)
    



     <input type="file" id="files-89fd1f1e-2e66-41a1-ae2c-1ea13c2f5f44" name="files[]" multiple disabled
        style="border:none" />
     <output id="result-89fd1f1e-2e66-41a1-ae2c-1ea13c2f5f44">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script>// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Helpers for google.colab Python module.
 */
(function(scope) {
function span(text, styleAttributes = {}) {
  const element = document.createElement('span');
  element.textContent = text;
  for (const key of Object.keys(styleAttributes)) {
    element.style[key] = styleAttributes[key];
  }
  return element;
}

// Max number of bytes which will be uploaded at a time.
const MAX_PAYLOAD_SIZE = 100 * 1024;

function _uploadFiles(inputId, outputId) {
  const steps = uploadFilesStep(inputId, outputId);
  const outputElement = document.getElementById(outputId);
  // Cache steps on the outputElement to make it available for the next call
  // to uploadFilesContinue from Python.
  outputElement.steps = steps;

  return _uploadFilesContinue(outputId);
}

// This is roughly an async generator (not supported in the browser yet),
// where there are multiple asynchronous steps and the Python side is going
// to poll for completion of each step.
// This uses a Promise to block the python side on completion of each step,
// then passes the result of the previous step as the input to the next step.
function _uploadFilesContinue(outputId) {
  const outputElement = document.getElementById(outputId);
  const steps = outputElement.steps;

  const next = steps.next(outputElement.lastPromiseValue);
  return Promise.resolve(next.value.promise).then((value) => {
    // Cache the last promise value to make it available to the next
    // step of the generator.
    outputElement.lastPromiseValue = value;
    return next.value.response;
  });
}

/**
 * Generator function which is called between each async step of the upload
 * process.
 * @param {string} inputId Element ID of the input file picker element.
 * @param {string} outputId Element ID of the output display.
 * @return {!Iterable<!Object>} Iterable of next steps.
 */
function* uploadFilesStep(inputId, outputId) {
  const inputElement = document.getElementById(inputId);
  inputElement.disabled = false;

  const outputElement = document.getElementById(outputId);
  outputElement.innerHTML = '';

  const pickedPromise = new Promise((resolve) => {
    inputElement.addEventListener('change', (e) => {
      resolve(e.target.files);
    });
  });

  const cancel = document.createElement('button');
  inputElement.parentElement.appendChild(cancel);
  cancel.textContent = 'Cancel upload';
  const cancelPromise = new Promise((resolve) => {
    cancel.onclick = () => {
      resolve(null);
    };
  });

  // Wait for the user to pick the files.
  const files = yield {
    promise: Promise.race([pickedPromise, cancelPromise]),
    response: {
      action: 'starting',
    }
  };

  cancel.remove();

  // Disable the input element since further picks are not allowed.
  inputElement.disabled = true;

  if (!files) {
    return {
      response: {
        action: 'complete',
      }
    };
  }

  for (const file of files) {
    const li = document.createElement('li');
    li.append(span(file.name, {fontWeight: 'bold'}));
    li.append(span(
        `(${file.type || 'n/a'}) - ${file.size} bytes, ` +
        `last modified: ${
            file.lastModifiedDate ? file.lastModifiedDate.toLocaleDateString() :
                                    'n/a'} - `));
    const percent = span('0% done');
    li.appendChild(percent);

    outputElement.appendChild(li);

    const fileDataPromise = new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        resolve(e.target.result);
      };
      reader.readAsArrayBuffer(file);
    });
    // Wait for the data to be ready.
    let fileData = yield {
      promise: fileDataPromise,
      response: {
        action: 'continue',
      }
    };

    // Use a chunked sending to avoid message size limits. See b/62115660.
    let position = 0;
    do {
      const length = Math.min(fileData.byteLength - position, MAX_PAYLOAD_SIZE);
      const chunk = new Uint8Array(fileData, position, length);
      position += length;

      const base64 = btoa(String.fromCharCode.apply(null, chunk));
      yield {
        response: {
          action: 'append',
          file: file.name,
          data: base64,
        },
      };

      let percentDone = fileData.byteLength === 0 ?
          100 :
          Math.round((position / fileData.byteLength) * 100);
      percent.textContent = `${percentDone}% done`;

    } while (position < fileData.byteLength);
  }

  // All done.
  yield {
    response: {
      action: 'complete',
    }
  };
}

scope.google = scope.google || {};
scope.google.colab = scope.google.colab || {};
scope.google.colab._files = {
  _uploadFiles,
  _uploadFilesContinue,
};
})(self);
</script> 


    Saving kaggle.json to kaggle.json
    Downloading ag-news-classification-dataset.zip to /content
     88% 10.0M/11.4M [00:01<00:00, 12.4MB/s]
    100% 11.4M/11.4M [00:01<00:00, 7.71MB/s]
    

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

        # dictionary 초기화화
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
        self.add_description()

    def add_word(self, word):
        if not word in self.word2idx:
            idx = self.idx
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
        return idx

    def add_description(self):
        counter = Counter()
        for description in self.news_df.Description:
            tokens = nltk.tokenize.word_tokenize(description.lower())
            counter.update(tokens)

        for word, cnt in counter.items():
            if cnt >= self.vocab_threshold:
                self.add_word(word)

        print("description_vocab 길이 :", len(self.word2idx))

    def __call__(self, word): # lookup word
        return self.word2idx.get(word, self.unk_index)

    def __len__(self):
        return len(self.word2idx)
```


```python
#data_dir = '/kaggle/input/ag-news-classification-dataset/'
data_dir = './dataset/'
```


```python
train_news_csv= data_dir + "train.csv"
test_news_csv= data_dir + "test.csv"
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

    description_vocab 길이 : 10320
    


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
vectorize("I am a boy", -1)
```




    array([   1,  464, 7647,   21, 4123,    2])




```python
class NewsDataset(Dataset):
  def __init__(self, mode, batch_size, vocab_threshold, vocab_file,
               mask_word, start_word, end_word, unk_word,
               news_df, vocab_from_file):
      self.news_df = news_df
      self.batch_size = batch_size
      self.description_vocab = Vocabulary(vocab_threshold, vocab_file,
                                    mask_word, start_word, end_word, unk_word,
                                    train_news_df, vocab_from_file)
      # (1) 문자열을 max_length 로 고정해서 벡터화할 때
      # measure_len = lambda context : len(nltk.tokenize.word_tokenize(context.lower()))
      # self.max_seq_length = max(map(measure_len, train_news_df.Title)) + 2

      # (2) 문자열을 가변적으로 벡터화할 때
      self.description_lengths = [len(nltk.tokenize.word_tokenize(description.lower()))
                                         for description in self.news_df.Description]

      self.description_vectors = [vectorize(self.news_df.iloc[index].Description, -1)
                                   for index in range(len(self.news_df)) ]

  def __getitem__(self, index):
      row = self.news_df.iloc[index]
      # description_vector = vectorize(row.Description, -1)
      description_vector = self.description_vectors[index]
      category_index = row['Class Index'] - 1
      return {'x_data' : description_vector,
              'y_target' : category_index }

  def __len__(self):
      return len(self.news_df)

  def get_train_indices(self):
      # 전체 데이터에서 description의 길이 중 하나를 선택해서
      # 그 길이와 같은 description들의 indices를 반환
      sel_length = np.random.choice(self.description_lengths)
      condition = [self.description_lengths[i] == sel_length for i in np.arange(len(self.description_lengths))]
      all_indices = np.where(condition)[0]
      indices = list(np.random.choice(all_indices, size=self.batch_size))
      return indices

```


```python
train_batch_size = 32
valid_batch_size = 32
test_batch_size = 32
```


```python
vocab_threshold = 25
mask_word = "<mask>"
start_word = "<start>"
end_word = "<end>"
unk_word = "<unk>"
vocab_file = './vocab.pkl'
vocab_from_file = False

trainset = NewsDataset("train", train_batch_size, vocab_threshold, vocab_file,
                       mask_word, start_word, end_word, unk_word,
                       train_df, vocab_from_file)
```

    description_vocab 길이 : 10320
    


```python
vocab_from_file = True
validset = NewsDataset("valid", valid_batch_size, vocab_threshold, vocab_file,
                       mask_word, start_word, end_word, unk_word,
                       valid_df, vocab_from_file)
testset = NewsDataset("test", test_batch_size, vocab_threshold, vocab_file,
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
indices = trainset.get_train_indices() # description의 길이가 같은 indices (batch_size)
initial_sampler = data.sampler.SubsetRandomSampler(indices = indices) # random하게 뒤섞음
batch_sampler = data.sampler.BatchSampler(sampler=initial_sampler, batch_size=train_batch_size, drop_last=False)
trainloader = DataLoader(dataset=trainset, batch_sampler=batch_sampler, num_workers=2)
```


```python
# !lscpu
```


```python
indices = validset.get_train_indices() # description의 길이가 같은 indices (batch_size)
initial_sampler = data.sampler.SubsetRandomSampler(indices = indices) # random하게 뒤섞음
batch_sampler = data.sampler.BatchSampler(sampler=initial_sampler, batch_size=valid_batch_size, drop_last=False)
validloader = DataLoader(dataset=validset, batch_sampler=batch_sampler, num_workers=2)
```


```python
indices = testset.get_train_indices() # description의 길이가 같은 indices (batch_size)
initial_sampler = data.sampler.SubsetRandomSampler(indices = indices) # random하게 뒤섞음
batch_sampler = data.sampler.BatchSampler(sampler=initial_sampler, batch_size=test_batch_size, drop_last=False)
testloader = DataLoader(dataset=testset, batch_sampler=batch_sampler, num_workers=2)
```


```python
batch = next(iter(trainloader))
batch['x_data'].size(), batch['y_target'].size()
```




    (torch.Size([32, 33]), torch.Size([32]))




```python
batch = next(iter(validloader))
batch['x_data'].size(), batch['y_target'].size()
```




    (torch.Size([32, 23]), torch.Size([32]))




```python
batch = next(iter(testloader))
batch['x_data'].size(), batch['y_target'].size()
```




    (torch.Size([32, 40]), torch.Size([32]))




```python
len(trainloader), len(validloader), len(testloader)
```




    (1, 1, 1)



## 5. 모델 생성: `NewsClassifier`

![image.png](data:image/png;base64,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)


```python
class NewsClassifier(nn.Module):
    def __init__(self, embedding_size, vocab_size,
                 rnn_hidden_dim, hidden_dim, num_layers, num_classes, dropout):

        super().__init__()

        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.lstm = nn.LSTM(input_size = embedding_size, hidden_size = rnn_hidden_dim,
                           num_layers = num_layers,
                           dropout = dropout,
                           batch_first = True)
        self.classifier = nn.Sequential(
                            nn.Linear(in_features=rnn_hidden_dim, out_features = hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Linear(in_features= hidden_dim, out_features = num_classes)
                           )


    def forward(self, inputs, apply_softmax=False): # input : (batch_size, description_length)
        embeddings = self.emb(inputs) # embeddings : (batch_size, description_length, embedding_size)
        _, (hidden, cell) = self.lstm(embeddings) # outputs : (batch_size, description_length, rnn_hidden_dim)
                                               # hidden : (num_layers , batch_size, rnn_hidden_dim)
        hidden = hidden[1] # hidden : (batch_size, rnn_hidden_dim)
        outputs = self.classifier(hidden) # outputs : (batch_size, num_classes)

        if apply_softmax:
            outputs = F.softmax(outputs, dim=1)
        return outputs
```

**하이퍼 파라미터 설정**


```python
embedding_size=100
rnn_hidden_dim = 100
hidden_dim = 50
num_layers = 2
dropout = 0.2
learning_rate=0.001
num_epochs=12
```


```python
classifier = NewsClassifier(embedding_size=embedding_size,
                            vocab_size=len(vocab),
                            rnn_hidden_dim = rnn_hidden_dim,
                            hidden_dim = hidden_dim,
                            num_layers = num_layers,
                            num_classes=4, dropout=dropout)
```


```python
classifier = classifier.to(device)
classifier
```




    NewsClassifier(
      (emb): Embedding(10320, 100)
      (lstm): LSTM(100, 100, num_layers=2, batch_first=True, dropout=0.2)
      (classifier): Sequential(
        (0): Linear(in_features=100, out_features=50, bias=True)
        (1): ReLU()
        (2): Dropout(p=0.2, inplace=False)
        (3): Linear(in_features=50, out_features=4, bias=True)
      )
    )




```python
out = classifier(batch['x_data'].to(device))
out.shape
```




    torch.Size([32, 4])



## 6. 모델 컴파일 (손실함수, 옵티마이저 선택)


```python
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                           mode='min', factor=0.01,
                                           patience=1, verbose=True)
```

## 7. 모델 훈련


```python
train_step = len(trainset) // train_batch_size
valid_step = len(validset) // valid_batch_size
test_step = len(testset) // test_batch_size
train_step, valid_step, test_step
```




    (3000, 750, 237)




```python
def validate(model, validloader, loss_fn):
    model.eval()
    total = 0
    correct = 0
    valid_loss = []
    valid_epoch_loss=0
    valid_accuracy = 0

    with torch.no_grad():
        for step in range(1, valid_step+1):
            indices = validset.get_train_indices()
            initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                    batch_size=valid_batch_size,
                                                    drop_last=False)
            validloader= data.DataLoader(dataset=validset, num_workers=0,
                                         batch_sampler=batch_sampler)
            # Obtain the batch.
            batch_dict = next(iter(validloader))
            inputs = batch_dict['x_data'].to(device)
            labels = batch_dict['y_target'].to(device)

            # 전방향 예측과 손실
            logits = model(inputs)
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

        for step in range(1, train_step+1):
            indices = trainset.get_train_indices()
            initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                    batch_size=train_batch_size,
                                                    drop_last=False)
            trainloader= data.DataLoader(dataset=trainset, num_workers=2,
                                         batch_sampler = batch_sampler)

            # Obtain the batch.
            batch_dict = next(iter(trainloader))
            inputs = batch_dict['x_data'].to(device)
            labels = batch_dict['y_target'].to(device)


            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, labels)
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
          best_model_state = deepcopy(model.state_dict())
          torch.save(best_model_state, 'best_checkpoint.pth')
        # -------------------------------------------

        # Learning Rate Scheduler
        scheduler.step(valid_epoch_loss)
        # -------------------------------------------
```


```python
total_loss = {"train": [], "val": []}
%time train_loop(classifier, trainloader, loss_fn, num_epochs, optimizer)
```

    Epoch: 1/12, Train Loss=0.5833, Val Loss=0.4604, Val Accyracy=0.8458
    Epoch: 2/12, Train Loss=0.3168, Val Loss=0.3307, Val Accyracy=0.8917
    Epoch: 3/12, Train Loss=0.2435, Val Loss=0.3096, Val Accyracy=0.8998
    Epoch: 4/12, Train Loss=0.1943, Val Loss=0.3056, Val Accyracy=0.9028
    Epoch: 5/12, Train Loss=0.1647, Val Loss=0.3115, Val Accyracy=0.8996
    trigger :  1
    Epoch: 6/12, Train Loss=0.1352, Val Loss=0.3383, Val Accyracy=0.9022
    trigger :  2
    Epoch 00006: reducing learning rate of group 0 to 1.0000e-05.
    Epoch: 7/12, Train Loss=0.1224, Val Loss=0.3317, Val Accyracy=0.9032
    trigger :  3
    Epoch: 8/12, Train Loss=0.1160, Val Loss=0.3162, Val Accyracy=0.9085
    trigger :  4
    Early Stopping !!!
    Training loop is finished !!
    CPU times: user 26min 30s, sys: 26min 55s, total: 53min 25s
    Wall time: 1h 37min 8s
    


```python
import matplotlib.pyplot as plt

plt.plot(total_loss['train'], label="train_loss")
plt.plot(total_loss['val'], label="vallid_loss")
plt.legend()
plt.show()
```


    
![png](/assets/images/2023-04-24-RNN 12 (Predict News Category using LSTM in Pytorch)/output_51_0.png)
    


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
        for step in range(1, test_step+1):
            indices = testset.get_train_indices()
            initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                    batch_size=test_batch_size,
                                                    drop_last=False)
            testloader= data.DataLoader(dataset=testset, num_workers=2,
                                        batch_sampler=batch_sampler)

            # Obtain the batch.
            batch_dict = next(iter(testloader))
            inputs = batch_dict['x_data'].to(device)
            labels = batch_dict['y_target'].to(device)

            # 전방향 예측과 손실
            logits = model(inputs)
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

    Test Loss : 0.32579 Test Accuracy : 0.90730
    


```python
# valid loss or accuracy 기준 best model
best_state_dict = torch.load('best_checkpoint.pth')
best_classifier = classifier
best_classifier.to(device)
best_classifier.load_state_dict(best_state_dict)

evaluate(best_classifier, testloader, loss_fn)
```

    Test Loss : 0.33211 Test Accuracy : 0.88898
    

## 9. 모델 예측


```python
def predict_category(text, classifier, max_length):
    # 뉴스 제목을 기반으로 카테고리를 예측

    # 1. vetororize
    vectorized_text = vectorize(text, vector_length=max_length)
    vectorized_text = torch.tensor(vectorized_text).unsqueeze(0) # tensor로 바꾸고, 배치처리를 위해 차원 늘림

    # 2. model의 예측
    result = classifier(vectorized_text, apply_softmax=True) # result : 예측 확률
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
        samples[category]= testset.news_df.Description[testset.news_df['Class Index'] == category].tolist()[-5:]

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
        prediction = predict_category(sample, classifier, max_length = -1)
        print("예측: {} (p={:0.2f})".format(prediction['category'], prediction['probability']))
        print("샘플: {}".format(sample))
        print('-'*30)
    print()
```

    True Category: Business
    ==================================================
    예측: Business (p=0.77)
    샘플: MOSCOW (AFP) - Russia forged ahead with the weekend auction of the core asset of crippled oil giant Yukos despite a disputed US court order barring the sale, with state-controlled gas giant Gazprom entering the bidding.
    ------------------------------
    예측: Business (p=1.00)
    샘플: The head of plane maker Airbus yesterday won a bitter battle to oust his boss from the helm of parent aerospace group Eads after winning the support of a key shareholder.
    ------------------------------
    예측: Business (p=0.88)
    샘플: Standard  amp; Poor #39;s Equity Research said the purchase of Rent.com by eBay (nasdaq: EBAY - news - people ) could be a bit of a miscalculation.
    ------------------------------
    예측: Business (p=0.92)
    샘플: SINGAPORE : Doctors in the United States have warned that painkillers Bextra and Celebrex may be linked to major cardiovascular problems and should not be prescribed.
    ------------------------------
    예측: Sci/Tech (p=0.60)
    샘플: EBay plans to buy the apartment and home rental service Rent.com for \$415 million, adding to its already exhaustive breadth of offerings.
    ------------------------------
    
    True Category: Sci/Tech
    ==================================================
    예측: Sci/Tech (p=0.71)
    샘플: A software company that Microsoft acquired this week to help beef up computer security may come with a bug of its own--a company claiming ownership of the programs.
    ------------------------------
    예측: Sci/Tech (p=0.70)
    샘플: The U.S. Army has struck a deal with IBM and other companies to create an automated record-keeping system that ends the need for electronic forms to be printed out, signed and delivered up the military service's chain of command.
    ------------------------------
    예측: Sci/Tech (p=0.97)
    샘플: InfoWorld - The great debate over the impact of Oracle's hostile takeover of PeopleSoft has all the big industry analyst organizations weighing in. However, in most of the analysis one group's opinion seems to have been overlooked: that of PeopleSoft users.
    ------------------------------
    예측: Sci/Tech (p=0.94)
    샘플: AP - Australian scientists who helped discover a species of tiny humans nicknamed Hobbits have been hailed for making the second most important scientific achievement of 2004.
    ------------------------------
    예측: Sci/Tech (p=0.98)
    샘플: Internet search providers are reacting to users #39; rising interest in finding video content on the Web, while acknowledging that there are steep challenges that need to be overcome.
    ------------------------------
    
    True Category: Sports
    ==================================================
    예측: Sports (p=0.63)
    샘플: NEW YORK - The TV lights were on, the cameras rolled and the symphony of cameras flashing in his face blinded Pedro Martinez - but not for long.
    ------------------------------
    예측: Sports (p=0.99)
    샘플: DAVIE - The Dolphins want Nick Saban, and the LSU coach could be on his way. Although LSU Athletic Director Skip Bertman said Friday that  quot;an offer is very imminent, quot; the Dolphins are committed to adhering 
    ------------------------------
    예측: Sports (p=1.00)
    샘플: Paceman Mashrafe Mortaza claimed two prize scalps, including Sachin Tendulkar with the day #39;s first ball, to lead a Bangladesh fightback in the second and final test against India on Saturday.
    ------------------------------
    예측: Sports (p=0.98)
    샘플: With the supply of attractive pitching options dwindling daily -- they lost Pedro Martinez to the Mets, missed on Tim Hudson, and are resigned to Randy Johnson becoming a Yankee -- the Red Sox struck again last night, coming to terms with free agent Matt Clement on a three-year deal that will pay the righthander in the neighborhood of \$25 ...
    ------------------------------
    예측: Sports (p=0.99)
    샘플: Like Roger Clemens did almost exactly eight years earlier, Pedro Martinez has left the Red Sox apparently bitter about the way he was treated by management.
    ------------------------------
    
    True Category: World
    ==================================================
    예측: Business (p=0.97)
    샘플: The \$500 billion drug industry is stumbling badly in its core business of finding new medicines, while aggressively marketing existing drugs.
    ------------------------------
    예측: World (p=1.00)
    샘플: Canadian Press - BANJA LUKA, Bosnia-Herzegovina (AP) - The prime minister of the Serbian half of Bosnia resigned Friday, a day after the U.S. government and Bosnia's top international administrator sanctioned Bosnian Serbs for failing to arrest and hand over war crimes suspects to the UN tribunal.
    ------------------------------
    예측: World (p=0.98)
    샘플: The European Union's decision to hold entry talks with Turkey receives a widespread welcome.
    ------------------------------
    예측: World (p=1.00)
    샘플: WASHINGTON -- Outgoing Secretary of State Colin L. Powell said yesterday he doesn't regret being the public face for the Bush administration's international call to war in Iraq. He also believes diplomacy is making headway in containing nuclear threats in Iran and North Korea, he said in an interview.
    ------------------------------
    예측: World (p=0.83)
    샘플: Ukrainian presidential candidate Viktor Yushchenko was poisoned with the most harmful known dioxin, which is contained in Agent Orange, a scientist who analyzed his blood said Friday.
    ------------------------------
    
    


```python

```

## 참고문법

**Counter**


```python
from collections import Counter
```


```python
# 사용 예 (1)
s = 'life is short, so python is easy.'

counter = Counter(s)
counter
```


```python
# 사용 예 (2)
s = 'life is short, so python is easy.'

counter = Counter()
tokens = s.split()
for token in tokens:
    counter[token] += 1
counter
```




    Counter({'life': 1, 'is': 2, 'short,': 1, 'so': 1, 'python': 1, 'easy.': 1})




```python
# 사용 예 (3)
s = 'life is short, so python is easy.'

counter = Counter()
tokens = s.split()
counter.update(tokens)
counter
```




    Counter({'life': 1, 'is': 2, 'short,': 1, 'so': 1, 'python': 1, 'easy.': 1})




```python
# 사용 예 (4)
s = 'life is short, so python is easy.'

counter = Counter()
tokens = nltk.tokenize.word_tokenize(s)
counter.update(tokens)
counter
```




    Counter({'life': 1,
             'is': 2,
             'short': 1,
             ',': 1,
             'so': 1,
             'python': 1,
             'easy': 1,
             '.': 1})




```python
# 아래 문자열에 대해 소문자로 변환전 tokenize 결과후 변환후 결과가 다름
s = "AP - Environmentalists asked the U.S. Fish and Wildlife Service on Wednesday to grant protected status to the California spotted owl, claiming the bird's old-growth forest habitat is threatened by logging."
counter = Counter()
tokens = nltk.tokenize.word_tokenize(s)
counter.update(tokens)
counter
```


```python
s = "AP - Environmentalists asked the U.S. Fish and Wildlife Service on Wednesday to grant protected status to the California spotted owl, claiming the bird's old-growth forest habitat is threatened by logging."
counter = Counter()
tokens = nltk.tokenize.word_tokenize(s.lower())
counter.update(tokens)
counter
```

**np.where**


```python
a = np.arange(10)
cond = a < 5
cond
```




    array([ True,  True,  True,  True,  True, False, False, False, False,
           False])




```python
np.where(cond, a, a*10)
```




    array([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])




```python
np.where(cond) # condition만 적으면 아래 np.asarray(cond).nonzero()와 동일한 결과과
```




    (array([0, 1, 2, 3, 4]),)




```python
np.asarray(cond).nonzero()
```




    (array([0, 1, 2, 3, 4]),)




```python
description_lengths = [37, 38, 45, 2, 3, 37, 37, 45, 45, 50]
sel_length = 37
cond = [description_lengths[i] == sel_length for i in np.arange(len(description_lengths))]
indices = np.where(cond)
indices
```




    (array([0, 5, 6]),)



**BatchSampler**


```python
indices = range(10)
initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
batch_sampler = data.sampler.BatchSampler(sampler=initial_sampler, batch_size=3, drop_last=False)
list(batch_sampler)
```




    [[9, 2, 0], [6, 7, 5], [3, 8, 4], [1]]




```python
indices = range(32) # 같은 길이인 description들의 indices
initial_sampler = data.sampler.SubsetRandomSampler(indices=indices) # random하게 뒤섞음
batch_sampler = data.sampler.BatchSampler(sampler=initial_sampler, batch_size=32, drop_last=True) # initial sampler에서 샘플링된 데이터를 배치 단위로 만들어줌
list(batch_sampler)
```


```python

```
