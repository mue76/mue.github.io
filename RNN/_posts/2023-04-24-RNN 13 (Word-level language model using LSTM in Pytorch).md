---
tag: [Deep Learning, 딥러닝, RNN, 자연어처리, 순환신경망, pytorch, 파이토치, 언어모델, Language Model, PTB Dataset, LSTM]
toc: true
toc_sticky: true
toc_label: 목차
author_profile: false

---

# Word-level language model using LSTM
- 참고 : https://github.com/pytorch/examples/tree/main/word_language_model


```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
```


```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
```




    device(type='cuda')



## 1. 데이터 다운로드


```python
!mkdir ptb_dataset
!wget https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt -P ./ptb_dataset
!wget https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt -P ./ptb_dataset
!wget https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt -P ./ptb_dataset
```

    --2023-04-28 05:48:37--  https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 5101618 (4.9M) [text/plain]
    Saving to: ‘./ptb_dataset/ptb.train.txt’
    
    ptb.train.txt       100%[===================>]   4.87M  --.-KB/s    in 0.02s   
    
    2023-04-28 05:48:39 (236 MB/s) - ‘./ptb_dataset/ptb.train.txt’ saved [5101618/5101618]
    
    --2023-04-28 05:48:39--  https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 399782 (390K) [text/plain]
    Saving to: ‘./ptb_dataset/ptb.valid.txt’
    
    ptb.valid.txt       100%[===================>] 390.41K  --.-KB/s    in 0.003s  
    
    2023-04-28 05:48:39 (143 MB/s) - ‘./ptb_dataset/ptb.valid.txt’ saved [399782/399782]
    
    --2023-04-28 05:48:39--  https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 449945 (439K) [text/plain]
    Saving to: ‘./ptb_dataset/ptb.test.txt’
    
    ptb.test.txt        100%[===================>] 439.40K  --.-KB/s    in 0.004s  
    
    2023-04-28 05:48:40 (100 MB/s) - ‘./ptb_dataset/ptb.test.txt’ saved [449945/449945]
    
    

## 2. 데이터 불러오기


```python
dataset_dir = './ptb_dataset/'
train_file_name = 'ptb.train.txt'
valid_file_name = 'ptb.valid.txt'
test_file_name = 'ptb.test.txt'
```


```python
word_to_id = {}
id_to_word = {}
def load_vocab(data_type="train"):
    if data_type == 'train':
        file_path = dataset_dir + train_file_name
        words = open(file_path).read().replace('\n', '<eos>').strip().split()
        for i, word in enumerate(words):
            if word not in word_to_id:
                tmp_id = len(word_to_id)
                word_to_id[word] = tmp_id
                id_to_word[tmp_id] = word    
        corpus = np.array([word_to_id[w] for w in words])        
        print("vocab size : ", len(id_to_word))
        print("corpus size : ", len(corpus))
        return corpus, word_to_id, id_to_word
    elif data_type == 'valid':
        file_path = dataset_dir + valid_file_name
        words = open(file_path).read().replace('\n', '<eos>').strip().split()
        corpus = np.array([word_to_id[w] for w in words])        
        print("corpus size : ", len(corpus))
        return corpus, word_to_id, id_to_word
    else:
        file_path = dataset_dir + test_file_name
        words = open(file_path).read().replace('\n', '<eos>').strip().split()
        corpus = np.array([word_to_id[w] for w in words])        
        print("corpus size : ", len(corpus))
        return corpus, word_to_id, id_to_word
```


```python
corpus, word_to_id, id_to_word = load_vocab("train")
corpus_val, _, _ = load_vocab("valid")
corpus_test, _, _ = load_vocab("test")
```

    vocab size :  10000
    corpus size :  929589
    corpus size :  73760
    corpus size :  82430
    

## 3. 데이터 적재

## 4. 모델 생성


```python
class BetterRnnlm(nn.Module):
    def __init__(self, vocab_size, wordvec_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.encoder = nn.Embedding(num_embeddings=vocab_size, embedding_dim = wordvec_size)
        # self.encoder.weight shape : (vocab_size, wordvec_size)
        
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(input_size=wordvec_size, hidden_size= hidden_size,
                           num_layers = num_layers, 
                           dropout = dropout,
                           batch_first = True)
        self.decoder = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        # self.encoder.weight shape : (vocab_size, hidden_size)

        self.decoder.weight = self.encoder.weight # 가중치 공유

    def forward(self, inputs, hidden, cell): # inputs :(batch_size, time_size)
        embedded = self.encoder(inputs) # embeddings : (batch_size, time_size, wordvec_size)
        embedded = self.dropout(embedded)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell)) # outputs : (batch_size, time_size, hidden_size)
                                                        # hidden : (num_layers , batch_size, hidden_size)
        outputs = self.dropout(outputs)                                                        
        decoded = self.decoder(outputs) # decoded : (batch_size, time_size, vocab_size)                                                  
        decoded = decoded.view(-1, self.vocab_size) # decoded : (batch_size*time_size, vocab_size)
        decoded = F.log_softmax(decoded, dim=1)

        return decoded, (hidden, cell)
     

    def init_hidden(self, batch_size):
        # https://pytorch.org/docs/stable/generated/torch.Tensor.new_zeros.html
        # 아래 hidden 벡터를 생성할 때 weight와 동일한 device, dtype으로 만들기 위해
        # weight.new_zeros()를 사용
        weight = next(self.parameters())
        return (weight.new_zeros(num_layers, batch_size, self.hidden_size), # hidden
                weight.new_zeros(num_layers, batch_size, self.hidden_size)) # cell
```


```python
vocab_size = 10000
wordvec_size = 650
hidden_size = 650 
num_layers = 2
dropout = 0.5

model = BetterRnnlm(vocab_size=vocab_size, wordvec_size=wordvec_size, 
                    hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
model.to(device)
```




    BetterRnnlm(
      (encoder): Embedding(10000, 650)
      (dropout): Dropout(p=0.5, inplace=False)
      (lstm): LSTM(650, 650, num_layers=2, batch_first=True, dropout=0.5)
      (decoder): Linear(in_features=650, out_features=10000, bias=True)
    )



## 5. 모델 설정 (손실함수, 옵티마이저 선택)


```python
lr = 0.001
loss_fn = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 mode='min', factor=0.25, 
                                                 patience=1, verbose=True)
```

## 6. 모델 훈련


```python
def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
```


```python
def get_batch(xs, ts, batch_size, time_size):
    global time_idx   

    data_size = len(xs)
    jump = data_size // batch_size
    offsets = [i * jump for i in range(batch_size)] 

    batch_x = np.empty((batch_size, time_size), dtype=np.int64)
    batch_t = np.empty((batch_size, time_size), dtype=np.int64)
    for t in range(time_size):
        for i, offset in enumerate(offsets):
            batch_x[i, t] = xs[(offset + time_idx) % data_size]
            batch_t[i, t] = ts[(offset + time_idx) % data_size]
        time_idx += 1
    batch_x = torch.from_numpy(batch_x).to(device)
    batch_t = torch.from_numpy(batch_t).to(device)
    return batch_x, batch_t
```


```python
def eval_perplexity(model, corpus, batch_size=10, time_size=35):
    corpus_size = len(corpus)
    total_loss, loss_cnt = 0, 0
    max_iters = (corpus_size - 1) // (batch_size * time_size)
    jump = (corpus_size - 1) // batch_size
    
    hidden, cell = model.init_hidden(batch_size)
    model.eval()
    for iters in range(max_iters):
        xs = np.zeros((batch_size, time_size), dtype=np.int64)
        ts = np.zeros((batch_size, time_size), dtype=np.int64)
        time_offset = iters * time_size
        offsets = [time_offset + (i * jump) for i in range(batch_size)]
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                xs[i, t] = corpus[(offset + t) % corpus_size]
                ts[i, t] = corpus[(offset + t + 1) % corpus_size]
        xs = torch.from_numpy(xs).to(device)
        ts = torch.from_numpy(ts).to(device)
        
        hidden, cell = repackage_hidden((hidden, cell))
        outputs, (hidden, cell) = model(xs, hidden, cell)
        loss = loss_fn(outputs, ts.view(-1))
               
        total_loss += loss.item()
    
    valid_epoch_loss = total_loss / max_iters
    ppl = np.exp(valid_epoch_loss)
    return ppl
```


```python
batch_size=20
time_size = 35
xs = corpus[:-1]  # 입력
ts = corpus[1:]   # 출력(정답 레이블)
data_size = len(xs)
max_iters = data_size // (batch_size * time_size)
time_idx = 0
max_grad = 0.25

def train_loop(model, loss_fn, epochs, optimizer):      
    ppl_list = []
    hidden, cell = model.init_hidden(batch_size)

    for epoch in range(epochs):
        model.train()
        train_loss = []
        for iter in range(max_iters):            
            batch_x, batch_t = get_batch(xs, ts, batch_size, time_size)
            optimizer.zero_grad()
            hidden, cell = repackage_hidden((hidden, cell))
            outputs, (hidden, cell) = model(batch_x, hidden, cell)
            loss = loss_fn(outputs, batch_t.view(-1))
            loss.backward()
            
            # clipping gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
            
            optimizer.step()

            train_loss.append(loss.item())

        train_epoch_loss = np.mean(train_loss)
        ppl = np.exp(train_epoch_loss)
        ppl_list.append(float(ppl))

        # 검증 퍼플렉서티
        eval_ppl = eval_perplexity(model, corpus_val, batch_size=batch_size, time_size=time_size )

        print('| 에폭 %d | 퍼플렉서티 %.2f | 검증 퍼플렉서티 %.2f'
                % (epoch+1, ppl, eval_ppl))
        
        scheduler.step(eval_ppl)
```


```python
epochs = 40
%time train_loop(model, loss_fn, epochs, optimizer)
```

    | 에폭 1 | 퍼플렉서티 522.13 | 검증 퍼플렉서티 264.31
    | 에폭 2 | 퍼플렉서티 1762.87 | 검증 퍼플렉서티 216.24
    | 에폭 3 | 퍼플렉서티 57524.32 | 검증 퍼플렉서티 197.17
    | 에폭 4 | 퍼플렉서티 6861.26 | 검증 퍼플렉서티 183.74
    | 에폭 5 | 퍼플렉서티 4273.91 | 검증 퍼플렉서티 173.72
    | 에폭 6 | 퍼플렉서티 2279.65 | 검증 퍼플렉서티 165.38
    | 에폭 7 | 퍼플렉서티 1115.97 | 검증 퍼플렉서티 157.66
    | 에폭 8 | 퍼플렉서티 414.15 | 검증 퍼플렉서티 151.46
    | 에폭 9 | 퍼플렉서티 230.04 | 검증 퍼플렉서티 144.73
    | 에폭 10 | 퍼플렉서티 196.24 | 검증 퍼플렉서티 138.86
    | 에폭 11 | 퍼플렉서티 181.57 | 검증 퍼플렉서티 133.85
    | 에폭 12 | 퍼플렉서티 173.09 | 검증 퍼플렉서티 130.00
    | 에폭 13 | 퍼플렉서티 165.42 | 검증 퍼플렉서티 125.68
    | 에폭 14 | 퍼플렉서티 158.72 | 검증 퍼플렉서티 122.71
    | 에폭 15 | 퍼플렉서티 151.74 | 검증 퍼플렉서티 119.76
    | 에폭 16 | 퍼플렉서티 146.39 | 검증 퍼플렉서티 116.87
    | 에폭 17 | 퍼플렉서티 141.62 | 검증 퍼플렉서티 114.71
    | 에폭 18 | 퍼플렉서티 136.33 | 검증 퍼플렉서티 112.62
    | 에폭 19 | 퍼플렉서티 132.77 | 검증 퍼플렉서티 110.47
    | 에폭 20 | 퍼플렉서티 129.66 | 검증 퍼플렉서티 108.53
    | 에폭 21 | 퍼플렉서티 125.09 | 검증 퍼플렉서티 106.92
    | 에폭 22 | 퍼플렉서티 123.70 | 검증 퍼플렉서티 105.30
    | 에폭 23 | 퍼플렉서티 119.64 | 검증 퍼플렉서티 104.30
    | 에폭 24 | 퍼플렉서티 115.77 | 검증 퍼플렉서티 102.63
    | 에폭 25 | 퍼플렉서티 114.75 | 검증 퍼플렉서티 101.52
    | 에폭 26 | 퍼플렉서티 111.39 | 검증 퍼플렉서티 100.28
    | 에폭 27 | 퍼플렉서티 109.50 | 검증 퍼플렉서티 99.70
    | 에폭 28 | 퍼플렉서티 107.02 | 검증 퍼플렉서티 98.48
    | 에폭 29 | 퍼플렉서티 104.64 | 검증 퍼플렉서티 97.11
    | 에폭 30 | 퍼플렉서티 103.20 | 검증 퍼플렉서티 96.64
    | 에폭 31 | 퍼플렉서티 100.48 | 검증 퍼플렉서티 95.95
    | 에폭 32 | 퍼플렉서티 98.91 | 검증 퍼플렉서티 95.21
    | 에폭 33 | 퍼플렉서티 97.72 | 검증 퍼플렉서티 94.91
    | 에폭 34 | 퍼플렉서티 96.35 | 검증 퍼플렉서티 94.16
    | 에폭 35 | 퍼플렉서티 94.25 | 검증 퍼플렉서티 93.54
    | 에폭 36 | 퍼플렉서티 92.86 | 검증 퍼플렉서티 92.90
    | 에폭 37 | 퍼플렉서티 91.08 | 검증 퍼플렉서티 92.46
    | 에폭 38 | 퍼플렉서티 89.19 | 검증 퍼플렉서티 91.95
    | 에폭 39 | 퍼플렉서티 88.18 | 검증 퍼플렉서티 91.47
    | 에폭 40 | 퍼플렉서티 87.25 | 검증 퍼플렉서티 91.30
    CPU times: user 29min 12s, sys: 7.26 s, total: 29min 19s
    Wall time: 29min 26s
    


```python
test_ppl = eval_perplexity(model, corpus_test, batch_size=batch_size, time_size=time_size )
test_ppl
```




    87.42089143300728



## 7. 문장 생성 실험


```python
class BetterRnnlmGen(BetterRnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100, hidden=None, cell=None):
        word_ids = [start_id]
        x = start_id
        while len(word_ids) < sample_size:
            sample_x = torch.tensor(x).reshape(1, 1)
            hidden, cell = repackage_hidden((hidden, cell))
            log_p, (hidden, cell) = model(sample_x, hidden, cell)
            log_p = log_p.detach().numpy().flatten()
            p = np.exp(log_p)
            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids
```


```python
gen_model = BetterRnnlmGen(vocab_size=vocab_size, wordvec_size=wordvec_size, hidden_size=hidden_size, 
                           num_layers=num_layers, dropout=dropout)
gen_model.to('cpu')
model.to('cpu')
```




    BetterRnnlm(
      (encoder): Embedding(10000, 650)
      (dropout): Dropout(p=0.5, inplace=False)
      (lstm): LSTM(650, 650, num_layers=2, batch_first=True, dropout=0.5)
      (decoder): Linear(in_features=650, out_features=10000, bias=True)
    )




```python
sampe_batch = 1
hidden, cell = model.init_hidden(sampe_batch)
# start 문자와 skip 문자 설정
start_word = 'you'
start_id = word_to_id[start_word]
skip_words = ['N', '<unk>', '$']
skip_ids = [word_to_id[w] for w in skip_words]
# 문장 생성
word_ids = gen_model.generate(start_id, skip_ids, hidden=hidden, cell=cell)
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')
print(txt)
```

    you 'd love themselves and said it does n't have a quick impact on deal with ibm.
     since the big three ratings companies argue that data 's social business really say mr. hahn told reporters that ford 's refinery complain that using its countries ' good technology to focus.
     tough profits have caused his recent tables in southeast asia.
     a plagued business in this case process or having always a ghost he said scott reitman a new york developer for firm owners richard dodd.
     mr. hunt commissioned his own family.
    .
     in the same of
    


```python
sample_batch = 1
hidden, cell = model.init_hidden(sample_batch)
start_words = 'the meaning of life is'
start_ids = [word_to_id[w] for w in start_words.split(' ')]

for x in start_ids[:-1]:
    sample_x = torch.tensor(x).reshape(1, 1)
    hidden = repackage_hidden(hidden)
    model(sample_x, hidden, cell)

word_ids = gen_model.generate(start_ids[-1], skip_ids, hidden=hidden, cell=cell)
word_ids = start_ids[:-1] + word_ids
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')
print('-' * 50)
print(txt)
```

    --------------------------------------------------
    the meaning of life is getting a much compound position.
     the securities and exchange commission gives a spending provision expires by the company that was he sold a company long service at about # million million checks and raise interest in losses.
     many soviet executives have been counting representing losses from a restructuring that will want expansion funds and fruit because of banks in part because of fujitsu 's disciplinary action to acquire a new york city.
     their ratio is less likely to change said airline katz research analyst at first boston corp.
     the transaction is still worth preferred shares
    


```python

```
