---
tag: [Deep Learning, 딥러닝, RNN, 자연어처리, 순환신경망, pytorch, 파이토치, 언어모델, Language Model, PTB Dataset]
toc: true
toc_sticky: true
toc_label: 목차
author_profile: false

---

# Word-level language model using RNN
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
class SimpleRnnlm(nn.Module):
    def __init__(self, vocab_size, wordvec_size, hidden_size, num_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.encoder = nn.Embedding(num_embeddings=vocab_size, embedding_dim = wordvec_size)
        self.rnn = nn.RNN(input_size=wordvec_size, hidden_size= hidden_size,
                          num_layers = num_layers, batch_first = True)
        self.decoder = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, inputs, hidden): # inputs :(batch_size, time_size)
        embedded = self.encoder(inputs) # embeddings : (batch_size, time_size, wordvec_size)
        outputs, hidden = self.rnn(embedded, hidden) # outputs : (batch_size, time_size, hidden_size)
                                                        # hidden : (num_layers , batch_size, hidden_size)
        decoded = self.decoder(outputs) # decoded : (batch_size, time_size, vocab_size)                                                  
        decoded = decoded.view(-1, self.vocab_size) # decoded : (batch_size*time_size, vocab_size)
        decoded = F.log_softmax(decoded, dim=1)

        return decoded, hidden
     

    def init_hidden(self, batch_size):
        # https://pytorch.org/docs/stable/generated/torch.Tensor.new_zeros.html
        # 아래 hidden 벡터를 생성할 때 weight와 동일한 device, dtype으로 만들기 위해
        # weight.new_zeros()를 사용
        weight = next(self.parameters())
        return weight.new_zeros(num_layers, batch_size, self.hidden_size)        
```


```python
vocab_size = 10000
wordvec_size = 100
hidden_size = 100 
num_layers = 1

model = SimpleRnnlm(vocab_size=vocab_size, wordvec_size=wordvec_size, 
                    hidden_size=hidden_size, num_layers=num_layers)
model.to(device)
```




    SimpleRnnlm(
      (encoder): Embedding(10000, 100)
      (rnn): RNN(100, 100, batch_first=True)
      (decoder): Linear(in_features=100, out_features=10000, bias=True)
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
    
    hidden = model.init_hidden(batch_size)
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
        
        hidden = repackage_hidden(hidden)
        outputs, hidden = model(xs, hidden)
        loss = loss_fn(outputs, ts.view(-1))
               
        total_loss += loss.item()
    
    valid_epoch_loss = total_loss / max_iters
    ppl = np.exp(valid_epoch_loss)
    return ppl
```


```python
batch_size = 10
time_size = 5 
xs = corpus[:-1]  # 입력
ts = corpus[1:]   # 출력(정답 레이블)
data_size = len(xs)
max_iters = data_size // (batch_size * time_size)
time_idx = 0

def train_loop(model, loss_fn, epochs, optimizer):      
    ppl_list = []
    hidden = model.init_hidden(batch_size)

    for epoch in range(epochs):
        model.train()
        train_loss = []
        for iter in range(max_iters):            
            batch_x, batch_t = get_batch(xs, ts, batch_size, time_size)
            optimizer.zero_grad()
            hidden = repackage_hidden(hidden)
            outputs, hidden = model(batch_x, hidden)
            loss = loss_fn(outputs, batch_t.view(-1))
            loss.backward()            
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
epochs = 15
%time train_loop(model, loss_fn, epochs, optimizer)
```

    | 에폭 1 | 퍼플렉서티 280.96 | 검증 퍼플렉서티 233.11
    | 에폭 2 | 퍼플렉서티 170.23 | 검증 퍼플렉서티 209.86
    | 에폭 3 | 퍼플렉서티 141.56 | 검증 퍼플렉서티 201.83
    | 에폭 4 | 퍼플렉서티 126.95 | 검증 퍼플렉서티 200.18
    | 에폭 5 | 퍼플렉서티 116.64 | 검증 퍼플렉서티 200.70
    | 에폭 6 | 퍼플렉서티 108.73 | 검증 퍼플렉서티 200.99
    Epoch 00006: reducing learning rate of group 0 to 2.5000e-04.
    | 에폭 7 | 퍼플렉서티 92.79 | 검증 퍼플렉서티 181.59
    | 에폭 8 | 퍼플렉서티 88.51 | 검증 퍼플렉서티 180.36
    | 에폭 9 | 퍼플렉서티 86.45 | 검증 퍼플렉서티 179.95
    | 에폭 10 | 퍼플렉서티 84.89 | 검증 퍼플렉서티 180.22
    | 에폭 11 | 퍼플렉서티 83.62 | 검증 퍼플렉서티 180.52
    Epoch 00011: reducing learning rate of group 0 to 6.2500e-05.
    | 에폭 12 | 퍼플렉서티 81.16 | 검증 퍼플렉서티 175.54
    | 에폭 13 | 퍼플렉서티 80.32 | 검증 퍼플렉서티 175.28
    | 에폭 14 | 퍼플렉서티 80.00 | 검증 퍼플렉서티 175.17
    | 에폭 15 | 퍼플렉서티 79.71 | 검증 퍼플렉서티 175.10
    CPU times: user 10min 27s, sys: 11.2 s, total: 10min 38s
    Wall time: 10min 39s
    


```python
# test_ppl = eval_perplexity(model, corpus_test, batch_size=batch_size, time_size=time_size )
```

## 7. 문장 생성 실험


```python
class RnnlmGen(SimpleRnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100, hidden=None):
        word_ids = [start_id]
        x = start_id
        while len(word_ids) < sample_size:
            sample_x = torch.tensor(x).reshape(1, 1)
            hidden = repackage_hidden(hidden)
            log_p, hidden = model(sample_x, hidden)
            log_p = log_p.detach().numpy().flatten()
            p = np.exp(log_p)
            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids
```


```python
gen_model = RnnlmGen(vocab_size=vocab_size, wordvec_size=wordvec_size, hidden_size=hidden_size, num_layers=num_layers)
gen_model.to('cpu')
model.to('cpu')
```




    SimpleRnnlm(
      (encoder): Embedding(10000, 100)
      (rnn): RNN(100, 100, batch_first=True)
      (decoder): Linear(in_features=100, out_features=10000, bias=True)
    )




```python
sampe_batch = 1
hidden = model.init_hidden(sampe_batch)
# start 문자와 skip 문자 설정
start_word = 'you'
start_id = word_to_id[start_word]
skip_words = ['N', '<unk>', '$']
skip_ids = [word_to_id[w] for w in skip_words]
# 문장 생성
word_ids = gen_model.generate(start_id, skip_ids, hidden=hidden)
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')
print(txt)
```

    you know you know the supreme court turn quickly been to cooperate with the quota disappearance without change a partner will be available to price may reflect the transaction with six-month lehman 's october it is half the issue of.
     among ibm position was closed at c$ late yesterday.
     both shearson 's parent of proposed late last selling them.
     but others raise funds for both companies rising mistakenly a healthy investment.
     southern california companies yet seen as soon as a strong mitchell to pegged finances about slipped to # a transaction plan to focus about a
    


```python
sample_batch = 1
hidden = model.init_hidden(sample_batch)
start_words = 'the meaning of life is'
start_ids = [word_to_id[w] for w in start_words.split(' ')]

for x in start_ids[:-1]:
    sample_x = torch.tensor(x).reshape(1, 1)
    hidden = repackage_hidden(hidden)
    model(sample_x, hidden)

word_ids = gen_model.generate(start_ids[-1], skip_ids, hidden=hidden)
word_ids = start_ids[:-1] + word_ids
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')
print('-' * 50)
print(txt)
```

    --------------------------------------------------
    the meaning of life is with taking late with automotive demand.
     for its service disposal is many more than it discovered that if you can make sense to a former affordable personally party.
     if we did in an annual appropriations committees of services has the equipment and a u.s. car to be sold or later.
     it 's a uncertainty averaged bonds series b double its common shares closed to yield after increased interest has been withheld nationwide of total existing national scuttle a older example of american express or control and state u.s. pharmaceutical tend to hold who had said the
    


```python

```
