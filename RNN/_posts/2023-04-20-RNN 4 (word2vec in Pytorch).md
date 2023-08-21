---
tag: [Deep Learning, 딥러닝, RNN, 자연어처리, word2vec, pytorch, 파이토치, PTB Dataset]
toc: true
toc_sticky: true
toc_label: 목차
author_profile: false

---


# CBOW with PTB Dataset


```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
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
```

    --2023-04-24 03:38:56--  https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.111.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.111.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 5101618 (4.9M) [text/plain]
    Saving to: ‘./ptb_dataset/ptb.train.txt’
    
    ptb.train.txt       100%[===================>]   4.87M  --.-KB/s    in 0.08s   
    
    2023-04-24 03:38:56 (62.6 MB/s) - ‘./ptb_dataset/ptb.train.txt’ saved [5101618/5101618]
    
    

## 2. 데이터 불러오기


```python
dataset_dir = './ptb_dataset/'
train_file_name = 'ptb.train.txt'
```


```python
def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)
```


```python
class PTBDataset(Dataset):
    def __init__(self, file_path, window_size):
        self.file_path = file_path
        self.word_to_id, self.id_to_word, self.words = self.load_vocab()

        corpus = np.array([self.word_to_id[w] for w in self.words])
        print('corpus size :', len(corpus))

        self.contexts, self.target = create_contexts_target(corpus, window_size)
        print('context.shpape:', self.contexts.shape, 'target.shape:', self.target.shape)


    def load_vocab(self):
        words = open(file_path).read().replace('\n', '<eos>').strip().split()
        word_to_id = {}
        id_to_word = {}

        for i, word in enumerate(words):
            if word not in word_to_id:
                new_id = len(word_to_id)
                word_to_id[word] = new_id
                id_to_word[new_id] = word
        print('vocab size:', len(id_to_word))        
        return word_to_id, id_to_word, words

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return self.contexts[index], self.target[index]
```

**학습 후 유사도 측정을 위해 준비**


```python
dataset_dir = './ptb_dataset/'
train_file_name = 'ptb.train.txt'
file_path = dataset_dir + train_file_name
```


```python
words = open(file_path).read().replace('\n', '<eos>').strip().split()
words # 말뭉치에 있는 데이터를 단어 단위로 쪼개어 리스트로 보관
```


```python
word_to_id = {}
id_to_word = {}

for i, word in enumerate(words):
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word
print('corpus size :', len(words))        
print('vocab size : ', len(id_to_word))        
```

    corpus size : 929589
    vocab size :  10000
    


```python
def load_vocab():
    words = open(file_path).read().replace('\n', '<eos>').strip().split()
    word_to_id = {}
    id_to_word = {}

    for i, word in enumerate(words):
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    # print('corpus size :', len(words))        
    print('vocab size : ', len(id_to_word))       

    return word_to_id, id_to_word
```


```python
word_to_id, id_to_word = load_vocab()
```

    corpus size : 929589
    vocab size :  10000
    


```python
dataset_dir = './ptb_dataset/'
train_file_name = 'ptb.train.txt'
```


```python
window_size = 5
dataset = PTBDataset(file_path, window_size)
```

    vocab size: 10000
    corpus size : 929589
    context.shpape: (929579, 10) target.shape: (929579,)
    


```python
len(dataset)
```




    929579




```python
dataset[100] # contexts, target
```




    (array([76, 77, 64, 78, 79, 27, 28, 81, 82, 83]), 80)



## 3. 데이터 적재


```python
batch_size=100
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
```


```python
contexts, target = next(iter(dataloader))
contexts.size(), target.size()
```




    (torch.Size([100, 10]), torch.Size([100]))



## 4. 모델 생성


```python
vocab_size = 10000
hidden_size = 100
embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = hidden_size)
emb_out = embedding(contexts)
emb_out.shape
```




    torch.Size([100, 10, 100])




```python
h_mean = emb_out.mean(axis=1)
h_mean.shape 
```




    torch.Size([100, 100])




```python
class CBOW_Model(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(CBOW_Model, self).__init__()
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = hidden_size)
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)


    def forward(self, inputs): # input (batch_size, context_size)
        h = self.embedding(inputs) # h (batch_size, context_size, hidden_size)
        h_mean = h.mean(axis=1) # h_mean (batch_size, hidden_size)
        out = self.linear(h_mean) # out (batch_size, vocab_size)
        return out
```


```python
vocab_size = 10000
hidden_size = 100

model = CBOW_Model(vocab_size=vocab_size, hidden_size=hidden_size)
model.to(device)
```




    CBOW_Model(
      (embedding): Embedding(10000, 100)
      (linear): Linear(in_features=100, out_features=10000, bias=True)
    )




```python
contexts, target = contexts.to(device), target.to(device)
out = model(contexts)
out.shape, out.dtype
```




    (torch.Size([100, 10000]), torch.float32)



## 5. 모델 컴파일 (손실함수, 옵티마이저 선택)


```python
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

## 6. 모델 훈련


```python
def train_loop(model, trainloader, loss_fn, epochs, optimizer):  
    min_loss = 1000000  
    trigger = 0
    patience = 4     

    for epoch in range(epochs):
        model.train()
        train_loss = []

        for batch_data in trainloader:
            contexts = batch_data[0].to(device)
            target = batch_data[1].to(device)

            optimizer.zero_grad()
            outputs = model(contexts)
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        train_epoch_loss = np.mean(train_loss)
        total_loss["train"].append(train_epoch_loss)

        print( "Epoch: {}/{}, Train Loss={:.5f}".format(                    
                    epoch + 1, epochs,
                    total_loss["train"][-1]))  
```


```python
epochs = 15
total_loss = {"train": []}
%time train_loop(model, dataloader, loss_fn, epochs, optimizer)
```

    Epoch: 1/15, Train Loss=5.86334
    Epoch: 2/15, Train Loss=5.12299
    Epoch: 3/15, Train Loss=4.84547
    Epoch: 4/15, Train Loss=4.69440
    Epoch: 5/15, Train Loss=4.60174
    Epoch: 6/15, Train Loss=4.53866
    Epoch: 7/15, Train Loss=4.49440
    Epoch: 8/15, Train Loss=4.46302
    Epoch: 9/15, Train Loss=4.43796
    Epoch: 10/15, Train Loss=4.41853
    Epoch: 11/15, Train Loss=4.40023
    Epoch: 12/15, Train Loss=4.38645
    Epoch: 13/15, Train Loss=4.37518
    Epoch: 14/15, Train Loss=4.36635
    Epoch: 15/15, Train Loss=4.35788
    CPU times: user 5min 2s, sys: 8.68 s, total: 5min 11s
    Wall time: 5min 12s
    

## 7. 유사도 측정


```python
list(model.parameters())[0].shape
```




    torch.Size([10000, 100])




```python
# embedding from first model layer
embeddings = list(model.parameters())[0]
```




    torch.Size([10000, 100])




```python
# detach : https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
embeddings = embeddings.cpu().detach().numpy()
embeddings
```




    array([[-3.4201128 ,  1.0614903 , -3.2150905 , ...,  2.100835  ,
             1.3760728 ,  2.2108448 ],
           [ 0.46403676, -2.0405898 , -2.4745762 , ...,  0.32860556,
            -1.334212  , -0.22706386],
           [ 3.47059   , -0.6248425 ,  0.9439361 , ...,  3.689676  ,
            -0.8413572 , -0.39743933],
           ...,
           [ 1.8678907 ,  6.9253416 , -0.58181703, ...,  1.1381289 ,
            -3.7073238 , -0.4031666 ],
           [ 2.3469465 ,  3.015586  , -3.6032927 , ...,  1.1200864 ,
             2.8106098 , -4.200683  ],
           [ 0.34840223,  0.5831198 , -0.88251144, ..., -2.5415118 ,
            -3.3060513 ,  0.03533731]], dtype=float32)




```python
def cos_similarity(x, y, eps=1e-8):
    '''코사인 유사도 산출

    :param x: 벡터
    :param y: 벡터
    :param eps: '0으로 나누기'를 방지하기 위한 작은 값
    :return:
    '''
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)
```


```python
def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    # 1. 검색어를 꺼낸다
    if query not in word_to_id:
        print('%s(을)를 찾을 수 없습니다.' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 2. 코사인 유사도 계산
    vocab_size = len(id_to_word)

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # 3. 코사인 유사도를 기준으로 내림차순으로 출력
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return
```


```python
word_vecs = embeddings

# 가장 비슷한(most similar) 단어 뽑기
querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
```

    
    [query] you
     we: 0.7654329538345337
     they: 0.7229254841804504
     i: 0.6785027384757996
     she: 0.4186348617076874
     he: 0.36235079169273376
    
    [query] year
     week: 0.7872598171234131
     month: 0.7634924650192261
     summer: 0.5889633297920227
     decade: 0.5301210880279541
     spring: 0.5232934951782227
    
    [query] car
     cars: 0.499886691570282
     clothing: 0.4101100265979767
     glass: 0.3856881260871887
     family: 0.3844160735607147
     cigarette: 0.38434258103370667
    
    [query] toyota
     factory: 0.44822514057159424
     mitsubishi: 0.42368119955062866
     potatoes: 0.39004334807395935
     vehicles: 0.3864486813545227
     loan-loss: 0.38212400674819946
    

