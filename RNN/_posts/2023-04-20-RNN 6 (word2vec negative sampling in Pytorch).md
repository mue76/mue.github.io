---
tag: [Deep Learning, 딥러닝, RNN, 자연어처리, word2vec, negative sampling, PTB Dataset, pytorch, 파이토치]
toc: true
toc_sticky: true
toc_label: 목차
author_profile: false

---

# CBOW Negative Sampling with PTB Dataset


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
```

    --2023-04-25 01:52:20--  https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 5101618 (4.9M) [text/plain]
    Saving to: ‘./ptb_dataset/ptb.train.txt’
    
    ptb.train.txt       100%[===================>]   4.87M  --.-KB/s    in 0.03s   
    
    2023-04-25 01:52:21 (145 MB/s) - ‘./ptb_dataset/ptb.train.txt’ saved [5101618/5101618]
    
    

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
import collections
GPU = True
class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0
                p /= p.sum()
                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        else:
            # GPU(cupy）로 계산할 때는 속도를 우선한다.
            # 부정적 예에 타깃이 포함될 수 있다.
            negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
                                               replace=True, p=self.word_p)

        return negative_sample
```


```python
class PTBDataset(Dataset):
    def __init__(self, file_path, window_size, sample_size):
        self.file_path = file_path
        self.word_to_id, self.id_to_word, self.words = self.load_vocab()

        corpus = np.array([self.word_to_id[w] for w in self.words])
        print('corpus size :', len(corpus))

        self.contexts, self.target = create_contexts_target(corpus, window_size)
        print('context.shpape:', self.contexts.shape, 'target.shape:', self.target.shape)
        
        self.sampler = UnigramSampler(corpus, 0.75, sample_size)

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




    ['aer',
     'banknote',
     'berlitz',
     'calloway',
     'centrust',
     'cluett',
     'fromstein',
     'gitano',
     'guterman',
     'hydro-quebec',
     'ipo',
     'kia',
     'memotec',
     'mlx',
     'nahb',
     'punts',
     'rake',
     'regatta',
     'rubens',
     'sim',
     'snack-food',
     'ssangyong',
     'swapo',
     'wachter',
     '<eos>',
     'pierre',
     '<unk>',
     'N',
     'years',
     'old',
     'will',
     'join',
     'the',
     'board',
     'as',
     'a',
     'nonexecutive',
     'director',
     'nov.',
     'N',
     '<eos>',
     'mr.',
     '<unk>',
     'is',
     'chairman',
     'of',
     '<unk>',
     'n.v.',
     'the',
     'dutch',
     'publishing',
     'group',
     '<eos>',
     'rudolph',
     '<unk>',
     'N',
     'years',
     'old',
     'and',
     'former',
     'chairman',
     'of',
     'consolidated',
     'gold',
     'fields',
     'plc',
     'was',
     'named',
     'a',
     'nonexecutive',
     'director',
     'of',
     'this',
     'british',
     'industrial',
     'conglomerate',
     '<eos>',
     'a',
     'form',
     'of',
     'asbestos',
     'once',
     'used',
     'to',
     'make',
     'kent',
     'cigarette',
     'filters',
     'has',
     'caused',
     'a',
     'high',
     'percentage',
     'of',
     'cancer',
     'deaths',
     'among',
     'a',
     'group',
     'of',
     'workers',
     'exposed',
     'to',
     'it',
     'more',
     'than',
     'N',
     'years',
     'ago',
     'researchers',
     'reported',
     '<eos>',
     'the',
     'asbestos',
     'fiber',
     '<unk>',
     'is',
     'unusually',
     '<unk>',
     'once',
     'it',
     'enters',
     'the',
     '<unk>',
     'with',
     'even',
     'brief',
     'exposures',
     'to',
     'it',
     'causing',
     'symptoms',
     'that',
     'show',
     'up',
     'decades',
     'later',
     'researchers',
     'said',
     '<eos>',
     '<unk>',
     'inc.',
     'the',
     'unit',
     'of',
     'new',
     'york-based',
     '<unk>',
     'corp.',
     'that',
     'makes',
     'kent',
     'cigarettes',
     'stopped',
     'using',
     '<unk>',
     'in',
     'its',
     '<unk>',
     'cigarette',
     'filters',
     'in',
     'N',
     '<eos>',
     'although',
     'preliminary',
     'findings',
     'were',
     'reported',
     'more',
     'than',
     'a',
     'year',
     'ago',
     'the',
     'latest',
     'results',
     'appear',
     'in',
     'today',
     "'s",
     'new',
     'england',
     'journal',
     'of',
     'medicine',
     'a',
     'forum',
     'likely',
     'to',
     'bring',
     'new',
     'attention',
     'to',
     'the',
     'problem',
     '<eos>',
     'a',
     '<unk>',
     '<unk>',
     'said',
     'this',
     'is',
     'an',
     'old',
     'story',
     '<eos>',
     'we',
     "'re",
     'talking',
     'about',
     'years',
     'ago',
     'before',
     'anyone',
     'heard',
     'of',
     'asbestos',
     'having',
     'any',
     'questionable',
     'properties',
     '<eos>',
     'there',
     'is',
     'no',
     'asbestos',
     'in',
     'our',
     'products',
     'now',
     '<eos>',
     'neither',
     '<unk>',
     'nor',
     'the',
     'researchers',
     'who',
     'studied',
     'the',
     'workers',
     'were',
     'aware',
     'of',
     'any',
     'research',
     'on',
     'smokers',
     'of',
     'the',
     'kent',
     'cigarettes',
     '<eos>',
     'we',
     'have',
     'no',
     'useful',
     'information',
     'on',
     'whether',
     'users',
     'are',
     'at',
     'risk',
     'said',
     'james',
     'a.',
     '<unk>',
     'of',
     'boston',
     "'s",
     '<unk>',
     'cancer',
     'institute',
     '<eos>',
     'dr.',
     '<unk>',
     'led',
     'a',
     'team',
     'of',
     'researchers',
     'from',
     'the',
     'national',
     'cancer',
     'institute',
     'and',
     'the',
     'medical',
     'schools',
     'of',
     'harvard',
     'university',
     'and',
     'boston',
     'university',
     '<eos>',
     'the',
     '<unk>',
     'spokeswoman',
     'said',
     'asbestos',
     'was',
     'used',
     'in',
     'very',
     'modest',
     'amounts',
     'in',
     'making',
     'paper',
     'for',
     'the',
     'filters',
     'in',
     'the',
     'early',
     '1950s',
     'and',
     'replaced',
     'with',
     'a',
     'different',
     'type',
     'of',
     '<unk>',
     'in',
     'N',
     '<eos>',
     'from',
     'N',
     'to',
     'N',
     'N',
     'billion',
     'kent',
     'cigarettes',
     'with',
     'the',
     'filters',
     'were',
     'sold',
     'the',
     'company',
     'said',
     '<eos>',
     'among',
     'N',
     'men',
     'who',
     'worked',
     'closely',
     'with',
     'the',
     'substance',
     'N',
     'have',
     'died',
     'more',
     'than',
     'three',
     'times',
     'the',
     'expected',
     'number',
     '<eos>',
     'four',
     'of',
     'the',
     'five',
     'surviving',
     'workers',
     'have',
     '<unk>',
     'diseases',
     'including',
     'three',
     'with',
     'recently',
     '<unk>',
     'cancer',
     '<eos>',
     'the',
     'total',
     'of',
     'N',
     'deaths',
     'from',
     'malignant',
     '<unk>',
     'lung',
     'cancer',
     'and',
     '<unk>',
     'was',
     'far',
     'higher',
     'than',
     'expected',
     'the',
     'researchers',
     'said',
     '<eos>',
     'the',
     '<unk>',
     'rate',
     'is',
     'a',
     'striking',
     'finding',
     'among',
     'those',
     'of',
     'us',
     'who',
     'study',
     '<unk>',
     'diseases',
     'said',
     'dr.',
     '<unk>',
     '<eos>',
     'the',
     'percentage',
     'of',
     'lung',
     'cancer',
     'deaths',
     'among',
     'the',
     'workers',
     'at',
     'the',
     'west',
     '<unk>',
     'mass.',
     'paper',
     'factory',
     'appears',
     'to',
     'be',
     'the',
     'highest',
     'for',
     'any',
     'asbestos',
     'workers',
     'studied',
     'in',
     'western',
     'industrialized',
     'countries',
     'he',
     'said',
     '<eos>',
     'the',
     'plant',
     'which',
     'is',
     'owned',
     'by',
     '<unk>',
     '&',
     '<unk>',
     'co.',
     'was',
     'under',
     'contract',
     'with',
     '<unk>',
     'to',
     'make',
     'the',
     'cigarette',
     'filters',
     '<eos>',
     'the',
     'finding',
     'probably',
     'will',
     'support',
     'those',
     'who',
     'argue',
     'that',
     'the',
     'u.s.',
     'should',
     'regulate',
     'the',
     'class',
     'of',
     'asbestos',
     'including',
     '<unk>',
     'more',
     '<unk>',
     'than',
     'the',
     'common',
     'kind',
     'of',
     'asbestos',
     '<unk>',
     'found',
     'in',
     'most',
     'schools',
     'and',
     'other',
     'buildings',
     'dr.',
     '<unk>',
     'said',
     '<eos>',
     'the',
     'u.s.',
     'is',
     'one',
     'of',
     'the',
     'few',
     'industrialized',
     'nations',
     'that',
     'does',
     "n't",
     'have',
     'a',
     'higher',
     'standard',
     'of',
     'regulation',
     'for',
     'the',
     'smooth',
     '<unk>',
     'fibers',
     'such',
     'as',
     '<unk>',
     'that',
     'are',
     'classified',
     'as',
     '<unk>',
     'according',
     'to',
     '<unk>',
     't.',
     '<unk>',
     'a',
     'professor',
     'of',
     '<unk>',
     'at',
     'the',
     'university',
     'of',
     'vermont',
     'college',
     'of',
     'medicine',
     '<eos>',
     'more',
     'common',
     '<unk>',
     'fibers',
     'are',
     '<unk>',
     'and',
     'are',
     'more',
     'easily',
     'rejected',
     'by',
     'the',
     'body',
     'dr.',
     '<unk>',
     'explained',
     '<eos>',
     'in',
     'july',
     'the',
     'environmental',
     'protection',
     'agency',
     'imposed',
     'a',
     'gradual',
     'ban',
     'on',
     'virtually',
     'all',
     'uses',
     'of',
     'asbestos',
     '<eos>',
     'by',
     'N',
     'almost',
     'all',
     'remaining',
     'uses',
     'of',
     '<unk>',
     'asbestos',
     'will',
     'be',
     'outlawed',
     '<eos>',
     'about',
     'N',
     'workers',
     'at',
     'a',
     'factory',
     'that',
     'made',
     'paper',
     'for',
     'the',
     'kent',
     'filters',
     'were',
     'exposed',
     'to',
     'asbestos',
     'in',
     'the',
     '1950s',
     '<eos>',
     'areas',
     'of',
     'the',
     'factory',
     'were',
     'particularly',
     'dusty',
     'where',
     'the',
     '<unk>',
     'was',
     'used',
     '<eos>',
     'workers',
     'dumped',
     'large',
     '<unk>',
     '<unk>',
     'of',
     'the',
     'imported',
     'material',
     'into',
     'a',
     'huge',
     '<unk>',
     'poured',
     'in',
     'cotton',
     'and',
     '<unk>',
     'fibers',
     'and',
     '<unk>',
     'mixed',
     'the',
     'dry',
     'fibers',
     'in',
     'a',
     'process',
     'used',
     'to',
     'make',
     'filters',
     '<eos>',
     'workers',
     'described',
     'clouds',
     'of',
     'blue',
     'dust',
     'that',
     'hung',
     'over',
     'parts',
     'of',
     'the',
     'factory',
     'even',
     'though',
     '<unk>',
     'fans',
     '<unk>',
     'the',
     'area',
     '<eos>',
     'there',
     "'s",
     'no',
     'question',
     'that',
     'some',
     'of',
     'those',
     'workers',
     'and',
     'managers',
     'contracted',
     '<unk>',
     'diseases',
     'said',
     '<unk>',
     'phillips',
     'vice',
     'president',
     'of',
     'human',
     'resources',
     'for',
     '<unk>',
     '&',
     '<unk>',
     '<eos>',
     'but',
     'you',
     'have',
     'to',
     'recognize',
     'that',
     'these',
     'events',
     'took',
     'place',
     'N',
     'years',
     'ago',
     '<eos>',
     'it',
     'has',
     'no',
     'bearing',
     'on',
     'our',
     'work',
     'force',
     'today',
     '<eos>',
     'yields',
     'on',
     'money-market',
     'mutual',
     'funds',
     'continued',
     'to',
     'slide',
     'amid',
     'signs',
     'that',
     'portfolio',
     'managers',
     'expect',
     'further',
     'declines',
     'in',
     'interest',
     'rates',
     '<eos>',
     'the',
     'average',
     'seven-day',
     'compound',
     'yield',
     'of',
     'the',
     'N',
     'taxable',
     'funds',
     'tracked',
     'by',
     '<unk>',
     "'s",
     'money',
     'fund',
     'report',
     'eased',
     'a',
     'fraction',
     'of',
     'a',
     'percentage',
     'point',
     'to',
     'N',
     'N',
     'from',
     'N',
     'N',
     'for',
     'the',
     'week',
     'ended',
     'tuesday',
     '<eos>',
     'compound',
     'yields',
     'assume',
     'reinvestment',
     'of',
     'dividends',
     'and',
     'that',
     'the',
     'current',
     'yield',
     'continues',
     'for',
     'a',
     'year',
     '<eos>',
     'average',
     'maturity',
     'of',
     'the',
     'funds',
     "'",
     'investments',
     '<unk>',
     'by',
     'a',
     'day',
     'to',
     'N',
     'days',
     'the',
     'longest',
     'since',
     'early',
     'august',
     'according',
     'to',
     'donoghue',
     "'s",
     '<eos>',
     'longer',
     'maturities',
     'are',
     'thought',
     'to',
     'indicate',
     'declining',
     'interest',
     'rates',
     'because',
     'they',
     'permit',
     'portfolio',
     'managers',
     'to',
     'retain',
     'relatively',
     'higher',
     'rates',
     'for',
     'a',
     'longer',
     'period',
     '<eos>',
     'shorter',
     'maturities',
     'are',
     'considered',
     'a',
     'sign',
     'of',
     'rising',
     'rates',
     'because',
     'portfolio',
     'managers',
     'can',
     'capture',
     'higher',
     'rates',
     'sooner',
     '<eos>',
     'the',
     'average',
     'maturity',
     'for',
     'funds',
     'open',
     'only',
     'to',
     'institutions',
     'considered',
     'by',
     'some',
     'to',
     'be',
     'a',
     'stronger',
     'indicator',
     'because',
     'those',
     'managers',
     'watch',
     'the',
     'market',
     'closely',
     'reached',
     'a',
     'high',
     'point',
     'for',
     'the',
     'year',
     'N',
     'days',
     '<eos>',
     'nevertheless',
     'said',
     '<unk>',
     '<unk>',
     '<unk>',
     'editor',
     'of',
     'money',
     'fund',
     'report',
     'yields',
     'may',
     '<unk>',
     'up',
     'again',
     'before',
     'they',
     '<unk>',
     'down',
     'because',
     'of',
     'recent',
     'rises',
     'in',
     'short-term',
     'interest',
     'rates',
     '<eos>',
     'the',
     'yield',
     'on',
     'six-month',
     'treasury',
     'bills',
     'sold',
     'at',
     'monday',
     "'s",
     'auction',
     'for',
     'example',
     'rose',
     'to',
     'N',
     'N',
     'from',
     'N',
     'N',
     '<eos>',
     'despite',
     'recent',
     'declines',
     'in',
     'yields',
     'investors',
     'continue',
     'to',
     'pour',
     'cash',
     'into',
     'money',
     'funds',
     '<eos>',
     'assets',
     'of',
     'the',
     'N',
     'taxable',
     'funds',
     'grew',
     'by',
     '$',
     'N',
     'billion',
     'during',
     'the',
     ...]




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

    vocab size :  10000
    


```python
dataset_dir = './ptb_dataset/'
train_file_name = 'ptb.train.txt'
```


```python
window_size = 5
sample_size = 5
dataset = PTBDataset(file_path, window_size, sample_size)
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
class CBOW_NS_Model(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(CBOW_NS_Model, self).__init__()
        self.embedding_in = nn.Embedding(num_embeddings=vocab_size, embedding_dim = hidden_size)
        self.embedding_out = nn.Embedding(num_embeddings=vocab_size, embedding_dim = hidden_size)

    def forward(self, inputs, targets): 
        h = self.embedding_in(inputs) # h : (batch_size, hidden_size)
        h = h.mean(axis=1)
        target_W = self.embedding_out(targets) # target_W : (batch_size, hidden_size)
        out = torch.sum(target_W * h, axis=1)
        prob = F.sigmoid(out)
        return prob
```


```python
vocab_size = 10000
hidden_size = 100

model = CBOW_NS_Model(vocab_size=vocab_size, hidden_size=hidden_size)
model.to(device)
```




    CBOW_NS_Model(
      (embedding_in): Embedding(10000, 100)
      (embedding_out): Embedding(10000, 100)
    )




```python
contexts, target = contexts.to(device), target.to(device)
out = model(contexts, target)
out.shape, out.dtype
```




    (torch.Size([100]), torch.float32)



## 5. 모델 컴파일 (손실함수, 옵티마이저 선택)


```python
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

## 6. 모델 훈련


```python
def train_loop(model, dataloader, loss_fn, epochs, optimizer):  
    min_loss = 1000000  
    trigger = 0
    patience = 4     

    for epoch in range(epochs):
        model.train()
        train_loss = []

        for batch_data in dataloader:
            contexts = batch_data[0].to(device)
            target = batch_data[1].to(device)
            
            negative_sample = dataset.sampler.get_negative_sample(target)
            negative_sample = torch.LongTensor(negative_sample).to(device)
            
            optimizer.zero_grad()
            
            # positive sample 순전파
            positive_prob = model(contexts, target)
            correct_label = torch.ones(target.shape[0]).to(device)
            positive_loss = loss_fn(positive_prob, correct_label)

            # negative samples 순전파
            # negative_sample.shape : (batch_size, sample_size)
            negative_label = torch.zeros(target.shape[0]).to(device)
            negative_loss = 0
            for i in range(sample_size):
                negative_target = negative_sample[:, i]
                negative_prob = model(contexts, negative_target)
                negative_loss += loss_fn(negative_prob, negative_label)
                
            loss = positive_loss + negative_loss
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        train_epoch_loss = np.mean(train_loss)
        total_loss["train"].append(train_epoch_loss)

        print("Epoch: {}/{}, Train Loss={:.5f}".format(                    
               epoch + 1, epochs,
               total_loss["train"][-1]))                  
```


```python
epochs = 12
total_loss = {"train": []}
%time train_loop(model, dataloader, loss_fn, epochs, optimizer)
```

    Epoch: 1/12, Train Loss=4.46790
    Epoch: 2/12, Train Loss=2.73813
    Epoch: 3/12, Train Loss=2.32509
    Epoch: 4/12, Train Loss=2.07252
    Epoch: 5/12, Train Loss=1.90162
    Epoch: 6/12, Train Loss=1.77771
    Epoch: 7/12, Train Loss=1.68148
    Epoch: 8/12, Train Loss=1.60607
    Epoch: 9/12, Train Loss=1.54479
    Epoch: 10/12, Train Loss=1.49309
    Epoch: 11/12, Train Loss=1.44821
    Epoch: 12/12, Train Loss=1.40948
    CPU times: user 9min 1s, sys: 6.54 s, total: 9min 8s
    Wall time: 9min 9s
    

## 7. 유사도 측정


```python
# embedding from first model layer
# detach : https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html

embeddings = list(model.parameters())[0]
embeddings = embeddings.cpu().detach().numpy()
embeddings.shape
```




    (10000, 100)




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
     we: 0.7707014679908752
     i: 0.6373926401138306
     they: 0.5769265294075012
     triple-a: 0.43831726908683777
     she: 0.42899152636528015
    
    [query] year
     month: 0.839719295501709
     week: 0.7326676249504089
     summer: 0.5758762359619141
     decade: 0.56444251537323
     spring: 0.5531997680664062
    
    [query] car
     move: 0.5004615187644958
     buildings: 0.38829562067985535
     furor: 0.3863994777202606
     record: 0.36925071477890015
     plant: 0.3521578013896942
    
    [query] toyota
     strengths: 0.43140339851379395
     marble: 0.38097456097602844
     sdi: 0.3780283033847809
     entertaining: 0.3772747218608856
     ford: 0.3746541738510132
    


```python

```
