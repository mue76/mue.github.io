---
tag: [Deep Learning, 딥러닝, RNN, 자연어처리, 순환신경망, pytorch, 파이토치, 언어모델, Language Model, MS COCO Dataset, LSTM, Sequence2Seqence, Seq2Seq, 시퀀스투시퀀스, 문장생성]
toc: true
toc_sticky: true
toc_label: 목차
author_profile: false

---

## [MS COCO DataSet](https://cocodataset.org/#home)

**MS COCO Dataset의 특징**
- 모든 이미지들에 대해서 1개의 Annotation 파일을 가짐

**다운로드 링크**
- [홈페이지의 Download 메뉴](https://cocodataset.org/#download)에서 다운로드 가능


```python
!mkdir ./mscoco
```

**train 데이터**


```python
!wget http://images.cocodataset.org/zips/train2014.zip
!unzip -q "train2014.zip" -d ./mscoco/
!rm train2014.zip
```

    --2023-05-03 05:31:52--  http://images.cocodataset.org/zips/train2014.zip
    Resolving images.cocodataset.org (images.cocodataset.org)... 52.217.236.73, 54.231.235.153, 52.217.91.233, ...
    Connecting to images.cocodataset.org (images.cocodataset.org)|52.217.236.73|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 13510573713 (13G) [application/zip]
    Saving to: ‘train2014.zip’
    
    train2014.zip       100%[===================>]  12.58G  60.5MB/s    in 3m 27s  
    
    2023-05-03 05:35:19 (62.3 MB/s) - ‘train2014.zip’ saved [13510573713/13510573713]
    
    

**valid 데이터**


```python
!wget http://images.cocodataset.org/zips/val2014.zip
!unzip -q "val2014.zip" -d ./mscoco/
!rm val2014.zip
```

    --2023-05-03 05:37:38--  http://images.cocodataset.org/zips/val2014.zip
    Resolving images.cocodataset.org (images.cocodataset.org)... 52.216.236.83, 3.5.19.125, 3.5.29.160, ...
    Connecting to images.cocodataset.org (images.cocodataset.org)|52.216.236.83|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 6645013297 (6.2G) [application/zip]
    Saving to: ‘val2014.zip’
    
    val2014.zip         100%[===================>]   6.19G  56.1MB/s    in 2m 6s   
    
    2023-05-03 05:39:45 (50.1 MB/s) - ‘val2014.zip’ saved [6645013297/6645013297]
    
    

**test 데이터**


```python
!wget http://images.cocodataset.org/zips/test2014.zip
!unzip -q "test2014.zip" -d ./mscoco/
!rm test2014.zip
```

    --2023-05-03 05:41:16--  http://images.cocodataset.org/zips/test2014.zip
    Resolving images.cocodataset.org (images.cocodataset.org)... 52.217.170.201, 52.217.4.20, 54.231.202.225, ...
    Connecting to images.cocodataset.org (images.cocodataset.org)|52.217.170.201|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 6660437059 (6.2G) [application/zip]
    Saving to: ‘test2014.zip’
    
    test2014.zip        100%[===================>]   6.20G  49.3MB/s    in 2m 4s   
    
    2023-05-03 05:43:21 (51.1 MB/s) - ‘test2014.zip’ saved [6660437059/6660437059]
    
    

**train, valid 용 정답 데이터**


```python
!wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
!unzip -q annotations_trainval2014.zip -d ./mscoco
```

    --2023-05-03 05:44:35--  http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    Resolving images.cocodataset.org (images.cocodataset.org)... 54.231.230.113, 52.217.128.209, 52.216.60.33, ...
    Connecting to images.cocodataset.org (images.cocodataset.org)|54.231.230.113|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 252872794 (241M) [application/zip]
    Saving to: ‘annotations_trainval2014.zip’
    
    annotations_trainva 100%[===================>] 241.16M  96.2MB/s    in 2.5s    
    
    2023-05-03 05:44:38 (96.2 MB/s) - ‘annotations_trainval2014.zip’ saved [252872794/252872794]
    
    

**test용 정답 데이터**


```python
!wget http://images.cocodataset.org/annotations/image_info_test2014.zip
!unzip -q "image_info_test2014.zip" -d ./mscoco/
```

    --2023-05-03 05:44:47--  http://images.cocodataset.org/annotations/image_info_test2014.zip
    Resolving images.cocodataset.org (images.cocodataset.org)... 3.5.11.124, 3.5.28.18, 52.216.12.28, ...
    Connecting to images.cocodataset.org (images.cocodataset.org)|3.5.11.124|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 763464 (746K) [application/zip]
    Saving to: ‘image_info_test2014.zip’
    
    image_info_test2014 100%[===================>] 745.57K  --.-KB/s    in 0.03s   
    
    2023-05-03 05:44:47 (21.5 MB/s) - ‘image_info_test2014.zip’ saved [763464/763464]
    
    


```python
# http://json.parser.online.fr/ (사이트에서 변환)
!sudo apt-get install jq
```


```python
!jq . ./mscoco/annotations/captions_train2014.json > train_output.json
!jq . ./mscoco/annotations/captions_val2014.json > val_output.json
!jq . ./mscoco/annotations/image_info_test2014.json > test_output.json
```


```python
!head train_output.json
```

    {
      "info": {
        "description": "COCO 2014 Dataset",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": 2014,
        "contributor": "COCO Consortium",
        "date_created": "2017/09/01"
      },
      "images": [
    


```python
# json 파일을 dictionary로 바꾸기
import json
with open('./mscoco/annotations/captions_train2014.json') as fd:
    train_dict = json.load(fd)   

with open('./mscoco/annotations/captions_val2014.json') as fd:
    val_dict = json.load(fd)   

with open('./mscoco/annotations/image_info_test2014.json') as fd:
    test_dict = json.load(fd)       
```


```python
# captions_train2014.json 의 키 값들
train_dict.keys()
```




    dict_keys(['info', 'images', 'licenses', 'annotations'])




```python
# image_info_test2014.json 의 키 값들 (annotations가 없음)
test_dict.keys()
```




    dict_keys(['info', 'images', 'licenses', 'categories'])




```python
# images 키로 색인
train_dict['images'][:2]
```




    [{'license': 5,
      'file_name': 'COCO_train2014_000000057870.jpg',
      'coco_url': 'http://images.cocodataset.org/train2014/COCO_train2014_000000057870.jpg',
      'height': 480,
      'width': 640,
      'date_captured': '2013-11-14 16:28:13',
      'flickr_url': 'http://farm4.staticflickr.com/3153/2970773875_164f0c0b83_z.jpg',
      'id': 57870},
     {'license': 5,
      'file_name': 'COCO_train2014_000000384029.jpg',
      'coco_url': 'http://images.cocodataset.org/train2014/COCO_train2014_000000384029.jpg',
      'height': 429,
      'width': 640,
      'date_captured': '2013-11-14 16:29:45',
      'flickr_url': 'http://farm3.staticflickr.com/2422/3577229611_3a3235458a_z.jpg',
      'id': 384029}]




```python
len(train_dict['images'])
```




    82783




```python
# annotations 키로 색인
train_dict['annotations'][:2]
```




    [{'image_id': 318556,
      'id': 48,
      'caption': 'A very clean and well decorated empty bathroom'},
     {'image_id': 116100,
      'id': 67,
      'caption': 'A panoramic view of a kitchen and all of its appliances.'}]




```python
len(train_dict['annotations'])
```




    414113




```python
# image_id 318556가 있는 라인 출력하기 (한 이미지에 대해 여러 캡션이 있음)
# grep은 특정 문자열을 찾아 출력하는 명령어, -n 옵션은 매칭되는 각 라인의 번호를 출력
!grep -n '318556' train_output.json
```

    146228:      "flickr_url": "http://farm4.staticflickr.com/3478/3185566473_82b8b10891_z.jpg",
    533153:      "file_name": "COCO_train2014_000000318556.jpg",
    533154:      "coco_url": "http://images.cocodataset.org/train2014/COCO_train2014_000000318556.jpg",
    533159:      "id": 318556
    827886:      "image_id": 318556,
    827896:      "image_id": 318556,
    827916:      "image_id": 318556,
    827921:      "image_id": 318556,
    828451:      "image_id": 318556,
    2531332:      "id": 318556,
    


```python
# 533153번 라인부터 533159번 라인까지의 내용을 출력
!sed -n '533153, 533159p' train_output.json
```

          "file_name": "COCO_train2014_000000318556.jpg",
          "coco_url": "http://images.cocodataset.org/train2014/COCO_train2014_000000318556.jpg",
          "height": 640,
          "width": 480,
          "date_captured": "2013-11-15 05:00:35",
          "flickr_url": "http://farm4.staticflickr.com/3133/3378902101_3c9fa16b84_z.jpg",
          "id": 318556
    


```python
!sed -n '827886, 827889p' train_output.json
```

          "image_id": 318556,
          "id": 48,
          "caption": "A very clean and well decorated empty bathroom"
        },
    


```python
!sed -n '827896, 827899p' train_output.json
```

          "image_id": 318556,
          "id": 126,
          "caption": "A blue and white bathroom with butterfly themed wall tiles."
        },
    

**pycocotools를 이용한 COCO 데이터 액세스**


```python
from pycocotools.coco import COCO
# annotation 파일을 COCO객체로 로드하면 다양한 COCO객체의 API들을 이용하여 
# COCO 데이터셋에서 제공하는 모든 annotation 정보를 쉽게 접근
annotations_file = './mscoco/annotations/captions_train2014.json'
# COCO API 객체 생성 
coco = COCO(annotations_file) # 어노테이션 정보를 사용하여 생성된 COCO 객체
```

    loading annotations into memory...
    Done (t=1.28s)
    creating index...
    index created!
    


```python
# coco.anns는 COCO 데이터셋의 어노테이션 정보를 저장한 딕셔너리
coco.anns
```


```python
# 모든 어노테이션 key(ID)를 출력
coco.anns.keys()
```


```python
ids = list(coco.anns.keys())
len(ids)
```




    414113




```python
type(ids)
```




    list




```python
ids[:3]
```




    [48, 67, 126]




```python
ids[0]
```




    48




```python
coco.anns[48] # 인덱스가 48인 어노테이션 정보 가져오기
```




    {'image_id': 318556,
     'id': 48,
     'caption': 'A very clean and well decorated empty bathroom'}




```python
coco.anns[48]['caption']
```




    'A very clean and well decorated empty bathroom'



**Vocabulary 구축 준비**


```python
import nltk
nltk.download('punkt')
```

    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt.zip.
    




    True




```python
# COCO Data에서 추출한 Caption 정보를 nltk를 이용하여 tokenize 할 수 있음
nltk.tokenize.word_tokenize(str(coco.anns[48]['caption']).lower())
```




    ['a', 'very', 'clean', 'and', 'well', 'decorated', 'empty', 'bathroom']




```python
# Annotation 정보 모두를 tokenize하기기
import numpy as np
all_tokens = [nltk.tokenize.word_tokenize(str(coco.anns[ids[index]]['caption']).lower()) for index in np.arange(len(ids))]
```


```python
len(all_tokens)
```




    414113




```python
all_tokens[0]
```




    ['a', 'very', 'clean', 'and', 'well', 'decorated', 'empty', 'bathroom']




```python
all_tokens[-1]
```




    ['a', 'dinner', 'plate', 'has', 'a', 'lemon', 'wedge', 'garnishment', '.']




```python
# 캡션 한개에 대해 단어 빈도수 구하기
from collections import Counter
counter = Counter()
counter.update(all_tokens[0])
```


```python
counter
```




    Counter({'a': 1,
             'very': 1,
             'clean': 1,
             'and': 1,
             'well': 1,
             'decorated': 1,
             'empty': 1,
             'bathroom': 1})




```python
# 캡션 전체에 대해 단어 빈도수 구하기
# Vocabulary 생성시 일정 빈도수 이상 단어만 사용할 수 있음
counter = Counter()
ids = coco.anns.keys()
for i, id in enumerate(ids):
    caption = str(coco.anns[id]['caption'])
    tokens = nltk.tokenize.word_tokenize(caption.lower())
    counter.update(tokens)
counter    
```

**Dataset 클래스를 작성하기 위해 필요한 정보 준비**
- image file path와 caption


```python
ids = list(coco.anns.keys())
index = 0
ann_id = ids[index]

# (1) image file path
img_id = coco.anns[ann_id]['image_id']
# coco.loadImgs() 메서드는 COCO 데이터셋에서 이미지 ID에 해당하는 이미지 정보
path = coco.loadImgs(img_id)[0]['file_name'] 

# (2) caption
caption = coco.anns[ann_id]['caption']
```


```python
img_id = coco.anns[48]['image_id']
img_id
```




    318556




```python
coco.loadImgs(img_id)[0]['file_name']
```




    'COCO_train2014_000000318556.jpg'




## Reference
- [MS COCO DataSet](https://cocodataset.org/#home)
