---
tag: [python, programming, 파이썬 문법, 기초코딩]
toc: true
toc_sticky: true
toc_label: 목차
---
# 시퀀스 객체 응용

## 리스트 응용하기

- append : 요소 하나를 추가
- extend : 리스트를 연결하여 확장
- insert : 특정 인덱스에 요소 추가


```python
a = [10, 20, 30]
```


```python
a.append(500)
```


```python
a
```




    [10, 20, 30, 500]




```python
a = []
a.append(10)
a
```




    [10]




```python
a = [10, 20, 30]
b = [40, 50, 60]
a.extend(b)
a
```




    [10, 20, 30, 40, 50, 60]




```python
# 참고
# 시퀀스 객체 + 시퀀스 객체
# 리스트 + 리스트
```


```python
# + 이 extend와 다른점은 새로운 결과가 나옴
# 반면에 extend는 a에 결과가 적용되어 있음
a = [10, 20, 30]
b = [40, 50, 60]
c = a + b
c
```




    [10, 20, 30, 40, 50, 60]




```python
a
```




    [10, 20, 30]




```python
b
```




    [40, 50, 60]




```python
a.insert(1, 10000)
a
```




    [10, 10000, 20, 30]



- a.insert(0, 요소) : 리스트의 맨 처음에 요소를 추가
- a.insert(len(리스트), 요소) : 리스트의 맨 끝에 요소를 추가


```python
a = [10, 20, 30]
a.insert(0, 1000)
a
```




    [1000, 10, 20, 30]




```python
a.insert(len(a), 2000)
a
```




    [1000, 10, 20, 30, 2000]




```python
len(a)
```




    5




```python
a.insert(5, 3000)
a
```




    [1000, 10, 20, 30, 2000, 3000]




```python
a.insert(-1, 4000) # -1 은 리스트의 마지막 인덱스
a
```




    [1000, 10, 20, 30, 2000, 4000, 3000]



- pop : 마지막 요소 또는 특정 인덱스의 요소를 삭제
- remove : 특정 값을 찾아서 삭제


```python
a = [10, 20, 30]
a.pop() # 인덱스를 넣지 않으면 마지막 요소 삭제
a
```




    [10, 20]




```python
a = [10, 20, 30]
a.pop(1) # 인덱스 넣으면 해당 인덱스의 요소가 삭제
a
```




    [10, 30]




```python
a = [10, 20, 30]
result = a.pop() # pop의 결과로 삭제된 요소를 반환
result
```




    30




```python
a = [10, 20, 30]
result = a.pop(1) # pop의 결과로 삭제된 요소를 반환
result
```




    20




```python
a = [10, 20, 30]
a.remove(20) # 값 자체를 넣어줌
a
```




    [10, 30]



- index(값) : 리스트에서 특정 값의 인덱스를 구함
- count(값) : 리스트에서 특정 값의 개수를 구함


```python
a = [10, 20, 30]
a.index(20)
```




    1




```python
a.index(40) # error
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-22-ff9934e1dbe7> in <module>
    ----> 1 a.index(40)
    

    ValueError: 40 is not in list



```python
a = [10, 10, 20, 30, 30, 30]
a.count(30)
```




    3



- reverse() : 리스트에서 요소의 순서를 반대로 뒤집음
- sort() : 리스트의 요소를 정렬함(오름차순 기본값)
- clear() : 리스트의 모든 요소를 삭제함


```python
a = [10, 20, 30]
a.reverse()
a
```




    [30, 20, 10]




```python
a = [10, 50, -1, 0, 4, 1000]
a.sort()
a
```




    [-1, 0, 4, 10, 50, 1000]




```python
a = [10, 50, -1, 0, 4, 1000]
a.sort(reverse=True)
a
```




    [1000, 50, 10, 4, 0, -1]




```python
# sorted() : a에는 정렬결과가 반영 안됨
a = [10, 50, -1, 0, 4, 1000]
result = sorted(a)
result
```




    [-1, 0, 4, 10, 50, 1000]




```python
a
```




    [10, 50, -1, 0, 4, 1000]



- =
- copy()


```python
a = [0, 0, 0, 0]
b = a
```


```python
a is b
```




    True




```python
b[1] = 1000
b
```




    [0, 1000, 0, 0]




```python
a # b만 변경을 했는데도 같은 메모리를 공유하므로 a에도 변화가 생겼음
```




    [0, 1000, 0, 0]




```python
a = [0, 0, 0, 0]
b = a.copy()
```


```python
a is b
```




    False




```python
a
```




    [0, 0, 0, 0]




```python
b
```




    [0, 0, 0, 0]




```python
b[1] = 1000
b
```




    [0, 1000, 0, 0]




```python
a # b와 a가 서로 다른 메모리가 할당되서 b의 변화가 a에 영향을 주지 않았음
```




    [0, 0, 0, 0]




```python
a = [[1, 2, 3], [4, 5, 6]] # 2차원 리스트
a
```




    [[1, 2, 3], [4, 5, 6]]




```python
b = a
```


```python
b[0] # 2차원 리스트를 색인하면 1차원 리스트가 결과로 나옴
```




    [1, 2, 3]




```python
b[1]
```




    [4, 5, 6]




```python
x = b[0]
x[2]
```




    3




```python
b[0][2]
```




    3




```python
b[0][2] = 1000
b
```




    [[1, 2, 1000], [4, 5, 6]]




```python
a
```




    [[1, 2, 1000], [4, 5, 6]]




```python
a = [[1, 2, 3], [4, 5, 6]] # 2차원 리스트

b = a.copy()
```


```python
b[0][2] = 1000
b
```




    [[1, 2, 1000], [4, 5, 6]]




```python
a # copy로 b를 생성했는데도, b의 변화가 a에도 적용이 되었음 (2차원 리스트에서는 copy가 제대로 동작안함)
```




    [[1, 2, 1000], [4, 5, 6]]




```python
import copy
a = [[1, 2, 3], [4, 5, 6]] # 2차원 리스트

b = copy.deepcopy(a)
```


```python
b[0][2] = 1000
b
```




    [[1, 2, 1000], [4, 5, 6]]




```python
a # deepcopy로 b를 생성하면 b의 변화가 a에 영향을 미치지 않았음
```




    [[1, 2, 3], [4, 5, 6]]



**반복문으로 리스트 요소 출력하기**

- for 요소 in 리스트:


```python
a = [10, 20, 30]
for v in a:
    print(v)
```

    10
    20
    30
    

- for 인덱스, 요소 in enumerate(리스트):


```python
a = [10, 20, 30]
for i, v in enumerate(a):
    print(i, v)
```

    0 10
    1 20
    2 30
    


```python
a = [10, 20, 30]
for i, v in enumerate(a, start=1):
    print(i, v)
```

    1 10
    2 20
    3 30
    

**리스트의 가장 작은수, 가장 큰수, 합계**


```python
a = [10, 20, 30]
```


```python
min(a)
```




    10




```python
max(a)
```




    30




```python
sum(a)
```




    60




```python
sum(a)/len(a)
```




    20.0



**리스트 표현식(List Comprehension)**

- [식 for 변수 in 리스트]


```python
result = []
for i in range(10): # 0~ 9 순회
    result.append(float(i))
```


```python
result
```




    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]




```python
[float(i) for i in range(10)]
```




    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]




```python
word_list = ['python', 'is', 'easy']
```


```python
# 반복문
result = []
for word in word_list:
    result.append(len(word))
result
```




    [6, 2, 4]




```python
# 리스트표현식
[len(word) for word in word_list]
```




    [6, 2, 4]



- [식 for 변수 in 리스트 if 조건식]


```python
[i for i in range(10) if i % 2 == 0]
```




    [0, 2, 4, 6, 8]




```python
result = []
for i in range(10):
    if i % 2 == 0:
        result.append(i)
result        
```




    [0, 2, 4, 6, 8]



**리스트에서 특정 요소만 뽑아내기**
- 리스트 a에 들어있는 문자열 중에서 길이가 5인 것들만 리스트 형태로 출력되게 만드세요


```python
a = ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf', 'hotel', 'india']
```


```python
b = [word for word in a if len(word) == 5]
print(b)
```

    ['alpha', 'bravo', 'delta', 'hotel', 'india']
    


```python
b = []
for word in a:
    if len(word) == 5:
        b.append(word)
print(b)        
```

    ['alpha', 'bravo', 'delta', 'hotel', 'india']
    

**리스트에서 map 사용하기**


```python
a = [1.2, 2.5, 3.7, 4.6]
b = []
for i in a:
    b.append(int(i))
b    
```




    [1, 2, 3, 4]




```python
b = list(map(int, a))
b
```




    [1, 2, 3, 4]



## 튜플 응용하기

- index(값) : 특정값의 인덱스 구하기


```python
a = (10, 20, 30)
```


```python
a.index(30)
```




    2



- count(값) : 특정값의 개수 구하기


```python
a = (10, 10, 10, 20, 20, 30, 30, 30, 30)
a.count(30)
```




    4



## 문자열 응용하기

- replace('바꿀문자열', '새문자열') : 문자열 바꾸기


```python
s = 'Hello World!'
result = s.replace('World', 'Python')
result
```




    'Hello Python!'




```python
s
```




    'Hello World!'



- split('기준문자열') : 문자열 분리하기


```python
s = 'apple pear grape pineapple orange'
result = s.split()
result
```




    ['apple', 'pear', 'grape', 'pineapple', 'orange']




```python
s
```




    'apple pear grape pineapple orange'




```python
s = 'apple.pear.grape.pineapple.orange'
result = s.split('.')
result
```




    ['apple', 'pear', 'grape', 'pineapple', 'orange']



- '구분자'.join(리스트)


```python
'.'.join(result)
```




    'apple.pear.grape.pineapple.orange'



- upper() : 대문자로 바꿈
- lower() : 소문자로 바꿈
- title() : 첫글자만 대문자로 바꿈
- strip() : 문자열 양쪽에 있는 연속된 모든 공백을 삭제
- lstrip() : 문자열 왼쪽에 있는 연속된 모든 공백을 삭제
- rstrip() : 문자열 오른쪽에 있는 연속된 모든 공백을 삭제
- center(길이) : 문자열을 길이만큼 사이즈를 확보하고 중간에 배치


```python
s = 'python'
s.upper()
```




    'PYTHON'




```python
s
```




    'python'




```python
S = 'PYTHON'
S.lower()
```




    'python'




```python
S.title()
```




    'Python'




```python
'    Python     '.strip()
```




    'Python'




```python
'    Python     '.lstrip()
```




    'Python     '




```python
'    Python     '.rstrip()
```




    '    Python'




```python
'  ,,.  Python ,...    '.strip(',. ')
```




    'Python'




```python
'python'.center(20)
```




    '       python       '



- index('찾을문자열') : 문자열에서 특정문자열을 찾아서 인덱스를 반환하고, 없으면 에러
- find('찾을문자열') : 문자열에서 특정문자열을 찾아서 인덱스를 반환하고, 없으면 -1 반환


```python
s = 'apple pear grape pineapple orange'
s.index('pl') # 같은 문자열이 여러개면 먼저나온 문자열의 인덱스 반환
```




    2




```python
s.rindex('pl') # 오른쪽부터 찾음
```




    23




```python
s = 'apple pear grape pineapple orange'
s.index('pppp') # error
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-95-35c74c9e7c37> in <module>
          1 s = 'apple pear grape pineapple orange'
    ----> 2 s.index('pppp')
    

    ValueError: substring not found



```python
s = 'apple pear grape pineapple orange'
s.find('pppp')
```




    -1




```python
s = 'apple pear grape pineapple orange'
s.find('pl')
```




    2




```python
s = 'apple pear grape pineapple orange'
s.rfind('pl')
```




    23



- count('문자열') : 현재 문자열에서 특정 문자열이 몇 번 나오는지 알아냄


```python
s = 'apple pear grape pineapple orange'
s.count('pl')
```




    2



**서식지정자**

- %


```python
name = '장경희'

'나는 %s입니다'%name
```




    '나는 장경희입니다'




```python
age = 20
'나이는 %d입니다'%age
```




    '나이는 20입니다'




```python
score = 4.5
'내 학점은 %.1f입니다'%score
```




    '내 학점은 4.5입니다'



- format()


```python
name = '장경희'

'나는 {}입니다'.format(name)
```




    '나는 장경희입니다'




```python
age = 20
'나이는 {}입니다'.format(age)
```




    '나이는 20입니다'




```python
score = 4.5
'내 학점은 {}입니다'.format(score)
```




    '내 학점은 4.5입니다'




```python
name = '장경희'
age = 20
score = 4.5

'나는 {0}입니다. 나이는 {1}입니다. 내 학점은 {2}입니다.'.format(name, age, score)
```




    '나는 장경희입니다. 나이는 20입니다. 내 학점은 4.5입니다.'



- f'문자열'


```python
name = '장경희'
age = 20
score = 4.5
f'나는 {name}입니다. 나이는 {age}입니다. 내 학점은 {score}입니다.'
```




    '나는 장경희입니다. 나이는 20입니다. 내 학점은 4.5입니다.'



## 딕셔너리 응용하기

- setdefault : 키-값 쌍 추가
- update : 키의 값 수정, 키가 없으면 키-값 쌍 추가   


```python
x = {'a':10, 'b':20, 'c':30, 'd':40}
```


```python
type(x)
```




    dict




```python
x.setdefault('e')
x
```




    {'a': 10, 'b': 20, 'c': 30, 'd': 40, 'e': None}




```python
x.setdefault('f', 50)
x
```




    {'a': 10, 'b': 20, 'c': 30, 'd': 40, 'e': None, 'f': 50}




```python
x.update({'a':100, 'b':200, 'g':1000})
x
```




    {'a': 100, 'b': 200, 'c': 30, 'd': 40, 'e': None, 'f': 50, 'g': 1000}



- pop(키) : 특정 키-값 쌍을 삭제한 뒤 삭제한 값을 반환
- pop(키, 기본값) : 키가 없을 때 기본값을 반환
- clear() : 딕셔너리의 모든 키-값 쌍을 삭제


```python
x
```




    {'a': 100, 'b': 200, 'c': 30, 'd': 40, 'e': None, 'f': 50, 'g': 1000}




```python
result = x.pop('a')
result
```




    100




```python
x
```




    {'b': 200, 'c': 30, 'd': 40, 'e': None, 'f': 50, 'g': 1000}




```python
result = x.pop('a', 0)
result
```




    0



- get(키) : 특정 키의 값을 가져옴
- get(키, 기본값) : 키가 없을 때 기본값을 반환


```python
x.get('alpha')
```


```python
x.get('alphaaa', 0)
```




    0




```python
x
```




    {'b': 200, 'c': 30, 'd': 40, 'e': None, 'f': 50, 'g': 1000}



- items() : 키-값 쌍의 모두 가져옴
- keys() : 키를 모두 가져옴
- values() : 값을 모두 가져옴


```python
x
```




    {'b': 200, 'c': 30, 'd': 40, 'e': None, 'f': 50, 'g': 1000}




```python
x.items()
```




    dict_items([('b', 200), ('c', 30), ('d', 40), ('e', None), ('f', 50), ('g', 1000)])




```python
x.keys()
```




    dict_keys(['b', 'c', 'd', 'e', 'f', 'g'])




```python
x.values()
```




    dict_values([200, 30, 40, None, 50, 1000])




```python
for k, v in x.items():
    print(k, v)
```

    b 200
    c 30
    d 40
    e None
    f 50
    g 1000
    


```python
for k in x.keys():
    print(k)
```

    b
    c
    d
    e
    f
    g
    


```python
for v in x.values():
    print(v)
```

    200
    30
    40
    None
    50
    1000
    

- [식 for 변수 in 시퀀스객체 if 조건식] <-- 리스트 표현식
- {키:값 for 키, 값 in 딕셔너리.items() if 조건식} <-- 딕셔너리 표현식


```python
x = {'alpha': 10, 'bravo': 20, 'charlie': 30, 'delta': 40}
```


```python
{k:v for k, v in x.items() if k!='delta' and v != 30}
```




    {'alpha': 10, 'bravo': 20}



## 세트

- 세트 = {값1, 값2, 값3, ...}


```python
fruits = {'strawberry', 'grape', 'orange', 'pineapple', 'cherry'}
```


```python
type(fruits)
```




    set




```python
# 빈 리스트 만들기
l = []
l = list()

# 빈 튜플 만들기
t = ()
t = tuple()

# 빈 딕셔너리 만들기
d = {}
d = dict()

# 빈 세트 만들기
s = set()

```


```python
# 세트는 중복이 없고(unique한 원소), 순서가 없음
```


```python
fruits = {'strawberry', 'grape', 'orange', 'pineapple', 'cherry', 'cherry', 'cherry'}
fruits
```




    {'cherry', 'grape', 'orange', 'pineapple', 'strawberry'}




```python
a = {'a', 'a', 'b'}
b = {'a', 'b', 'a'}
c = {'b', 'a'}
```


```python
a == b
```




    True




```python
b == c
```




    True

## Reference
[파이썬 코딩 도장](https://dojang.io/course/view.php?id=7)
