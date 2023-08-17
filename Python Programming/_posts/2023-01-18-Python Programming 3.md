# Workshop

**알파벳별 빈도 사전 등록**
- 다음의 주어진 문자열 s에서 알파벳 별 빈도를 사전형인 d에 등록하는 파이썬 코드블럭을 작성하시오. 

```
s = ‘life is short, so python is easy.’ 
punct = ‘,. ’ 
d = {} 

결과 print(d) (순서는 상관 없음)
{'a': 1, 'e': 2, 'f': 1, 'i': 3, 'h': 2, 'l': 1, 'o': 3, 'n': 1, 'p': 1, 's': 5, 'r': 1, 't': 2, 'y': 2}

```


```python
{}
{'l':1}
{'l':1, 'i':1}
{'l':1, 'i':1, 'f':1}
{'l':1, 'i':1, 'f':1, 'e':1}
{'l':1, 'i':2, 'f':1, 'e':1}
```


```python
s = 'life is short, so python is easy.'
punct = ',. '
d = {} 

for c in s:
    if c in punct:
        continue
    if c not in d: # 'l' not in d:
        d[c] = 1
    else: # 'i' in d:   
        d[c] += 1
    
print(d)
```

    {'l': 1, 'i': 3, 'f': 1, 'e': 2, 's': 5, 'h': 2, 'o': 3, 'r': 1, 't': 2, 'p': 1, 'y': 2, 'n': 1, 'a': 1}
    


```python
# (참고) 파이썬에서 사용할 수 있는 collections 모듈 이용
from collections import Counter
counter = Counter(s)
print(counter)
```

# 중첩루프 사용하기


```python
for i in range(5): # 세로처리 위한 for 반복문
    for j in range(5): # 가로처리 위한 for 반복문
        print('*', end='')
    print()
```

    *****
    *****
    *****
    *****
    *****
    

```
# 계단식으로 별 출력하기 (1)
*
**
***
****
*****
```


```python
for i in range(5): # 세로처리 위한 for 반복문 (i : 0->1->2->3->4)
    for j in range(i+1): # 가로처리 위한 for 반복문 (*의 출력갯수 : 1->2->3->4->5)
        print('*', end='')
    print()
```

    *
    **
    ***
    ****
    *****
    

```
# 계단식으로 별 출력하기 (2)
*****
 ****
  ***
   **
    *
```   


```python
for i in range(5): # 세로처리 위한 for 반복문 (i : 0->1->2->3->4)
    for j in range(i): # 가로처리 위한 for 반복문 (' '의 출력갯수 : 0->1->2->3->4)
        print(' ', end='')
    for k in range(5-i): # 가로처리 위한 for 반복문 ('*'의 출력갯수 : 5->4->3->2->1)
        print("*", end='')
    print()
```

    *****
     ****
      ***
       **
        *
    

```
# 별로 산만들기 (3)
    *    
   ***   
  *****  
 ******* 
*********
```


```python
for i in range(1, 6): # 세로처리 위한 for 반복문 (i : 1->2->3->4->5)
    for j in range(5-i): # 가로처리 위한 for 반복문 (' '의 출력갯수 : 4->3->2->1->0)
        print(' ', end='')
    for k in range(2*i-1): # 가로처리 위한 for 반복문 ('*'의 출력갯수 : 1->3->5->7->9)
        print('*', end='')
    print()    
```

        *
       ***
      *****
     *******
    *********
    

**FizzBuzz 문제**

- 1에서 100까지 출력
- 3의 배수는 Fizz 출력
- 5의 배수는 Buzz 출력
- 3과 5의 공배수는 FizzBuzz 출력


```python
for i in range(1, 101):
    if (i % 3 == 0) and (i % 5 == 0):
        print('FizzBuzz')    
    elif i % 3 == 0: # 3의 배수
        print('Fizz')        
    elif i % 5 == 0: # 5의 배수
        print('Buzz')
    else:
        print(i)
```

    1
    2
    Fizz
    4
    Buzz
    Fizz
    7
    8
    Fizz
    Buzz
    11
    Fizz
    13
    14
    FizzBuzz
    16
    17
    Fizz
    19
    Buzz
    Fizz
    22
    23
    Fizz
    Buzz
    26
    Fizz
    28
    29
    FizzBuzz
    31
    32
    Fizz
    34
    Buzz
    Fizz
    37
    38
    Fizz
    Buzz
    41
    Fizz
    43
    44
    FizzBuzz
    46
    47
    Fizz
    49
    Buzz
    Fizz
    52
    53
    Fizz
    Buzz
    56
    Fizz
    58
    59
    FizzBuzz
    61
    62
    Fizz
    64
    Buzz
    Fizz
    67
    68
    Fizz
    Buzz
    71
    Fizz
    73
    74
    FizzBuzz
    76
    77
    Fizz
    79
    Buzz
    Fizz
    82
    83
    Fizz
    Buzz
    86
    Fizz
    88
    89
    FizzBuzz
    91
    92
    Fizz
    94
    Buzz
    Fizz
    97
    98
    Fizz
    Buzz
    

# 리스트 응용하기

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
a.index(40)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_10568\4247496908.py in <module>
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




    [1, 2, 3]




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



**반복문으로 리스트 요소 출력하가ㅣ**

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
    if i % 2 == 0:f
        result.append(i)
result        
```




    [0, 2, 4, 6, 8]



# Workshop

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



# 튜플 응용하기

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



# Workshop

**자동 로또 번호 생성기**
- 1~45 숫자 중에서 6개를 고르는 로또 번호를 자동으로 만들어 주는 프로그램 작성하기
- 사용자가 입력한 개수만큼 번호 쌍을 생성하기(예: 5를 입력하면 5 세트의 번호가 생성되도록 하기)
- 한번 뽑히것은 뽑히지 않도록 하고, 최종 출력은 오름차순 정렬해서 보여주기

```
입출력 예시
로또 몇회 뽑으시겠습니까? : 5
[[1, 6, 17, 21, 34, 39],
 [6, 10, 17, 32, 36, 45],
 [10, 16, 17, 20, 24, 33],
 [3, 9, 17, 25, 34, 40],
 [8, 14, 29, 31, 39, 45]]
```


```python
import random

count = int(input("로또 몇회 뽑으시겠습니까? : "))

total = []
for _ in range(count):
    lotto = []
    while True:
        pick = random.randint(1, 45)
        if pick not in lotto:
            lotto.append(pick)

        if len(lotto) == 6:
            break
    lotto.sort()
    total.append(lotto)

total
```

    로또 몇회 뽑으시겠습니까? : 5
    




    [[4, 7, 9, 11, 29, 45],
     [2, 4, 5, 11, 15, 17],
     [2, 10, 13, 30, 32, 40],
     [2, 6, 17, 21, 22, 42],
     [7, 12, 13, 21, 30, 44]]




```python
import random

count = int(input("로또 몇회 뽑으시겠습니까? : "))

total = []
for _ in range(count):
    lotto = []
    lotto = random.sample(range(1, 46), 6) # sample 함수는 unique한 요소를 뽑아줌
    lotto.sort()
    total.append(lotto)

total
```

    로또 몇회 뽑으시겠습니까? : 5
    




    [[10, 15, 16, 18, 23, 36],
     [10, 12, 29, 33, 36, 42],
     [3, 5, 11, 25, 26, 44],
     [21, 22, 25, 29, 35, 42],
     [8, 10, 21, 38, 39, 42]]



# 문자열 응용하기

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
s.index('pppp')
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_10568\1337320204.py in <module>
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



# 딕셔너리 응용하기

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




    {'alpha': 10, 'bravo': 20, 'charlie': 30, 'delta': 40}




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




    10




```python
x.get('alphaaa', 0)
```




    0




```python
x
```




    {'alpha': 10, 'bravo': 20, 'charlie': 30, 'delta': 40}



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
    

# 세트

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



# Workshop

**리스트와 반복문**
- 아래에 주어진 리스트 l1의 요소 중에서 l2의 요소와 값이 같은 경우 삭제하는 파이썬 코드 작성하기

```
l1 = ['a', 'b', 'c', 'd', 'a', 'b', 'a', 'b']
l2 = ['b', 'a']

결과 l1
['c', 'd']

```


```python
# option 1
l1 = ['a', 'b', 'c', 'd', 'a', 'b', 'a', 'b']
l2 = ['b', 'a']

list(set(l1) - set(l2))
```




    ['c', 'd']




```python
# option 2
l1 = ['a', 'b', 'c', 'd', 'a', 'b', 'a', 'b']
l2 = ['b', 'a']
for c in l2: # 'b'->'a'
    while c in l1:
        l1.remove(c)
        # print(l1)
print(l1)        
```

    ['c', 'd']
    


```python
# option 3
l1 = ['a', 'b', 'c', 'd', 'a', 'b', 'a', 'b']
l2 = ['b', 'a']

[i for i in l1 if i not in l2]
```




    ['c', 'd']



**'the'의 개수 출력하기**
- 아래 문자열에서 'the'의 개수를 출력하는 프로그램을 만드세요. 단 , 모든 문자가 소문자인 'the'만 찾으면 되며 'them', 'there', 'their' 등은 포함되지 않아야 합니다.

the grown-ups' response, this time, was to advise me to lay aside my drawings of boa constrictors, whether from the inside or the outside, and devote myself instead to geography, history, arithmetic, and grammar. That is why, at the, age of six, I gave up what might have been a magnificent career as a painter. I had been disheartened by the failure of my Drawing Number One and my Drawing Number Two. Grown-ups never understand anything by themselves, and it is tiresome for children to be always and forever explaining things to the.


```python
paragraph = "the grown-ups' response, this time, was to advise me to lay aside my drawings of boa constrictors, whether from the inside or the outside, and devote myself instead to geography, history, arithmetic, and grammar. That is why, at the, age of six, I gave up what might have been a magnificent career as a painter. I had been disheartened by the failure of my Drawing Number One and my Drawing Number Two. Grown-ups never understand anything by themselves, and it is tiresome for children to be always and forever explaining things to the."
```


```python
word_list = paragraph.split()
```


```python
count = 0
for word in word_list:
    if word.strip('.,') == 'the':
        count += 1
print(count)        
```

    6
    

**평균 점수 구하기**
- 다음 소스 코드를 완성하여 평균 점수가 출력되게 만드세요


```python
maria = {'korean' : 94, 'english': 91, 'mathmatics': 89, 'science': 83 }
```


```python
maria.values()
```




    dict_values([94, 91, 89, 83])




```python
total_v = 0
for v in maria.values():
    total_v += v
average = total_v / len(maria)    
print(average)
```

    89.25
    


```python
average = sum(maria.values())/len(maria)
print(average)
```

    89.25
    

**딕셔너리에서 특정값 삭제하기**
- 표준 입력으로 문자열 여러개와 숫자 여러개가 두 줄로 입력되고, 첫 번째 줄은 키, 두번째 줄은 값으로 하여 딕셔너리를 생성합니다. 다음 코드를 완성하여 딕셔너리에서 키가 'delta'인 키-값 쌍과 값이 30인 키-값 쌍을 삭제하도록 만드세요.
```
입력
alpha bravo charlie delta
10 20 30 40
결과
{'alpha':10, 'bravo':20}
```


```python
keys = input("key : ").split()
values = list(map(int, input('value : ').split()))
x = dict(zip(keys, values))
```

    key : alpha bravo charlie delta
    


```python
x
```




    {'alpha': 10, 'bravo': 20, 'charlie': 30, 'delta': 40}




```python
# 키가 delta 인 요소 삭제
x.pop('delta')
x
```




    {'alpha': 10, 'bravo': 20, 'charlie': 30}




```python
# 값이 30아닌 것만 새로운 딕셔너리에 추가
y = {}
for k, v in x.items():
    if v != 30:
        y.setdefault(k, v)
print(y)         
```

    {'alpha': 10, 'bravo': 20}
    

- [식 for 변수 in 시퀀스객체 if 조건식] <-- 리스트 표현식
- {키:값 for 키, 값 in 딕셔너리.items() if 조건식} <-- 딕셔너리 표현식


```python
x = {'alpha': 10, 'bravo': 20, 'charlie': 30, 'delta': 40}
```


```python
{k:v for k, v in x.items() if k!='delta' and v != 30}
```




    {'alpha': 10, 'bravo': 20}


