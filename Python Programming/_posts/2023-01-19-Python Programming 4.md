---
tag: [python, programming, 파이썬 문법, 기초코딩]
toc: true
toc_sticky: true
toc_label: 목차
author_profile: false
---
# 파일과 함수

## 파일 사용하기

- 파일객체 = open('파일이름', '파일모드')
- 파일객체.write('문자열')
- 파일객체.close()

**텍스트모드로 파일 읽고 저장하기**


```python
fd = open('hello.txt', 'w') # hello.txt 파일을 쓰기모드로 열기
fd.write('Hello World')
fd.close()
```

- 파일객체 = open('파일이름', '파일모드')
- 파일객체.read('문자열')
- 파일객체.close()


```python
fd = open('hello.txt', 'r')
s = fd.read()
print(s)
fd.close()
```

    Hello World
    

```
with open('파일이름', '파일모드') as 파일객체:
    코드
    ...
    ...
    ...

새로운코드 # 이때는 파일객체가 닫혀있음
```


```python
with open('hello.txt', 'r') as fd:
    s = fd.read()
    print(s)
```

    Hello World
    


```python
with open('hello.txt', 'w') as fd:
    for i in range(3):
        fd.write('Hello World\n')
```


```python
l = ['여기는 ', '플레이 데이터입니다. \n', '오늘도 ', '화이팅!!']

with open('play.txt', 'w', encoding='utf8') as fd:
    fd.writelines(l) # write() : 문자열을 기대, writelines() : 리스트를 기대
```


```python
with open('play.txt', 'r', encoding='utf8') as fd:
    lines = fd.readlines() # read() : 파일안에 있는 모든 '문자열'을 반환, readlines() : 리스트를 반환
lines    
```




    ['여기는 플레이 데이터입니다. \n', '오늘도 화이팅!!']




```python
with open('play.txt', 'r', encoding='utf8') as fd:
    line = fd.readline()
line    
```




    '여기는 플레이 데이터입니다. \n'




```python
with open('play.txt', 'r', encoding='utf8') as fd:
    line = None
    while line != '':
        line = fd.readline()
        print(line.strip('\n'))
```

    여기는 플레이 데이터입니다. 
    오늘도 화이팅!!
    
    


```python
# fd.read() : 파일에 있는 모든 데이터를 문자열로 반환
# fd.readline() : 파일에 있는 한줄 문자열을 반환
# fd.readlines() : 파일에 있는 여러줄의 문자열을 리스트 형태로 반환
```

**이진 모드로 파일 읽고 쓰기**


```python
import pickle
```


```python
scroes = {'korean' : 90, 'english' : 80} # 파이썬의 딕셔너리 객체
```


```python
scroes
```




    {'korean': 90, 'english': 80}




```python
with open('scores.pkl', 'wb') as fd:
    # pickle.dump(객체명, 파일객체)
    pickle.dump(scroes, fd)    
```


```python
with open('scores.pkl', 'rb') as fd:
    # pickle.load(파일객체)
    loaded_scores = pickle.load(fd)
```


```python
loaded_scores
```




    {'korean': 90, 'english': 80}




```python
name = 'James'
age = 17
scores = {'korean' : 90, 'english' : 80} 
```


```python
with open('student.pkl', 'wb') as fd:
    pickle.dump(name, fd) # 1
    pickle.dump(age, fd)  # 2
    pickle.dump(scores, fd)  # 3
```


```python
with open('student.pkl', 'rb') as fd:
    loaded_name = pickle.load(fd)
    loaded_age = pickle.load(fd)
    loaded_scores = pickle.load(fd)
```


```python
loaded_name
```




    'James'




```python
loaded_age
```




    17




```python
loaded_scores
```




    {'korean': 90, 'english': 80}




```python
from IPython.display import Image
Image('file_mode.png', width=400)
```




    
![png](/assets/images/2023-01-19-Python Programming 4/output_28_0.png)
    



## 함수

- 코드의 용도 구분
- 코드를 재사용
- 실수를 줄일 수 있음

```
def 함수이름():
    코드
```    


```python
# 함수 정의
def print_hello():
    print('hello')
```


```python
# 함수 호출
print_hello()
```

    hello
    

```
def 함수이름(매개변수1, 매개변수2, .....):
    코드
```   


```python
# 함수 정의
def print_hello(name):
    print('hello', name)
```


```python
# 함수 호출
print_hello('James')
```

    hello James
    


```python
print_hello('Selly')
```

    hello Selly
    


```python
# 함수 정의 : 덧셈의 결과를 출력하는 함수
def add(a, b):
    c = a + b
    print(c)
```


```python
# 함수 호출
add(5, 3)
```

    8
    


```python
add(8, 3)
```

    11
    

```
def 함수이름(매개변수1, 매개변수2, .....):
    코드
    return 반환값
```  


```python
# 함수 정의 : 덧셈의 결과를 반환하는 함수
def add(a, b):
    c = a + b
    return c
```


```python
result = add(5, 3)
result
```




    8



```
def 함수이름(매개변수1, 매개변수2, .....):
    코드
    return 반환값1, 반환값2
``` 


```python
# 함수 정의 : 덧셈의 결과와 뺄셈의 결과를 반환하는 함수
def add_sub(a, b):
    c = a + b
    d = a - b
    return c, d
```


```python
add_r, sub_r = add_sub(5, 3)
add_r, sub_r
```




    (8, 2)



**(참고) 딕셔너리 정렬**


```python
lst = [1, 4, 5, 6, -1, -10]
lst.sort() # 리스트에서 제공하는 sort() 메서드를 사용
```


```python
lst # 오름차순으로 정렬된 결과과 lst에 반영
```




    [-10, -1, 1, 4, 5, 6]




```python
lst = [1, 4, 5, 6, -1, -10]
result = sorted(lst)
result
```




    [-10, -1, 1, 4, 5, 6]




```python
lst # 오름차순으로 정렬된 결과가 lst에 반영되지 않음
```




    [1, 4, 5, 6, -1, -10]




```python
# 단어의 리스트
word_list = ['abc', 'abcd', 'lmn', 'opqrstu', 'z', 'wwwww']

sorted(word_list, reverse=True)
```




    ['z', 'wwwww', 'opqrstu', 'lmn', 'abcd', 'abc']




```python
len('abc'), len('abcd')
```




    (3, 4)




```python
# key 매개변수에 설정한 함수의 결과를 기준으로 정렬
sorted(word_list, key=len)
```




    ['z', 'abc', 'lmn', 'abcd', 'wwwww', 'opqrstu']




```python
# 딕셔너리 정렬
scores = {'James':90, 'Selly':80, 'Jun':100}
```


```python
def return_score(item):
    return item[1] # score
```


```python
scores.items()
```




    dict_items([('James', 90), ('Selly', 80), ('Jun', 100)])




```python
sorted(scores.items(), key=return_score, reverse=True)
```




    [('Jun', 100), ('James', 90), ('Selly', 80)]




```python
# 이름없는 함수 : lambda 식
lambda item:item[1]
```




    <function __main__.<lambda>(item)>




```python
sorted(scores.items(), key=lambda item:item[1], reverse=True)
```




    [('Jun', 100), ('James', 90), ('Selly', 80)]



**(참고) 단어리스트에서 개별 단어들을 strip 하려면..**


```python
def str_strip(x):
    return x.strip(',.')
```


```python
lambda x: x.strip(',.')
```




    <function __main__.<lambda>(x)>




```python
# option 1 : list 전체에 일괄 적용하는 map 함수 사용
word_list = ['i.', 'am,.', 'happy.,']
list(map(lambda x: x.strip(',.'), word_list))
```




    ['i', 'am', 'happy']




```python
# option 2 : list 표현식
word_list = ['i.', 'am,.', 'happy.,']
[word.strip(',.') for word in word_list]
```




    ['i', 'am', 'happy']



## 위치 인수와 키워드 인수


```python
# 함수 정의
def print_number(a, b, c):
    print(a)
    print(b)
    print(c)
```


```python
# 함수 호출
print_number(10, 20, 30)
```

    10
    20
    30
    


```python
lst = [10, 20, 30]
print_number(*lst) # *는 언패킹 해주는 연산자
```

    10
    20
    30
    


```python
# 가변 인수 함수 정의
def print_number(*args): # args 는 집합체인데, 원소의 개수가 가변적일때 unpacking해서 사용
    for arg in args:
        print(arg)
```


```python
print_number(10, 20)
```

    10
    20
    


```python
print_number(10, 20, 30)
```

    10
    20
    30
    


```python
print_number(10, 20, 30, 40)
```

    10
    20
    30
    40
    


```python
def personal_info(name, age, address):
    print(name)
    print(age)
    print(address)
```


```python
personal_info('조민호', 28, '가산동')
```

    조민호
    28
    가산동
    


```python
# 키워드 인수 사용하기
personal_info(name='조민호', age=28, address='가산동')
```

    조민호
    28
    가산동
    


```python
# 키워드 인수를 사용하게 되면 순서를 고려하지 않아도 됨
personal_info(age=28, address='가산동', name='조민호')
```

    조민호
    28
    가산동
    


```python
# 기본값 설정
def personal_info(name, age, address='가산동'):
    print(name)
    print(age)
    print(address)
```


```python
personal_info(age=28, name='조민호') # 기본값이 설정되어 있어서 address를 넣지 않아도 기본값 사용
```

    조민호
    28
    가산동
    


```python
personal_info(age=28, name='조민호', address='서초동') # 호출하는곳에서 설정한 address로 사용
```

    조민호
    28
    서초동
    


```python
# 딕셔너리를 사용하여 호출하는 경우
d = {'name': '조민호', 'age':28, 'address':'가산동'}
personal_info(*d) # 언패킹을 한번 했을 경우 키가 출력
```

    name
    age
    address
    


```python
personal_info(**d) # 언패킹을 두번 했을 경우 값이 출력
```

    조민호
    28
    가산동
    


```python
# 키워드 인수를 가변적으로 처리
def personal_info(**kwargs): # kwargs에는 가변적인 개수의 키-값 쌍이 들어가 있음
    for k, v in kwargs.items(): # kwargs는 딕셔너리로 간주
        print(k, v)
```


```python
personal_info(name='장경희')
```

    name 장경희
    


```python
personal_info(name='장경희', age=47)
```

    name 장경희
    age 47
    


```python
personal_info(name='장경희', age=47, address='세곡동')
```

    name 장경희
    age 47
    address 세곡동
    


```python
personal_info(name='장경희', age=47, address='세곡동', phone='0101234567')
```

    name 장경희
    age 47
    address 세곡동
    phone 0101234567
    


```python
personal_info(name='장경희', age=47, address='세곡동', phone='0101234567', etc='없음')
```

    name 장경희
    age 47
    address 세곡동
    phone 0101234567
    etc 없음
    

## 함수에서 재귀 호출 사용하기


```python
# 재귀 함수 호출시 스택 깊이 제한으로 오류 발생
# def hello():
#     print('hello world')
#     hello()

# hello()
```


```python
# 재귀 호출을 사용하여 hello world 5번 출력
def hello(count):
    if count == 0:
        return    
    print('hello world', count)
    count -= 1
    hello(count) # count : 4 -> 3 -> 2 -> 1 -> 0
hello(5) # count : 5
```

    hello world 5
    hello world 4
    hello world 3
    hello world 2
    hello world 1
    


```python
# 반복문을 사용하여 hello world 5번 출력
def hello(count):
    for i in range(count):
        print('hello world')
hello(5)        
```

    hello world
    hello world
    hello world
    hello world
    hello world
    


```python
# 10 + 9 + 8 + 7 .... + 1
```


```python
# 반복문을 사용하여 10~1 까지의 합 구하기
```


```python
def add_sum(number):
    total = 0
    for i in range(number, 0, -1):
        total = total + i
    return total
        
add_sum(10)
```




    55




```python
# 재귀호출을 사용하여 10~1 까지의 합 구하기
```


```python
def add_sum(number):
    if number == 1:
        return 1
    
    total = number + add_sum(number-1)
    return total
    
add_sum(10)    
```




    55




```python
# 팩토리얼 
# n! = n * (n-1) * (n-2) *....* 1
# 10! = 10*9*8*7*....*1
```


```python
# 반복문으로 팩토리얼 구현
def factorial(number):
    total = 1
    for i in range(number, 0, -1):
        total = total * i
    return total
factorial(5)    
```




    120




```python
# 재귀 호출로 팩토리얼 구현
def factorial(number):
    if number == 1:
        return 1
    
    total = number * factorial(number-1)
    return total
    
factorial(5)     
```




    120




```python
# 반복문으로로 구구단 출력 (2-9단까지)
```


```python
def gugu(x):
    for i in range(1, 10):
        print(x, '*', i, '=', x*i)

for dan in range(2, 10):
    print('----%d단----'%dan)
    gugu(dan)
```

    ----2단----
    2 * 1 = 2
    2 * 2 = 4
    2 * 3 = 6
    2 * 4 = 8
    2 * 5 = 10
    2 * 6 = 12
    2 * 7 = 14
    2 * 8 = 16
    2 * 9 = 18
    ----3단----
    3 * 1 = 3
    3 * 2 = 6
    3 * 3 = 9
    3 * 4 = 12
    3 * 5 = 15
    3 * 6 = 18
    3 * 7 = 21
    3 * 8 = 24
    3 * 9 = 27
    ----4단----
    4 * 1 = 4
    4 * 2 = 8
    4 * 3 = 12
    4 * 4 = 16
    4 * 5 = 20
    4 * 6 = 24
    4 * 7 = 28
    4 * 8 = 32
    4 * 9 = 36
    ----5단----
    5 * 1 = 5
    5 * 2 = 10
    5 * 3 = 15
    5 * 4 = 20
    5 * 5 = 25
    5 * 6 = 30
    5 * 7 = 35
    5 * 8 = 40
    5 * 9 = 45
    ----6단----
    6 * 1 = 6
    6 * 2 = 12
    6 * 3 = 18
    6 * 4 = 24
    6 * 5 = 30
    6 * 6 = 36
    6 * 7 = 42
    6 * 8 = 48
    6 * 9 = 54
    ----7단----
    7 * 1 = 7
    7 * 2 = 14
    7 * 3 = 21
    7 * 4 = 28
    7 * 5 = 35
    7 * 6 = 42
    7 * 7 = 49
    7 * 8 = 56
    7 * 9 = 63
    ----8단----
    8 * 1 = 8
    8 * 2 = 16
    8 * 3 = 24
    8 * 4 = 32
    8 * 5 = 40
    8 * 6 = 48
    8 * 7 = 56
    8 * 8 = 64
    8 * 9 = 72
    ----9단----
    9 * 1 = 9
    9 * 2 = 18
    9 * 3 = 27
    9 * 4 = 36
    9 * 5 = 45
    9 * 6 = 54
    9 * 7 = 63
    9 * 8 = 72
    9 * 9 = 81
    


```python
# 재귀 호출로 구구단 출력 (2-9단까지)
```


```python
def gugu(i):
    print(2, '*', i, '=', 2*i)
    if i >= 9:
        return 
    gugu(i+1)
    
gugu(1)    
```

    2 * 1 = 2
    2 * 2 = 4
    2 * 3 = 6
    2 * 4 = 8
    2 * 5 = 10
    2 * 6 = 12
    2 * 7 = 14
    2 * 8 = 16
    2 * 9 = 18
    


```python
def gugu(dan, i):
    print(dan, '*', i, '=', dan*i)
    if i >= 9:
        return 
    gugu(dan, i+1)

for dan in range(2, 10):
    print('----%d단----'%dan)
    gugu(dan, 1) 
```

    ----2단----
    2 * 1 = 2
    2 * 2 = 4
    2 * 3 = 6
    2 * 4 = 8
    2 * 5 = 10
    2 * 6 = 12
    2 * 7 = 14
    2 * 8 = 16
    2 * 9 = 18
    ----3단----
    3 * 1 = 3
    3 * 2 = 6
    3 * 3 = 9
    3 * 4 = 12
    3 * 5 = 15
    3 * 6 = 18
    3 * 7 = 21
    3 * 8 = 24
    3 * 9 = 27
    ----4단----
    4 * 1 = 4
    4 * 2 = 8
    4 * 3 = 12
    4 * 4 = 16
    4 * 5 = 20
    4 * 6 = 24
    4 * 7 = 28
    4 * 8 = 32
    4 * 9 = 36
    ----5단----
    5 * 1 = 5
    5 * 2 = 10
    5 * 3 = 15
    5 * 4 = 20
    5 * 5 = 25
    5 * 6 = 30
    5 * 7 = 35
    5 * 8 = 40
    5 * 9 = 45
    ----6단----
    6 * 1 = 6
    6 * 2 = 12
    6 * 3 = 18
    6 * 4 = 24
    6 * 5 = 30
    6 * 6 = 36
    6 * 7 = 42
    6 * 8 = 48
    6 * 9 = 54
    ----7단----
    7 * 1 = 7
    7 * 2 = 14
    7 * 3 = 21
    7 * 4 = 28
    7 * 5 = 35
    7 * 6 = 42
    7 * 7 = 49
    7 * 8 = 56
    7 * 9 = 63
    ----8단----
    8 * 1 = 8
    8 * 2 = 16
    8 * 3 = 24
    8 * 4 = 32
    8 * 5 = 40
    8 * 6 = 48
    8 * 7 = 56
    8 * 8 = 64
    8 * 9 = 72
    ----9단----
    9 * 1 = 9
    9 * 2 = 18
    9 * 3 = 27
    9 * 4 = 36
    9 * 5 = 45
    9 * 6 = 54
    9 * 7 = 63
    9 * 8 = 72
    9 * 9 = 81
    
## Reference
[파이썬 코딩 도장](https://dojang.io/course/view.php?id=7)