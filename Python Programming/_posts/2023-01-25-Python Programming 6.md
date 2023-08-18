---
tag: [python, 기초코딩]
---
# 예외처리하기, 이터레이터

## 예외 처리하기

- 예외 처리는 에러가 발생하더라도 스크립트 실행을 중단하지 않고 계속 실행하고자 할 때 사용함

```
try:
    실행할 코드
except:
    예외가 발생했을 때 처리할 코드    
```    


```python
x = int(input("나눌 숫자를 입력하세요 : "))

y = 10/x
print(y)
```

    나눌 숫자를 입력하세요 : 0
    


    ---------------------------------------------------------------------------

    ZeroDivisionError                         Traceback (most recent call last)

    <ipython-input-1-4cfe023edecd> in <module>
          1 x = int(input("나눌 숫자를 입력하세요 : "))
          2 
    ----> 3 y = 10/x
          4 print(y)
    

    ZeroDivisionError: division by zero



```python
try:
    x = int(input("나눌 숫자를 입력하세요 : "))
    y = 10/x
    print(y) 
except:
    print("예외가 발생했습니다.")    
```

    나눌 숫자를 입력하세요 : 0
    예외가 발생했습니다.
    

```
# 특정 예외만 처리하기
try:
    실행할 코드
except 예외이름1:
    예외가 발생했을 때 처리할 코드    
except 예외이름2:    
    예외가 발생했을 때 처리할 코드  
```  


```python
y = [10, 20 , 30]

index = int(input("인덱스를 입력해주세요 : "))
r = y[index]/5
print(r)
```

    인덱스를 입력해주세요 : 4
    


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-3-7afbc2e37935> in <module>
          2 
          3 index = int(input("인덱스를 입력해주세요 : "))
    ----> 4 r = y[index]/5
          5 print(r)
    

    IndexError: list index out of range



```python
y = [10, 20 , 30]

index, x = map(int, input("인덱스와 나눌 숫자를 입력해주세요 : ").split())
r = y[index]/x
print(r)
```

    인덱스와 나눌 숫자를 입력해주세요 : 4 0
    


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-4-691609081ed3> in <module>
          2 
          3 index, x = map(int, input("인덱스와 나눌 숫자를 입력해주세요 : ").split())
    ----> 4 r = y[index]/x
          5 print(r)
    

    IndexError: list index out of range



```python
try:
    y = [10, 20 , 30]

    index, x = map(int, input("인덱스와 나눌 숫자를 입력해주세요 : ").split())
    r = y[index]/x
    print(r)
except ZeroDivisionError:
    print("0으로 나눌 수 없습니다.")
except IndexError:
    print("잘못된 인덱스입니다.")    
```

    인덱스와 나눌 숫자를 입력해주세요 : 2 0
    0으로 나눌 수 없습니다.
    

```
# 예외 메세지를 변수로 받기
try:
    실행할 코드
except 예외이름 as 변수:
    예외가 발생했을 때 처리할 코드    
```  


```python
try:
    y = [10, 20 , 30]

    index, x = map(int, input("인덱스와 나눌 숫자를 입력해주세요 : ").split())
    r = y[index]/x
    print(r)
except ZeroDivisionError as e:
    print(e)
except IndexError as e:
    print(e)  
```

    인덱스와 나눌 숫자를 입력해주세요 : 4 0
    list index out of range
    

```
# 예외가 발생하지 않았을 때 처리
try:
    실행할 코드
except 예외이름 as 변수:
    예외가 발생했을 때 처리할 코드    
else:
    예외가 발생하지 않았을 때 실행할 코드
```  


```python
try:
    y = [10, 20 , 30]

    index, x = map(int, input("인덱스와 나눌 숫자를 입력해주세요 : ").split())
    r = y[index]/x    
except ZeroDivisionError as e:
    print(e)
except IndexError as e:
    print(e) 
else:
    print(r)
```

    인덱스와 나눌 숫자를 입력해주세요 : 3 0
    list index out of range
    

```
# 예외가 발생하지 않았을 때 처리
try:
    실행할 코드
except 예외이름 as 변수:
    예외가 발생했을 때 처리할 코드    
else:
    예외가 발생하지 않았을 때 실행할 코드
finally:
    예외 발생 여부와 상관없이 항상 실행할 코드
```  


```python
try:
    y = [10, 20 , 30]

    index, x = map(int, input("인덱스와 나눌 숫자를 입력해주세요 : ").split())
    r = y[index]/x    
except ZeroDivisionError as e:
    print(e)
except IndexError as e:
    print(e) 
else:
    print(r)
finally:
    print('프로그램 실행이 끝났습니다.')
```

    인덱스와 나눌 숫자를 입력해주세요 : 3 0
    list index out of range
    프로그램 실행이 끝났습니다.
    

```
# 예외 발생시키기
raise Exception("에러메세지")
```


```python
try:
    x = int(input("3의 배수를 입력하세요."))
    if (x % 3) != 0: # 3으로 나눴을 때 나머지가 0이 아님, 3의 배수가 아님
        raise Exception("3의 배수가 아닙니다")
except Exception as e:    
    print("예외가 발생했습니다.", e)

```

    3의 배수를 입력하세요.4
    예외가 발생했습니다. 3의 배수가 아닙니다
    

## 이터레이터

- **이터레이터(iterator)**는 값을 차례대로 꺼낼 수 있는 객체
- 파이썬에서는 이터레이터만 생성하고 값이 '필요한 시점'이 되었을 때 값을 만드는 방식을 사용
- 데이터 생성을 뒤로 미루는 방식을 지연 평가(lazy evaluation)라고 함

- **반복 가능한 객체**는 말 그대로 반복할 수 있는 개체인데 문자열, 리스트, 딕셔너리, 세트가 그 예임
- 객체 안에 요소가 여러개 들어있고, 한 번에 하나씩 꺼낼 수 있는 객체

- 객체가 반복 가능한 객체인이 알아보는 방법은 객체의 ```__iter__()``` 메서드가 들어 있는지 확인해 보면 됨


```python
lst = [10, 20, 30]
lst
```




    [10, 20, 30]




```python
dir(lst)
```




    ['__add__',
     '__class__',
     '__contains__',
     '__delattr__',
     '__delitem__',
     '__dir__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__getitem__',
     '__gt__',
     '__hash__',
     '__iadd__',
     '__imul__',
     '__init__',
     '__init_subclass__',
     '__iter__',
     '__le__',
     '__len__',
     '__lt__',
     '__mul__',
     '__ne__',
     '__new__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__reversed__',
     '__rmul__',
     '__setattr__',
     '__setitem__',
     '__sizeof__',
     '__str__',
     '__subclasshook__',
     'append',
     'clear',
     'copy',
     'count',
     'extend',
     'index',
     'insert',
     'pop',
     'remove',
     'reverse',
     'sort']




```python
# 반복가능한 객체에서만 __iter__()함수가 나타남
# i = 10
# dir(i)
```


```python
# 반복가능한 객체를 이터레이터로 만들기
lst_it = lst.__iter__()
```


```python
lst_it # 이터레이터이므로 값을 바로 꺼내 볼 수는 없음
```




    <list_iterator at 0x26cbc987bb0>




```python
lst # 반복 가능한 객체이므로 바로 값이 확인이 됨
```




    [10, 20, 30]




```python
dir(lst_it)
```




    ['__class__',
     '__delattr__',
     '__dir__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__gt__',
     '__hash__',
     '__init__',
     '__init_subclass__',
     '__iter__',
     '__le__',
     '__length_hint__',
     '__lt__',
     '__ne__',
     '__new__',
     '__next__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__setattr__',
     '__setstate__',
     '__sizeof__',
     '__str__',
     '__subclasshook__']




```python
lst_it.__next__()
```




    10




```python
lst_it.__next__()
```




    20




```python
lst_it.__next__()
```




    30




```python
lst_it.__next__() # 값이 더이상 없을 때 StopIteration 예외 발생
```


    ---------------------------------------------------------------------------

    StopIteration                             Traceback (most recent call last)

    <ipython-input-20-b12096e9e653> in <module>
    ----> 1 lst_it.__next__() # 값이 더이상 없을 때 StopIteration 예외 발생
    

    StopIteration: 



```python
lst_it = lst.__iter__()
for i in lst_it: # 이터레이터의 경우에는 필요한 시점에 값을 가져다 사용, 값을 가져올 때마다 __next__() 함수가 호출
    print(i)
```

    10
    20
    30
    


```python
# 문자열
s = 'Hello World!'
dir(s)
```




    ['__add__',
     '__class__',
     '__contains__',
     '__delattr__',
     '__dir__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__getitem__',
     '__getnewargs__',
     '__gt__',
     '__hash__',
     '__init__',
     '__init_subclass__',
     '__iter__',
     '__le__',
     '__len__',
     '__lt__',
     '__mod__',
     '__mul__',
     '__ne__',
     '__new__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__rmod__',
     '__rmul__',
     '__setattr__',
     '__sizeof__',
     '__str__',
     '__subclasshook__',
     'capitalize',
     'casefold',
     'center',
     'count',
     'encode',
     'endswith',
     'expandtabs',
     'find',
     'format',
     'format_map',
     'index',
     'isalnum',
     'isalpha',
     'isascii',
     'isdecimal',
     'isdigit',
     'isidentifier',
     'islower',
     'isnumeric',
     'isprintable',
     'isspace',
     'istitle',
     'isupper',
     'join',
     'ljust',
     'lower',
     'lstrip',
     'maketrans',
     'partition',
     'replace',
     'rfind',
     'rindex',
     'rjust',
     'rpartition',
     'rsplit',
     'rstrip',
     'split',
     'splitlines',
     'startswith',
     'strip',
     'swapcase',
     'title',
     'translate',
     'upper',
     'zfill']




```python
s_it = s.__iter__() # s_it = iter(s) 와 같은 결과
s_it # __iter__ 이 있음
```




    <str_iterator at 0x26cbc987910>




```python
dir(s_it) # __next__ 이 있음
```




    ['__class__',
     '__delattr__',
     '__dir__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__gt__',
     '__hash__',
     '__init__',
     '__init_subclass__',
     '__iter__',
     '__le__',
     '__length_hint__',
     '__lt__',
     '__ne__',
     '__new__',
     '__next__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__setattr__',
     '__setstate__',
     '__sizeof__',
     '__str__',
     '__subclasshook__']




```python
s_it.__next__() # next(s_it)와 같은 결과
```




    'H'




```python
s_it.__next__()
```




    'e'




```python
s_it.__next__()
```




    'l'




```python
s_it = s.__iter__()
for i in s_it: # s_it로 순회를 하더라도 내부적으로 __next__가 호출되서 개별 요소를 가져옴
    print(i, end='')
```

    Hello World!


```python
# 딕셔너리
d = {'a':1, 'b':2}
```


```python
dir(d) # __iter__ 가 있음
```




    ['__class__',
     '__contains__',
     '__delattr__',
     '__delitem__',
     '__dir__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__getitem__',
     '__gt__',
     '__hash__',
     '__init__',
     '__init_subclass__',
     '__iter__',
     '__le__',
     '__len__',
     '__lt__',
     '__ne__',
     '__new__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__reversed__',
     '__setattr__',
     '__setitem__',
     '__sizeof__',
     '__str__',
     '__subclasshook__',
     'clear',
     'copy',
     'fromkeys',
     'get',
     'items',
     'keys',
     'pop',
     'popitem',
     'setdefault',
     'update',
     'values']




```python
d_it = iter(d) # d.__iter__() 와 같은 결과
```


```python
dir(d_it) # __next__가 있음
```




    ['__class__',
     '__delattr__',
     '__dir__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__gt__',
     '__hash__',
     '__init__',
     '__init_subclass__',
     '__iter__',
     '__le__',
     '__length_hint__',
     '__lt__',
     '__ne__',
     '__new__',
     '__next__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__setattr__',
     '__sizeof__',
     '__str__',
     '__subclasshook__']




```python
d_it
```




    <dict_keyiterator at 0x26cbc999db0>




```python
next(d_it) # d.__next__() 와 같은 결과
```




    'a'




```python
next(d_it)
```




    'b'




```python
next(d_it)
```


    ---------------------------------------------------------------------------

    StopIteration                             Traceback (most recent call last)

    <ipython-input-36-31659235bc8d> in <module>
    ----> 1 next(d_it)
    

    StopIteration: 



```python
# 딕셔너리의 key, value 다 가져오기
d_it = iter(d.items())
d_it
```




    <dict_itemiterator at 0x26cbc9946d0>




```python
# 딕셔너리의 key만 가져오기
d_it = iter(d.keys())
d_it
```




    <dict_keyiterator at 0x26cbc99d540>




```python
# 딕셔너리의 value만 가져오기
d_it = iter(d.values())
d_it
```




    <dict_valueiterator at 0x26cbc5d48b0>




```python
d_it = iter(d.items())
d_it
```




    <dict_itemiterator at 0x26cbc99dcc0>




```python
next(d_it)
```




    ('a', 1)




```python
d_it = iter(d.items())
for k, v in d_it:
    print(k, v)
```

    a 1
    b 2
    

**이터레이터 만들기 1** (iterable-style)

```
class 이터레이터이름:
    def __iter__(self):
        코드
    def __next__(self):
        코드
```        


```python
range(10)
```




    range(0, 10)




```python
# 파이썬에서 제공하는 range()와 유사한 이터레이터 myrange() 만들기

class myrange:
    def __init__(self, stop):
        print("myrange is initialized", stop)
        self.current = 0
        self.stop = stop
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current < self.stop:
            r = self.current
            self.current += 1
            return r
        else: # self.current >= self.stop:
            raise StopIteration
```

- range 사용예 


```python
r = range(10)
```


```python
dir(r) # __iter__가 있으므로 반복가능한 객체
```




    ['__bool__',
     '__class__',
     '__contains__',
     '__delattr__',
     '__dir__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__getitem__',
     '__gt__',
     '__hash__',
     '__init__',
     '__init_subclass__',
     '__iter__',
     '__le__',
     '__len__',
     '__lt__',
     '__ne__',
     '__new__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__reversed__',
     '__setattr__',
     '__sizeof__',
     '__str__',
     '__subclasshook__',
     'count',
     'index',
     'start',
     'step',
     'stop']




```python
r_it = iter(r) # r.__iter__() 와 같은 결과
dir(r_it) # __next__ 가 있음
```




    ['__class__',
     '__delattr__',
     '__dir__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__gt__',
     '__hash__',
     '__init__',
     '__init_subclass__',
     '__iter__',
     '__le__',
     '__length_hint__',
     '__lt__',
     '__ne__',
     '__new__',
     '__next__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__setattr__',
     '__setstate__',
     '__sizeof__',
     '__str__',
     '__subclasshook__']




```python
next(r_it)
```




    0




```python
next(r_it)
```




    1




```python
next(r_it)
```




    2




```python
# 원래 range에는 색인 기능이 지원
range(10)[0]
```




    0




```python
range(10)[9]
```




    9




```python
range(10)[10]
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-53-0e54eb2a55f8> in <module>
    ----> 1 range(10)[10]
    

    IndexError: range object index out of range


- myrange 확인


```python
r = myrange(3) 
```

    myrange is initialized 3
    


```python
r2 = myrange(20)
```

    myrange is initialized 20
    


```python
type(r)
```




    __main__.myrange




```python
dir(r) # myrange로 만들어진 r 인스터는 __iter__, __next__ 
```




    ['__class__',
     '__delattr__',
     '__dict__',
     '__dir__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__gt__',
     '__hash__',
     '__init__',
     '__init_subclass__',
     '__iter__',
     '__le__',
     '__lt__',
     '__module__',
     '__ne__',
     '__new__',
     '__next__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__setattr__',
     '__sizeof__',
     '__str__',
     '__subclasshook__',
     '__weakref__',
     'current',
     'stop']




```python
r.__next__()
```




    0




```python
r.__next__()
```




    1




```python
r.__next__()
```




    2




```python
r.__next__()
```


    ---------------------------------------------------------------------------

    StopIteration                             Traceback (most recent call last)

    <ipython-input-61-3a998f5f1be5> in <module>
    ----> 1 r.__next__()
    

    <ipython-input-44-2e47fad56a10> in __next__(self)
         16             return r
         17         else: # self.current >= self.stop:
    ---> 18             raise StopIteration
    

    StopIteration: 



```python
# myrange는 range와 다르게 색인 기능이 지원되지 않음
myrange(10)[0]
```

    myrange is initialized 10
    


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-62-6d24f963bcc0> in <module>
          1 # myrange는 range와 다르게 색인 기능이 지원되지 않음
    ----> 2 myrange(10)[0]
    

    TypeError: 'myrange' object is not subscriptable


**이터레이터 만들기 2** (map-style)

```
class 이터레이터이름:
    def __getitem__(self, 인덱스):
        코드
```  


```python
class myrange:
    def __init__(self, stop):
        self.stop = stop
        
    def __getitem__(self, index):
        if index < self.stop:
            return index
        
        else: # index >= self.stop:
            raise IndexError
    
```


```python
r = myrange(10)
```


```python
r[1] # r.__getitem__(1)
```




    1




```python
r[3] # r.__getitem__(3)
```




    3




```python
r[10] # r.__getitem__(10)
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-67-3df2bca72313> in <module>
    ----> 1 r[10] # r.__getitem__(10)
    

    <ipython-input-63-f3109e6b02d2> in __getitem__(self, index)
          8 
          9         else: # index >= self.stop:
    ---> 10             raise IndexError
         11 
    

    IndexError: 



```python
for i in range(10):
    print(i)
```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    


```python
for i in myrange(10): # 내부적으로 __getitem()__가 호출되서 순회
    print(i)
```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    

## 제너레이터

- 이터레이터를 만들어주는 또 다른 방식(**함수**를 사용)
- 제너레이터는 이터레이터를 생성해 주는 함수
- 앞서 만든 이터레이터는 클래스 안에 `__iter__()`, `__next__()`, `__getitem()__` 메서들을 구현해야 했지만
- 제너레이터는 함수 안에서 yield 라는 키워드만 사용하면 간단히 작성할 수 있음


```python
def myrange2():
    yield 0
    yield 1
    yield 2
    yield 3
    yield 4
    yield 5
    yield 6
    yield 7
    yield 8
    yield 9
```


```python
for i in myrange2():
    print(i)
```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    

## 모듈과 패키지

**모듈을 가져오는 방법**

- import 모듈
- import 모듈1, 모듈2, ....

**모듈을 사용하는 방법**
- 모듈.변수
- 모듈.함수()
- 모듈.클래스()


```python
import math
```


```python
# 모듈.함수()
math.sqrt(5)
```




    2.23606797749979




```python
# 모듈.변수
math.pi
```




    3.141592653589793



- import 모듈 as 별칭


```python
import math as m
```


```python
m.sqrt(5)
```




    2.23606797749979




```python
m.pi
```




    3.141592653589793




```python
# 참고 (데이터 분석에서 사용하는 모듈)
import numpy as np # 관례적으로 사용
import pandas as pd
import seaborn as sns
```

- from 모듈 import 변수
- from 모듈 import 함수
- from 모듈 import 클래스


```python
from math import sqrt
```


```python
sqrt(4)
```




    2.0




```python
from math import pi
pi
```




    3.141592653589793



- from 모듈 import 변수, 함수, 클래스....


```python
from math import sqrt, pi
```


```python
sqrt(4)
```




    2.0




```python
pi
```




    3.141592653589793




```python
from math import *
```


```python
sqrt(4)
```




    2.0




```python
pi
```




    3.141592653589793



- from 모듈 import 변수 as 별칭


```python
from math import pi as PI
```


```python
PI
```




    3.141592653589793



**패키지 안의 모듈 가져오는 방법**
- import 패키지.모듈
- import 패키지.모듈1, 모듈2,...

**패캐지 안의 모듈 사용하는 방법**
- 패키지.모듈.변수
- 패키지.모듈.함수()
- 패키지.모듈.클래스()


```python
import urllib.request
```


```python
response = urllib.request.urlopen("http://www.google.co.kr")
```


```python
response.status
```




    200



- import 패키지.모듈 as 별칭


```python
import urllib.request as r
```


```python
response = r.urlopen("http://www.google.co.kr")
response.status
```




    200



- from 패키지.모듈 import 변수
- from 패키지.모듈 import 함수
- from 패키지.모듈 import 클래스
- from 패키지.모듈 import 변수, 함수, 클래스


```python
from urllib.request import urlopen
```


```python
response = urlopen('http://google.co.kr')
response.status
```




    200



- from 패키지.모듈 import *


```python
response = urlopen('http://google.co.kr')
response.status
```




    200



**나만의 모듈 만들어서 사용하기**


```python
import math2
```


```python
math2.base # 모듈.변수
```




    2




```python
math2.square2(10) # 2**10을 계산
```




    1024




```python
from math2 import square2
```


```python
square2(10)
```




    1024




```python
import person
```


```python
hong = person.Person("홍길동", 17, "서초동")
```


```python
hong.greeting()
```

    안녕하세요
    저는 홍길동입니다.
    나이는 17살입니다.
    


```python
# %load hello2.py
print('hello2 모듈 시작')
print(__name__)
print('hello2 모듈 끝')

```

    hello2 모듈 시작
    __main__
    hello2 모듈 끝
    


```python
# hello2.py의 __name__ 에 __main__이 출력
%run hello2.py  
```

    hello2 모듈 시작
    __main__
    hello2 모듈 끝
    


```python
# hello2.py의 __name__에 hello2가 출력
# test.py의 __name__에 __main__이 출력
%run test.py 
```

    hello2 모듈 시작
    hello2
    hello2 모듈 끝
    test 모듈 시작
    __main__
    test 모듈 끝
    


```python
# 실행의 시작이 되는 파이썬 스크립트의 __name__은 __main__이 됨을 알 수 있음
# 모듈의 역할을 하는 파이썬 스크립트는 __name__이 모듈 이름이 됨
```


```python
# case 1 : calc.py가 모듈로 사용
import calc
```


```python
# case 2 : calc.py가 실행의 시작이 되는 파이썬 스크립트
%run calc.py
```

    30
    200
    


```python
import calcpkg.operation # 패키지.모듈
```


```python
calcpkg.operation.add(10, 20)
```




    30




```python
import calcpkg.geometry
```


```python
calcpkg.geometry.triangle_area(10, 20)
```




    100.0




```python
from calcpkg.operation import add, mul
```


```python
add(10, 20)
```




    30




```python
mul(10, 20)
```




    200




```python
from calcpkg.geometry import *
```


```python
triangle_area(10, 20)
```




    100.0




```python
rectangle_area(10, 20)
```




    200



## 정규표현식


```python
import re
```


```python
p = re.compile('ab*c') # a로 시작해서 b가 0개 이상이면 만족하는 패턴
p
```




    re.compile(r'ab*c', re.UNICODE)




```python
p.match('abc') # 매치가 잘 되었을 때
```




    <re.Match object; span=(0, 3), match='abc'>




```python
result = p.match('ab') # 매치가 잘 되지 않았을 때
print(result)
```

    None
    


```python
p = re.compile('[a-z]+')
p.match('python')
```




    <re.Match object; span=(0, 6), match='python'>




```python
p = re.compile('[a-z]+')
p.match('python123')
```




    <re.Match object; span=(0, 6), match='python'>




```python
p = re.compile('[a-z]+')
p.match('3 python')
```


```python
p.search('3 python')
```




    <re.Match object; span=(2, 8), match='python'>




```python
p = re.compile('[a-z]+')
p.findall('life is too short')
```




    ['life', 'is', 'too', 'short']




```python
p = re.compile('[a-z]+')
result_it = p.finditer('life is too short')
```


```python
for r in result_it:
    print(r)
```

    <re.Match object; span=(0, 4), match='life'>
    <re.Match object; span=(5, 7), match='is'>
    <re.Match object; span=(8, 11), match='too'>
    <re.Match object; span=(12, 17), match='short'>
    


```python
p = re.compile('[a-z]+') # pattern 만들기
m = p.match('python') # pattern으로 새로운 데이터 매치하는지 확인하기
m # match 객체
```




    <re.Match object; span=(0, 6), match='python'>




```python
m.group() # 매치된 문자열
```




    'python'




```python
m.start() # 매치된 문자열의 첫번째 인덱스
```




    0




```python
m.end() # 매치된 문자열의 마지막 인덱스
```




    6




```python
m.span()
```




    (0, 6)



**(참고) 문자열에서 역슬래시('\') 찾기**


```python
# https://wikidocs.net/4308#_7 참고
import re

p = re.compile('[a-z]*\\\\[a-z]*') # 파이썬에서는 \\가 \로 인식, \\\\가 \\로 인식
print(p.match('a\\a'))

p = re.compile(r'[a-z]*\\[a-z]*')
print(p.match('a\\a'))
```

    <re.Match object; span=(0, 3), match='a\\a'>
    <re.Match object; span=(0, 3), match='a\\a'>
    
## Reference
[파이썬 코딩 도장](https://dojang.io/course/view.php?id=7)