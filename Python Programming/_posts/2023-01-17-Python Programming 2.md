# 문자열 사용하기


```python
s1 = 'hello'
s2 = "hello"
s3 = '''hello'''
s4 = """hello"""
type(s1), type(s2), type(s3), type(s4)
```




    (str, str, str, str)




```python
print(s1, s2, s3, s4)
```

    hello hello hello hello
    

hello 'python'


```python
s1 = "hello 'python'"
print(s1)
```

    hello 'python'
    

hello "python"


```python
s2 = 'hello "python"'
print(s2)
```

    hello "python"
    

hello 'python'


```python
s3 = '''hello 'python' '''
print(s3)
```

    hello 'python' 
    


```python
s4 = """hello 'python'"""
print(s4)
```

    hello 'python'
    


```python
s5 = '''python is a programming language
that lets you work quickly 
and integrate systems more effectively.
'''
print(s5)
```

    python is a programming language
    that lets you work quickly 
    and integrate systems more effectively.
    
    


```python
s6 = """python is a programming language
that lets you work quickly 
and integrate systems more effectively.
"""
print(s6)
```

    python is a programming language
    that lets you work quickly 
    and integrate systems more effectively.
    
    


```python
s6.count('\n')
```




    3



hello 'python'


```python
s7 = 'hello \'python\''
print(s7)
```

    hello 'python'
    

```
'Python' is a "programming language"
that lets you work quicky
and
integrate systems more effectively.
```


```python
s = """'Python' is a "programming language"
that lets you work quicky
and
integrate systems more effectively."""
print(s)
```

    'Python' is a "programming language"
    that lets you work quicky
    and
    integrate systems more effectively.
    

# 리스트와 튜플

- 리스트 = [값1, 값2, ....]


```python
scores = [90, 95, 100, 80, 70, 100]
scores
```




    [90, 95, 100, 80, 70, 100]




```python
type(scores)
```




    list




```python
person1 = ["James", 17, 175.3, True]
person1
```




    ['James', 17, 175.3, True]




```python
type(person1)
```




    list



**빈 리스트 만들기**

- 리스트 = []
- 리스트 = list()


```python
empty_list1 = []
empty_list1
```




    []




```python
type(empty_list1)
```




    list




```python
empty_list2 = list()
empty_list2
```




    []




```python
type(empty_list2)
```




    list



- range(횟수)


```python
range(10)
```




    range(0, 10)



- 리스트 = list(range(횟수))


```python
a = list(range(10))
a
```




    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]



- 리스트 = list(range(시작, 끝))


```python
b = list(range(0, 10))
b
```




    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]




```python
c = list(range(5, 10))
c
```




    [5, 6, 7, 8, 9]



- 리스트 = list(range(시작, 끝, 증가폭))


```python
d = list(range(0, 10, 2))
d
```




    [0, 2, 4, 6, 8]




```python
e = list(range(10, 0, -2))
e
```




    [10, 8, 6, 4, 2]



- 튜플 = (값1, 값2, ....)
- 튜플 = 값1, 값2, ...


```python
a = (1, 2, 3, 4)
a
```




    (1, 2, 3, 4)




```python
type(a)
```




    tuple




```python
b = 1, 2, 3, 4
b
```




    (1, 2, 3, 4)




```python
type(b)
```




    tuple




```python
c = (1,)
c
```




    (1,)




```python
type(c)
```




    tuple



- 튜플 = tuple(range(횟수))


```python
a = tuple(range(10))
a
```




    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)



- 튜플 = tuple(range(시작, 끝))


```python
b = tuple(range(0, 10))
b
```




    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)



- 튜플 = tuple(range(시작, 끝, 증가폭))


```python
c = tuple(range(0, 10, 3))
c
```




    (0, 3, 6, 9)




```python
# 튜플을 리스트로
list_c = list(c)
list_c
```




    [0, 3, 6, 9]




```python
# 리스트를 튜플로
tuple_c = tuple(list_c)
tuple_c
```




    (0, 3, 6, 9)



# Workshop

**range로 리스트 만들기**
- [5, 3, 1, -1, -3, -5, -7, -9]가 출력되게 만드세요


```python
a = list(range(5, -10, -2))
print(a)
```

    [5, 3, 1, -1, -3, -5, -7, -9]
    

**range로 튜플 만들기**
- 표준입력으로 정수가 입력됩니다. range의 시작하는 숫자는 -10, 끝나는 숫자는 10이며 입력된 정수만큼 증가하는 숫자가 들어가도록 튜플을 만들고, 해당 튜플을 출력하는 프로그램을 만드세요(input에서 안내 문자열은 출력하지 않아야 합니다.)
- 입력 : 2, 출력: (-10, -8, -6, -4, -2, 0, 2, 4, 6, 8)


```python
step = int(input())
b = tuple(range(-10, 10, step))
print(b)
```

    2
    (-10, -8, -6, -4, -2, 0, 2, 4, 6, 8)
    

# 시퀀스 자료형 활용하기

- 시퀀스 자료형 : list, tuple, range, str

- 값 in 시퀀스객체


```python
a = [1, 2, 3, 4, 5, 6]
1 in a
```




    True




```python
10 in a
```




    False




```python
10 not in a
```




    True




```python
b = (1, 2, 3, 4, 5, 6)
1 in b
```




    True




```python
10 not in b
```




    True




```python
c = 'Hello'
'H' in c
```




    True




```python
'h' in c
```




    False




```python
0 in range(10)
```




    True




```python
10 in range(10)
```




    False



- 시퀀스객체1 + 시퀀스객체2


```python
a = [1, 2, 3, 4, 5]
b = [6, 7, 8, 9, 10]
```


```python
a + b
```




    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]




```python
c = (1, 2, 3, 4, 5)
d = (6, 7, 8, 9, 10)
```


```python
c + d
```




    (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)




```python
range(5) + range(5)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_2268\731926869.py in <module>
    ----> 1 range(5) + range(5)
    

    TypeError: unsupported operand type(s) for +: 'range' and 'range'



```python
list(range(5)) + list(range(5))
```




    [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]




```python
s1 = 'Hello '
s2 = 'World'
```


```python
s1 + s2
```




    'Hello World'



- 시퀀스객체 * 정수
- 정수 * 시퀀스객체


```python
[1, 2, 3] * 3
```




    [1, 2, 3, 1, 2, 3, 1, 2, 3]




```python
3 * [1, 2, 3]
```




    [1, 2, 3, 1, 2, 3, 1, 2, 3]




```python
range(5) * 2
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_2268\761196676.py in <module>
    ----> 1 range(5) * 2
    

    TypeError: unsupported operand type(s) for *: 'range' and 'int'



```python
'Hello ' * 3
```




    'Hello Hello Hello '




```python
3 * 'Hello '
```




    'Hello Hello Hello '



- len(시퀀스객체)


```python
a = [1, 2, 3]
len(a)
```




    3




```python
b = (1, 2, 3, 4, 5)
len(b)
```




    5




```python
len(range(10))
```




    10




```python
len('hello')
```




    5



- 시퀀스[인덱스]


```python
scores = [95, 80, 100, 90] # 2번째에 있는 영어 점수만 조회
```


```python
scores[1] # 0번째부터 시작하므로 1을 사용하면 두번째 요소를 가져옴
```




    80




```python
scores[-1] == scores[3] # 음수 인덱스는 마지막 원소부터 가져올 때 편리
```




    True




```python
scores[-3] == scores[1]
```




    True




```python
len(scores) - 1 # 마지막 인덱스를 구하는 방법
```




    3




```python
scores[len(scores) - 1]
```




    90




```python
b = (1, 2, 3)
```


```python
b[0]
```




    1




```python
b[4]
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_2268\3059827491.py in <module>
    ----> 1 b[4]
    

    IndexError: tuple index out of range



```python
range(10, 100, 10)[2] # range도 색인이 가능
```




    30




```python
s = 'Hello'
```


```python
s[0]
```




    'H'




```python
s[-1]
```




    'o'




```python
s[-2]
```




    'l'



- 시퀀스객체[인덱스] = 값
- 위의 문법은 list 객체에서만 유효, tuple, range, str에서는 요소 값을 변경할 수 없음


```python
scroes = [90, 100, 80, 70]
```


```python
scores[0] = 95
scores
```




    [95, 80, 100, 90]



- del 시퀀스객체[인덱스]
- 위의 문법은 list 객체에서만 유효, tuple, range, str에서는 요소 값을 삭제할 수 없음


```python
a = [1, 2, 3, 4]
```


```python
del a[1]
```


```python
a
```




    [1, 3, 4]




```python
b = (1, 2, 3, 4)
del b[0]
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_2268\2049846757.py in <module>
          1 b = (1, 2, 3, 4)
    ----> 2 del b[0]
    

    TypeError: 'tuple' object doesn't support item deletion



```python
s = 'Hello'
del s[0]
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_2268\1843072722.py in <module>
          1 s = 'Hello'
    ----> 2 del s[0]
    

    TypeError: 'str' object doesn't support item deletion



```python
del range(10)[0]
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_2268\849124569.py in <module>
    ----> 1 del range(10)[0]
    

    TypeError: 'range' object doesn't support item deletion


- 시퀀스객체[시작인덱스:끝인덱스]


```python
a = [10, 20, 30, 40, 50, 60]
a[0:3] # 끝인덱스는 포함되지 않음
```




    [10, 20, 30]




```python
a[:3] # 0번 인덱스는 생략 가능
```




    [10, 20, 30]




```python
a[3:6] # 3번째부터 5번(끝번)째까지 가져옴
```




    [40, 50, 60]




```python
a[3:] # 마지막 인덱스도 생략 가능
```




    [40, 50, 60]




```python
a[:] # 시작인덱스와 끝인덱스를 생략하면 처음부터 마지막까지 가져옴
```




    [10, 20, 30, 40, 50, 60]




```python
a[3:-1] #3번째 인덱스부터 -2번째까지 가져옴, -1번째는 포함이 안됨
```




    [40, 50]




```python
a[3:5] # 3번째부터 4번째까지 가져옴. 5번째는 포함이 안됨
```




    [40, 50]



- 시퀀스객체[시작인덱스:끝인덱스:증가폭]


```python
a = [0, 10, 20, 30, 40, 50, 60, 70]
```


```python
a[0:5] # 0~4번째까지 가져옴
```




    [0, 10, 20, 30, 40]




```python
a[0:5:1] 
```




    [0, 10, 20, 30, 40]




```python
a[0:5:2] 
```




    [0, 20, 40]




```python
a[-1::-1]
```




    [70, 60, 50, 40, 30, 20, 10, 0]



# Workshop

**최근 3년간 인구 출력하기**
- 리스트 year에 연도, population에 서울시 인구수가 저장되어 있습니다. 
- 아래 소스 코드를 완성하여 최근 3년간 연도와 인구수를 리스트로 출력되게 만드세요.


```python
year = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
population = [10249679, 10195318, 10143645, 10103233, 10022181, 9930616, 9857426, 9838892]
```


```python
print(year[-3:])
print(population[-3:])
```

    [2016, 2017, 2018]
    [9930616, 9857426, 9838892]
    


```python
# year와 population을 각 원소끼리 연결지어서 출력할 수 있음
list(zip(year[-3:], population[-3:]))
```




    [(2016, 9930616), (2017, 9857426), (2018, 9838892)]




```python
tuple(zip(year[-3:], population[-3:]))
```




    ((2016, 9930616), (2017, 9857426), (2018, 9838892))




```python
dict(zip(year[-3:], population[-3:]))
```




    {2016: 9930616, 2017: 9857426, 2018: 9838892}



**인덱스가 홀수인 요소 출력하기**
- 다음 소스 코드를 완성하여 튜플 n에서 인덱스가 홀수인 요소들이 출력되게 만드세요.


```python
n = -32, 75, 97, -10, 9, 32, 4, -15, 0, 76, 14, 2
```


```python
#n[시작:끝:보폭]
n[1:len(n):2]
```




    (75, -10, 32, -15, 76, 2)




```python
n[1::2]
```




    (75, -10, 32, -15, 76, 2)



- 시퀀스객체[시작인덱스:끝인덱스] = 시퀀스객체
- tuple, range, str은 슬라이스 범위에 요소를 할당할 수 없음


```python
a = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
a[2:5] = ['a', 'b', 'c']
a
```




    [0, 10, 'a', 'b', 'c', 50, 60, 70, 80, 90]




```python
b = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90)
b[2:5] = ['a', 'b', 'c']
b
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_2268\1822267862.py in <module>
          1 b = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90)
    ----> 2 b[2:5] = ['a', 'b', 'c']
          3 b
    

    TypeError: 'tuple' object does not support item assignment


- del 시퀀스객체[시작인덱스:끝인덱스]


```python
a
```




    [0, 10, 'a', 'b', 'c', 50, 60, 70, 80, 90]




```python
del a[2:5]
a
```




    [0, 10, 50, 60, 70, 80, 90]




```python
del b[2:5]
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_2268\223786831.py in <module>
    ----> 1 del b[2:5]
    

    TypeError: 'tuple' object does not support item deletion


# Workshop

**리스트의 마지막 요소 5개 삭제하기**
- 표준 입력으로 숫자 또는 문자열 여러개가 입력되어 리스트 x에 저장됩니다(입력되는 숫자 또는 문자열의 개수는 정해져 있지 않음). 
- 리스트 x의 마지막 요소 5개를 삭제한 뒤 튜플로 출력되게 만드세요.


```python
x = input()
```

    a b c d 1 2 3 4 5 e f
    


```python
x_list = x.split()
x_list
```




    ['a', 'b', 'c', 'd', '1', '2', '3', '4', '5', 'e', 'f']




```python
del x_list[-5:]
```


```python
tuple(x_list)
```




    ('a', 'b', 'c', 'd', '1', '2')




```python
# 함수를 한번에 연결
x = input().split()
del x[-5:]
print(tuple(x))
```

    1 2 3 4 5 6 7 7 9 10
    ('1', '2', '3', '4', '5')
    

**주민번호로 성별 구분하기**
- 아래와 같은 홍길동씨의 주민번호를 대쉬(-) 기호 전후로 쪼개보자. 성별이 남자면 'male', 여자면 'female'을 출력하세요.


```python
pin= '881204-1068234' # 1일 경우 남자, 2일 경우 여자
pit_list = pin.split('-')
if pit_list[1][0] == '1':
    print('male')
else:
    print('female')
```

    male
    

**핸드폰 번호 가리기**
- 전화번호가 문자열 phone_number로 주어졌을 때, 전화번호의 뒷 4자리를 제외한 나머지 숫자를 전부 *으로 가린 문자열을 출력하세요.

- 제한 조건
- s는 길이 4 이상, 20이하인 문자열입니다.
- 입출력 예
```
phone_number 출력 문자열
"01033334444" "*******4444"
"027778888" "*****8888"
```


```python
phone_number = input()
```

    01033334444
    


```python
phone_number
```




    '01033334444'




```python
phone_number[-4:]
```




    '4444'




```python
phone_number[:-4]
```




    '0103333'




```python
s1 = '*' * len(phone_number[:-4])
s2 = phone_number[-4:]
```


```python
s1
```




    '*******'




```python
s2
```




    '4444'




```python
s1 + s2
```




    '*******4444'




```python
# 최종
phone_number = input()
s1 = '*' * len(phone_number[:-4])
s2 = phone_number[-4:]
print(s1 + s2)
```

    027778888
    *****8888
    

# 딕셔너리

- 딕셔너리 = {키1:값1, 키2:값2, 키3:값3, ....}


```python
year_pop = {2016: 9930616, 2017: 9857426, 2018: 9838892}
```


```python
type(year_pop)
```




    dict




```python
# 딕셔너리에서는 어떤 값을 찾아가기 위한 유일한 수단이 키가 됨
# 리스트, 튜플, 문자열 등에서는 어떤 값을 찾아가기 위해 원소의 위치를 직접 계산을 했음
```


```python
# 2016년도의 population을 알고 싶을 때
year_pop[2016]
```




    9930616




```python
# 딕셔너리에서는 어떤 값을 찾아가기 위한 유일한 수단이 키이므로
# 키 값은 중복되어서는 안됨
```


```python
year_pop = {2016: 9930616, 2016:9940616, 2017: 9857426, 2018: 9838892}
```


```python
year_pop[2016] # 문법적으로 오류는 없지만 논리적으로 문제가 있음
```




    9940616




```python
year_pop = { 2017: 9857426, 2018: 9838892, 2016: 9930616}
```


```python
year_pop
```




    {2017: 9857426, 2018: 9838892, 2016: 9930616}




```python
year_pop[2016]
```




    9930616




```python
# 딕셔너리의 키 : 문자열, 정수, 실수, 불, 튜플 등은 사용할 수 있음 (리스트, 딕셔너리 포함안됨)
# 딕셔너리의 값 : 리스트, 딕셔너리 포함해서 모든 자료형을 사용할 수 있음
```


```python
# 딕셔너리 키값에는 읽기만 가능한 자료형 와야함(리스트, 딕셔너리 올 수 없음)
```


```python
lux = {'heath' : 490, 'melee':500, 'armor':18.72}
type(lux)
```




    dict




```python
lux = {[1, 2, 3] : 490, 'melee':500, 'armor':18.72} # 키 값에 리스트가 들어간 경우
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_2268\3510054695.py in <module>
    ----> 1 lux = {[1, 2, 3] : 490, 'melee':500, 'armor':18.72} # 키 값에 리스트가 들어간 경우
    

    TypeError: unhashable type: 'list'




```python
lux = {(1, 2) : 490, 'melee':500, 'armor':18.72}  # 키 값에 튜플이 들어간 경우
```

**빈 딕셔너리 만들기**
- 딕셔너리 = {}
- 딕셔너리 = dict()


```python
x = {}
type(x)
```




    dict




```python
x = dict()
type(x)
```




    dict



**dict()로 딕셔너리 만들기**
- 딕셔너리 = dict(zip([키1, 키2], [값1, 값2])
- 딕셔너리 = dict([(키1, 값1), (키2, 값2)])


```python
# lux = {'heath' : 490, 'melee':500, 'armor':18.72}
```


```python
lux = dict(zip(['heath', 'melee', 'armor'], [490, 500, 18.72]))
lux
```




    {'heath': 490, 'melee': 500, 'armor': 18.72}




```python
lux = dict([('health', 490), ('melee', 500), ('armor', 18.72)])
lux
```




    {'health': 490, 'melee': 500, 'armor': 18.72}



**딕셔너리의 키로 값 조회하기**
- 딕셔너리[키]


```python
year_pop[2016]
```




    9930616




```python
lux['armor']
```




    18.72



**딕셔너리의 값 변경하기**
- 딕셔너리[키] = 값


```python
lux
```




    {'health': 490, 'melee': 500, 'armor': 18.72}




```python
lux['melee'] = 1000
lux
```




    {'health': 490, 'melee': 1000, 'armor': 18.72}




```python
lux['attack_speed'] = 500
lux
```




    {'health': 490, 'melee': 1000, 'armor': 18.72, 'attack_speed': 500}




```python
lux['power']
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_2268\4007741939.py in <module>
    ----> 1 lux['power']
    

    KeyError: 'power'


**딕셔너리에 키가 있는지 확인하기**
- 키 in 딕셔너리


```python
# list의 경우
a = [1, 2, 3]
1 in a
```




    True




```python
'power' in lux
```




    False




```python
'melee' in lux
```




    True



**딕셔너리 키 개수 구하기**
- len(딕셔너리)


```python
lux
```




    {'health': 490, 'melee': 1000, 'armor': 18.72, 'attack_speed': 500}




```python
len(lux)
```




    4



**게임 캐릭터 능력 저장**
- 표준 입력으로 문자열 여러 개와 숫자(실수) 여러 개가 두 줄로 입력됩니다. 입력된 첫 번째 줄은 키, 두 번째 줄은 값으로 하여 딕셔너리를 생성한 뒤 딕셔너리를 출력하는 프로그램을 만드세요. input().split()의 결과를 변수 한개에 저장하면 리스트로 저장됩니다.

```
(입력예)
health health_regen mana mana_regen
575.6 1.7 338.8 1.63


(결과)
{'health':575.6, 'health_regen':1.7, 'mana':338.8, 'mana_regen':1.63}

```


```python
k_list = input().split()
v_list = input().split()
```

    health health_regen mana mana_regen
    


```python
dict(zip(k_list, v_list))
```




    {'health': '576.6',
     'health_regen': '1.7',
     'mana': '338.8',
     'mana_regen': '1.63'}



# 흐름제어

# if, else, elif

**교통 카드 시스템 만들기**

표준 입력으로 나이(만 나이)가 입력됩니다(입력 값은 7 이상 입력됨). 교통카드 시스템에서 시내버스 요금은 다음과 같으며 각 나이에 맞게 요금을 차감한 뒤 잔액이 출력되게 만드세요(if, elif 사용). 현재 교통카드에는 9,000원이 들어있습니다. 

- 어린이(초등학생, 만 7세 이상 12세 이하): 650원 
- 청소년(중∙고등학생, 만 13세 이상 18세 이하): 1,050원 
- 어른(일반, 만 19세 이상): 1,250원

```
age = int(input()) 
balance = 9000 # 교통카드 잔액 
________________ 
________________ 
________________ 
print(balance)
```


```python
age = int(input())
balance = 9000

if 7 <= age <= 12: # 어린이
    balance -= 650
elif 13 <= age <= 18: # 청소년
    balance -= 1050
elif age >= 19:
    balance -= 1250 # 어른
else:
    print("오류입니다.")
print(balance)    
```

    0
    오류입니다.
    9000
    

# for와 range 사용하기

```
for 변수 in range(횟수):
    반복할 코드
```    


```python
for i in range(10):
    print('hello world')
```

    hello world
    hello world
    hello world
    hello world
    hello world
    hello world
    hello world
    hello world
    hello world
    hello world
    

```
for 변수 in range(시작, 끝, 증가폭):
    반복할 코드
```  


```python
for i in range(0, 10, 2): # range(10) 과 동일
    print(i, 'hello world')
```

    0 hello world
    2 hello world
    4 hello world
    6 hello world
    8 hello world
    


```python
for i in range(10, 0, -2): # range(10) 과 동일
    print(i, 'hello world')
```

    10 hello world
    8 hello world
    6 hello world
    4 hello world
    2 hello world
    


```python
# enumerate() : 시퀀스 객체의 값뿐만 아니라 인덱스까지 반환해줌
for i, v in enumerate(range(0, 10, 2)):
    print(i, v, 'hello world')
```

    0 0 hello world
    1 2 hello world
    2 4 hello world
    3 6 hello world
    4 8 hello world
    

```
for 변수 in 시퀀스객체:
    반복할 코드
```  


```python
for i in [100, 200, 300]: # 총 3회 반복하되, i에는 시퀀스 객체의 개별 원소가 전달
    print(i)
```

    100
    200
    300
    


```python
# lst = [100, 200, 300]
# for i in range(len(lst)):
#     print(lst[i])
```

    100
    200
    300
    


```python
for i, v in enumerate([100, 200, 300]):
    print(i, v)
```

    0 100
    1 200
    2 300
    


```python
for v in 'Hello':
    print(v)
```

    H
    e
    l
    l
    o
    


```python
# reversed() : 시퀀스 객체를 반전
for v in reversed('Hello'):
    print(v)
```

    o
    l
    l
    e
    H
    

# Workshop

**구구단 출력 프로그램**
- 표준 입력으로 정수가 입력됩니다. 입력된 정수의 구구단을 출력하는 프로그램을 만드세요(input에서 안내 문자열은 출력하지 않아야 합니다). 출력형식은 숫자 * 숫자 = 숫자 처럼 만들고 숫자와 *, = 사이는 공백을 한칸 띄웁니다.

```
입력 예
2

결과
2 * 1 = 2
2 * 2 = 4
2 * 3 = 6
2 * 4 = 8
2 * 5 = 10
2 * 6 = 12
2 * 7 = 14
2 * 8 = 16
2 * 9 = 18
```


```python
x = int(input())
```

    2
    


```python
for i in range(1, 10):
    print(x, '*', i, '=', x*i)
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
for i in range(1, 10):
    print("{0} * {1} = {2}".format(x, i, x*i))
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
    

# while 반복문 사용하기

```
초기식
while 조건식:
    반복할 코드
    변화식
```    


```python
# while 반복문을 사용하여 'hello world'를 10번 출력하세요
i = 10
while i > 0:
    print(i, 'helllo world')
    i -= 1
```

    10 helllo world
    9 helllo world
    8 helllo world
    7 helllo world
    6 helllo world
    5 helllo world
    4 helllo world
    3 helllo world
    2 helllo world
    1 helllo world
    

# continue, break

# Workshop

- 주사위 던지고 눈을 출력하는 동작을 계속 반복하다가
- "3"이 나왔을때 멈추기


```python
import random
```


```python
n = random.randint(1, 6) # 1에서 6사이의 무작위수 추출
```




    5




```python
i = 0
while i != 3:
    i = random.randint(1, 6)
    print(i)
```

    4
    5
    4
    1
    4
    6
    1
    1
    1
    2
    6
    4
    4
    6
    1
    3
    

**교통카드 잔액 출력하기**
- 표준입력으로 금액(정수)이 입력됩니다. 1회당 요금은 1,350이고, 교통카드를 사용했을 때마다의 잔액을 각 줄에 출력하는 프로그램을 만드세요(input에서 안내 문자열은 출려갛지 않아야 합니다). 단, 최초 금액은 출력하지 않아야 합니다. 그리고 잔액은 음수가 될 수 없으며 잔액이 부족하면 출력을 끝냅니다.


```python
balance = int(input())
```

    10000
    


```python
while balance >= 1350:
    balance -= 1350
    print(balance)
```

    8650
    7300
    5950
    4600
    3250
    1900
    550
    

**3으로 끝나는 숫자만 출력하기**
- 0과 73 사이의 숫자 중 3으로 끝나는 숫자만 출력되게 만드세요.
```
출력예
3 13 23 33 43 53 63 73 
```


```python
i = 0
while True:
    # 3으로 끝나지 않는 숫자는 출력하지 않게
    if i % 10 != 3:
        i += 1
        continue
    
    if i > 73:
        break
    
    print(i)
    i += 1        
    
```

    3
    13
    23
    33
    43
    53
    63
    73
    


```python
i = 0
while True:
    # 3으로 끝나는 숫자만 출력하게
    if i % 10 == 3:
        print(i)    
    if i > 73:
        break        
    i += 1    
```

    3
    13
    23
    33
    43
    53
    63
    73
    


```python

```
