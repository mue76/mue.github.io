# 많이 사용하는 단축키

- Shift + Enter : 실행
- Alt + Enter : 실행(후에 셀이 추가)
- a : 위로 셀이 추가
- b : 아래로 셀이 추가
- m : 마크다운 모드로 변경
- y : 코드 모드로 변경
- x : 셀 삭제
- dd : 셀 삭제
- Ctrl + Shift + - : 커서한 위치한 곳에서 셀이 나뉨
- Shift + M : 선택된 셀들이 병합
- Esc : 에디터 모드에서 셀포커스 모드 전환

# Python Code 실행하기

**프로그램 실행방법 option 1**


```python
print('Hello World!')
```

    Hello World!
    

**프로그램 실행 방법 option 2**


```python
# cmd 창 (or anaconda prompt 창)에서
# python hello.py
```

**프로그램 실행 방법 option 3**


```python
%run hello.py
```

    Hello World!
    

# Markdown Language
[Markdown Cheat Sheet](https://www.markdownguide.org/cheat-sheet/)

![python logo](python.jpg)

![image.png](attachment:image.png)


```python
from IPython.display import Image
Image('python.jpg', width=60)
```




    
![jpeg](/assets/images/output_12_0.jpg)
    



# 탭 완성 기능


```python
an_apple = True
```


```python
an_apple, type(an_apple)
```




    (True, bool)




```python
# python list 자료형
# 변수명 = [재료1, 재료2, 재료3.....]
score = [90, 100, 85]
```


```python
score
```




    [90, 100, 85]




```python
type(score)
```




    list




```python
score.<탭키> # score 객체에서 제공하는 메소드(함수) 리스트를 볼 수 있음
```

# 도움말


```python
score?
```


```python
?score
```


```python
print('hello world!')
```

    hello world!
    


```python
print?
```

```
# 내가 만든 함수

# 1. 함수에 대한 정의
def 함수이름(a, b):
    c = (a + b)/2
    return c


# 2. 함수를 호출
결과1 = 함수이름(90, 100) # A 클래스의 평균
결과2 = 함수이름(70, 60) # B 클래스의 평균
```


```python
def mean_korean(a, b):
    """
    return mean of a and b
    """
    c = (a+b)/2
    return c
```


```python
result = mean_korean(100, 80)
result
```




    90.0




```python
mean_korean? # doc string
```


```python
mean_korean?? # doc string + source code
```

# 매직 명령어


```python
%run hello.py # 파이썬 스크립트를 실행할 때 사용
```

    Hello World!
    


```python
%pwd # 현재 작업하고 있는 노트북이 있는 위치
```




    'C:\\Users\\Playdata\\Documents\\데이터 분석 28기\\Python Programming'




```python
# %load hello.py
print('Hello World!')
```


```python
a = 2
b = 3
```


```python
%time c = a + b
```

    Wall time: 0 ns
    


```python
%timeit c = a + b
```

    112 ns ± 6.37 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
    

# 기본 문법


```python
# 파이썬의 타입(자료형)은 미리 지정이 되어 있지 않음
# 입력되는 데이터의 종류에 따라서 결정
a = 3
b = 'b'
```


```python
type(a)
```




    int




```python
type(b)
```




    str




```python
a = 3; b = 'b' # 한줄에 여러 문장을 작성할 때는 세미콜론 사용
```


```python
# 들여쓰기
a = 3
if a == 2: # a가 2와 같다면
    print("a에 2가 들어있습니다!")
    
else: # a가 2와 같지 않다면
    print("a는 2가 아닙니다!!")
    print(a)
    print('프로그램 끝')
```

    a는 2가 아닙니다!!
    3
    프로그램 끝
    

# 숫자 계산하기


```python
1+1
```




    2




```python
2-1
```




    1




```python
2*3
```




    6




```python
4/2
```




    2.0




```python
5/2
```




    2.5




```python
5//2 # 몫
```




    2




```python
5%2 # 나머지
```




    1




```python
2**10
```




    1024




```python
2**0.5
```




    1.4142135623730951




```python
a = 2
print(type(a))
```

    <class 'int'>
    


```python
b = 2.3
print(type(b))
```

    <class 'float'>
    


```python
# 자료형 변환 (type casting)
c = float(a)
print(type(c))
```

    <class 'float'>
    


```python
# b : float 형 -> int형
d = int(b)
print(type(d))
```

    <class 'int'>
    


```python
s = '10'
print(type(s))
```

    <class 'str'>
    


```python
i = int(s)
print(type(i))
```

    <class 'int'>
    


```python
xxx = 3 # 오른쪽에 있는 값을 왼쪽으로 할당
```


```python
xxx == 3 # 오른쪽값과 왼쪽값이 같은지 비교 (True/False)
```




    True




```python
# 영문자와 숫자를 함께 사용한 변수
a1 = 4
```


```python
A1 = 10
```


```python
# 대소문자 구분
a1 == A1
```




    False




```python
# 숫자로 시작하는 변수는 사용 불가
1a = 10
```


      File "C:\Users\Playdata\AppData\Local\Temp\ipykernel_10460\2682391529.py", line 2
        1a = 10
         ^
    SyntaxError: invalid syntax
    



```python
_a = 10
```


```python
# 특수문자(+, -, /, $, @, %, &) 는 변수에 사용할 수 없음
$a
```


      File "C:\Users\Playdata\AppData\Local\Temp\ipykernel_10460\2632885823.py", line 2
        $a
        ^
    SyntaxError: invalid syntax
    



```python
# 파이썬에 이미 예약된 키워드는 사용할 수 없음(if, for, while, and, or, str)

# str = 2
```


      File "C:\Users\Playdata\AppData\Local\Temp\ipykernel_10460\1295833656.py", line 3
        if = 2
           ^
    SyntaxError: invalid syntax
    



```python
a = 2
b = 3
```


```python
# a와 b를 바꾸고 싶을 때
temp = a
a = b
b = temp
```


```python
a, b
```




    (3, 2)




```python
a, b = b, a # a와 b를 맞바꿈
```


```python
a, b
```




    (2, 3)




```python
# 할당할 값의 개수 맞추기
a, b, c = 10, 20
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_10460\2432644643.py in <module>
    ----> 1 a, b, c = 10, 20
    

    ValueError: not enough values to unpack (expected 3, got 2)



```python
a
```




    2




```python
del a
```


```python
a
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_10460\2167009006.py in <module>
    ----> 1 a
    

    NameError: name 'a' is not defined



```python
x = None
```


```python
print(type(x))
```

    <class 'NoneType'>
    


```python
a = 10
a = a + 10
a
```




    20




```python
a = 10
a += 10
a
```




    20




```python
a = 10
a = a - 10
a
```




    0




```python
a = 10
a -= 10 # a = a - 10
a
```




    0



# Workshop

**아파트에서 소음이 가장 심한 층수 출력하기**
- 국립환경과학원에서는 아파트에서 소음이 가장 심한 층수를 구하는 계산식을 발표했습니다. 소음이 가장 심한 층은 0.2467 * 도로와의 거리(m) + 4.159입니다. 소음이 가장 심한 층수가 출력되게 만드세요. 단, 층수를 출력할 때는 소수점 이하 자리는 버립니다(정수로 출력).
- 도로와의 거리: 12m


```python
int(0.2467 * 12 + 4.159)
```




    7



**근의 공식 구하기**
- 근의 공식을 이용하여 x**2 + x - 2 = 0의 해를 구하라.


```python
Image('근의공식.jpg', width=300)
```




    
![jpeg](/assets/images/output_87_0.jpg)
    




```python
a=1; b=1; c=-2

x1 = (-b + (b**2 - 4*a*c)**0.5)/2*a
x2 = (-b - (b**2 - 4*a*c)**0.5)/2*a
x1, x2
```




    (1.0, -2.0)



# 입력값을 변수에 저장하기


```python
a = input(prompt="숫자만 입력해주세요.")
```

    숫자만 입력해주세요.100
    


```python
a, type(a) # input의 결과물은 문자열임을 알 수 있음
```




    ('100', str)




```python
a = input(prompt="숫자만 입력해주세요.")
b = input(prompt="숫자만 입력해주세요.")
```

    숫자만 입력해주세요.10
    숫자만 입력해주세요.20
    


```python
a, b
```




    ('10', '20')




```python
a+b
```




    '1020'




```python
int(a) + int(b)
```




    30




```python
ten_no = input("열개의 숫자를 입력하세요")
```

    열개의 숫자를 입력하세요10 20 30 40 50 60 70 80 90 100
    


```python
ten_no
```




    '10 20 30 40 50 60 70 80 90 100'




```python
type(ten_no)
```




    str




```python
# 문자열에서 제공하는 split() 함수는 whitespace를 기준으로 나눈 결과를 리스트로 반환
ten_no_list = ten_no.split()
ten_no_list
```




    ['10', '20', '30', '40', '50', '60', '70', '80', '90', '100']




```python
# split() 함수 참고
s = 'abc-def'
s.split('-')
```




    ['abc', 'def']




```python
# map()
# map(적용하고자하는 함수, 리스트(시퀀스객체))
ten_no_list = list(map(int, ten_no_list)) # list 형태로 unpacking
ten_no_list
```




    [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]




```python
# (1) 두개 숫자 입력하기
two_no = input('숫자 두개를 입력하세요 : ')

# (2) 구분자(공백)를 통해서 나누기
two_no_list = two_no.split()
two_no_list

# (3) 일괄 자료형 변경
n1, n2 = map(int, two_no_list)

n1, n2
```

    숫자 두개를 입력하세요 : 10 20
    

# Workshop

**정수 세개를 입력 받고 합계 출력하기**


```python
three_no = input("세 개의 입력하세요 : ")
three_no_list = three_no.split()
x, y, z= map(int, three_no_list)
print(x+y+z)
```

    세 개의 입력하세요 : 10 20 30
    60
    


```python
x, y, z = map(int, input("세 개의 입력하세요 : ").split())
print(x+y+z)
```

    세 개의 입력하세요 : 10 20 30
    60
    

**평균 점수 구하기**
- 표준 입력으로 국어, 영어, 수학, 과학 점수가 입력됩니다. 평균 점수를 출력하는 프로그램을 만드세요(input에서 안내 문자열은 출력하지 않아야 합니다). 단, 평균 점수를 출력할 때는 소수점 이하 자리는 버립니다(정수로 출력).


```python
k, e, m, s = map(int, input().split())
avg = (k+e+m+s)//4
print(avg)
```

    100 100 100 100
    100
    

# 출력 방법 알아보기

- print(값1, 값2, 값3, ...)
- print(변수1, 변수2, 변수3, ...)


```python
print(1, 2, 3, 4)
```

    1 2 3 4
    


```python
print(k, e, m, s)
```

    100 100 100 100
    

- print(값1, 값2, 값3, ..., sep='문자' 또는 '문자열')
- print(변수1, 변수2, 변수3, ..., sep='문자' 또는 '문자열')


```python
print(1, 2, 3, 4, sep='*')
```

    1*2*3*4
    


```python
print(k, e, m, s, sep='.')
```

    100.100.100.100
    


```python
print(1, 2, 3, sep='\n') # \n 줄바꿈
```

    1
    2
    3
    

- print(값1, 값2, 값3, ..., end='문자' 또는 '문자열')
- print(변수1, 변수2, 변수3, ..., end='문자' 또는 '문자열')


```python
print(1, 2, 3, 4, end=' ')
print(5, 6, 7, 8)
```

    1 2 3 4 5 6 7 8
    


```python
print(1, 2, 3, 4)
print(5, 6, 7, 8)
```

    1 2 3 4
    5 6 7 8
    


```python
print(1, end=' ')
print(2, end=' ')
print(3)
```

    1 2 3
    

# Workshop

**날짜와 시간 출력하기**
 - 2023/1/16 16:02:06


```python
year = 2023
month = 1
day = 16
hour = 16
minute = 2
second = 6
```


```python
print(year, month, day, sep='/', end=' ')
print(hour, minute, second, sep=':')
```

    2023/1/16 16:2:6
    


```python
# 서식지정자
print(year, month, day, sep='/', end=' ')
print(hour, end=':')
print('%02d'%minute, '%02d'%second, sep=':')
```

    2023/1/16 16:02:06
    

# Bool 자료형과 비교연산자
- ==, !=, >, <


```python
a == '10' # 비교의 결과로서 True/False 반환
```




    True




```python
a != '10'
```




    False




```python
result = (a == '10')
result
```




    True




```python
type(result) # boolean 타입
```




    bool




```python
'Python' == 'python'
```




    False




```python
10 > 20
```




    False




```python
1 == 1.0 # 값을 비교
```




    True




```python
1 is 1.0 # 객체 자체를 비교
```

    <>:1: SyntaxWarning: "is" with a literal. Did you mean "=="?
    <>:1: SyntaxWarning: "is" with a literal. Did you mean "=="?
    C:\Users\Playdata\AppData\Local\Temp\ipykernel_10460\3336455986.py:1: SyntaxWarning: "is" with a literal. Did you mean "=="?
      1 is 1.0
    




    False




```python
id(1)
```




    2098125367600




```python
id(1.0)
```




    2098241604880



# 논리 연산자
- a and b
- a or b
- not a


```python
True and True
```




    True




```python
True and False
```




    False




```python
True or False
```




    True




```python
False or False
```




    False




```python
not False
```




    True



# 비교 연산자와 논리 연산자를 함께 사용


```python
a=10; b=9
```


```python
(a == 10) and (b == 10)
```




    False




```python
# 자료형 변환(type casting)
# int(1.5)
# float(1)
# str(1)
```


```python
# bool 타입으로 바꾸기
bool(1)
```




    True




```python
bool(0)
```




    False



# Workshop

**합격 여부 출력하기**
- 국어, 영어, 수학, 과학 점수가 있을 때 한 과목이라도 50점 미만이면 불합격
이라고 정했습니다. 합격이면 True, 불합격이
면 False가 출력되게 만드세요.


```python
korean = 92 
english = 47 
mathmatics = 86 
science = 81 
```


```python
result = korean >= 50 and english >= 50 and mathmatics >= 50 and science >= 50
print(result)
```

    False
    


```python
result = not(korean < 50 or english < 50 or mathmatics < 50 and science < 50)
print(result)
```

    False
    

- all(), any()


```python
# 과목별 pass 여부를 list로 만들기
# is_pass = [True, False, True,......]
is_pass = [korean >= 50, english >= 50, mathmatics >= 50, science >= 50]
is_pass
```




    [True, False, True, True]




```python
# all(불리안 결과 집합체)
all(is_pass)
```




    False




```python
# 과목별 fail 여부를 list로 만들기
# is_fail = [True, False, True,......]
is_fail = [korean < 50, english < 50, mathmatics < 50, science < 50]
is_fail
```




    [False, True, False, False]




```python
# any(불리안 결과 집합체)
not(any(is_fail))
```




    False


