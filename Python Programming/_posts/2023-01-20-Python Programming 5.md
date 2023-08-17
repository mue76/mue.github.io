# 람다 표현식


```python
def plus_ten(x):
    return x+10

plus_ten(10)
```




    20



- lambda 매개변수:식


```python
plus_one = lambda x : x+1
```


```python
plus_one(10)
```




    11




```python
# 함수 정의/호출
(lambda x : x+1)(10)
```




    11




```python
list(map(int, ['1', '2', '3']))
```




    [1, 2, 3]




```python
# def type_int(x):
#     return int(x)

# lambda x:int(x)
```


```python
list(map(lambda x:int(x), ['1', '2', '3']))
```




    [1, 2, 3]




```python
# def plus_ten(x):
#     return x +10

# lambda x:x+10
```


```python
list(map(lambda x:x+10, [1, 2, 3]))
```




    [11, 12, 13]




```python
list(map(plus_ten, [1, 2, 3]))
```




    [11, 12, 13]



- lambda 매개변수:식1 if 조건식 else 식2


```python
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```


```python
# 3의 배수이면 문자열로 만들고, 그렇지 않으면 있는 그대로 출력
# lambda식 이용
```


```python
# def myformat(x):
#     if x % 3 == 0:
#         return str(x)
#     else:
#         return x
```


```python
myformat = lambda x: str(x) if x % 3 == 0 else x
```


```python
list(map(myformat, a))
```




    [1, 2, '3', 4, 5, '6', 7, 8, '9', 10]



- lambda 매개변수1, 매개변수2 : 식


```python
# def mul(x, y):
#     return x*y
```


```python
# lambda x, y : x*y
```


```python
a = [1, 2, 3, 4, 5]
b = [1, 2, 3, 4, 5]
```


```python
list(map(lambda x, y : x*y, a, b))
```




    [1, 4, 9, 16, 25]



- map(함수, 반복 가능한 객체) : 반복 가능한 객체에 함수를 일괄 적용
- filter(함수, 반복 가능한 객체) : 반복 가능한 객체에 함수에서 출력하는 조건에 맞는것만 가져옴    


```python
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```


```python
def f(x):
    return 5<x<10
```


```python
list(filter(f, a))
```




    [6, 7, 8, 9]



# Workshop

**거리계산**
- 리스트를 이용하여 점 사이의 거리를 계산해 보자.
- 직교 좌표 위에서 A는 (1, 1), B는 (3, 2), C는 (5,7)일 때 X(2, 3)와  A/B/C와의 거리를 각각 구하여라.
- distMeasure 함수 작성하기


```python
from math import sqrt # math라는 모듈에서 sqrt 함수를 가져옴
```


```python
# option 1 : 함수 사용
points = [(1, 1), (3, 2), (5, 7)] # A, B, C
X = (2, 3)

def distMeasure(pts):    
    result = []
    for pt in pts : # A->B->C
        dist = sqrt((X[0] - pt[0])**2 + (X[1] - pt[1])**2)
        result.append(dist)  
    return result

dist_all = distMeasure(points)
dist_all
```




    [2.23606797749979, 1.4142135623730951, 5.0]




```python
# option 2 : map 을 적용
points = [(1, 1), (3, 2), (5, 7)] # A, B, C
X = (2, 3)

def distMeasure_1p(pt):    
    dist = sqrt((X[0] - pt[0])**2 + (X[1] - pt[1])**2)
    return dist

list(map(distMeasure_1p, points))
```




    [2.23606797749979, 1.4142135623730951, 5.0]




```python
# option 3 : lambda 적용
list(map(lambda pt:sqrt((X[0] - pt[0])**2 + (X[1] - pt[1])**2), points))
```




    [2.23606797749979, 1.4142135623730951, 5.0]



**이미지 파일만 가져오기**
- 다음 소스 코드를 완성하여 확장자가 .jpg, .png인 이미지 파일만 출력되게 만드세요. 여기서는 람다 표현식을 사용해야 하며 출력 결과는 리스트 형태라야 합니다. 람다 표현식에서 확장자를 검사할 때는 문자열 메서드를 활용하세요.

```
files = ['font', '1.png', '10.jpg', '11.gif', '2.jpg', '3.png', 'table.xslx', 'spec.docx']

실행결과
['1.png', '10.jpg', '2.jpg', '3.png']
```


```python
files = ['font', '1.png', '10.jpg', '11.gif', '2.jpg', '3.png', 'table.xslx', 'spec.docx']

def filter_img(x): # 파일명 예 : 'font', '1.png', '10.jpg'
    return (x.find('.png') != -1) or (x.find('.jpg') != -1) # .png 이거나 .jpg인 파일인 경우 True 반환
```


```python
# map은 함수의 결과물이 그대로 적용
# list(map(filter_img, files))
```




    [False, True, True, False, True, True, False, False]




```python
# filter은 함수에서 True로 리턴된 요소만 반환
list(filter(filter_img, files))
```




    ['1.png', '10.jpg', '2.jpg', '3.png']




```python
# filter_img 함수를 lambda 식으로 대체
list(filter(lambda x:(x.find('.png') != -1) or (x.find('.jpg') != -1), files))
```




    ['1.png', '10.jpg', '2.jpg', '3.png']



**파일 이름을 한꺼번에 바꾸기**
- 표준 입력으로 숫자.확장자 형식으로 된 파일 이름 여러 개가 입력됩니다. 파일 이름이 숫자 3개이면서 앞에 0이 들어가는 형식으로 출력되게 만드세요. 예를 들어 1.png는 001.png, 99.docx는 099.docx, 100.xlsx는 100.xlsx처럼 출력되어야 합니다. 그리고 람다 표현식을 사용해야 하며 출력 결과는 리스트 형태라야 합니다. 람다 표현식에서 파일명을 처리할 때는 문자열 포매팅과 문자열 메서드를 활용하세요.

```
입력 예
1.jpg 10.png 11.png 2.jpg 3.png

결과
['001.jpg', '010.png', '011.png', '002.jpg', '003.png']

입력 예
97.xlsx 98.docx 99.docx 100.xlsx 101.docx 102.docx

결과
['097.xlsx', '098.docx', '099.docx', '100.xlsx', '101.docx', '102.docx']
```


```python
file_names = input().split()
```

    97.xlsx 98.docx 99.docx 100.xlsx 101.docx 102.docx
    


```python
file_names
```




    ['97.xlsx', '98.docx', '99.docx', '100.xlsx', '101.docx', '102.docx']




```python
names = []
exts = []

for file_name in file_names:
    name = int(file_name.split('.')[0])
    ext = file_name.split('.')[1]
    
    names.append(name)
    exts.append(ext)
```


```python
names, exts
```




    ([97, 98, 99, 100, 101, 102], ['xlsx', 'docx', 'docx', 'xlsx', 'docx', 'docx'])




```python
# 1개 샘플 포매팅 예시
name = 1
ext = 'jpg'

'{0:03d}.{1}'.format(name, ext)
```




    '001.jpg'




```python
# name = '1'
# ext = 'jpg'

# '{0:>03s}.{1}'.format(name, ext)
```




    '001.jpg'




```python
def myformat(name, ext):
    return '{0:03d}.{1}'.format(name, ext)
```


```python
list(map(myformat, names, exts))
```




    ['097.xlsx', '098.docx', '099.docx', '100.xlsx', '101.docx', '102.docx']



# 변수의 사용 범위


```python
# x는 전역 변수
x = 10

def foo():
    # x는 지역 변수
    x = 1
    print("foo() : ", x) # foo() : 1

foo()    
print("main() : ", x) # main() : 10
```

    foo() :  1
    main() :  10
    


```python
# x는 전역 변수
x = 10

def foo():
    print("foo() : ", x) # foo() : 10

foo()    
print("main() : ", x) # main() : 10
```

    foo() :  10
    main() :  10
    


```python
# x는 전역 변수
x = 10

def foo():
    global x
    x = 1
    print("foo() : ", x) # foo() : 1

foo()    
print("main() : ", x) # main() : 1
```

    foo() :  1
    main() :  1
    


```python
def print_hello():
    hello = 'hello world'
    
    def print_message():
        print(hello)
        
    print_message()
    
print_hello()    
```

    hello world
    


```python
def print_hello():
    hello = 'hello world'
    
    def print_message():
        hello = 'hello country'
        print(hello)
        
    print_message()
    
print_hello()   
```

    hello country
    


```python
def print_hello():
    hello = 'hello world'
    
    def print_message():
        nonlocal hello
        hello = 'hello country'
        print(hello)
        
    print_message()
    print(hello)
    
print_hello()  
```

    hello country
    hello country
    

# 클래스

```
class 클래스이름: # 붕어빵 틀
    def 메서드(self):
        코드
```        


```python
class Person:
    def greeting(self):
        print("Hello")
```

```
인스턴스 = 클래스() # 붕어빵
```


```python
James = Person()
```


```python
James.greeting()
```

    Hello
    


```python
# 파이썬 흔히 볼 수 있는 클래스

a = int(10)
print(type(a))

b = int(9)
print(type(b))
```

    <class 'int'>
    <class 'int'>
    


```python
c = list([1, 2, 3])
print(type(c))
```

    <class 'list'>
    


```python
d = dict({'x':10})
print(type(d))
```

    <class 'dict'>
    

```
class 클래스이름: # 붕어빵 틀
    def __init__(self):
        self.속성 = 값
        
    def 메서드(self):
        코드
```   


```python
class Person:
    def __init__(self):
        self.hello = '안녕하세요'
    
    def greeting(self):
        print(self.hello)
```


```python
James = Person()
James.greeting()
```

    안녕하세요
    


```python
Selly = Person()
Selly.greeting()
```

    안녕하세요
    

```
class 클래스이름: # 붕어빵 틀
    def __init__(self, 매개변수1, 매개변수2, ...):
        self.속성1 = 매개변수1
        self.속성2 = 매개변수2
        
    def 메서드(self):
        코드
```  


```python
class Person:
    def __init__(self, hello):
        self.hello = hello
    
    def greeting(self):
        print(self.hello)
```


```python
James = Person("안녕하십니까?")
James.greeting()
```

    안녕하십니까?
    


```python
Selly = Person("안녕하세요.")
Selly.greeting()
```

    안녕하세요.
    


```python
class Person:
    def __init__(self, hello, name, age, address):
        self.hello = hello
        self.name = name
        self.age = age
        self.address = address
    
    def greeting(self):
        print(self.hello)
        print("저는 {}입니다.".format(self.name))
        print("나이는 {}살입니다.".format(self.age))
        print("{}에 삽니다.".format(self.address))
```


```python
James = Person("안녕하십니까", "제임스", 17, "서초동")
```


```python
James.greeting()
```

    안녕하십니까
    저는 제임스입니다.
    나이는 17살입니다.
    서초동에 삽니다.
    


```python
Selly = Person("안녕하세요.", "샐리", 20, "가산동")
Selly.greeting()
```

    안녕하세요.
    저는 샐리입니다.
    나이는 20살입니다.
    가산동에 삽니다.
    


```python
Selly.address
```




    '가산동'




```python
Selly.age
```




    20



```
class 클래스이름: # 붕어빵 틀
    def __init__(self):
        self.__속성 = 값 # 비공개속성

        
    def 메서드(self):
        코드
```  


```python
class Person:
    def __init__(self, hello, name, age, address, wallet):
        self.hello = hello
        self.name = name
        self.age = age
        self.address = address
        self.__wallet = wallet
    
    def greeting(self):
        print(self.hello)
        print("저는 {}입니다.".format(self.name))
        print("나이는 {}살입니다.".format(self.age))
        print("{}에 삽니다.".format(self.address))
        
    def pay(self, amount):
        self.__wallet -= amount
        print("지갑에 {}원 남았습니다.".format(self.__wallet))
```


```python
Maria = Person("안녕?", "마리아", 30, "역삼동", 10000)
```


```python
# 비공개 속성은 외부에서 접근하면 에러가 생김
Maria.__wallet
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_8812\1432878972.py in <module>
          1 # 비공개 속성은 외부에서 접근하면 에러가 생김
    ----> 2 Maria.__wallet
    

    AttributeError: 'Person' object has no attribute '__wallet'



```python
Maria.pay(2000)
```

    지갑에 8000원 남았습니다.
    


```python
Maria.pay(3000)
```

    지갑에 5000원 남았습니다.
    

**클래스 속성과 인스턴스 속성**

```
class 클래스이름:
    속성 = 값
```    


```python
class Person:
    bag = []
    
    def put_bag(self, stuff):
        Person.bag.append(stuff) 
```


```python
James = Person()
James.put_bag("책")
```


```python
James.bag
```




    ['책']




```python
Selly = Person()
Selly.put_bag("열쇠")
```


```python
Selly.bag
```




    ['책', '열쇠']




```python
class Person:
    def __init__(self):
        self.bag = []
    
    def put_bag(self, stuff):
        self.bag.append(stuff) 
```


```python
James = Person()
James.put_bag("책")
```


```python
James.bag
```




    ['책']




```python
Selly = Person()
Selly.put_bag("열쇠")
```


```python
Selly.bag
```




    ['열쇠']



- 클래스 속성 : 모든 인스턴스들이 공유. 인스턴스 전체가 사용해하는 값을 저장할 때 사용
- 인스턴스 속성 : 인스턴스별로 독립되어 있음. 각 인스턴스가 값을 따로 저장해야 할 때 사용

```
class 클래스이름:
    __속성 = 값 # 비공개 클래스 속성
```  

**정적 메서드와 인스턴스 메서드**


```python
# 인스턴스 메서드의 예
a = list([1, 2, 3])
a.append(100)
a
```




    [1, 2, 3, 100]




```python
# 정적 메서드의 예
a = [1, 2, 3]
list.append(a, 100)
```


```python
a
```




    [1, 2, 3, 100]



**정적메서드**

```
class 클래스이름:
    @staticmethod
    def 메서드(매개변수1, 매개변수2...):
        코드
```    


```python
class Calc:
    @staticmethod
    def add(a, b):
        print(a+b)
    @staticmethod    
    def mul(a, b):
        print(a*b)
```


```python
Calc.add(10, 20) # 클래스에서 바로 메서드를 호출(인스턴스 만들 필요 없음)
```

    30
    

# 클래스 상속 사용하기

```
class 기반클래스이름:
    코드
    
class 파생클래스이름(기반클래스이름):
    코드
```    


```python
class Person():
    def greeting(self):
        print("안녕하세요")
```


```python
class Student(Person):
    def study(self):
        print("공부중입니다.")
```


```python
James = Student()
```


```python
James.greeting() # 상속받은 부모클래스(Person)의 기능까지도 사용할 수 있음
```

    안녕하세요
    


```python
James.study()
```

    공부중입니다.
    


```python
class Person():
    def __init__(self):
        print("Person initialized")
        
    def greeting(self):
        print("안녕하세요")
        
        
class Student(Person):
    def __init__(self):
        print('Student initialized')
    
    def study(self):
        print("공부중입니다.")        
```


```python
James = Student() # 자식 클래스에서 __init__()하게 되면 부모 클래스의 __init__()는 사용하지 않음
```

    Student initialized
    


```python
class Person():
    def __init__(self):
        print("Person initialized")
        
    def greeting(self):
        print("안녕하세요")
        
        
class Student(Person):
#     def __init__(self):
#         print('Student initialized')
    
    def study(self):
        print("공부중입니다.")  
```


```python
James = Student() # 자식 클래스에 __init__()가 없으면 부모 클래스의 __init__() 가 그대로 적용
```

    Person initialized
    


```python
class Person():
    def __init__(self):
        print("Person initialized")
        self.hello = '안녕하세요'
        
    def greeting(self):
        print(self.hello)
        
        
class Student(Person):
    def __init__(self):
        print('Student initialized')
    
    def study(self):
        print("공부중입니다.")    
```


```python
James = Student()
James.greeting() # 자식 클래스에서 __init__()를 사용하면서 부모클래스의 __init__()가 사용이 안됨
                 # self.hello가 설정되지 않은 상태로 greeting()이 호출되면서 오류
```

    Student initialized
    


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_8812\1104609217.py in <module>
          1 James = Student()
    ----> 2 James.greeting()
    

    ~\AppData\Local\Temp\ipykernel_8812\4008125588.py in greeting(self)
          5 
          6     def greeting(self):
    ----> 7         print(self.hello)
          8 
          9 
    

    AttributeError: 'Student' object has no attribute 'hello'



```python
class Person():
    def __init__(self):
        print("Person initialized")
        self.hello = '안녕하세요'
        
    def greeting(self):
        print(self.hello)
        
        
class Student(Person):
    def __init__(self):
        super().__init__() # 부모클래스의 __init__()를 강제로 호출
        print('Student initialized')
    
    def study(self):
        print("공부중입니다.")  
```


```python
James = Student()
James.greeting()
```

    Person initialized
    Student initialized
    안녕하세요
    

**메서드 오버라이딩**


```python
class Person:
    def greeting(self):
        print('안녕하세요')
        
class Student(Person):
    def greeting(self):
        print('안녕하세요. 저는 분석과정 28기 학생입니다.')
```


```python
James = Student()
James.greeting() # 부모 클래스와 동일한 함수인 경우 자식 클래스의 함수가 호출
```

    안녕하세요. 저는 분석과정 28기 학생입니다.
    


```python
class Person:
    def greeting(self):
        print('안녕하세요')
        
class Student(Person):
    def greeting(self):
        super().greeting() # 부모클래스의 greeting()도 활용
        print('저는 분석과정 28기 학생입니다.')
```


```python
James = Student()
James.greeting()
```

    안녕하세요
    저는 분석과정 28기 학생입니다.
    

**다중 상속 사용하기**

```
class 기반클래스1:
    코드
    
class 기반클래스2:
    코드
    
class 파생클래스(기반클래스1, 기반클래스2):
    코드
```    


```python
class Person:
    def greeting(self):
        print('안녕하세요.')
class University:
    def manage_credit(self):
        print('학점 관리')

class Undergraduate(Person, University):
    def study(self):
        print('공부하기')
```


```python
James = Undergraduate()
James.study()
```

    공부하기
    


```python
James.manage_credit()
```

    학점 관리
    


```python
James.greeting()
```

    안녕하세요.
    

**추상클래스**

```
from abc import *

class 추상클래스(metaclass=ABCMeta):
    @abstractmethod
    def 메서드이름(self):
        코드
```        


```python
from abc import *

class StdudentBase(metaclass=ABCMeta):
    @abstractmethod
    def study(self):
        pass
    
    @abstractmethod
    def gotoschool(self):
        pass
```


```python
class Student(StdudentBase):
    def study(self):
        print("공부하기")
        
    def gotoschool(self):
        print("학교가기")
```


```python
James = Student()
```


```python
James.gotoschool()
```

    학교가기
    

# Workshop

**리스트에 기능 추가하기**
- 아래 예시와 같이 리스트(list)에 replace 메서드를 추가한 AdvancedList 클래스를 작성하세요. AdvancedList는 list를 상속받아서 만들고, replace 메서드는 리스트에서 특정 값으로 된 요소를 찾아서 다른 값으로 바꾸도록 만드세요.

```
x = AdvancedList([1, 2, 3, 1, 2, 3, 1, 2, 3])
x.replace(1, 100)
print(x)

결과
[100, 2, 3, 100, 2, 3, 100, 2, 3]
```


```python
class AdvancedList(list):
    def replace(self, old, new):
        for i, v in enumerate(self):
            if v == old:
                self[i] = new        
```


```python
x = AdvancedList([1, 2, 3, 1, 2, 3, 1, 2, 3])
```


```python
x.replace(1, 100)
```


```python
x
```




    [100, 2, 3, 100, 2, 3, 100, 2, 3]




```python
from IPython.display import Image
Image('인스턴스와 self.PNG', width=400)
```




    
![png](/assets/images/output_142_0.png)
    




```python

```
