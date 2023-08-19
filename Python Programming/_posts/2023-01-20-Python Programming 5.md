---
tag: [python, 기초코딩]
---
# 람다표현식, 클래스

## 람다 표현식


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



## 변수의 사용 범위


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
    

## 클래스

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
from IPython.display import Image
Image('인스턴스와 self.PNG', width=400)
```




    
![png](/assets/images/2023-01-20-Python Programming 5/output_47_0.png)
    




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

    <ipython-input-49-18a9baf0c01e> in <module>
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
    

## 클래스 상속 사용하기

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

    <ipython-input-77-7364268d9046> in <module>
          1 James = Student()
    ----> 2 James.greeting() # 자식 클래스에서 __init__()를 사용하면서 부모클래스의 __init__()가 사용이 안됨
          3                  # self.hello가 설정되지 않은 상태로 greeting()이 호출되면서 오류
    

    <ipython-input-76-763e0460dabe> in greeting(self)
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
    


```python

```
## Reference
[파이썬 코딩 도장](https://dojang.io/course/view.php?id=7)