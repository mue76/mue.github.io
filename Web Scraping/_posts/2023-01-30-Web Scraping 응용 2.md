---
tag: [python, 웹스크래핑, Web Scraping, BeautifulSoup, Selenium]
toc: true
toc_sticky: true
toc_label: 목차
author_profile: false
sidebar:
  nav: "counts"
---
# Selenium을 이용한 Web scraping

**셀레니움**
- 웹페이지 테스트 자동화를 할 수 있는 프레임워크
- 셀레니움 사용하면 완전한 형태의 웹 페이지 소스를 볼 수 있기 때문에 스크래핑할 때 유용

**사용 예**
- 자바스크립트가 동적으로 만든 데이터 스크래핑할 때
- 사이트의 다양한 HTML 요소에 클릭, 키보드 입력 등 이벤트를 줄 때

**사용 방법**
- 셀레니움을 사용하려면 사용중인 웹 브라우저의 드라이버 파일을 다운로드해야 함
- 크롬 드라이버 파일 다운로드 받기
```
chrome://version/ 버전 확인
https://chromedriver.chromium.org/downloads 에서 chrome 버전과 운영체제에 맞는 드라이버 다운로드
압축 푼 뒤 응용 프로그램을 현재 작업 디렉토리에 카피
```

- selenium 설치
```
conda install selenium
```

* 참고사이트
```
https://www.selenium.dev/documentation/webdriver/getting_started/
```    


```python
from IPython.display import Image
Image('webdriver.png')
```




    
![png](/assets/images/2023-01-30-Web Scraping 응용 2/output_5_0.png)
    




```python
import selenium
```


```python
from selenium import webdriver
```


```python
selenium.__version__
```




    '3.141.0'




```python
import requests
import time
```


```python
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
```


```python
from bs4 import BeautifulSoup
```


```python
driver = webdriver.Chrome() # driver가 있는 path를 넣어줄 수 있음
```


```python
driver.get('http://naver.com')
```


```python
driver.back()
```


```python
driver.forward()
```


```python
driver.refresh()
```


```python
driver.quit() # driver.close() 한탭만 종료
```

- 더 자세한 내용은
```
https://www.selenium.dev/documentation/webdriver/elements/
```

**find_element_by_tag_name**


```python
driver = webdriver.Chrome()
driver.get('http://naver.com')
```


```python
# a tag에 접근
elem = driver.find_element_by_tag_name('a')
```


```python
# a tag 요소로부터 href 속성의 값을 조회
elem.get_attribute('href') # bs 의 a['href']
```




    'https://www.naver.com/#newsstand'




```python
# 여러개의 a tag에 접근
elems = driver.find_elements_by_tag_name('a')
for elem in elems[:10]:
    print(elem.get_attribute('href'))
```




    [<selenium.webdriver.remote.webelement.WebElement (session="2034f411e6c074c90ddec6a1c8ac24d5", element="d744e0d9-b489-4d14-b5d7-5257de26ed1c")>,
     <selenium.webdriver.remote.webelement.WebElement (session="2034f411e6c074c90ddec6a1c8ac24d5", element="b36b24c7-3b5d-43ba-bf41-3b07c8e89536")>,
     <selenium.webdriver.remote.webelement.WebElement (session="2034f411e6c074c90ddec6a1c8ac24d5", element="d2a97a2f-134b-400f-84ed-63c955f4205c")>,
     <selenium.webdriver.remote.webelement.WebElement (session="2034f411e6c074c90ddec6a1c8ac24d5", element="dc2ff99e-2688-4c0d-87dd-9bf97d70be3a")>,
     <selenium.webdriver.remote.webelement.WebElement (session="2034f411e6c074c90ddec6a1c8ac24d5", element="3e191220-9d87-497a-b753-41b1a1ca1aef")>,
     <selenium.webdriver.remote.webelement.WebElement (session="2034f411e6c074c90ddec6a1c8ac24d5", element="ad224d24-b1b2-490c-b448-667cfe33b6eb")>,
     <selenium.webdriver.remote.webelement.WebElement (session="2034f411e6c074c90ddec6a1c8ac24d5", element="81fa7f20-732d-4ca0-908c-f1d6a193ecc7")>,
     <selenium.webdriver.remote.webelement.WebElement (session="2034f411e6c074c90ddec6a1c8ac24d5", element="f913a5e6-207a-4ff3-82e8-12042be96f5e")>,
     <selenium.webdriver.remote.webelement.WebElement (session="2034f411e6c074c90ddec6a1c8ac24d5", element="3127a677-3132-441d-affa-723a2dc777bb")>,
     <selenium.webdriver.remote.webelement.WebElement (session="2034f411e6c074c90ddec6a1c8ac24d5", element="54444165-b784-4b56-995f-9b86fbf3aa78")>]




```python
driver.quit()
```

**find_element_by_id**


```python
driver = webdriver.Chrome()
driver.get('http://naver.com')
```


```python
#<input id="query" name="query" type="text" title="검색어 입력" maxlength="255" class="input_text" tabindex="1" accesskey="s" style="ime-mode:active;" autocomplete="off" placeholder="검색어를 입력해 주세요." onclick="document.getElementById('fbm').value=1;" value="" data-atcmp-element="">
```


```python
elem = driver.find_element_by_id('query')
```


```python
elem.send_keys('파이썬')
```


```python
from selenium.webdriver.common.keys import Keys
elem.send_keys(Keys.ENTER)
```


```python
driver.quit()
```

**find_element_by_name**


```python
driver = webdriver.Chrome()
driver.get('http://naver.com')
```


```python
elem = driver.find_element_by_name('query')
```


```python
elem.send_keys('파이썬')
```


```python
elem = driver.find_element_by_id('search_btn')
```


```python
elem.click()
```


```python
driver.back()
```

**find_element_by_xpath**


```python
elem = driver.find_element_by_name('query')
elem.send_keys('파이썬')
```


```python
elem = driver.find_element_by_xpath('//*[@id="search_btn"]')
elem.click()
```


```python
driver.back()
```

**find_element_by_css_selector**


```python
elem = driver.find_element_by_name('query')
elem.send_keys('파이썬')
```


```python
elem = driver.find_element_by_css_selector('#search_btn')
elem.click()
```


```python
driver.back()
```

**find_element_by_link_text**


```python
elem = driver.find_element_by_link_text("지도")
elem.click()
```


```python
driver.quit()
```

### 실습 1
**네이버 로그인**


```python
import time
```


```python
options = webdriver.ChromeOptions()
options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36')

driver = webdriver.Chrome(options=options)
driver.get('http://naver.com')
```


```python
#elem = driver.find_element_by_css_selector('a.link_login')
elem = driver.find_element_by_css_selector('#account > a')
elem.click()
```


```python
elem = driver.find_element_by_id('id')
elem.send_keys('mue')

time.sleep(2)

elem = driver.find_element_by_id('pw')
elem.send_keys('xxxx')
```


```python
elem = driver.find_element_by_class_name('btn_login')
elem.click()
```


```python
driver.quit()
```

### 실습 2

**구글 이미지 검색**


```python
url = "https://www.google.co.kr/imghp?hl=ko&tab=ri&ogbl"

options = webdriver.ChromeOptions()
options.headless = True
options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36')

driver = webdriver.Chrome(options=options)
driver.get(url)

# 검색어 입력 
keyword = '감자'

elem = driver.find_element_by_name('q')
elem.send_keys(keyword)
elem.submit() # elem.send_keys(Keys.ENTER) 와 동일한 효과
```

**이미지 한장 저장**
- 왼쪽 작은 이미지 클릭후 오른쪽 큰 이미지 저장


```python
# 작은 이미지
elem = driver.find_element_by_css_selector('img.rg_i.Q4LuWd')
elem.click()
time.sleep(1)

# 오른쪽 큰 이미지 URL을 알아오기
elem = driver.find_element_by_xpath('//*[@id="Sva75c"]/div[2]/div/div[2]/div[2]/div[2]/c-wiz/div[2]/div[1]/div[1]/div[2]/div/a/img')
image_url = elem.get_attribute('src')

# 위에서 구한 URL 요청해서 이미지 저장하기
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
headers = {'User-Agent' : user_agent}

response = requests.get(image_url, headers)
response.raise_for_status()

with open('./potato/1.jpg', 'wb') as fd:
    fd.write(response.content)

driver.quit()
```

**이미지 여러장 저장**


```python
# 이미지 URL에 base64 포맷의 embedded image가 내려온 경우
# https://acdongpgm.tistory.com/159
# https://stackoverflow.com/questions/16214190/how-to-convert-base64-string-to-image

import base64
test_url = 'data:image/jpeg;base64,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'

imgdata = base64.b64decode(test_url[23:])

with open('test_img.jpg', 'wb') as fd:
    fd.write(imgdata)
```


```python
url = "https://www.google.co.kr/imghp?hl=ko&tab=ri&ogbl"

options = webdriver.ChromeOptions()
options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36')

driver = webdriver.Chrome(options=options)
driver.get(url)

# 검색어 입력 
keyword = '사과'

elem = driver.find_element_by_name('q')
elem.send_keys(keyword)
elem.submit() # elem.send_keys(Keys.ENTER) 와 동일한 효과

# 검색하고 난 뒤 다음 페이지 요소들(예:CSS Selector 요소들)이 나올때까지 기다리기
elems = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'img.rg_i.Q4LuWd')))

# 왼쪽 작은 이미지들에 대한 요소 가져오기
# elems = driver.find_elements_by_css_selector('img.rg_i.Q4LuWd')

for i, elem in enumerate(elems, start=1): # 검색된 이미지들 순회
    #print(i)
    elem.click()
    time.sleep(1)
    #img_elem = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="Sva75c"]/div[2]/div/div[2]/div[2]/div[2]/c-wiz/div[2]/div[1]/div[1]/div[2]/div/a/img')))

    # 오른쪽 큰 이미지 URL을 알아오기
    img_elem = driver.find_element_by_xpath('//*[@id="Sva75c"]/div[2]/div/div[2]/div[2]/div[2]/c-wiz/div[2]/div[1]/div[1]/div[2]/div/a/img')
    image_url = img_elem.get_attribute('src')
    if image_url.startswith('data:'):
        continue
#         embedded_img = base64.b64decode(image_url[23:])
#         with open(f'./apple/{i}__.jpg', 'wb') as fd:
#             fd.write(embedded_img)
    else:   
        # 위에서 구한 URL 요청해서 이미지 저장하기
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
        headers = {'User-Agent' : user_agent}

        try:
            response = requests.get(image_url, headers)
        except Exception as e :
            print('request 실패 예외 발생', e, i)

        try:
            response.raise_for_status()
        except Exception as e : # 모든 예외 표시
            print('response 예외 발생', e, i)

        else:
            with open(f'./apple/{i}.jpg', 'wb') as fd:
                fd.write(response.content)
    if i > 20:
        break

driver.quit()
```

    response 예외 발생 403 Client Error: Forbidden for url: http://m.grfarm.co.kr/web/product/medium/202204/c3b1939cd60caed066f428a4dfcc1d1f.jpg?User-Agent=Mozilla%2F5.0+%28Windows+NT+10.0%3B+Win64%3B+x64%29+AppleWebKit%2F537.36+%28KHTML%2C+like+Gecko%29+Chrome%2F109.0.0.0+Safari%2F537.36 8
    response 예외 발생 403 Client Error: Forbidden for url: https://dmzfarm.com/web/product/big/201611/78_shop1_157279.jpg?User-Agent=Mozilla%2F5.0+%28Windows+NT+10.0%3B+Win64%3B+x64%29+AppleWebKit%2F537.36+%28KHTML%2C+like+Gecko%29+Chrome%2F109.0.0.0+Safari%2F537.36 15
    

**스크롤 테스트**


```python
url = "https://www.google.co.kr/imghp?hl=ko&tab=ri&ogbl"

options = webdriver.ChromeOptions()
options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36')

driver = webdriver.Chrome(options=options)
driver.get(url)

# 검색어 입력 
keyword = '사과'

elem = driver.find_element_by_name('q')
elem.send_keys(keyword)
elem.submit() # elem.send_keys(Keys.ENTER) 와 동일한 효과
```


```python
prev_height = driver.execute_script('return document.body.scrollHeight')
print(prev_height)

while True:
    driver.execute_script('window,scrollTo(0, document.body.scrollHeight)')
    time.sleep(1)
    curr_height = driver.execute_script('return document.body.scrollHeight')
    print(curr_height)
    if prev_height == curr_height:
        break
    prev_height = curr_height
print('**scroll done**')    
```

    3618
    6531
    12593
    18897
    25201
    25282
    25282
    **scroll done**
    

**이미지 여러장 저장 (스크롤 포함)**


```python
url = "https://www.google.co.kr/imghp?hl=ko&tab=ri&ogbl"

options = webdriver.ChromeOptions()
# tions.headless = True  # Browser 띄우지 않고 자동화할 때 넣는 옵션
options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36')

driver = webdriver.Chrome(options=options)
driver.maximize_window() # 윈도우 크기를 최대화한다.
driver.get(url)

# 검색어 입력 
keyword = '사과'

elem = driver.find_element_by_name('q')
elem.send_keys(keyword)
elem.submit() # elem.send_keys(Keys.ENTER) 와 동일한 효과

# 검색하고 난 뒤 다음 페이지 요소들(예:CSS Selector 요소들)이 나올때까지 기다리기
elems = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'img.rg_i.Q4LuWd')))
print("스크롤전 : ", len(elems))

# 스크롤해서 더 많은 이미지 가져오기
prev_height = driver.execute_script('return document.body.scrollHeight')
print(prev_height)

while True:
    driver.execute_script('window,scrollTo(0, document.body.scrollHeight)')
    time.sleep(1)
    curr_height = driver.execute_script('return document.body.scrollHeight')
    print(curr_height)
    if prev_height == curr_height:
        break
    prev_height = curr_height
print('**scroll done**')    

# 왼쪽 작은 이미지들에 대한 요소 가져오기
elems = driver.find_elements_by_css_selector('img.rg_i.Q4LuWd')
print("스크롤후 : ", len(elems))

for i, elem in enumerate(elems, start=1): # 검색된 이미지들 순회
    #print(i)
    elem.click()
    time.sleep(1)
    # 아래와 같이 컨디션을 사용하면 너무 빠른 요청으로 인해 Embedded img가 내려오는 경향이 있어 삭제
    #img_elem = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="Sva75c"]/div[2]/div/div[2]/div[2]/div[2]/c-wiz/div[2]/div[1]/div[1]/div[2]/div/a/img')))

    # 오른쪽 큰 이미지 URL을 알아오기
    img_elem = driver.find_element_by_xpath('//*[@id="Sva75c"]/div[2]/div/div[2]/div[2]/div[2]/c-wiz/div[2]/div[1]/div[1]/div[2]/div/a/img')
    image_url = img_elem.get_attribute('src')
    if image_url.startswith('data:'):
        continue
#         embedded_img = base64.b64decode(image_url[23:])
#         with open(f'./apple/{i}__.jpg', 'wb') as fd:
#            fd.write(embedded_img)
    else:   
        # 위에서 구한 URL 요청해서 이미지 저장하기
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
        headers = {'User-Agent' : user_agent}

        try:
            response = requests.get(image_url, headers)
        except Exception as e :
            print('request 실패 예외 발생', e, i)

        try:
            response.raise_for_status()
        except Exception as e : # 모든 예외 표시
            print('response 예외 발생', e, i)

        else:
            with open(f'./apple/{i}.jpg', 'wb') as fd:
                fd.write(response.content)
    if i > 20:
        break


driver.quit()
```

    스크롤전 :  48
    2650
    4595
    8721
    12847
    16973
    17054
    17054
    **scroll done**
    스크롤후 :  400
    response 예외 발생 403 Client Error: Forbidden for url: http://m.grfarm.co.kr/web/product/medium/202204/c3b1939cd60caed066f428a4dfcc1d1f.jpg?User-Agent=Mozilla%2F5.0+%28Windows+NT+10.0%3B+Win64%3B+x64%29+AppleWebKit%2F537.36+%28KHTML%2C+like+Gecko%29+Chrome%2F109.0.0.0+Safari%2F537.36 8
    response 예외 발생 403 Client Error: Forbidden for url: http://img1.tmon.kr/cdn3/deals/2020/09/20/3252248318/front_d7334_cug82.jpg?User-Agent=Mozilla%2F5.0+%28Windows+NT+10.0%3B+Win64%3B+x64%29+AppleWebKit%2F537.36+%28KHTML%2C+like+Gecko%29+Chrome%2F109.0.0.0+Safari%2F537.36 9
    response 예외 발생 403 Client Error: Forbidden for url: https://dmzfarm.com/web/product/big/201611/78_shop1_157279.jpg?User-Agent=Mozilla%2F5.0+%28Windows+NT+10.0%3B+Win64%3B+x64%29+AppleWebKit%2F537.36+%28KHTML%2C+like+Gecko%29+Chrome%2F109.0.0.0+Safari%2F537.36 15
    

### 실습 3

**구글 무비**
- https://play.google.com/store/movies/top
- 스크롤된 모든 데이터에 대해
- 가격이 할인된 영화만 출력하되
- 출력 양식 : 영화제목, 할인전/후 가격, 링크


```python
# Option 1 (Selenium 만 이용 가능)
```


```python
# 1. 첫 화면 접속
options = webdriver.ChromeOptions()
options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36 Edg/106.0.1370.47')

driver = webdriver.Chrome(options=options)
driver.maximize_window()

# 구글 이미지검색 사이트 이동
driver.get("https://play.google.com/store/movies/top")


elems = driver.find_elements_by_class_name("VfPpkd-EScbFb-JIbuQc.UVEnyf")
print("스크롤전 : ", len(elems))

# 2. 페이지 스크롤
# Java Script의 스크롤 기능을 통해서 더 많은 이미지를 가져옴
prev_height = driver.execute_script('return document.body.scrollHeight')
print(prev_height)

while True:
    driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')
    time.sleep(1)
    curr_height = driver.execute_script('return document.body.scrollHeight')
    print(curr_height)
    if prev_height == curr_height:
        break
    prev_height = curr_height
    
elems = driver.find_elements_by_class_name("VfPpkd-EScbFb-JIbuQc.UVEnyf")
print("스크롤후 : ", len(elems))  

print("전체영화수", len(elems))
count=0
for elem in elems:
    title = elem.find_element_by_class_name('Epkrse').text
    try:
        original_price= elem.find_element_by_class_name('SUZt4c.P8AFK').text
    except: # original_price가 없는 항목들은 skip
        continue
    
    discount_price= elem.find_element_by_class_name('VfPpfd.VixbEe').text
    link = elem.find_element_by_class_name('Si6A0c.ZD8Cqc')
    link = link.get_attribute('href') # get_attribute를 통해 얻은 href가 절대 경로임
    
    print('제목 : ', title)
    print('할인전 가격 : ', original_price)
    print('할인후 가격 : ', discount_price)
    print('링크 : ', link)
    print('-'*100)
    count += 1
print(count)   

driver.quit()    
```

    스크롤전 :  60
    2591
    3772
    4042
    3531
    4048
    4048
    스크롤후 :  120
    전체영화수 120
    제목 :  뮬란: 더 레전드
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EB%AE%AC%EB%9E%80_%EB%8D%94_%EB%A0%88%EC%A0%84%EB%93%9C?id=lfqlk0qtt8k.P
    ----------------------------------------------------------------------------------------------------
    제목 :  돈 워리 달링
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EB%8F%88_%EC%9B%8C%EB%A6%AC_%EB%8B%AC%EB%A7%81?id=95u-gG-mt0o.P
    ----------------------------------------------------------------------------------------------------
    제목 :  마이선
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EB%A7%88%EC%9D%B4%EC%84%A0?id=oiqIk6k27OM.P
    ----------------------------------------------------------------------------------------------------
    제목 :  에브리씽 에브리웨어 올 앳 원스
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EC%97%90%EB%B8%8C%EB%A6%AC%EC%94%BD_%EC%97%90%EB%B8%8C%EB%A6%AC%EC%9B%A8%EC%96%B4_%EC%98%AC_%EC%95%B3_%EC%9B%90%EC%8A%A4?id=9lJwvD-5x-s.P
    ----------------------------------------------------------------------------------------------------
    제목 :  캅시크릿
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EC%BA%85%EC%8B%9C%ED%81%AC%EB%A6%BF?id=D1ZXLHaEMJ4.P
    ----------------------------------------------------------------------------------------------------
    제목 :  해리포터 시리즈 완결 패키지
    할인전 가격 :  ₩36,000
    할인후 가격 :  ₩18,900
    링크 :  https://play.google.com/store/movies/details/%ED%95%B4%EB%A6%AC%ED%8F%AC%ED%84%B0_%EC%8B%9C%EB%A6%AC%EC%A6%88_%EC%99%84%EA%B2%B0_%ED%8C%A8%ED%82%A4%EC%A7%80?id=DsVgRu5dDdY
    ----------------------------------------------------------------------------------------------------
    제목 :  인디아나 존스: 모험 컬렉션
    할인전 가격 :  ₩20,000
    할인후 가격 :  ₩11,400
    링크 :  https://play.google.com/store/movies/details/%EC%9D%B8%EB%94%94%EC%95%84%EB%82%98_%EC%A1%B4%EC%8A%A4_%EB%AA%A8%ED%97%98_%EC%BB%AC%EB%A0%89%EC%85%98?id=Yv_OthKEftU
    ----------------------------------------------------------------------------------------------------
    제목 :  반지의 제왕: 3 영화 컬렉션 확장판 (자막판)
    할인전 가격 :  ₩13,500
    할인후 가격 :  ₩11,900
    링크 :  https://play.google.com/store/movies/details/%EB%B0%98%EC%A7%80%EC%9D%98_%EC%A0%9C%EC%99%95_3_%EC%98%81%ED%99%94_%EC%BB%AC%EB%A0%89%EC%85%98_%ED%99%95%EC%9E%A5%ED%8C%90_%EC%9E%90%EB%A7%89%ED%8C%90?id=80DJns6yqWs
    ----------------------------------------------------------------------------------------------------
    제목 :  탑건 2 무비 컬렉션
    할인전 가격 :  ₩13,500
    할인후 가격 :  ₩12,000
    링크 :  https://play.google.com/store/movies/details/%ED%83%91%EA%B1%B4_2_%EB%AC%B4%EB%B9%84_%EC%BB%AC%EB%A0%89%EC%85%98?id=1aLgJ4syNGk.P
    ----------------------------------------------------------------------------------------------------
    제목 :  신비한 동물 영화 3편 패키지
    할인전 가격 :  ₩15,000
    할인후 가격 :  ₩11,900
    링크 :  https://play.google.com/store/movies/details/%EC%8B%A0%EB%B9%84%ED%95%9C_%EB%8F%99%EB%AC%BC_%EC%98%81%ED%99%94_3%ED%8E%B8_%ED%8C%A8%ED%82%A4%EC%A7%80?id=_b1ycFl_mFU.P
    ----------------------------------------------------------------------------------------------------
    제목 :  해리포터풀패키지(1-7B)
    할인전 가격 :  ₩36,000
    할인후 가격 :  ₩30,400
    링크 :  https://play.google.com/store/movies/details/%ED%95%B4%EB%A6%AC%ED%8F%AC%ED%84%B0%ED%92%80%ED%8C%A8%ED%82%A4%EC%A7%80_1_7B?id=oV2yIteiUjY.P
    ----------------------------------------------------------------------------------------------------
    제목 :  맨인블랙 시리즈 패키지 (자막판)
    할인전 가격 :  ₩12,000
    할인후 가격 :  ₩7,500
    링크 :  https://play.google.com/store/movies/details/%EB%A7%A8%EC%9D%B8%EB%B8%94%EB%9E%99_%EC%8B%9C%EB%A6%AC%EC%A6%88_%ED%8C%A8%ED%82%A4%EC%A7%80_%EC%9E%90%EB%A7%89%ED%8C%90?id=TiFtAEp2KVo
    ----------------------------------------------------------------------------------------------------
    제목 :  트랜스포머 영화 5편 컬렉션
    할인전 가격 :  ₩25,000
    할인후 가격 :  ₩16,000
    링크 :  https://play.google.com/store/movies/details/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B8_%EC%98%81%ED%99%94_5%ED%8E%B8_%EC%BB%AC%EB%A0%89%EC%85%98?id=-wBC9omKCv0
    ----------------------------------------------------------------------------------------------------
    제목 :  토르 4 무비 컬렉션
    할인전 가격 :  ₩26,400
    할인후 가격 :  ₩18,900
    링크 :  https://play.google.com/store/movies/details/%ED%86%A0%EB%A5%B4_4_%EB%AC%B4%EB%B9%84_%EC%BB%AC%EB%A0%89%EC%85%98?id=qk4UsCD3K6c.P
    ----------------------------------------------------------------------------------------------------
    제목 :  조커/ 더 배트맨 영화 컬렉션
    할인전 가격 :  ₩10,500
    할인후 가격 :  ₩8,600
    링크 :  https://play.google.com/store/movies/details/%EC%A1%B0%EC%BB%A4_%EB%8D%94_%EB%B0%B0%ED%8A%B8%EB%A7%A8_%EC%98%81%ED%99%94_%EC%BB%AC%EB%A0%89%EC%85%98?id=ETFncwTsz40.P
    ----------------------------------------------------------------------------------------------------
    제목 :  미션 임파서블 1-5 컬렉션 (자막판)
    할인전 가격 :  ₩25,000
    할인후 가격 :  ₩12,400
    링크 :  https://play.google.com/store/movies/details/%EB%AF%B8%EC%85%98_%EC%9E%84%ED%8C%8C%EC%84%9C%EB%B8%94_1_5_%EC%BB%AC%EB%A0%89%EC%85%98_%EC%9E%90%EB%A7%89%ED%8C%90?id=Nkdb-Bl3WeU
    ----------------------------------------------------------------------------------------------------
    제목 :  라이트이어+토이 스토리(1~4) 무비 컬렉션
    할인전 가격 :  ₩42,000
    할인후 가격 :  ₩25,900
    링크 :  https://play.google.com/store/movies/details/%EB%9D%BC%EC%9D%B4%ED%8A%B8%EC%9D%B4%EC%96%B4_%ED%86%A0%EC%9D%B4_%EC%8A%A4%ED%86%A0%EB%A6%AC_1_4_%EB%AC%B4%EB%B9%84_%EC%BB%AC%EB%A0%89%EC%85%98?id=CN9QGsn95zA.P
    ----------------------------------------------------------------------------------------------------
    제목 :  MIB 맨 인 블랙 풀 패키지 (자막판)
    할인전 가격 :  ₩16,000
    할인후 가격 :  ₩8,550
    링크 :  https://play.google.com/store/movies/details/MIB_%EB%A7%A8_%EC%9D%B8_%EB%B8%94%EB%9E%99_%ED%92%80_%ED%8C%A8%ED%82%A4%EC%A7%80_%EC%9E%90%EB%A7%89%ED%8C%90?id=gjNbS93yMyg
    ----------------------------------------------------------------------------------------------------
    제목 :  50가지 그림자: 3 무비 콜렉션
    할인전 가격 :  ₩10,500
    할인후 가격 :  ₩7,400
    링크 :  https://play.google.com/store/movies/details/50%EA%B0%80%EC%A7%80_%EA%B7%B8%EB%A6%BC%EC%9E%90_3_%EB%AC%B4%EB%B9%84_%EC%BD%9C%EB%A0%89%EC%85%98?id=mFyAj0YjCKM
    ----------------------------------------------------------------------------------------------------
    제목 :  저스티스 리그 패키지 (자막판)
    할인전 가격 :  ₩13,500
    할인후 가격 :  ₩13,400
    링크 :  https://play.google.com/store/movies/details/%EC%A0%80%EC%8A%A4%ED%8B%B0%EC%8A%A4_%EB%A6%AC%EA%B7%B8_%ED%8C%A8%ED%82%A4%EC%A7%80_%EC%9E%90%EB%A7%89%ED%8C%90?id=UoFllXXcy0w
    ----------------------------------------------------------------------------------------------------
    제목 :  슈퍼배드 & 미니언즈: 4편의 무비 컬렉션 (더빙판)
    할인전 가격 :  ₩14,000
    할인후 가격 :  ₩9,800
    링크 :  https://play.google.com/store/movies/details/%EC%8A%88%ED%8D%BC%EB%B0%B0%EB%93%9C_%EB%AF%B8%EB%8B%88%EC%96%B8%EC%A6%88_4%ED%8E%B8%EC%9D%98_%EB%AC%B4%EB%B9%84_%EC%BB%AC%EB%A0%89%EC%85%98_%EB%8D%94%EB%B9%99%ED%8C%90?id=YJFC3dXPH9Y
    ----------------------------------------------------------------------------------------------------
    제목 :  드래곤 길들이기 3부작 (더빙판)
    할인전 가격 :  ₩13,500
    할인후 가격 :  ₩12,000
    링크 :  https://play.google.com/store/movies/details/%EB%93%9C%EB%9E%98%EA%B3%A4_%EA%B8%B8%EB%93%A4%EC%9D%B4%EA%B8%B0_3%EB%B6%80%EC%9E%91_%EB%8D%94%EB%B9%99%ED%8C%90?id=lbGJnOFGtu4
    ----------------------------------------------------------------------------------------------------
    제목 :  돈 워리 달링
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EB%8F%88_%EC%9B%8C%EB%A6%AC_%EB%8B%AC%EB%A7%81?id=95u-gG-mt0o.P
    ----------------------------------------------------------------------------------------------------
    제목 :  오펀: 천사의 탄생
    할인전 가격 :  ₩7,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EC%98%A4%ED%8E%80_%EC%B2%9C%EC%82%AC%EC%9D%98_%ED%83%84%EC%83%9D?id=LZZ0BoZNVvg.P
    ----------------------------------------------------------------------------------------------------
    제목 :  돈 워리 달링
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EB%8F%88_%EC%9B%8C%EB%A6%AC_%EB%8B%AC%EB%A7%81?id=95u-gG-mt0o.P
    ----------------------------------------------------------------------------------------------------
    제목 :  블레이드 퍼피 워리어
    할인전 가격 :  ₩5,000
    할인후 가격 :  ₩1,800
    링크 :  https://play.google.com/store/movies/details/%EB%B8%94%EB%A0%88%EC%9D%B4%EB%93%9C_%ED%8D%BC%ED%94%BC_%EC%9B%8C%EB%A6%AC%EC%96%B4?id=ZtlnwvrgOzM.P
    ----------------------------------------------------------------------------------------------------
    제목 :  마이선
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EB%A7%88%EC%9D%B4%EC%84%A0?id=oiqIk6k27OM.P
    ----------------------------------------------------------------------------------------------------
    제목 :  에브리씽 에브리웨어 올 앳 원스
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EC%97%90%EB%B8%8C%EB%A6%AC%EC%94%BD_%EC%97%90%EB%B8%8C%EB%A6%AC%EC%9B%A8%EC%96%B4_%EC%98%AC_%EC%95%B3_%EC%9B%90%EC%8A%A4?id=9lJwvD-5x-s.P
    ----------------------------------------------------------------------------------------------------
    제목 :  이스터 선데이
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩7,700
    링크 :  https://play.google.com/store/movies/details/%EC%9D%B4%EC%8A%A4%ED%84%B0_%EC%84%A0%EB%8D%B0%EC%9D%B4?id=X9bXIOPclkU.P
    ----------------------------------------------------------------------------------------------------
    29
    


```python
# Option 2 (Selenium 통해 page scroll 마친 뒤, driver.page_source 를 BS에 전달)
```


```python
# 1. 첫 화면 접속
options = webdriver.ChromeOptions()
options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36 Edg/106.0.1370.47')

driver = webdriver.Chrome(options=options)
driver.maximize_window()

# 구글 이미지검색 사이트 이동
driver.get("https://play.google.com/store/movies/top")


elems = driver.find_elements_by_class_name("VfPpkd-EScbFb-JIbuQc.UVEnyf")
print("스크롤전 : ", len(elems))

# 2. 페이지 스크롤
# Java Script의 스크롤 기능을 통해서 더 많은 이미지를 가져옴
prev_height = driver.execute_script('return document.body.scrollHeight')
print(prev_height)

while True:
    driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')
    time.sleep(1)
    curr_height = driver.execute_script('return document.body.scrollHeight')
    print(curr_height)
    if prev_height == curr_height:
        break
    prev_height = curr_height
    
elems = driver.find_elements_by_class_name("VfPpkd-EScbFb-JIbuQc.UVEnyf")
print("스크롤후 : ", len(elems))    

# 3. 필요한 정보 가져오기 위해 구문 분석
soup = BeautifulSoup(driver.page_source, 'lxml')

# print(soup.prettify())

movies = soup.select('div.VfPpkd-EScbFb-JIbuQc.UVEnyf')

print(len(movies))

count = 0
for movie in movies:
    # 영화제목
    title = movie.select_one('div.Epkrse')
    title = title.text

    # 할인 전 가격
    original_price= movie.select_one('span.SUZt4c.P8AFK')
    if original_price: 
        original_price = original_price.text

    else: # 할인 전 가격이 없으면 세일하는 항목이 아님
        continue
        
    # 할인 후 가격
    discount_price= movie.select_one('span.VfPpfd.VixbEe')
    discount_price = discount_price.text


    # 링크 정보
    link = movie.select_one('a.Si6A0c.ZD8Cqc')
    link = 'https://play.google.com' + link['href']

    
    print('제목 : ', title)
    print('할인전 가격 : ', original_price)
    print('할인후 가격 : ', discount_price)
    print('링크 : ', link)
    print('-'*100)
    count += 1
print(count)   

driver.quit()
```

    스크롤전 :  60
    2591
    3772
    4042
    4042
    스크롤후 :  120
    120
    제목 :  뮬란: 더 레전드
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EB%AE%AC%EB%9E%80_%EB%8D%94_%EB%A0%88%EC%A0%84%EB%93%9C?id=lfqlk0qtt8k.P
    ----------------------------------------------------------------------------------------------------
    제목 :  돈 워리 달링
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EB%8F%88_%EC%9B%8C%EB%A6%AC_%EB%8B%AC%EB%A7%81?id=95u-gG-mt0o.P
    ----------------------------------------------------------------------------------------------------
    제목 :  마이선
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EB%A7%88%EC%9D%B4%EC%84%A0?id=oiqIk6k27OM.P
    ----------------------------------------------------------------------------------------------------
    제목 :  에브리씽 에브리웨어 올 앳 원스
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EC%97%90%EB%B8%8C%EB%A6%AC%EC%94%BD_%EC%97%90%EB%B8%8C%EB%A6%AC%EC%9B%A8%EC%96%B4_%EC%98%AC_%EC%95%B3_%EC%9B%90%EC%8A%A4?id=9lJwvD-5x-s.P
    ----------------------------------------------------------------------------------------------------
    제목 :  캅시크릿
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EC%BA%85%EC%8B%9C%ED%81%AC%EB%A6%BF?id=D1ZXLHaEMJ4.P
    ----------------------------------------------------------------------------------------------------
    제목 :  해리포터 시리즈 완결 패키지
    할인전 가격 :  ₩36,000
    할인후 가격 :  ₩18,900
    링크 :  https://play.google.com/store/movies/details/%ED%95%B4%EB%A6%AC%ED%8F%AC%ED%84%B0_%EC%8B%9C%EB%A6%AC%EC%A6%88_%EC%99%84%EA%B2%B0_%ED%8C%A8%ED%82%A4%EC%A7%80?id=DsVgRu5dDdY
    ----------------------------------------------------------------------------------------------------
    제목 :  인디아나 존스: 모험 컬렉션
    할인전 가격 :  ₩20,000
    할인후 가격 :  ₩11,400
    링크 :  https://play.google.com/store/movies/details/%EC%9D%B8%EB%94%94%EC%95%84%EB%82%98_%EC%A1%B4%EC%8A%A4_%EB%AA%A8%ED%97%98_%EC%BB%AC%EB%A0%89%EC%85%98?id=Yv_OthKEftU
    ----------------------------------------------------------------------------------------------------
    제목 :  반지의 제왕: 3 영화 컬렉션 확장판 (자막판)
    할인전 가격 :  ₩13,500
    할인후 가격 :  ₩11,900
    링크 :  https://play.google.com/store/movies/details/%EB%B0%98%EC%A7%80%EC%9D%98_%EC%A0%9C%EC%99%95_3_%EC%98%81%ED%99%94_%EC%BB%AC%EB%A0%89%EC%85%98_%ED%99%95%EC%9E%A5%ED%8C%90_%EC%9E%90%EB%A7%89%ED%8C%90?id=80DJns6yqWs
    ----------------------------------------------------------------------------------------------------
    제목 :  탑건 2 무비 컬렉션
    할인전 가격 :  ₩13,500
    할인후 가격 :  ₩12,000
    링크 :  https://play.google.com/store/movies/details/%ED%83%91%EA%B1%B4_2_%EB%AC%B4%EB%B9%84_%EC%BB%AC%EB%A0%89%EC%85%98?id=1aLgJ4syNGk.P
    ----------------------------------------------------------------------------------------------------
    제목 :  신비한 동물 영화 3편 패키지
    할인전 가격 :  ₩15,000
    할인후 가격 :  ₩11,900
    링크 :  https://play.google.com/store/movies/details/%EC%8B%A0%EB%B9%84%ED%95%9C_%EB%8F%99%EB%AC%BC_%EC%98%81%ED%99%94_3%ED%8E%B8_%ED%8C%A8%ED%82%A4%EC%A7%80?id=_b1ycFl_mFU.P
    ----------------------------------------------------------------------------------------------------
    제목 :  해리포터풀패키지(1-7B)
    할인전 가격 :  ₩36,000
    할인후 가격 :  ₩30,400
    링크 :  https://play.google.com/store/movies/details/%ED%95%B4%EB%A6%AC%ED%8F%AC%ED%84%B0%ED%92%80%ED%8C%A8%ED%82%A4%EC%A7%80_1_7B?id=oV2yIteiUjY.P
    ----------------------------------------------------------------------------------------------------
    제목 :  맨인블랙 시리즈 패키지 (자막판)
    할인전 가격 :  ₩12,000
    할인후 가격 :  ₩7,500
    링크 :  https://play.google.com/store/movies/details/%EB%A7%A8%EC%9D%B8%EB%B8%94%EB%9E%99_%EC%8B%9C%EB%A6%AC%EC%A6%88_%ED%8C%A8%ED%82%A4%EC%A7%80_%EC%9E%90%EB%A7%89%ED%8C%90?id=TiFtAEp2KVo
    ----------------------------------------------------------------------------------------------------
    제목 :  트랜스포머 영화 5편 컬렉션
    할인전 가격 :  ₩25,000
    할인후 가격 :  ₩16,000
    링크 :  https://play.google.com/store/movies/details/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B8_%EC%98%81%ED%99%94_5%ED%8E%B8_%EC%BB%AC%EB%A0%89%EC%85%98?id=-wBC9omKCv0
    ----------------------------------------------------------------------------------------------------
    제목 :  토르 4 무비 컬렉션
    할인전 가격 :  ₩26,400
    할인후 가격 :  ₩18,900
    링크 :  https://play.google.com/store/movies/details/%ED%86%A0%EB%A5%B4_4_%EB%AC%B4%EB%B9%84_%EC%BB%AC%EB%A0%89%EC%85%98?id=qk4UsCD3K6c.P
    ----------------------------------------------------------------------------------------------------
    제목 :  조커/ 더 배트맨 영화 컬렉션
    할인전 가격 :  ₩10,500
    할인후 가격 :  ₩8,600
    링크 :  https://play.google.com/store/movies/details/%EC%A1%B0%EC%BB%A4_%EB%8D%94_%EB%B0%B0%ED%8A%B8%EB%A7%A8_%EC%98%81%ED%99%94_%EC%BB%AC%EB%A0%89%EC%85%98?id=ETFncwTsz40.P
    ----------------------------------------------------------------------------------------------------
    제목 :  미션 임파서블 1-5 컬렉션 (자막판)
    할인전 가격 :  ₩25,000
    할인후 가격 :  ₩12,400
    링크 :  https://play.google.com/store/movies/details/%EB%AF%B8%EC%85%98_%EC%9E%84%ED%8C%8C%EC%84%9C%EB%B8%94_1_5_%EC%BB%AC%EB%A0%89%EC%85%98_%EC%9E%90%EB%A7%89%ED%8C%90?id=Nkdb-Bl3WeU
    ----------------------------------------------------------------------------------------------------
    제목 :  라이트이어+토이 스토리(1~4) 무비 컬렉션
    할인전 가격 :  ₩42,000
    할인후 가격 :  ₩25,900
    링크 :  https://play.google.com/store/movies/details/%EB%9D%BC%EC%9D%B4%ED%8A%B8%EC%9D%B4%EC%96%B4_%ED%86%A0%EC%9D%B4_%EC%8A%A4%ED%86%A0%EB%A6%AC_1_4_%EB%AC%B4%EB%B9%84_%EC%BB%AC%EB%A0%89%EC%85%98?id=CN9QGsn95zA.P
    ----------------------------------------------------------------------------------------------------
    제목 :  MIB 맨 인 블랙 풀 패키지 (자막판)
    할인전 가격 :  ₩16,000
    할인후 가격 :  ₩8,550
    링크 :  https://play.google.com/store/movies/details/MIB_%EB%A7%A8_%EC%9D%B8_%EB%B8%94%EB%9E%99_%ED%92%80_%ED%8C%A8%ED%82%A4%EC%A7%80_%EC%9E%90%EB%A7%89%ED%8C%90?id=gjNbS93yMyg
    ----------------------------------------------------------------------------------------------------
    제목 :  50가지 그림자: 3 무비 콜렉션
    할인전 가격 :  ₩10,500
    할인후 가격 :  ₩7,400
    링크 :  https://play.google.com/store/movies/details/50%EA%B0%80%EC%A7%80_%EA%B7%B8%EB%A6%BC%EC%9E%90_3_%EB%AC%B4%EB%B9%84_%EC%BD%9C%EB%A0%89%EC%85%98?id=mFyAj0YjCKM
    ----------------------------------------------------------------------------------------------------
    제목 :  저스티스 리그 패키지 (자막판)
    할인전 가격 :  ₩13,500
    할인후 가격 :  ₩13,400
    링크 :  https://play.google.com/store/movies/details/%EC%A0%80%EC%8A%A4%ED%8B%B0%EC%8A%A4_%EB%A6%AC%EA%B7%B8_%ED%8C%A8%ED%82%A4%EC%A7%80_%EC%9E%90%EB%A7%89%ED%8C%90?id=UoFllXXcy0w
    ----------------------------------------------------------------------------------------------------
    제목 :  슈퍼배드 & 미니언즈: 4편의 무비 컬렉션 (더빙판)
    할인전 가격 :  ₩14,000
    할인후 가격 :  ₩9,800
    링크 :  https://play.google.com/store/movies/details/%EC%8A%88%ED%8D%BC%EB%B0%B0%EB%93%9C_%EB%AF%B8%EB%8B%88%EC%96%B8%EC%A6%88_4%ED%8E%B8%EC%9D%98_%EB%AC%B4%EB%B9%84_%EC%BB%AC%EB%A0%89%EC%85%98_%EB%8D%94%EB%B9%99%ED%8C%90?id=YJFC3dXPH9Y
    ----------------------------------------------------------------------------------------------------
    제목 :  드래곤 길들이기 3부작 (더빙판)
    할인전 가격 :  ₩13,500
    할인후 가격 :  ₩12,000
    링크 :  https://play.google.com/store/movies/details/%EB%93%9C%EB%9E%98%EA%B3%A4_%EA%B8%B8%EB%93%A4%EC%9D%B4%EA%B8%B0_3%EB%B6%80%EC%9E%91_%EB%8D%94%EB%B9%99%ED%8C%90?id=lbGJnOFGtu4
    ----------------------------------------------------------------------------------------------------
    제목 :  돈 워리 달링
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EB%8F%88_%EC%9B%8C%EB%A6%AC_%EB%8B%AC%EB%A7%81?id=95u-gG-mt0o.P
    ----------------------------------------------------------------------------------------------------
    제목 :  오펀: 천사의 탄생
    할인전 가격 :  ₩7,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EC%98%A4%ED%8E%80_%EC%B2%9C%EC%82%AC%EC%9D%98_%ED%83%84%EC%83%9D?id=LZZ0BoZNVvg.P
    ----------------------------------------------------------------------------------------------------
    제목 :  돈 워리 달링
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EB%8F%88_%EC%9B%8C%EB%A6%AC_%EB%8B%AC%EB%A7%81?id=95u-gG-mt0o.P
    ----------------------------------------------------------------------------------------------------
    제목 :  블레이드 퍼피 워리어
    할인전 가격 :  ₩5,000
    할인후 가격 :  ₩1,800
    링크 :  https://play.google.com/store/movies/details/%EB%B8%94%EB%A0%88%EC%9D%B4%EB%93%9C_%ED%8D%BC%ED%94%BC_%EC%9B%8C%EB%A6%AC%EC%96%B4?id=ZtlnwvrgOzM.P
    ----------------------------------------------------------------------------------------------------
    제목 :  마이선
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EB%A7%88%EC%9D%B4%EC%84%A0?id=oiqIk6k27OM.P
    ----------------------------------------------------------------------------------------------------
    제목 :  에브리씽 에브리웨어 올 앳 원스
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EC%97%90%EB%B8%8C%EB%A6%AC%EC%94%BD_%EC%97%90%EB%B8%8C%EB%A6%AC%EC%9B%A8%EC%96%B4_%EC%98%AC_%EC%95%B3_%EC%9B%90%EC%8A%A4?id=9lJwvD-5x-s.P
    ----------------------------------------------------------------------------------------------------
    제목 :  이스터 선데이
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩7,700
    링크 :  https://play.google.com/store/movies/details/%EC%9D%B4%EC%8A%A4%ED%84%B0_%EC%84%A0%EB%8D%B0%EC%9D%B4?id=X9bXIOPclkU.P
    ----------------------------------------------------------------------------------------------------
    29
    


```python
# option 3 (우측버튼 눌러서 스크롤까지 되는 버전)
```


```python
# 1. 첫 화면 접속
options = webdriver.ChromeOptions()
options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36 Edg/106.0.1370.47')

driver = webdriver.Chrome(options=options)
driver.maximize_window()

# 구글 이미지검색 사이트 이동
driver.get("https://play.google.com/store/movies/top")


elems = driver.find_elements_by_class_name("VfPpkd-EScbFb-JIbuQc.UVEnyf")
print("스크롤전 : ", len(elems))

# 2. 페이지 스크롤
# Java Script의 스크롤 기능을 통해서 더 많은 이미지를 가져옴
prev_height = driver.execute_script('return document.body.scrollHeight')
print(prev_height)

while True:
    driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')
    time.sleep(1)
    curr_height = driver.execute_script('return document.body.scrollHeight')
    print(curr_height)
    if prev_height == curr_height:
        break
    prev_height = curr_height
    
elems = driver.find_elements_by_class_name("VfPpkd-EScbFb-JIbuQc.UVEnyf")
print("스크롤후 : ", len(elems))  

# 3. 우측 버튼 스크롤
buttons = driver.find_elements_by_class_name("VfPpkd-BIzmGd.SaBhMc.NNFoTc.zI3eKe.N7pe4e.PcY7Ff.DpB3re")
print(len(buttons))
# driver.execute_script("arguments[0].click();", button)

for button in buttons:    
    while True:
        try:
            driver.execute_script("arguments[0].click();", button)
            time.sleep(0.2)
        except Exception as e:
            print(e)
            break

elems = driver.find_elements_by_class_name("VfPpkd-EScbFb-JIbuQc.UVEnyf")            
print("우측버튼 스크롤후", len(elems))

# 4. 필요한 정보 가져오기 위해 구문 분석
count=0
for elem in elems:
    title = elem.find_element_by_class_name('Epkrse').text
    try:
        original_price= elem.find_element_by_class_name('SUZt4c.P8AFK').text
    except: # original_price가 없는 항목들은 skip
        continue
    
    discount_price= elem.find_element_by_class_name('VfPpfd.VixbEe').text
    link = elem.find_element_by_class_name('Si6A0c.ZD8Cqc')
    link = link.get_attribute('href') # get_attribute를 통해 얻은 href가 절대 경로임
    
    print('제목 : ', title)
    print('할인전 가격 : ', original_price)
    print('할인후 가격 : ', discount_price)
    print('링크 : ', link)
    print('-'*100)
    count += 1
print(count)   

driver.quit()    
```

    스크롤전 :  60
    2593
    3778
    4048
    4048
    스크롤후 :  120
    7
    Message: stale element reference: element is not attached to the page document
      (Session info: chrome=109.0.5414.120)
    
    Message: stale element reference: element is not attached to the page document
      (Session info: chrome=109.0.5414.120)
    
    Message: stale element reference: element is not attached to the page document
      (Session info: chrome=109.0.5414.120)
    
    Message: stale element reference: element is not attached to the page document
      (Session info: chrome=109.0.5414.120)
    
    Message: stale element reference: element is not attached to the page document
      (Session info: chrome=109.0.5414.120)
    
    Message: stale element reference: element is not attached to the page document
      (Session info: chrome=109.0.5414.120)
    
    Message: stale element reference: element is not attached to the page document
      (Session info: chrome=109.0.5414.120)
    
    우측버튼 스크롤후 517
    제목 :  뮬란: 더 레전드
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EB%AE%AC%EB%9E%80_%EB%8D%94_%EB%A0%88%EC%A0%84%EB%93%9C?id=lfqlk0qtt8k.P
    ----------------------------------------------------------------------------------------------------
    제목 :  프레이 포 더 데블
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩7,700
    링크 :  https://play.google.com/store/movies/details/%ED%94%84%EB%A0%88%EC%9D%B4_%ED%8F%AC_%EB%8D%94_%EB%8D%B0%EB%B8%94?id=QVhB7omoDd8.P
    ----------------------------------------------------------------------------------------------------
    제목 :  매복
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩7,700
    링크 :  https://play.google.com/store/movies/details/%EB%A7%A4%EB%B3%B5?id=k1-jHarOtYo.P
    ----------------------------------------------------------------------------------------------------
    제목 :  압꾸정
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩7,000
    링크 :  https://play.google.com/store/movies/details/%EC%95%95%EA%BE%B8%EC%A0%95?id=cfTIZM5KJrQ.P
    ----------------------------------------------------------------------------------------------------
    제목 :  몬스터 신부: 101번째 프로포즈
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩7,700
    링크 :  https://play.google.com/store/movies/details/%EB%AA%AC%EC%8A%A4%ED%84%B0_%EC%8B%A0%EB%B6%80_101%EB%B2%88%EC%A7%B8_%ED%94%84%EB%A1%9C%ED%8F%AC%EC%A6%88?id=dwBwvD2snDc.P
    ----------------------------------------------------------------------------------------------------
    제목 :  라스트필름
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩7,700
    링크 :  https://play.google.com/store/movies/details/%EB%9D%BC%EC%8A%A4%ED%8A%B8%ED%95%84%EB%A6%84?id=dWJlTVOKW8g.P
    ----------------------------------------------------------------------------------------------------
    제목 :  페르시아어 수업
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩7,000
    링크 :  https://play.google.com/store/movies/details/%ED%8E%98%EB%A5%B4%EC%8B%9C%EC%95%84%EC%96%B4_%EC%88%98%EC%97%85?id=iEfgHuCCXVU.P
    ----------------------------------------------------------------------------------------------------
    제목 :  당문: 신후쌍웅
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EB%8B%B9%EB%AC%B8_%EC%8B%A0%ED%9B%84%EC%8C%8D%EC%9B%85?id=oHYXAmuesik.P
    ----------------------------------------------------------------------------------------------------
    제목 :  오메르타
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EC%98%A4%EB%A9%94%EB%A5%B4%ED%83%80?id=FfIuBk-s2_E.P
    ----------------------------------------------------------------------------------------------------
    제목 :  아우슈비츠 챔피언
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EC%95%84%EC%9A%B0%EC%8A%88%EB%B9%84%EC%B8%A0_%EC%B1%94%ED%94%BC%EC%96%B8?id=kgacYWsTIAg.P
    ----------------------------------------------------------------------------------------------------
    제목 :  이스터 선데이
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩7,700
    링크 :  https://play.google.com/store/movies/details/%EC%9D%B4%EC%8A%A4%ED%84%B0_%EC%84%A0%EB%8D%B0%EC%9D%B4?id=X9bXIOPclkU.P
    ----------------------------------------------------------------------------------------------------
    제목 :  헤이지니&럭키강이 비밀의문
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩7,700
    링크 :  https://play.google.com/store/movies/details/%ED%97%A4%EC%9D%B4%EC%A7%80%EB%8B%88_%EB%9F%AD%ED%82%A4%EA%B0%95%EC%9D%B4_%EB%B9%84%EB%B0%80%EC%9D%98%EB%AC%B8?id=e7CZ5NpMx5U.P
    ----------------------------------------------------------------------------------------------------
    제목 :  소림나한
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EC%86%8C%EB%A6%BC%EB%82%98%ED%95%9C?id=xcJxbxc-lik.P
    ----------------------------------------------------------------------------------------------------
    제목 :  영웅천전
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EC%98%81%EC%9B%85%EC%B2%9C%EC%A0%84?id=2e04cdvssYI.P
    ----------------------------------------------------------------------------------------------------
    제목 :  간병: 사랑의 족쇄
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EA%B0%84%EB%B3%91_%EC%82%AC%EB%9E%91%EC%9D%98_%EC%A1%B1%EC%87%84?id=KX7NWrQpK2s.P
    ----------------------------------------------------------------------------------------------------
    제목 :  황제의 검
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%ED%99%A9%EC%A0%9C%EC%9D%98_%EA%B2%80?id=Hmq08t79z90.P
    ----------------------------------------------------------------------------------------------------
    제목 :  기문언갑사
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EA%B8%B0%EB%AC%B8%EC%96%B8%EA%B0%91%EC%82%AC?id=tvAHTQVNJnU.P
    ----------------------------------------------------------------------------------------------------
    제목 :  메가 스네이크2
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EB%A9%94%EA%B0%80_%EC%8A%A4%EB%84%A4%EC%9D%B4%ED%81%AC2?id=oaAU531h9t0.P
    ----------------------------------------------------------------------------------------------------
    제목 :  철갑기관병
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EC%B2%A0%EA%B0%91%EA%B8%B0%EA%B4%80%EB%B3%91?id=X65M-0eR1m0.P
    ----------------------------------------------------------------------------------------------------
    제목 :  지옥의 화원
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩7,000
    링크 :  https://play.google.com/store/movies/details/%EC%A7%80%EC%98%A5%EC%9D%98_%ED%99%94%EC%9B%90?id=T4QD41_G3Oo.P
    ----------------------------------------------------------------------------------------------------
    제목 :  은밀한 동거
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EC%9D%80%EB%B0%80%ED%95%9C_%EB%8F%99%EA%B1%B0?id=gkiLEib5SHQ.P
    ----------------------------------------------------------------------------------------------------
    제목 :  애프터 미투
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩7,700
    링크 :  https://play.google.com/store/movies/details/%EC%95%A0%ED%94%84%ED%84%B0_%EB%AF%B8%ED%88%AC?id=X3mAg1s2l_s.P
    ----------------------------------------------------------------------------------------------------
    제목 :  메가 파이톤
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩7,700
    링크 :  https://play.google.com/store/movies/details/%EB%A9%94%EA%B0%80_%ED%8C%8C%EC%9D%B4%ED%86%A4?id=zUjLgr2G1nk.P
    ----------------------------------------------------------------------------------------------------
    제목 :  영화감독 노동주
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩7,700
    링크 :  https://play.google.com/store/movies/details/%EC%98%81%ED%99%94%EA%B0%90%EB%8F%85_%EB%85%B8%EB%8F%99%EC%A3%BC?id=wo0TXUUuAzM.P
    ----------------------------------------------------------------------------------------------------
    제목 :  아머어택
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EC%95%84%EB%A8%B8%EC%96%B4%ED%83%9D?id=izAdCVb0EzU.P
    ----------------------------------------------------------------------------------------------------
    제목 :  홀로코스트: 더 셰퍼드
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%ED%99%80%EB%A1%9C%EC%BD%94%EC%8A%A4%ED%8A%B8_%EB%8D%94_%EC%85%B0%ED%8D%BC%EB%93%9C?id=gmjF70HkK64.P
    ----------------------------------------------------------------------------------------------------
    제목 :  액시던트 맨: 히트맨의 휴가 Accident Man" Hitman's Holiday
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩4,000
    링크 :  https://play.google.com/store/movies/details/%EC%95%A1%EC%8B%9C%EB%8D%98%ED%8A%B8_%EB%A7%A8_%ED%9E%88%ED%8A%B8%EB%A7%A8%EC%9D%98_%ED%9C%B4%EA%B0%80_Accident_Man_Hitman_s_Holiday?id=glEFWidsRxQ.P
    ----------------------------------------------------------------------------------------------------
    제목 :  나나
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩7,000
    링크 :  https://play.google.com/store/movies/details/%EB%82%98%EB%82%98?id=m-t8cusWooc.P
    ----------------------------------------------------------------------------------------------------
    제목 :  팬픽에서 연애까지
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%ED%8C%AC%ED%94%BD%EC%97%90%EC%84%9C_%EC%97%B0%EC%95%A0%EA%B9%8C%EC%A7%80?id=NVaMziFQIaA.P
    ----------------------------------------------------------------------------------------------------
    제목 :  적인걸: 안양의 비밀
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EC%A0%81%EC%9D%B8%EA%B1%B8_%EC%95%88%EC%96%91%EC%9D%98_%EB%B9%84%EB%B0%80?id=4qZXZp25MkI.P
    ----------------------------------------------------------------------------------------------------
    제목 :  단단한 끌림 Sharp Stick
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩4,000
    링크 :  https://play.google.com/store/movies/details/%EB%8B%A8%EB%8B%A8%ED%95%9C_%EB%81%8C%EB%A6%BC_Sharp_Stick?id=ws-scDaAEM0.P
    ----------------------------------------------------------------------------------------------------
    제목 :  프리즌 77
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%ED%94%84%EB%A6%AC%EC%A6%8C_77?id=F2pCFpVgj1A.P
    ----------------------------------------------------------------------------------------------------
    제목 :  스페이스 키드: 우주에서 살아남기
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EC%8A%A4%ED%8E%98%EC%9D%B4%EC%8A%A4_%ED%82%A4%EB%93%9C_%EC%9A%B0%EC%A3%BC%EC%97%90%EC%84%9C_%EC%82%B4%EC%95%84%EB%82%A8%EA%B8%B0?id=_cNXf-vWyiE.P
    ----------------------------------------------------------------------------------------------------
    제목 :  수프와 이데올로기
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EC%88%98%ED%94%84%EC%99%80_%EC%9D%B4%EB%8D%B0%EC%98%AC%EB%A1%9C%EA%B8%B0?id=oZw8a66Vawo.P
    ----------------------------------------------------------------------------------------------------
    제목 :  마이 리틀 레나타
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EB%A7%88%EC%9D%B4_%EB%A6%AC%ED%8B%80_%EB%A0%88%EB%82%98%ED%83%80?id=24s-4eHd5D8.P
    ----------------------------------------------------------------------------------------------------
    제목 :  멀티버스 프로젝트
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EB%A9%80%ED%8B%B0%EB%B2%84%EC%8A%A4_%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8?id=l1FfuwVQhiM.P
    ----------------------------------------------------------------------------------------------------
    제목 :  쥬라기 킹덤
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EC%A5%AC%EB%9D%BC%EA%B8%B0_%ED%82%B9%EB%8D%A4?id=W1zbYWlBap8.P
    ----------------------------------------------------------------------------------------------------
    제목 :  마이선
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EB%A7%88%EC%9D%B4%EC%84%A0?id=oiqIk6k27OM.P
    ----------------------------------------------------------------------------------------------------
    제목 :  홀리 쉣!
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%ED%99%80%EB%A6%AC_%EC%89%A3?id=smgqxWS4aQ0.P
    ----------------------------------------------------------------------------------------------------
    제목 :  만인의 연인
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩4,500
    링크 :  https://play.google.com/store/movies/details/%EB%A7%8C%EC%9D%B8%EC%9D%98_%EC%97%B0%EC%9D%B8?id=s8wS4_mwHUg.P
    ----------------------------------------------------------------------------------------------------
    제목 :  너만을 위하여
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EB%84%88%EB%A7%8C%EC%9D%84_%EC%9C%84%ED%95%98%EC%97%AC?id=Wqeqs4m2kSs.P
    ----------------------------------------------------------------------------------------------------
    제목 :  크리스마스 캐럴
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩7,700
    링크 :  https://play.google.com/store/movies/details/%ED%81%AC%EB%A6%AC%EC%8A%A4%EB%A7%88%EC%8A%A4_%EC%BA%90%EB%9F%B4?id=Pz-qF19t3Hk.P
    ----------------------------------------------------------------------------------------------------
    제목 :  플레져
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%ED%94%8C%EB%A0%88%EC%A0%B8?id=rKIG9m37ldQ.P
    ----------------------------------------------------------------------------------------------------
    제목 :  요정
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩7,000
    링크 :  https://play.google.com/store/movies/details/%EC%9A%94%EC%A0%95?id=SQBkkmHcyzY.P
    ----------------------------------------------------------------------------------------------------
    제목 :  화피사2: 요괴전쟁
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%ED%99%94%ED%94%BC%EC%82%AC2_%EC%9A%94%EA%B4%B4%EC%A0%84%EC%9F%81?id=5DVSILWiJiQ.P
    ----------------------------------------------------------------------------------------------------
    제목 :  특공 여전사
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%ED%8A%B9%EA%B3%B5_%EC%97%AC%EC%A0%84%EC%82%AC?id=VwWTpL6G528.P
    ----------------------------------------------------------------------------------------------------
    제목 :  단청낭자
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EB%8B%A8%EC%B2%AD%EB%82%AD%EC%9E%90?id=ny9aQnatfxM.P
    ----------------------------------------------------------------------------------------------------
    제목 :  돈 워리 달링
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EB%8F%88_%EC%9B%8C%EB%A6%AC_%EB%8B%AC%EB%A7%81?id=95u-gG-mt0o.P
    ----------------------------------------------------------------------------------------------------
    제목 :  마이선
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EB%A7%88%EC%9D%B4%EC%84%A0?id=oiqIk6k27OM.P
    ----------------------------------------------------------------------------------------------------
    제목 :  에브리씽 에브리웨어 올 앳 원스
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EC%97%90%EB%B8%8C%EB%A6%AC%EC%94%BD_%EC%97%90%EB%B8%8C%EB%A6%AC%EC%9B%A8%EC%96%B4_%EC%98%AC_%EC%95%B3_%EC%9B%90%EC%8A%A4?id=9lJwvD-5x-s.P
    ----------------------------------------------------------------------------------------------------
    제목 :  캅시크릿
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EC%BA%85%EC%8B%9C%ED%81%AC%EB%A6%BF?id=D1ZXLHaEMJ4.P
    ----------------------------------------------------------------------------------------------------
    제목 :  인생은 아름다워
    할인전 가격 :  ₩7,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EC%9D%B8%EC%83%9D%EC%9D%80_%EC%95%84%EB%A6%84%EB%8B%A4%EC%9B%8C?id=NHqf1ozuQBw.P
    ----------------------------------------------------------------------------------------------------
    제목 :  데시벨
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EB%8D%B0%EC%8B%9C%EB%B2%A8?id=oaATNN-65kU.P
    ----------------------------------------------------------------------------------------------------
    제목 :  리멤버
    할인전 가격 :  ₩7,700
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EB%A6%AC%EB%A9%A4%EB%B2%84?id=FtRkaKP7Vfg.P
    ----------------------------------------------------------------------------------------------------
    제목 :  같은 속옷을 입는 두 여자
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EA%B0%99%EC%9D%80_%EC%86%8D%EC%98%B7%EC%9D%84_%EC%9E%85%EB%8A%94_%EB%91%90_%EC%97%AC%EC%9E%90?id=ZCfGjX_z9FM.P
    ----------------------------------------------------------------------------------------------------
    제목 :  통영에서의 하루
    할인전 가격 :  ₩7,700
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%ED%86%B5%EC%98%81%EC%97%90%EC%84%9C%EC%9D%98_%ED%95%98%EB%A3%A8?id=oWPSKn2E-8A.P
    ----------------------------------------------------------------------------------------------------
    제목 :  창밖은 겨울
    할인전 가격 :  ₩7,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EC%B0%BD%EB%B0%96%EC%9D%80_%EA%B2%A8%EC%9A%B8?id=HoAz36p5mVw.P
    ----------------------------------------------------------------------------------------------------
    제목 :  간병: 사랑의 족쇄
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EA%B0%84%EB%B3%91_%EC%82%AC%EB%9E%91%EC%9D%98_%EC%A1%B1%EC%87%84?id=KX7NWrQpK2s.P
    ----------------------------------------------------------------------------------------------------
    제목 :  팬픽에서 연애까지
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%ED%8C%AC%ED%94%BD%EC%97%90%EC%84%9C_%EC%97%B0%EC%95%A0%EA%B9%8C%EC%A7%80?id=NVaMziFQIaA.P
    ----------------------------------------------------------------------------------------------------
    제목 :  애프터 미투
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩7,700
    링크 :  https://play.google.com/store/movies/details/%EC%95%A0%ED%94%84%ED%84%B0_%EB%AF%B8%ED%88%AC?id=X3mAg1s2l_s.P
    ----------------------------------------------------------------------------------------------------
    제목 :  오마이키스
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EC%98%A4%EB%A7%88%EC%9D%B4%ED%82%A4%EC%8A%A4?id=Zy3vHColTUs.P
    ----------------------------------------------------------------------------------------------------
    제목 :  춤추는 드래그 퀸
    할인전 가격 :  ₩5,500
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EC%B6%A4%EC%B6%94%EB%8A%94_%EB%93%9C%EB%9E%98%EA%B7%B8_%ED%80%B8?id=cMvMKDcHCJk.P
    ----------------------------------------------------------------------------------------------------
    제목 :  1번국도
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/1%EB%B2%88%EA%B5%AD%EB%8F%84?id=hsGNbNibCHA.P
    ----------------------------------------------------------------------------------------------------
    제목 :  행복 속 위험한 동행
    할인전 가격 :  ₩5,500
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%ED%96%89%EB%B3%B5_%EC%86%8D_%EC%9C%84%ED%97%98%ED%95%9C_%EB%8F%99%ED%96%89?id=oCvVUz8WyKM.P
    ----------------------------------------------------------------------------------------------------
    제목 :  기문언갑사
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EA%B8%B0%EB%AC%B8%EC%96%B8%EA%B0%91%EC%82%AC?id=tvAHTQVNJnU.P
    ----------------------------------------------------------------------------------------------------
    제목 :  영화감독 노동주
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩7,700
    링크 :  https://play.google.com/store/movies/details/%EC%98%81%ED%99%94%EA%B0%90%EB%8F%85_%EB%85%B8%EB%8F%99%EC%A3%BC?id=wo0TXUUuAzM.P
    ----------------------------------------------------------------------------------------------------
    제목 :  해리포터 시리즈 완결 패키지
    할인전 가격 :  ₩36,000
    할인후 가격 :  ₩18,900
    링크 :  https://play.google.com/store/movies/details/%ED%95%B4%EB%A6%AC%ED%8F%AC%ED%84%B0_%EC%8B%9C%EB%A6%AC%EC%A6%88_%EC%99%84%EA%B2%B0_%ED%8C%A8%ED%82%A4%EC%A7%80?id=DsVgRu5dDdY
    ----------------------------------------------------------------------------------------------------
    제목 :  인디아나 존스: 모험 컬렉션
    할인전 가격 :  ₩20,000
    할인후 가격 :  ₩11,400
    링크 :  https://play.google.com/store/movies/details/%EC%9D%B8%EB%94%94%EC%95%84%EB%82%98_%EC%A1%B4%EC%8A%A4_%EB%AA%A8%ED%97%98_%EC%BB%AC%EB%A0%89%EC%85%98?id=Yv_OthKEftU
    ----------------------------------------------------------------------------------------------------
    제목 :  반지의 제왕: 3 영화 컬렉션 확장판 (자막판)
    할인전 가격 :  ₩13,500
    할인후 가격 :  ₩11,900
    링크 :  https://play.google.com/store/movies/details/%EB%B0%98%EC%A7%80%EC%9D%98_%EC%A0%9C%EC%99%95_3_%EC%98%81%ED%99%94_%EC%BB%AC%EB%A0%89%EC%85%98_%ED%99%95%EC%9E%A5%ED%8C%90_%EC%9E%90%EB%A7%89%ED%8C%90?id=80DJns6yqWs
    ----------------------------------------------------------------------------------------------------
    제목 :  탑건 2 무비 컬렉션
    할인전 가격 :  ₩13,500
    할인후 가격 :  ₩12,000
    링크 :  https://play.google.com/store/movies/details/%ED%83%91%EA%B1%B4_2_%EB%AC%B4%EB%B9%84_%EC%BB%AC%EB%A0%89%EC%85%98?id=1aLgJ4syNGk.P
    ----------------------------------------------------------------------------------------------------
    제목 :  신비한 동물 영화 3편 패키지
    할인전 가격 :  ₩15,000
    할인후 가격 :  ₩11,900
    링크 :  https://play.google.com/store/movies/details/%EC%8B%A0%EB%B9%84%ED%95%9C_%EB%8F%99%EB%AC%BC_%EC%98%81%ED%99%94_3%ED%8E%B8_%ED%8C%A8%ED%82%A4%EC%A7%80?id=_b1ycFl_mFU.P
    ----------------------------------------------------------------------------------------------------
    제목 :  해리포터풀패키지(1-7B)
    할인전 가격 :  ₩36,000
    할인후 가격 :  ₩30,400
    링크 :  https://play.google.com/store/movies/details/%ED%95%B4%EB%A6%AC%ED%8F%AC%ED%84%B0%ED%92%80%ED%8C%A8%ED%82%A4%EC%A7%80_1_7B?id=oV2yIteiUjY.P
    ----------------------------------------------------------------------------------------------------
    제목 :  맨인블랙 시리즈 패키지 (자막판)
    할인전 가격 :  ₩12,000
    할인후 가격 :  ₩7,500
    링크 :  https://play.google.com/store/movies/details/%EB%A7%A8%EC%9D%B8%EB%B8%94%EB%9E%99_%EC%8B%9C%EB%A6%AC%EC%A6%88_%ED%8C%A8%ED%82%A4%EC%A7%80_%EC%9E%90%EB%A7%89%ED%8C%90?id=TiFtAEp2KVo
    ----------------------------------------------------------------------------------------------------
    제목 :  트랜스포머 영화 5편 컬렉션
    할인전 가격 :  ₩25,000
    할인후 가격 :  ₩16,000
    링크 :  https://play.google.com/store/movies/details/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B8_%EC%98%81%ED%99%94_5%ED%8E%B8_%EC%BB%AC%EB%A0%89%EC%85%98?id=-wBC9omKCv0
    ----------------------------------------------------------------------------------------------------
    제목 :  토르 4 무비 컬렉션
    할인전 가격 :  ₩26,400
    할인후 가격 :  ₩18,900
    링크 :  https://play.google.com/store/movies/details/%ED%86%A0%EB%A5%B4_4_%EB%AC%B4%EB%B9%84_%EC%BB%AC%EB%A0%89%EC%85%98?id=qk4UsCD3K6c.P
    ----------------------------------------------------------------------------------------------------
    제목 :  조커/ 더 배트맨 영화 컬렉션
    할인전 가격 :  ₩10,500
    할인후 가격 :  ₩8,600
    링크 :  https://play.google.com/store/movies/details/%EC%A1%B0%EC%BB%A4_%EB%8D%94_%EB%B0%B0%ED%8A%B8%EB%A7%A8_%EC%98%81%ED%99%94_%EC%BB%AC%EB%A0%89%EC%85%98?id=ETFncwTsz40.P
    ----------------------------------------------------------------------------------------------------
    제목 :  미션 임파서블 1-5 컬렉션 (자막판)
    할인전 가격 :  ₩25,000
    할인후 가격 :  ₩12,400
    링크 :  https://play.google.com/store/movies/details/%EB%AF%B8%EC%85%98_%EC%9E%84%ED%8C%8C%EC%84%9C%EB%B8%94_1_5_%EC%BB%AC%EB%A0%89%EC%85%98_%EC%9E%90%EB%A7%89%ED%8C%90?id=Nkdb-Bl3WeU
    ----------------------------------------------------------------------------------------------------
    제목 :  라이트이어+토이 스토리(1~4) 무비 컬렉션
    할인전 가격 :  ₩42,000
    할인후 가격 :  ₩25,900
    링크 :  https://play.google.com/store/movies/details/%EB%9D%BC%EC%9D%B4%ED%8A%B8%EC%9D%B4%EC%96%B4_%ED%86%A0%EC%9D%B4_%EC%8A%A4%ED%86%A0%EB%A6%AC_1_4_%EB%AC%B4%EB%B9%84_%EC%BB%AC%EB%A0%89%EC%85%98?id=CN9QGsn95zA.P
    ----------------------------------------------------------------------------------------------------
    제목 :  MIB 맨 인 블랙 풀 패키지 (자막판)
    할인전 가격 :  ₩16,000
    할인후 가격 :  ₩8,550
    링크 :  https://play.google.com/store/movies/details/MIB_%EB%A7%A8_%EC%9D%B8_%EB%B8%94%EB%9E%99_%ED%92%80_%ED%8C%A8%ED%82%A4%EC%A7%80_%EC%9E%90%EB%A7%89%ED%8C%90?id=gjNbS93yMyg
    ----------------------------------------------------------------------------------------------------
    제목 :  50가지 그림자: 3 무비 콜렉션
    할인전 가격 :  ₩10,500
    할인후 가격 :  ₩7,400
    링크 :  https://play.google.com/store/movies/details/50%EA%B0%80%EC%A7%80_%EA%B7%B8%EB%A6%BC%EC%9E%90_3_%EB%AC%B4%EB%B9%84_%EC%BD%9C%EB%A0%89%EC%85%98?id=mFyAj0YjCKM
    ----------------------------------------------------------------------------------------------------
    제목 :  저스티스 리그 패키지 (자막판)
    할인전 가격 :  ₩13,500
    할인후 가격 :  ₩13,400
    링크 :  https://play.google.com/store/movies/details/%EC%A0%80%EC%8A%A4%ED%8B%B0%EC%8A%A4_%EB%A6%AC%EA%B7%B8_%ED%8C%A8%ED%82%A4%EC%A7%80_%EC%9E%90%EB%A7%89%ED%8C%90?id=UoFllXXcy0w
    ----------------------------------------------------------------------------------------------------
    제목 :  슈퍼배드 & 미니언즈: 4편의 무비 컬렉션 (더빙판)
    할인전 가격 :  ₩14,000
    할인후 가격 :  ₩9,800
    링크 :  https://play.google.com/store/movies/details/%EC%8A%88%ED%8D%BC%EB%B0%B0%EB%93%9C_%EB%AF%B8%EB%8B%88%EC%96%B8%EC%A6%88_4%ED%8E%B8%EC%9D%98_%EB%AC%B4%EB%B9%84_%EC%BB%AC%EB%A0%89%EC%85%98_%EB%8D%94%EB%B9%99%ED%8C%90?id=YJFC3dXPH9Y
    ----------------------------------------------------------------------------------------------------
    제목 :  드래곤 길들이기 3부작 (더빙판)
    할인전 가격 :  ₩13,500
    할인후 가격 :  ₩12,000
    링크 :  https://play.google.com/store/movies/details/%EB%93%9C%EB%9E%98%EA%B3%A4_%EA%B8%B8%EB%93%A4%EC%9D%B4%EA%B8%B0_3%EB%B6%80%EC%9E%91_%EB%8D%94%EB%B9%99%ED%8C%90?id=lbGJnOFGtu4
    ----------------------------------------------------------------------------------------------------
    제목 :  해리포터 완결 패키지 / Wizarding World / 해리포터 20주년 기념: 리턴 투 호그와트
    할인전 가격 :  ₩42,000
    할인후 가격 :  ₩39,000
    링크 :  https://play.google.com/store/movies/details/%ED%95%B4%EB%A6%AC%ED%8F%AC%ED%84%B0_%EC%99%84%EA%B2%B0_%ED%8C%A8%ED%82%A4%EC%A7%80_Wizarding_World_%ED%95%B4%EB%A6%AC%ED%8F%AC%ED%84%B0_20%EC%A3%BC%EB%85%84_%EA%B8%B0%EB%85%90_%EB%A6%AC%ED%84%B4_%ED%88%AC_%ED%98%B8%EA%B7%B8%EC%99%80%ED%8A%B8?id=6OGQyXLo9QM.P
    ----------------------------------------------------------------------------------------------------
    제목 :  거미줄에 걸린 소녀 / 밀레니엄: 여자를 증오한 남자들 (자막판)
    할인전 가격 :  ₩8,000
    할인후 가격 :  ₩4,250
    링크 :  https://play.google.com/store/movies/details/%EA%B1%B0%EB%AF%B8%EC%A4%84%EC%97%90_%EA%B1%B8%EB%A6%B0_%EC%86%8C%EB%85%80_%EB%B0%80%EB%A0%88%EB%8B%88%EC%97%84_%EC%97%AC%EC%9E%90%EB%A5%BC_%EC%A6%9D%EC%98%A4%ED%95%9C_%EB%82%A8%EC%9E%90%EB%93%A4_%EC%9E%90%EB%A7%89%ED%8C%90?id=x1ZLmqkC408
    ----------------------------------------------------------------------------------------------------
    제목 :  일루미네이션 엔터테인먼트가 제공하는 6편의 무비 컬렉션. ('슈퍼배드', '슈퍼배드 2', '미니언즈 ...
    할인전 가격 :  ₩21,000
    할인후 가격 :  ₩14,900
    링크 :  https://play.google.com/store/movies/details/%EC%9D%BC%EB%A3%A8%EB%AF%B8%EB%84%A4%EC%9D%B4%EC%85%98_%EC%97%94%ED%84%B0%ED%85%8C%EC%9D%B8%EB%A8%BC%ED%8A%B8%EA%B0%80_%EC%A0%9C%EA%B3%B5%ED%95%98%EB%8A%94_6%ED%8E%B8%EC%9D%98_%EB%AC%B4%EB%B9%84_%EC%BB%AC%EB%A0%89%EC%85%98_%EC%8A%88%ED%8D%BC%EB%B0%B0%EB%93%9C_%EC%8A%88%ED%8D%BC%EB%B0%B0%EB%93%9C_2_%EB%AF%B8%EB%8B%88%EC%96%B8%EC%A6%88?id=CWA6UHtG4BA
    ----------------------------------------------------------------------------------------------------
    제목 :  도리 & 니모 패키지 (더빙판) (자막판)
    할인전 가격 :  ₩16,800
    할인후 가격 :  ₩10,900
    링크 :  https://play.google.com/store/movies/details/%EB%8F%84%EB%A6%AC_%EB%8B%88%EB%AA%A8_%ED%8C%A8%ED%82%A4%EC%A7%80_%EB%8D%94%EB%B9%99%ED%8C%90_%EC%9E%90%EB%A7%89%ED%8C%90?id=AFsVJMFh8nY
    ----------------------------------------------------------------------------------------------------
    제목 :  램페이지/샌 안드레아스 영화 패키지 (자막판)
    할인전 가격 :  ₩9,000
    할인후 가격 :  ₩8,600
    링크 :  https://play.google.com/store/movies/details/%EB%9E%A8%ED%8E%98%EC%9D%B4%EC%A7%80_%EC%83%8C_%EC%95%88%EB%93%9C%EB%A0%88%EC%95%84%EC%8A%A4_%EC%98%81%ED%99%94_%ED%8C%A8%ED%82%A4%EC%A7%80_%EC%9E%90%EB%A7%89%ED%8C%90?id=wyN7r4yPVwI
    ----------------------------------------------------------------------------------------------------
    제목 :  샤잠!/아쿠아맨 2 영화 컬렉션 (자막판)
    할인전 가격 :  ₩9,000
    할인후 가격 :  ₩8,600
    링크 :  https://play.google.com/store/movies/details/%EC%83%A4%EC%9E%A0_%EC%95%84%EC%BF%A0%EC%95%84%EB%A7%A8_2_%EC%98%81%ED%99%94_%EC%BB%AC%EB%A0%89%EC%85%98_%EC%9E%90%EB%A7%89%ED%8C%90?id=-7ZiNTHDrvI
    ----------------------------------------------------------------------------------------------------
    제목 :  킹스맨 무비 컬렉션
    할인전 가격 :  ₩12,000
    할인후 가격 :  ₩7,800
    링크 :  https://play.google.com/store/movies/details/%ED%82%B9%EC%8A%A4%EB%A7%A8_%EB%AC%B4%EB%B9%84_%EC%BB%AC%EB%A0%89%EC%85%98?id=mvAuoBJ9vU0
    ----------------------------------------------------------------------------------------------------
    제목 :  콩: 스컬 아일랜드 / 고질라 영화 패키지
    할인전 가격 :  ₩9,000
    할인후 가격 :  ₩8,600
    링크 :  https://play.google.com/store/movies/details/%EC%BD%A9_%EC%8A%A4%EC%BB%AC_%EC%95%84%EC%9D%BC%EB%9E%9C%EB%93%9C_%EA%B3%A0%EC%A7%88%EB%9D%BC_%EC%98%81%ED%99%94_%ED%8C%A8%ED%82%A4%EC%A7%80?id=AMV8EwqDFgM
    ----------------------------------------------------------------------------------------------------
    제목 :  수퍼 소닉 2 영화 패키지
    할인전 가격 :  ₩12,000
    할인후 가격 :  ₩10,000
    링크 :  https://play.google.com/store/movies/details/%EC%88%98%ED%8D%BC_%EC%86%8C%EB%8B%89_2_%EC%98%81%ED%99%94_%ED%8C%A8%ED%82%A4%EC%A7%80?id=NVFlbYIt29I.P
    ----------------------------------------------------------------------------------------------------
    제목 :  트롤: 2편의 무비 컬렉션
    할인전 가격 :  ₩9,000
    할인후 가격 :  ₩5,150
    링크 :  https://play.google.com/store/movies/details/%ED%8A%B8%EB%A1%A4_2%ED%8E%B8%EC%9D%98_%EB%AC%B4%EB%B9%84_%EC%BB%AC%EB%A0%89%EC%85%98?id=UuSaHprst-A.P
    ----------------------------------------------------------------------------------------------------
    제목 :  에이리언 VS. 프레데터 콜렉션 (자막판)
    할인전 가격 :  ₩12,000
    할인후 가격 :  ₩7,800
    링크 :  https://play.google.com/store/movies/details/%EC%97%90%EC%9D%B4%EB%A6%AC%EC%96%B8_VS_%ED%94%84%EB%A0%88%EB%8D%B0%ED%84%B0_%EC%BD%9C%EB%A0%89%EC%85%98_%EC%9E%90%EB%A7%89%ED%8C%90?id=VrU_V6a_Wo8
    ----------------------------------------------------------------------------------------------------
    제목 :  레고 배트맨 무비 / 레고 무비 : 레고영화패키지 (자막판)
    할인전 가격 :  ₩9,000
    할인후 가격 :  ₩8,600
    링크 :  https://play.google.com/store/movies/details/%EB%A0%88%EA%B3%A0_%EB%B0%B0%ED%8A%B8%EB%A7%A8_%EB%AC%B4%EB%B9%84_%EB%A0%88%EA%B3%A0_%EB%AC%B4%EB%B9%84_%EB%A0%88%EA%B3%A0%EC%98%81%ED%99%94%ED%8C%A8%ED%82%A4%EC%A7%80_%EC%9E%90%EB%A7%89%ED%8C%90?id=eSmkyBvaxW8
    ----------------------------------------------------------------------------------------------------
    제목 :  컨저링 유니버스 영화 컬렉션 (자막판)
    할인전 가격 :  ₩22,500
    할인후 가격 :  ₩21,400
    링크 :  https://play.google.com/store/movies/details/%EC%BB%A8%EC%A0%80%EB%A7%81_%EC%9C%A0%EB%8B%88%EB%B2%84%EC%8A%A4_%EC%98%81%ED%99%94_%EC%BB%AC%EB%A0%89%EC%85%98_%EC%9E%90%EB%A7%89%ED%8C%90?id=ExV_uBTCi18
    ----------------------------------------------------------------------------------------------------
    제목 :  벤 애플렉 영화 컬렉션
    할인전 가격 :  ₩18,000
    할인후 가격 :  ₩14,900
    링크 :  https://play.google.com/store/movies/details/%EB%B2%A4_%EC%95%A0%ED%94%8C%EB%A0%89_%EC%98%81%ED%99%94_%EC%BB%AC%EB%A0%89%EC%85%98?id=akIqqd8NMlQ
    ----------------------------------------------------------------------------------------------------
    제목 :  겟아웃 / 어스 컬렉션 (자막판)
    할인전 가격 :  ₩7,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EA%B2%9F%EC%95%84%EC%9B%83_%EC%96%B4%EC%8A%A4_%EC%BB%AC%EB%A0%89%EC%85%98_%EC%9E%90%EB%A7%89%ED%8C%90?id=3e_ILWEiEPA
    ----------------------------------------------------------------------------------------------------
    제목 :  타일러 페리의 부!, 부2! 마데의 핼러윈 (자막판)
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%ED%83%80%EC%9D%BC%EB%9F%AC_%ED%8E%98%EB%A6%AC%EC%9D%98_%EB%B6%80_%EB%B6%802_%EB%A7%88%EB%8D%B0%EC%9D%98_%ED%95%BC%EB%9F%AC%EC%9C%88_%EC%9E%90%EB%A7%89%ED%8C%90?id=R3uujTR-kMs
    ----------------------------------------------------------------------------------------------------
    제목 :  홈즈 & 왓슨 / 스텝 브라더스 / 탤러데가 나이트 - 릭키 바비의 발라드 (자막판)
    할인전 가격 :  ₩12,000
    할인후 가격 :  ₩7,000
    링크 :  https://play.google.com/store/movies/details/%ED%99%88%EC%A6%88_%EC%99%93%EC%8A%A8_%EC%8A%A4%ED%85%9D_%EB%B8%8C%EB%9D%BC%EB%8D%94%EC%8A%A4_%ED%83%A4%EB%9F%AC%EB%8D%B0%EA%B0%80_%EB%82%98%EC%9D%B4%ED%8A%B8_%EB%A6%AD%ED%82%A4_%EB%B0%94%EB%B9%84%EC%9D%98_%EB%B0%9C%EB%9D%BC%EB%93%9C_%EC%9E%90%EB%A7%89%ED%8C%90?id=rgj5DNnjCx8
    ----------------------------------------------------------------------------------------------------
    제목 :  킹 아서: 제왕의 검/레전드 오브 타잔 영화 패키지
    할인전 가격 :  ₩9,000
    할인후 가격 :  ₩8,600
    링크 :  https://play.google.com/store/movies/details/%ED%82%B9_%EC%95%84%EC%84%9C_%EC%A0%9C%EC%99%95%EC%9D%98_%EA%B2%80_%EB%A0%88%EC%A0%84%EB%93%9C_%EC%98%A4%EB%B8%8C_%ED%83%80%EC%9E%94_%EC%98%81%ED%99%94_%ED%8C%A8%ED%82%A4%EC%A7%80?id=iFeIgxO_rqo
    ----------------------------------------------------------------------------------------------------
    제목 :  신비한 동물들과 그린델왈드의 범죄/ 신비한 동물 사전 영화 패키지 (자막판)
    할인전 가격 :  ₩9,000
    할인후 가격 :  ₩8,600
    링크 :  https://play.google.com/store/movies/details/%EC%8B%A0%EB%B9%84%ED%95%9C_%EB%8F%99%EB%AC%BC%EB%93%A4%EA%B3%BC_%EA%B7%B8%EB%A6%B0%EB%8D%B8%EC%99%88%EB%93%9C%EC%9D%98_%EB%B2%94%EC%A3%84_%EC%8B%A0%EB%B9%84%ED%95%9C_%EB%8F%99%EB%AC%BC_%EC%82%AC%EC%A0%84_%EC%98%81%ED%99%94_%ED%8C%A8%ED%82%A4%EC%A7%80_%EC%9E%90%EB%A7%89%ED%8C%90?id=v8OGmxrY3oA
    ----------------------------------------------------------------------------------------------------
    제목 :  그것/ 샤이닝 영화 패키지 (자막판)
    할인전 가격 :  ₩9,000
    할인후 가격 :  ₩8,600
    링크 :  https://play.google.com/store/movies/details/%EA%B7%B8%EA%B2%83_%EC%83%A4%EC%9D%B4%EB%8B%9D_%EC%98%81%ED%99%94_%ED%8C%A8%ED%82%A4%EC%A7%80_%EC%9E%90%EB%A7%89%ED%8C%90?id=9Y7OuJJ4aYw
    ----------------------------------------------------------------------------------------------------
    제목 :  Space Jam 2-Film Collection
    할인전 가격 :  ₩9,000
    할인후 가격 :  ₩6,600
    링크 :  https://play.google.com/store/movies/details/Space_Jam_2_Film_Collection?id=ty3R99NWZR4.P
    ----------------------------------------------------------------------------------------------------
    제목 :  돈 워리 달링
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EB%8F%88_%EC%9B%8C%EB%A6%AC_%EB%8B%AC%EB%A7%81?id=95u-gG-mt0o.P
    ----------------------------------------------------------------------------------------------------
    제목 :  오펀: 천사의 탄생
    할인전 가격 :  ₩7,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EC%98%A4%ED%8E%80_%EC%B2%9C%EC%82%AC%EC%9D%98_%ED%83%84%EC%83%9D?id=LZZ0BoZNVvg.P
    ----------------------------------------------------------------------------------------------------
    제목 :  놈이 우리 안에 있다
    할인전 가격 :  ₩7,000
    할인후 가격 :  ₩4,500
    링크 :  https://play.google.com/store/movies/details/%EB%86%88%EC%9D%B4_%EC%9A%B0%EB%A6%AC_%EC%95%88%EC%97%90_%EC%9E%88%EB%8B%A4?id=qjB0ogGXito.P
    ----------------------------------------------------------------------------------------------------
    제목 :  프레이 포 더 데블
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩7,700
    링크 :  https://play.google.com/store/movies/details/%ED%94%84%EB%A0%88%EC%9D%B4_%ED%8F%AC_%EB%8D%94_%EB%8D%B0%EB%B8%94?id=QVhB7omoDd8.P
    ----------------------------------------------------------------------------------------------------
    제목 :  마이선
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EB%A7%88%EC%9D%B4%EC%84%A0?id=oiqIk6k27OM.P
    ----------------------------------------------------------------------------------------------------
    제목 :  엑소시즘 : 신과 퇴마사
    할인전 가격 :  ₩5,500
    할인후 가격 :  ₩2,750
    링크 :  https://play.google.com/store/movies/details/%EC%97%91%EC%86%8C%EC%8B%9C%EC%A6%98_%EC%8B%A0%EA%B3%BC_%ED%87%B4%EB%A7%88%EC%82%AC?id=ba-N4if20nI.P
    ----------------------------------------------------------------------------------------------------
    제목 :  헬 카운트
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%ED%97%AC_%EC%B9%B4%EC%9A%B4%ED%8A%B8?id=BTq3HQuwHKA.P
    ----------------------------------------------------------------------------------------------------
    제목 :  도쿄 괴담
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EB%8F%84%EC%BF%84_%EA%B4%B4%EB%8B%B4?id=AyjSIkuIiY0.P
    ----------------------------------------------------------------------------------------------------
    제목 :  너만을 위하여
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EB%84%88%EB%A7%8C%EC%9D%84_%EC%9C%84%ED%95%98%EC%97%AC?id=Wqeqs4m2kSs.P
    ----------------------------------------------------------------------------------------------------
    제목 :  인 드림즈
    할인전 가격 :  ₩5,000
    할인후 가격 :  ₩2,500
    링크 :  https://play.google.com/store/movies/details/%EC%9D%B8_%EB%93%9C%EB%A6%BC%EC%A6%88?id=XpjbF20cLzs.P
    ----------------------------------------------------------------------------------------------------
    제목 :  멀티버스 프로젝트
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EB%A9%80%ED%8B%B0%EB%B2%84%EC%8A%A4_%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8?id=l1FfuwVQhiM.P
    ----------------------------------------------------------------------------------------------------
    제목 :  주연
    할인전 가격 :  ₩9,900
    할인후 가격 :  ₩6,500
    링크 :  https://play.google.com/store/movies/details/%EC%A3%BC%EC%97%B0?id=soUO9betAWU.P
    ----------------------------------------------------------------------------------------------------
    제목 :  미혹
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EB%AF%B8%ED%98%B9?id=vBapmmd3TEs.P
    ----------------------------------------------------------------------------------------------------
    제목 :  돈 워리 달링
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EB%8F%88_%EC%9B%8C%EB%A6%AC_%EB%8B%AC%EB%A7%81?id=95u-gG-mt0o.P
    ----------------------------------------------------------------------------------------------------
    제목 :  블레이드 퍼피 워리어
    할인전 가격 :  ₩5,000
    할인후 가격 :  ₩1,800
    링크 :  https://play.google.com/store/movies/details/%EB%B8%94%EB%A0%88%EC%9D%B4%EB%93%9C_%ED%8D%BC%ED%94%BC_%EC%9B%8C%EB%A6%AC%EC%96%B4?id=ZtlnwvrgOzM.P
    ----------------------------------------------------------------------------------------------------
    제목 :  마이선
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EB%A7%88%EC%9D%B4%EC%84%A0?id=oiqIk6k27OM.P
    ----------------------------------------------------------------------------------------------------
    제목 :  에브리씽 에브리웨어 올 앳 원스
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EC%97%90%EB%B8%8C%EB%A6%AC%EC%94%BD_%EC%97%90%EB%B8%8C%EB%A6%AC%EC%9B%A8%EC%96%B4_%EC%98%AC_%EC%95%B3_%EC%9B%90%EC%8A%A4?id=9lJwvD-5x-s.P
    ----------------------------------------------------------------------------------------------------
    제목 :  이스터 선데이
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩7,700
    링크 :  https://play.google.com/store/movies/details/%EC%9D%B4%EC%8A%A4%ED%84%B0_%EC%84%A0%EB%8D%B0%EC%9D%B4?id=X9bXIOPclkU.P
    ----------------------------------------------------------------------------------------------------
    제목 :  아우슈비츠 챔피언
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EC%95%84%EC%9A%B0%EC%8A%88%EB%B9%84%EC%B8%A0_%EC%B1%94%ED%94%BC%EC%96%B8?id=kgacYWsTIAg.P
    ----------------------------------------------------------------------------------------------------
    제목 :  페르시아어 수업
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩7,000
    링크 :  https://play.google.com/store/movies/details/%ED%8E%98%EB%A5%B4%EC%8B%9C%EC%95%84%EC%96%B4_%EC%88%98%EC%97%85?id=iEfgHuCCXVU.P
    ----------------------------------------------------------------------------------------------------
    제목 :  프리즌 77
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%ED%94%84%EB%A6%AC%EC%A6%8C_77?id=F2pCFpVgj1A.P
    ----------------------------------------------------------------------------------------------------
    제목 :  지옥의 화원
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩7,000
    링크 :  https://play.google.com/store/movies/details/%EC%A7%80%EC%98%A5%EC%9D%98_%ED%99%94%EC%9B%90?id=T4QD41_G3Oo.P
    ----------------------------------------------------------------------------------------------------
    제목 :  매복
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩7,700
    링크 :  https://play.google.com/store/movies/details/%EB%A7%A4%EB%B3%B5?id=k1-jHarOtYo.P
    ----------------------------------------------------------------------------------------------------
    제목 :  홀로코스트: 더 셰퍼드
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%ED%99%80%EB%A1%9C%EC%BD%94%EC%8A%A4%ED%8A%B8_%EB%8D%94_%EC%85%B0%ED%8D%BC%EB%93%9C?id=gmjF70HkK64.P
    ----------------------------------------------------------------------------------------------------
    제목 :  몬스터 신부: 101번째 프로포즈
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩7,700
    링크 :  https://play.google.com/store/movies/details/%EB%AA%AC%EC%8A%A4%ED%84%B0_%EC%8B%A0%EB%B6%80_101%EB%B2%88%EC%A7%B8_%ED%94%84%EB%A1%9C%ED%8F%AC%EC%A6%88?id=dwBwvD2snDc.P
    ----------------------------------------------------------------------------------------------------
    제목 :  압꾸정
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩7,000
    링크 :  https://play.google.com/store/movies/details/%EC%95%95%EA%BE%B8%EC%A0%95?id=cfTIZM5KJrQ.P
    ----------------------------------------------------------------------------------------------------
    제목 :  리멤버
    할인전 가격 :  ₩7,700
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EB%A6%AC%EB%A9%A4%EB%B2%84?id=FtRkaKP7Vfg.P
    ----------------------------------------------------------------------------------------------------
    제목 :  간병: 사랑의 족쇄
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%EA%B0%84%EB%B3%91_%EC%82%AC%EB%9E%91%EC%9D%98_%EC%A1%B1%EC%87%84?id=KX7NWrQpK2s.P
    ----------------------------------------------------------------------------------------------------
    제목 :  팬픽에서 연애까지
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩5,000
    링크 :  https://play.google.com/store/movies/details/%ED%8C%AC%ED%94%BD%EC%97%90%EC%84%9C_%EC%97%B0%EC%95%A0%EA%B9%8C%EC%A7%80?id=NVaMziFQIaA.P
    ----------------------------------------------------------------------------------------------------
    제목 :  라스트필름
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩7,700
    링크 :  https://play.google.com/store/movies/details/%EB%9D%BC%EC%8A%A4%ED%8A%B8%ED%95%84%EB%A6%84?id=dWJlTVOKW8g.P
    ----------------------------------------------------------------------------------------------------
    제목 :  은밀한 동거
    할인전 가격 :  ₩11,000
    할인후 가격 :  ₩5,500
    링크 :  https://play.google.com/store/movies/details/%EC%9D%80%EB%B0%80%ED%95%9C_%EB%8F%99%EA%B1%B0?id=gkiLEib5SHQ.P
    ----------------------------------------------------------------------------------------------------
    제목 :  만인의 연인
    할인전 가격 :  ₩10,000
    할인후 가격 :  ₩4,500
    링크 :  https://play.google.com/store/movies/details/%EB%A7%8C%EC%9D%B8%EC%9D%98_%EC%97%B0%EC%9D%B8?id=s8wS4_mwHUg.P
    ----------------------------------------------------------------------------------------------------
    136
    


```python

```

## Reference
- [웹 크롤링 & 데이터 분석 with 파이썬 (장철원 저)](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=286684632)
- [데이터 분석을 위한 파이썬 철저 입문 (최은석 저)](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=189403433)
- [인프런 파이썬 무료 강의 (활용편3) - 웹 스크래핑 (5시간)](https://www.inflearn.com/course/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EC%9B%B9-%EC%8A%A4%ED%81%AC%EB%9E%98%ED%95%91)