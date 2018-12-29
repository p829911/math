# WAS (web application server)

서버에서 동작하게 하는 앱 client url로 요청하면 서버에서 어플이 실행된다.

python에서 가장 많이 쓰는 언어 장고 그다음으로 많이 쓰는것이 flask

장고는 무겁고 배우기가 어렵다 하지만 적은 코드로 많은 기능을 실행할 수 있다. 

플라스크는 가볍고 배우기 쉽다. 장고보다 2배 빠르다.

hello world를 얼마나 빨리 띄우느냐에 따라 성능 평가를 한다.

1. local에서 만들어 본다. 127.0.0.1:5000
2. 만든 코드를 서버에 올려서 IP로 통신할 수 있게 한다. 
3. 외부 아이피와 내부 아이피를 연결해 주는 proxy 서비스 - nginx
4. 나이브 베이지안을 이용해서 네이버 기사 카테고리 내용으로 학습 시켜본다.
5. 플라스크 서버 안에 4번의 model을 심는다.
6. 문장을 넣으면 서버의 model이 카테고리를 분류하여 알려준다.
7. slack을 이용하여 chatbot을 만든다. incoming webhook - outgoing webhook
8. client에서 slack 서버로 명령을 주면 slack 서버에서 우리가 만든 서버로 결과물을 전송한다.



# REST API

path안에 데이터를 포함해 보내는 것

- REST: Representational Safe Transfer
- API: Application Programming Interface
- Roy Fielding에 의해 웹의 장점을 최대한 활용할 수 있는 아키텍쳐



- Nomally API
  - `Http://naver.com/news.nhn?category=101&aid=62113`
  - key = value형태
- REST API
  - `Http://naver.com/news/101/62113`
  - Example: darksky api
  - `https://api.darksky.net/forecast/c259d4ae/37.8267,-112.4233`
  - key값이 없어 사용자가 URL구조를 쉽게 파악할 수 없고, 문자열이 줄어든다.



#### API Design

- 동사는 사용하지 않는다.
  - /getArticles (x)
  - /articles (0)
- 단수보다 복수가 좋다.
  - /article (x)
  - /articles (0)
- 추상적인 것보다 구체적인게 좋다.
  - /readablething (x)
  - /news, /cartoon, /movie, /music (0)
- 뒤에 query를 안쓰는 것은 아니다.
  - 입력 정보와 같은 데이터는 query를 사용한다.
  - /news/sports?q=평창올림픽
- 설계를 잘 해서 사용해야 한다.
  - 나중에 유지 보수를 할 때 편하다.



|                      | Create               | Read                      | Update                  | Delete              |
| -------------------- | -------------------- | ------------------------- | ----------------------- | ------------------- |
| Resource             | POST                 | GET                       | PUT                     | DELETE              |
| /group/              | 새로운 그룹 추가     | 해당 그룹 데이터 조회     | -                       | 해당 그룹 삭제      |
| /group/category/     | 새로운 카테고리 추가 | 해당 카테고리 데이터 조회 | -                       | 해당 카테고리 삭제  |
| /group/category/{id} | 새로운 데이터 추가   | 해당 id 데이터 조회       | 해당 id 데이터 업데이트 | 해당 id 데이터 삭제 |



| group   | category           | {id}           |
| ------- | ------------------ | -------------- |
| news    | sports, it,  world | 1, 2, 3, 8, 9  |
| cartoon | action, drama      | 4, 5, 6, 7, 10 |



# flask

hello

hello.py

stataic

templates

profile.html



nginx default port: 80, 8080

```bash
sudo apt install nginx
sudo systemctl status nginx
```



cd /etc/nginx

nginx.conf: 서버 몇초마다 끊어라

sites-available: 몇 번 포트로 들어가면 특정 html 띄워라

- default 파일 수정
- listen 80 default_server;
- root /var/www/html;
- 80번 포트로 들어갔을 때 root 폴더에 있는 html을 실행하겠다.
- 80번 포트를 사용할 것이기 때문에 원래 있던 코드의 포트를 9999로 바꿔준다.

```bash
vi /etc/nginx/sites-available/default
```





```bash
sudo systemctl restart nginx
```



freenom: 도메인 만들어 주는 사이트



## Incoming webhook, Outgoing webhook

개인 workspace -> manage apps

outgoing webhook: channel, Trigger Word, URL

URL: requests 할 때 사용되는 URL  

nginx: 외부망으로 접속할 수 있게 해주는 서비스

local pc 방화벽 뚫어 주는 서비스 local PC tunnel (ngrok)



