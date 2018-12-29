# AWS, local chatbot

Incoming webhook: slack server - > client

outgoing webhook: client -> slack server



TF-IDF

MNB

네이버 기사 데이터 Classification

pickle file 저장 -> flask를 이용해 웹 어플리케이션에 탑재 -> AWS에 올린다. -> bootstrap, highchart, 



### flask code

```python
from flask import *
from libs.slack import send_slack
from libs.forecast import forecast
from libs.naver import naver_keywords

app = Flask(__name__)

@app.route("/")
def index():
    return "running server..."

# slack outgoing webhook function
@app.route("/slack", methods=["POST"])
def slack():
    username = request.form.get("user_name")
    token = request.form.get("token")
    text = request.form.get("text")

    print(username, token, text)

    if "날씨" in text:
        weather = forecast()
        send_slack(weather)
    if "네이버" in text:
        keywords = naver_keywords()
        send_slack(keywords)

    return Response(), 200

app.run(debug=True)
```

ngrok.com

### ngrok 설치

```bash
sudo snap install ngrok
ngrok http 5000
```

slack outgoing webhook에 `ngrok http 5000` 실행 결과 나온 URL 써주기 뒤에는 /slack 써줘야함



### server 접속



slack 설정

incoming: slack 서버가 request를 받으면 client로 메세지를 보내주는 역할을 한다.

webhook URL이 있었다. 

outgoing: client에서 메세지를 받으면 슬랙 서버가 다른쪽으로 리퀘스트를 보내는 역할을 한다.