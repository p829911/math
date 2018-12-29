# Scikit-Learn의 전처리 기능



### 스케일링 - 전체 분포를 중심으로 스케일링

스케일링은 자료 집합에 적용되는 전처리 과정으로 모든 자료에 선형 변환을 적용하여 전체 자료의 분포를 평균 0, 분산 1이 되도록 만드는 과정이다. 스케일링은 자료의 오버플로우(overflow)나 언더플로우(underflow)를 방지하고 독립변수의 공분산 행렬의 조건수(condition number)를 감소시켜 최적화 과정에서의 안정성 및 수렴속도를 향상시킨다.



- `scale(X)`: 기본 스케일, 평균과 표준편차 사용
- `robust_scale(X)`: 중앙값(median)과 IQR(interquartile range) 사용. 아웃라이어의 영향을 최소화
- `minmax_scale(X)`: 최대/최소값이 각각 1, 0이 되도록 스케일링
- `maxabs_scale(X)`: 최대 절대값과 0이 각각 1, 0이 되도록 스케일링



위의 4가지 함수 안쓴다....

이거 대신 `StandardScaler` 클래스 객체를 쓰게 된다.

1. 클래스 객체 생성
2. `fit()`메서드와 트레이닝 데이터를 사용하여 변환 계수 추정
3. `transform()`메서드를 사용하여 실제로 자료를 변환

또는 `fit_transform()`메서드를 사용하여 계수 추정과 자료 변화를 동시에 실행할 수 있다.



```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(data1)
data2 = scaler.transform(data1)

data1.std(), data2.std()
```



### 정규화 (Normalization)  

x하나하나를 벡터라고 보았을 때 그 길이를 1로 조정

왜쓸까 정규화?

추천시스템 같은 것을 만들 때 x1 이랑 x2가 비슷한지

별점

x1 액션 영화에 대한 선호도

x2 드라마 로맨스에 대한 선호도

A = (2,7) 드라마를 액션보다 3배 

B = (5,5) 똑같은 정도로 좋아함

C = (1,3) 드라마를 액션보다 3배 (C가 점수를 짜개 준다.)

길이를 정규화 하여 없앤다. 



### One-Hot-Encoder





### 파이프라인

위에서 살펴 보았던 전처리 객체는 Scikit-Learn의 파이프라인(pipeline) 기능을 이용하여 분류 모형과 합칠 수 있다. 예를 들어 표준 스케일러와 로지스틱 회귀 모형은 다음처럼 구성요소의 이름을 문자열로 추가하여 합친다.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

model = Pipeline([
            ('scaler', StandardScaler()), 
            ('classifier', LogisticRegression()),
        ])
```



파이프라인으로 결합된 모형은 원래의 모형이 가지는 `fit`, `predict` 메서드를 가지며 각 메서드가 호출되면 그에 따른 적절한 메서드를 파이프라인의 각 객체에 대해서 호출한다. 예를 들어 파이프라인에 대해 `fit` 메서드를 호출하면 전처리 객체에는 `fit_transform`이 내부적으로 호출되고 분류 모형에서는 `fit` 메서드가 호출된다. 파이프라인에 대해 `predict` 메서드를 호출하면 전처리 객체에는 `transform`이 내부적으로 호출되고 분류 모형에서는 `predict` 메서드가 호출된다.