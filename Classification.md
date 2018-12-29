# Classification

### make_classification

- `n_features`: 독립 변수의 수 (default: 20)
- `n_classes`: 종속 변수의 클래스 수 (default: 2)
- `n_clusters_per_class`: 클래스 당 클러스터 수 (default: 2)
- `n_informative`: 독립 변수 중 종속 변수와 상관 관계가 있는 성분의 수 (default: 2)
- `n_redundant`: 독립 변수 중 다른 독립 변수의 선형 조합으로 나타내는 성분의 수 (default: 2)
- `weights`: 각 클래스에 할당된 표본 수

unbalanced case: 두 클래스의 데이터 수가 비슷하지 않고, 한 클래스의 데이터 수가 크게 작거나 많은 경우이다. (ex 희귀병)
희귀병을 나타내는 데이터, 차트, 엑스레이 사진들은 적다. 정상인은 많다.
우리가 생각하는 classification의 결과가 나쁘게 나올 수 있다.
내가 100개를 예측했는데 맞춘 갯수 (분류 결과의 성능) - 분류가 잘됐는지, 잘 안됐는지
unbalanced case에서는 위와 같은 성능 평가가 나쁘게 나올 수 있다.
그 이유는 나중에 말해 준다.

클래스가 3개 이상 - 다중 classification



### make_blobs

- clustering을 평가할 때 쓴다.
- 뭉쳐져 있는 데이터를 만드는 함수
- make_blobs(n_feature=2, centers=3)
- 피쳐의 개수 2개(x축, y축), 모여있는 집단 3개



### make_gaussian_quantiles

![1543312390825](/home/p829911/문서/git/math/1543312390825.png)

- 비선형 접근법만 gaussian quantiles를 풀 수 있다.



# 분류 모형

분류(classification)는 독립 변수 값이 주어졌을 때 그 독립 변수 값과 가장 연관성이 큰 종속변수 카테고리(클래스)를 계산하는 문제이다. 현실적인 문제로 바꾸어 말하면 어떤 표본에 대한 데이터가 주어졌을 때 그 표본이 어떤 카테고리 혹은 클래스에 알아내는 문제이기도 하다. 선택해야 할 카테고리 혹은 클래스가 미리 주어졌다는 점에서 보기가 주어진 시험 문제를 푸는 것과 비슷하다고 말할 수 있다.

- 결정론적 모형 (판별 함수 방법) : Decision function
- 확률적 모형 = 조건부확률 방법
  - 생성 모형: generative
  - 판별적 모형 : discriminative



### 분류 모형의 종류

분류 문제를 푸는 방법은 크게 두 가지로 나눌 수 있다. 하나는 주어진 데이터를 카테고리에 따라 서로 다른 영역으로 나누는 경계면(decision boundary)을 찾아낸 다음 이 경계면으로부터 주어진 데이터가 어느 위치에 있는지를 계산하는 판별함수(discriminant function)를 이용하는 **판별함수 모형**이고 또 다른 하나는 주어진 데이터에 대해(conditionally) 각 카테고리 혹은 클래스가 정답일 조건부확률(conditional probability)를 계산하는 **확률적 모형**이다. 조건부확률 기반 방법은 조건부확률을 계산하는 방법에 따라 직접 조건부확률 함수를 추정하는 **확률적 판별(discriminative) 모형**과 베이즈 정리를 사용하는 **확률적 생성(generative) 모형**으로 나누어진다.

| 모형                                      | 방법론                               |
| ----------------------------------------- | ------------------------------------ |
| Linear/Quadratic Discriminant Analysis    | 확률적 생성(generative) 모형         |
| 나이브 베이지안(Naive Bayes)              | 확률적 생성(generative) 모형         |
| 로지스틱 회귀(Logistic Regression)        | 확률적 판별(discriminative) 모형     |
| 의사결정나무(Decision Tree)               | 확률적 판별(discriminative) 모형     |
| 퍼셉트론(Perceptron)                      | 판별함수(discriminant function) 모형 |
| 서포트 벡터 머신 (Support Vector Machine) | 판별함수(discriminant function) 모형 |
| 신경망 (Neural Network)                   | 판별함수(discriminant function)모형  |



### 조건부 확률 방법 = 확률론적인 방법

출력 데이터 $y$가 $K$개의 클래스 $1, \cdots, K$ 중의 하나의 값을 가진다고 가정하자. 확률적 모형은 다음과 같은 순서로 $x$에 대한 클래스를 예측한다.

(1) 입력 $x$가 주어졌을 때 $y$가 클래스 $k$가 될 확률 $p(y=k \mid x)$를 모두 계산하고,
$$
\begin{eqnarray}
P_1 &=& P(y=1 \mid x ) \\
\vdots & & \vdots \\
P_K &=& P(y=K \mid x )\\
\end{eqnarray}
$$

(2) 다음으로 가장 확률이 큰  클래스를 선택하는 방법이다.
$$
y = \arg\max_{k} P(y=k \mid x)
$$

`predict_proba`

`predict_log_proba`

조건부확률을 계산하는 방법은 두가지가 있다.

1. 생성모형 (generative model)
2. 판별모형 (discriminative model)



### 확률적 생성모형

생성모형은 먼저 각 클래스 별 특징 데이터의 확률분포 $P(x \mid y = k)$을 추정한 다음 베이즈 정리를 사용하여 $P(y=k \mid x)$를 계산하는 방법이다. 
$$
P(y=k \mid x) = \dfrac{P(x \mid y=k)P(y=k)}{P(x)}
$$
생성모형에서는 전체 확률의 법칙을 이용하여 특징 데이터 $x$의 무조건부 확률분포 $P(x)$를 구할 수 있다.
$$
P(x) = \sum_{k=1}^K P(x|k)P(k)
$$
따라서 새로운 가상의 특징 데이터를 생성해내거나 특징 데이터만으로도 아웃라이어를 판단할 수 있다.

하지만 클래스가 많을 경우 모든 클래스에 대해 $P(x \mid y=k)$를 추정하는 것은 많은 데이터를 필요로 할 뿐더러 최종적으로는 사용하지도 않을 확률분포를 계산하는데 계산량을 너무 많이 필요로 한다는 단점이 있다.



단점: 클래스가 많은 경우 모든 클래스에 대해 확률을 추정해야 하기 때문에 많은 데이터를 필요로 할 뿐더러 최종적으로는 사용하지도 않을 확률분포를 계산하는데 계산량을 너무 많이 필요로 한다.

우도: 최종 문제를 풀기에는 너무 많은 정보를 포함하고 있다. 개와 고양이를 구분하려면 개와 고양이를 그릴수 있어야 한다. 구분만 할 수 있으면 될 때 생성모형 보다는 판별 모형을 쓰면 된다.



#### QDA(Quadratic Discriminant Analysis)

조건부확률 기반 생성(generative) 모형의 하나이다.



#### 나이브 베이지안 모형

조건부확률 기반 생성 모형의 장점 중 하나는 클래스가 3개 이상인 경우에도 바로 적용할 수 있다는 점이다.
나이브 베이지안(Naive Bayesian) 모형도 조건부확률 모형의 일종이다.



### 확률적 판별 모형



### 판별함수 기반 모형

동일한 클래스가 모여 있는 영역과 그 영역을 나누는 경계면(boundary plane)을 정의하는 것이다.
이 경계면은 경계면으로부터의 거리를 계산하는 $f(x)$형태의 함수인 판별함수(discriminant function)로 정의된다. 판별함수의 값의 부호에 따라 클래스가 나뉘어진다.



#### 퍼셉트론

가장 단순한 판별함수 모형이다. 직선이 경계선(boundary line)으로 데이터 영역을 나눈다.

![1543316695418](/home/p829911/문서/git/math/1543316695418.png)

만약 데이터의 차원이 3차원이라면 다음과 같이 경계면(boundary surface)을 가지게 된다. 이러한 경계면이나 경계선을 의사결정 하이퍼 플레인 (decision hyperplane)이라고 한다.

![1543316822965](/home/p829911/문서/git/math/1543316822965.png)



# 다중 클래스 분류

종속변수의 클래스가 2개인 경우를 이진(Binary Class) 분류 문제, 클래스가 3개 이상인 경우를 다중 클래스(Multi-Class) 분류 문제라고 한다. 다중 클래스 분류 문제는 OvO(One-Vs-One) 방법이나 OvR(One-Vs-the-Rest) 방법 등을 이용하면 여러개의 이진 클래스 분류 문제로 변환하여 풀 수 있다.



### OvO (One-Vs-One)

클래스가 많아지면 제대로 못푼다. 풀어야 할 이진 분류 문제가 너무 많아진다 제곱에 비례한다.



### OvR (One-vs-the-Rest)

$K$개의 클래스가 존재하는 경우, 각각의 클래스에 대해 표본이 속하는지 속하지 않는지 이진 분류 문제를 푼다. OVO와 달리 클래스 수만큼의 이진 분류 문제를 풀면 된다.
$K$개의 클래스가 존재하면 $K$개의 문제만 풀면 된다.



### Label Binarizer (Label = y)





# 분류 성능 평가

분류 문제는 회귀 분석과 달리 다양한 성능 평가 기준이 필요하다.

|                  | 사기 거래라고 예측 | 정상 거래라고 예측 |
| ---------------- | ------------------ | ------------------ |
| 실제로 사기 거래 | True Positive      | False Negative     |
| 실제로 정상 거래 | False Positive     | True Negative      |



### 평가 점수

**Accuracy 정확도**

- 전체 샘플 중 맞게 예측한 샘플 수의 비율
- 모형 트레이닝 즉 최적화에서 목적함수로 사용

$$
\text{accuracy} = \dfrac{\text{TP} + \text{TN}}{\text{TP}+\text{TN}+\text{FP}+\text{FN}}
$$

**Precision 정밀도**

- Positive 클래스에 속한다고 출력한 샘플 중 실제로 Positive 클래스에 속하는 샘플 수의 비율
- FDS의 경우, 사기 거래라고 판단한 거래 중 실제 사기 거래의 비율, 유죄율

$$
\text{precision} = \dfrac{\text{TP}}{\text{TP} + \text{FP}}
$$

precision을 높이는 가장 쉬운 방법 Threshold를 0보다 높게 만드는 것이다.
$$
f(x) = 0 \rightarrow f(x) = 10
$$

**Recall 재현율**

- 실제 Positive 클래스에 속한 샘플 중에 Positive 클래스에 속한다고 출력한 표본의 수
- FDS의 경우, 실제 사기 거래 중에서 실제 사기 거래라고 예측한 거래의 비율, 검거율
- TPR(true positive rate)
- sensitivity(민감도)

$$
\text{recall} = \dfrac{\text{TP}}{\text{TP} + \text{FN}}
$$



precision과 recall을 동시에 높이면 좋지만, 대체적으로 정밀도를 높이면 재현율이 떨어지고 재현율을 높이면 정밀도는 떨어진다.



**Fall-Out 위양성율**

- 실제 Positive 클래스에 속하지 않는 샘플 중에 Positive 클래스에 속한다고 출력한 표본의 수
- FDS의 경우, 실제 정상 거래 중에서 FDS가 사기 거래라고 예측한 거래의 비율, 원죄율
- FPR(False positive rate)
- specificity(특이도) = 1 - fall-out

$$
\text{fallout} = \dfrac{\text{FP}}{\text{FP} + \text{TN}}
$$

위양성율이 올라가면 재현율도 올라간다.



### F (beta) score

- 정밀도(precision)과 재현율(Recall)의 가중 조화 평균
  $$
  F_\beta = (1 + \beta^2)(\text{precision}\times\text{recall})\,/\,(\beta^2\text{precision} + \text{recall})
  $$

- F1 score

  - beta = 1

  $$
  F_1 = 2 \cdot \text{precision} \cdot \text{recall}\, / \,(\text{precision} + \text{recall})
  $$





위에서 설명한 각종 평가 점수들은 서로 밀접한 관계를 맺고 있다. 예를 들어

- 재현율(recall)과 위양성률(fall-out)은 양의 상관 관계가 있다.
- 정밀도(precision)와 재현율(recall)은 대략적으로 음의 상관 관계가 있다.

재현율을 높이기 위해서는 양성으로 판단하는 기준(threshold)을 낮추어 약간의 증거만 있어도 양성으로 판단하도록 하면 된다. 그러나 이렇게 되면 음성임에도 양성으로 판단되는 표본 데이터가 같이 증가하게 되어 위양성율이 동시에 증가한다. 반대로 위양성율을 낮추기 위해 양성을 판단하는 기준을 엄격하게 두게 되면 증거 부족으로 음성 판단을 받는 표본 데이터의 수가 같이 증가하므로 재현율이 떨어진다.

정밀도의 경우에는 재현율과 위양성률처럼 정확한 상관 관계는 아니지만 대략적으로 음의 상관 관계를 가진다. 즉 정밀도를 높이기 위해 판단 기준을 엄격하게 할수록 재현율이나 위양성율이 감소하는 경향을 띤다.



### ROC 커브

- Receiver Operator Characteristic 커브는 클래스 판별 기준값의 변화에 따른 위양성률(fall-out)과 재현율(recall)의 변화를 시각화한 것이다.
- 모든 이진 분류 모형은 판별 평면으로부터의 거리에 해당하는 판별 함수(discriminant function)를 가지며 판별 함수 값이 음수이면 0인 클래스, 양수이면 1인 클래스에 해당한다고 판별한다. 즉 0 이 클래스 판별 기준값이 된다. ROC 커브는 이 클래스 판별 기준값이 달라진다면 판별 결과가 어떻게 달라지는지를 표현한 것이다.
- Scikit-Learn 의 Classification 클래스는 다음처럼 판별 함수 값을 계산하는 `decision_function` 메서드를 제공한다. 다음 표는 분류 문제를 풀고 `decision_function` 메서드를 이용하여 모든 표본 데이터에 대해 판별 함수 값을 계산한 다음 계산된 판별 함수 값이 가장 큰 데이터부터 가장 작은 데이터 순서로 정렬한 것이다.

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=16, n_features=2,
                           n_informative=2, n_redundant=0,
                           random_state=0)

model = LogisticRegression().fit(X, y)
y_hat = model.predict(X)
f_value = model.decision_function(X)

df = pd.DataFrame(np.vstack([f_value, y_hat, y]).T,
                  columns=["f", "y_hat", "y"])
df.sort_values("f", ascending=False).reset_index(drop=True)
```

ROC 커브는 이 표를 이용하여 다음과 같이 작성한다.

1. 현재는 0을 기준값(threshold)으로 클래스를 구분하여 판별함수값이 0보다 크면 양성(Positive), 작으면 음성(negative)이 된다.
2. 데이터 분류가 다르게 되도록 기준값을 증가 혹은 감소시킨다. 위의 표에서는 기준값을 0.244729보다 크도록 올리면 6번 데이터는 더이상 양성이 아니다.
3. 기준값을 여러가지 방법으로 증가 혹은 감소시키면서 이를 반복하면 여러가지 다른 기준값에 대해 분류 결과가 달라지고 재현율, 위양성률 등의 성능평가 점수도 달라진다.



기준값 0을 사용하여 이진 분류 결과표, 재현율, 위양성율을 계산하면 다음과 같다.

```python
confusion_matrix(y, y_hat, labels=[1, 0])
```



|               | 예측 클래스 1 | 예측 클래스 0 |
| ------------- | ------------- | ------------- |
| 실제 클래스 1 | 6             | 2             |
| 실제 클래스 0 | 1             | 7             |



```python
recall = 6 / (6 + 2)
fallout = 1 / (1 + 7)
print("recall = ", recall)
print("fallout = ", fallout)
```

recall = 0.75

fallout = 0.125



Scikit-Learn는 위 과정을 자동화한 `roc_curve` 명령을 제공한다. 인수로는 타겟 y 벡터와 판별함수 벡터(혹은 확률 벡터)를 넣고 결과로는 변화되는 기준값과 그 기준값을 사용했을 때의 재현율과 위양성률을 반환한다.

```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y, model.decision_function(x))
fpr, tpr, thresholds
```



`decisino_function`메서드를 제공하지 않는 모형은 `predict_proba`명령을 써서 확률을 입력해도 된다.

```python
fpr, tpr, thresholds = roc_curve(y, model.predict_proba(x)[:, 1]
fpr, tpr, thresholds
```

