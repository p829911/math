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

