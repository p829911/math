# QDA와 LDA

QDA(quadratic discriminant analysis)와 LDA(linear discriminant analysis)는 대표적인 확률론적 생성 모형(generative model)이다. Likelihood, 즉 y의 클래스 값에 따른 x의 분포에 대한 정보를 먼저 알아낸 후, 베이즈 정리를 사용하여 주어진 x에 대한 y의 확률 분포를 찾아낸다.



### 생성모형

생성 모형에서는 베이즈 정리를 사용하여 조건부 확률 $p(y=k \mid x)$을 계산한다.
$$
P(y=k \mid x) = \dfrac{P(x \mid y=k)P(y=k)}{P(x)}
$$
분류 문제를 풀기 위해서는 각 클래스 $k$에 대한 확률을 비교하여 가장 큰 값을 선택한다. 따라서 모든 클래스에 대해 값이 같은 분모 $P(x)$은 굳이 계산하지 않아도 괜찮다.
$$
P(y=k \mid x) \,\,\propto\,\, P(x \mid y=k)P(y=k)
$$
여기에서 사전 확률(prior) $P(y=k)$는 특별한 정보가 없는 경우, 다음처럼 계산한다.
$$
P(y=k) \approx \dfrac{y=k\text{인 데이터의 수}}{\text{모든 데이터의 수}}
$$
만약 다른 지식이나 정보로 알고 있는 사전 확률값이 있다면 그 값을 사용하면 된다.



$y$에 대한 $x$의 조건부 확률 가능도(likelihood)는 다음과 같이 계산한다.

1. $P(x \mid y=k)$가 특정한 확률분포 모형을 따른다고 가정한다. 즉, 확률밀도함수의 형태를 가정한다.
2. $k$번째 클래스에 속하는 학습 데이터 $\{x_1, \cdots, x_N\}$을 사용하여 이 모형의 모수 값을 구한다.
3. 모수값을 알고 있으므로 $P(x \mid y=k)$의 확률 밀도 함수를 구한 것이다. 즉, 새로운 독립 변수 값 $x$이 어떤 값이 되더라도 $P(x \mid y=k)$의 값을 계산할 수 있다.



### QDA

