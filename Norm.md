# 거리 측정법

### 유클리드 거리 측정법

유클리드 거리는 가장 직관적이고 일반적으로 생각하는 거리 개념이다.

예를 들어 평면에서 주어진 두 점의 유클리드 거리를 자로 측정할 수도 있다. 수학적으로는 n-차원의 벡터 $(a_1, a_2, \dots, a_n)$ 와 $(b_1, b_2, \dots, b_n)$ 의 유클리드 거리는 다음과 같다.
$$
d = \sqrt{(a_1 - b_1)^2 + (a_2 - b_2)^2 + \cdots + (a_n - b_n)^2}
$$


### 맨하탄 거리 측정법

맨하탄 거리 방법은 Taxicab Distance, Rectlinear Distance, City Block Distance 등으로 불리며, 특히 L1 Distance라는 이름으로 유클리디안 거리와 함께 비교되는 공식이다.

맨하탄 거리 측정법은 격자무늬 도로를 가진 맨하탄에서 이름을 따왔다. 높은  빌딩으로 구성된 뉴욕에선, 2번가와 2번 스트리트의 교차점에서 6번가와 6번 스트리트의 교차점으로 빌딩을 통과해서 걸을 수는 없다. 4개의 블록을 갔지만 실제 걸은 거리는 4개 블록 이상이라고 볼 수 있다. n-차원의 벡터 $(a_1, a_2, \dots, a_n)$ 와 $(b_1, b_2, \dots, b_n)​$ 의 맨하탄 거리는 다음과 같다.
$$
d = |a_1 - b_1| + |a_2 - b_2| + \cdots + |a_n - b_n|
$$
![](https://user-images.githubusercontent.com/17154958/51818575-d6b3c580-2312-11e9-9799-f7fc1ca29609.png)

### 민코우스키 거리(Minkowski Distance)

민코우스키 거리 공식은 맨하탄 거리와 유클리디안 거리를 한번에 표현하는 공식이다. 맨하탄 거리와 유클리디안 거리가 각각 L1 Distance와 L2 Distance로 불리는 이유가 바로 이 공식 때문일 것이다.
$$
L_m = \sqrt[m]{\sum_{i=1}^n(|a_i - b_i|)^m}
$$


### 딥러닝을 위한 Norm

일반적으로 딥러닝에서 네트워크의 Overfitting(과적합) 문제를 해결하는 방법으로 다음과 같은 3가지 방법을 제시한다.

1. 더 많은 데이터를 사용할 것
2. Cross Validation
3. Regularization

더 이상 학습 데이터를 추가할 수 없거나 학습 데이터를 늘려도 과적합 문제가 해결되지 않을 때에는 3번 Regularization을 사용해야 한다. Regularization에서는 Lose 함수를 다음과 같이 변형하여 사용한다. 
$$
\text{cost}(W, b) = \frac{1}{m} \sum_{i=1}^m L(\hat{y^i}, y^i) + \lambda \frac{1}{2} \|w\|^2
$$
위 수식은 기존 Cost 함수에 L2 Regularization을 위한 새로운 항을 추가한 변형된 형태의 Cost 함수이다. 여기서 Weight의 Regularization을 위해서 Weight의 L2 Norm을 새로운 항으로 추가하고 있다. 딥러닝의 Regularization, kNN 알고리즘, kmean 알고리즘 등에서 L1 Norm / L2 Norm을 사용한다.



### Norm

Norm은 벡터의 길이 혹은 크기를 측정하는 방법(함수)이다. Norm이 측정한 벡터의 크기는 원점에서 벡터 좌표까지의 거리 혹은 Magnitude라고 한다.
$$
L_p = (\sum_i^n |x_i|^p)^{\frac{1}{p}}
$$

- p는 Norm의 차수를 의미한다. p가 1이면 L1 Norm 이고, p가 2이면 L2 Norm이다.
- n은 대상 벡터의 요소 수이다.

Norm은 각 요소별로 요소 절대값을 p번 곱한 값의 합을 p 제곱근한 값이다.

주로 사용되는 Norm은 L1 Norm과 L2 Norm, Maximum Norm이다.



### L1 Norm

L1 Norm은 p가 1인 Norm이다. L1 Norm 공식은 다음과 같다.
$$
\begin{eqnarray}
L_1 &=& (\sum_i^n |x_i|) \\
&=& |x_1| + |x_2| + |x_3| + \dots + |x_n|
\end{eqnarray}
$$
L1 Norm 을 Taxicab Norm 혹은 맨하튼 노름(Manhattan Norm) 이라고도 한다. L1 Norm은 벡터의 요소에 대한 절댓값의 합이다. 요소의 값 변화를 정확하게 파악할 수 있다.

L1 Norm은 다음과 같은 영역에서 사용된다.

- L1 Regularization
- Computer Vision



### L2 Norm

L2 Norm은 p가 2인 Norm이다. L2 Norm은 n 차원 좌표평면(유클리드 공간)에서의 벡터의 크기를 계산하기 때문에 유클리드 노름(Euclidean Norm)이라고도 한다. L2 Norm의 공식은 다음과 같다.
$$
\begin{eqnarray}
L_2 &=& \sqrt{\sum_i^n x_i^2} \\
&=& \sqrt{x_1^2 + x_2^2 + x_3^2 + \dots + x_n^2}
\end{eqnarray}
$$
추가로 L2 Norm 공식은 다음과 같이 표현할 수 있다.
$$
\begin{eqnarray}
L_2 &=& \sqrt{\sum_i^n x_i^2}\\
&=& \sqrt{x \cdot x} \\
&=& \sqrt{x^Tx} \\ 
&=& \sqrt{x_1 * x_1 + x_2 * x_2 + x_3 * x_3 + \cdots + x_n * x_n}
\end{eqnarray}
$$
피타고라스 정리는 2차원 좌표 평면상의 최단 거리를 계산하는 L2 Norm 이다.



### Maximum Norm(상한 노름)

상한 노름은 p 값을 무한대로 보냈을 때의 Norm이다. 벡터 성분의 최댓값을 구한다.
$$
L_\infty = \text{max}(|x_1|, |x_2|, |x_3|, \dots, |x_n|)
$$
L1 Norm은 각 요소 절댓값 크기의 합이고 L2 Norm은 해당 차원의 좌표평면에서 원점에서 벡터 좌표까지의 최단거리이다.

[출처](http://taewan.kim/post/norm/)