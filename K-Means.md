# K-Means 클러스터링

###  K-Means

K-Means 클러스터링 알고리즘은 가장 단순하고 빠른 클러스터링 알고리즘의 하나이다. 다음과 같은 목적함수 값이 최소화될 때까지 클러스터의 중심(centroid)의 위치와 각 데이터가 소속될 클러스터를 반복해서 찾는다. 이 값을 inertia라고도 한다.
$$
J = \sum_{k=1}^K\sum_{i \in C_k} d(x_i, u_k)
$$
이 식에서 $K$ 는 클러스터의 갯수이고 $C_k$ 는 $k$ 번째 클러스터에 속하는 데이터의 집합, $u_k$ 는 $k$ 번째 클러스터의 중심위치, $d$ 는 $x_i, u_k​$ 두 데이터 사이의 거리(distance) 혹은 비유사도(dissimilarity)로 정의한다. 만약 유클리드 거리를 사용한다면 다음과 같다.
$$
d(x_i, u_k) = || x_i - u_k || ^ 2
$$
세부 알고리즘은 다음과 같다.

1. 임의의 중심값 $u_k$ 를 고른다. 보통 데이터 샘플 중에서 $K$개를 선택한다.
2. 중심에서 각 데이터까지의 거리를 계산
3. 각 데이터에서 가장 가까운 중심을 선택하여 클러스터 갱신
4. 다시 만들어진 클러스터에 대해 중심을 다시 계산하고 1~4를 반복한다.



scikit-learn의 cluster 서브패키지는 Means 클러스터링을 위한 `KMeans` 클래스를 제공한다. 다음과 같은 인수를 받을 수 있다.

- `n_clusters`: 클러스터의 갯수
- `init`: 초기화 방법. `"random"` 이면 무작위, `"K-means++"` 이면 K-means++ 방법. 또는 각 데이터의 클러스터 라벨
- `n_init`:  초기 중심값 시도 횟수. 디폴트는 10이고 10개의 무작위 중심값 목록 중 가장 좋은 값을 선택한다.
- `max_iter`: 최대 반복 횟수.
- `random_state`: 시드값.

다음은 `make_blobs` 명령으로 만든 데이터를 2개로 K-means 클러스터링하는 과정을 나타낸 것이다. 마커(marker)의 모양은 클러스터를 나타내고 크기가 큰 마커가 중심값 위치이다. 각 단계에서 중심값은 전단계의 클러스터의 평균으로 다시 계산된다.

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=20, random_state=4)

def plot_KMeans(n):
    model = KMeans(n_clusters=2, init="random", n_init=1, max_iter=n, random_state=8).fit(X)
    c0, c1 = model.cluster_centers_
    plt.scatter(X[model.labels_ == 0, 0], X[model.labels_ == 0, 1], marker='v', facecolor='r', edgecolors='k')
    plt.scatter(X[model.labels_ == 1, 0], X[model.labels_ == 1, 1], marker='^', facecolor='y', edgecolors='k')
    plt.scatter(c0[0], c0[1], marker='v', c="r", s=200)
    plt.scatter(c1[0], c1[1], marker='^', c="y", s=200)
    plt.grid(False)
    plt.title("iteration={}, score={:5.2f}".format(n, model.score(X)))

plt.figure(figsize=(8, 8))
plt.subplot(321)
plot_KMeans(1)
plt.subplot(322)
plot_KMeans(2)
plt.subplot(323)
plot_KMeans(3)
plt.subplot(324)
plot_KMeans(4)
plt.tight_layout()
plt.show()
```

![](https://user-images.githubusercontent.com/17154958/50965456-579e4f00-1515-11e9-8167-b50a52c5e4ee.png)

### K-Means++

K-Means++ 알고리즘은 초기 중심값을 설정하기 위한 알고리즘이다. 다음과 같은 방법을 통해 되도록 멀리 떨어진 중심값 집합을 나타낸다.

1. 중심값을 저장할 집합 $M$ 준비
2. 일단 하나의 중심 $\mu_0$ 을 랜덤하게 선택하여 $M$ 에 넣는다.
3. $M$ 에 속하지 않는 모든 샘플 $x_i$ 에 대해 거리 $d(M,x_i)$ 를 계산. $d(M, x_i)$ 는 $M$ 안의 모든 샘플 $\mu_k$ 에 대해 $d(u_k, x_i)$ 를 계산하여 가장 작은 값 선택
4. $d(M, x_i)$ 에 비례한 확률로 다음 중심 $\mu$ 를 선택.
5. $K$ 개의 중심을 선택할 때까지 반복
6. K-Means 알고리즘 사용

다음은 KMean 방법을 사용하여 MNIST Digit 이미지 데이터를 클러스터링한 결과이다. 각 클러스터에서 10개씩의 데이터만 표시하였다. 

```python
from sklearn.datasets import load_digits

digits = load_digits()

model = KMeans(init="k-means++", n_clusters=10, random_state=0)
model.fit(digits.data)
y_pred = model.labels_

def show_digits(images, labels):
    f = plt.figure(figsize=(8, 2))
    i = 0
    while (i < 10 and i < images.shape[0]):
        ax = f.add_subplot(1, 10, i + 1)
        ax.imshow(images[i], cmap=plt.cm.bone)
        ax.grid(False)
        ax.set_title(labels[i])
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.tight_layout()
        i += 1
        
def show_cluster(images, y_pred, cluster_number):
    images = images[y_pred == cluster_number]
    y_pred = y_pred[y_pred == cluster_number]
    show_digits(images, y_pred)
    

for i in range(10):
    show_cluster(digits.images, y_pred, i)
```

![](https://user-images.githubusercontent.com/17154958/50965778-5588c000-1516-11e9-8106-f893180e08ab.png)

이미지의 제목에 있는 숫자는 클러스터 번호에 지나지 않으므로 원래 숫자의 번호와 일치하지 않는다. 하지만 이를 예측 문제라고 가정하고 분류 결과 행렬을 만들면 다음과 같다.

```python
from sklearn.metrics import confusion_matrix

confusion_matrix(digits.target, y_pred)

array([[  1,   0,   0,   0,   0, 177,   0,   0,   0,   0],
       [  0,   1,   1,   0,   0,   0,  55,  99,  24,   2],
       [  0,  13,   0,   2,   3,   1,   2,   8, 148,   0],
       [  0, 154,   2,  13,   7,   0,   0,   7,   0,   0],
       [163,   0,   0,   0,   7,   0,   7,   4,   0,   0],
       [  2,   0, 136,  43,   0,   0,   0,   0,   0,   1],
       [  0,   0,   0,   0,   0,   1,   1,   2,   0, 177],
       [  0,   0,   0,   0, 177,   0,   0,   2,   0,   0],
       [  0,   2,   4,  53,   5,   0,   5, 100,   3,   2],
       [  0,   6,   6, 139,   7,   0,  20,   2,   0,   0]])
```

이 클러스터링 결과의 adjusted Rand index와 adjusted mutual info값은 다음과 같다.

```python
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score

print(adjusted_rand_score(digits.target, y_pred))
print(adjusted_mutual_info_score(digits.target, y_pred))

array([[  1,   0,   0,   0,   0, 177,   0,   0,   0,   0],
       [  0,   1,   1,   0,   0,   0,  55,  99,  24,   2],
       [  0,  13,   0,   2,   3,   1,   2,   8, 148,   0],
       [  0, 154,   2,  13,   7,   0,   0,   7,   0,   0],
       [163,   0,   0,   0,   7,   0,   7,   4,   0,   0],
       [  2,   0, 136,  43,   0,   0,   0,   0,   0,   1],
       [  0,   0,   0,   0,   0,   1,   1,   2,   0, 177],
       [  0,   0,   0,   0, 177,   0,   0,   2,   0,   0],
       [  0,   2,   4,  53,   5,   0,   5, 100,   3,   2],
       [  0,   6,   6, 139,   7,   0,  20,   2,   0,   0]])
```

