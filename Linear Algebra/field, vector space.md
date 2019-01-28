# 선형대수학 01 (field, vector space)

#### vector space

- 물리과목: 크기와 방향이 있는 것

- 수학과목: [1, 2, 3, 4]



#### Field - Scalar 들의 집합

- Def: A field with two binary operation $(+, \cdot )$
  - 위의 두가지 연산에 대해 닫혀있다 (Closed)
  - $a + b = b + a,\;\; a \cdot b = b \cdot a$ (Commutative 교환법칙)
  - $(a + b) + c = a + (b + c), \;\; (a \cdot b) \cdot c = a \cdot (b \cdot c)$ (Associative 결합법칙)
  - Existence of additive identity and multiplicative identity 
    (덧셈의 항등원과 곱셈의 항등원이 존재해야 한다)
  - Existence of additive inverse and multiplicative inverse
    (덧셈에 대한 역원과 곱셈에 대한 역원이 존재해야 한다)
    - $a + (-a) = 0, \;\; b \cdot b^{-1} = 1$
  - $a \cdot (b + c) = a \cdot b + a \cdot c$   분배법칙이 가능해야 한다.



#### Algebra 대수학 - 방정식을 푸는 학문

##### **Example**

- $5 + x = 2$
- $-5 + (5+x) = -5 + 2 \;\;\; (\text{-5 = additive inverse of 5})$
- $(-5 + 5) + x = -5 + 2 \;\;\; (\text{associative law})​$
- $0 + x = -5 + 2 \;\;\; (\text{property of 0, 항등원의 존재})$
- $x = -5 + 2$
- $x = -3$



##### Examples of Field

- $R​$: 실수, $Q​$: 유리수, $C​$: 복소수
- $\{a+b\sqrt{2} \;|\;a, b \in a \}$
- $Z_2 = \{0, 1\} \rightarrow$ 정수를 2로 나누었을 때 나머지
- 연산테이블

| +    | 0    | 1    |
| ---- | ---- | ---- |
| 0    | 0    | 1    |
| 1    | 1    | 0    |

| $\cdot$ | 0    | 1    |
| ------- | ---- | ---- |
| 0       | 0    | 0    |
| 1       | 0    | 1    |



- $Z_3, Z_5, Z_7, Z_{11}, Z_{13}$  is a field
- $Z_4$ is not a field

$$
2 \cdot 0 = 0\\
2 \cdot 1 = 2\\
2 \cdot 2 = 0\\
2 \cdot 3 = 2
$$

- 1이 나오지 않는다 (2라는 원소는 곱셈에 대한 역원이 없다)
- Natural number, Integer not a field



#### Vector Space - vector 들의 집합

- Def: vector space $V$ over $F$
  벡터 스페이스를 정의하기 위해서는 Field도 같이 정의해 주어야 한다.
  = a set of vectors with two operation$(+, \cdot)$
  $x, y \in V, \;\; a, b \in F$
- Closed
- $x+y = y+x$
- $(x+y) + z = x + (y + z)$
- $\exists \;0 \in V\;\; s.t. x +0 = x​$   0 is vector
- $\forall x \in V\;\; s.t. 1 \cdot x = x$   1 is field
- $(a \cdot b) \cdot x = a \cdot(b \cdot x)$ 
- $a(x + y) = ax + ay$
- $(a+b) x = ax + bx$



- vector: vector space $V​$의 원소
- scalar: Field $F$의 원소
- n-tuple from $F$ ($a_1, a_2, a_3, \dots, a_n$)



##### Example of vector space



##### EX1

- $F^n = \{(a_1, a_2, \dots, a_n) \; | \; a_i \in F \}$
- $F^n \text{over F is a vector space} $
- $u = (a_1, a_2, \dots, a_n) \in F^n$
- $v = (b_1, b_2, \dots, b_n) \in F^n$
- $u + v = (a_1+b_1, a_2+b_2, \dots, a_n + b_n) \in F^n$
- $c \cdot v = (c \cdot b_1, c \cdot b_2, \dots, c \cdot b_n) \in F^n$
- $R^3 \text{over}\;\ R$
  - $(3, -2, 0) \in R^3$
- $C^4 \text{over} \; C$
  - $(3+i, -2i, 4, 0) \in c^4​$



##### EX2

- $M_{m \times n}(F) = \{ [a_{i,j}]_{m \times n}\;|\; a_{i,j} \in F\}​$

- $\begin{bmatrix} 2 & 3 & 4 \\ 5 & 6 & 7 \end{bmatrix} \in M_{2 \times 3}(F)​$
- $\text{zero in } M_{2 \times 3}(F) = \begin{bmatrix}0&0&0\\0&0&0\end{bmatrix} ​$



[출처 - 김영길님 선형대수 유튜브 강의](https://www.youtube.com/playlist?list=PL9k2wIz8VsfOjzW_nU_yRPFBoyS5C7ttG)

