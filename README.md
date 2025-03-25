# Logistic-Regression
Plot decision boundary using logistic regression and gradient descent

Class KMOOC:모두를위한머신러닝

## Regression Training

- Hypothsis Function
  - $m$개의 examples ${(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \cdots, (x^{(m)}, y^{(m)})}$ 가 주어졌을 때,
  - $h(x)=\frac{1}{1+e^{-w^{T}x}}$


- Linear Regression과 다르게 Cost Function을 사용하는 차이점 존재
  - $y=0$일 때 $\text{cost}(h(x), y)=-\log (h(x))$
  - $y=1$일 때 $\text{cost}(h(x), y)=-\log (1-h(x))$

- 비용함수 $J(w)$
  - $J(w)=-\frac{1}{m}\sum_{i=1}^{m}\text{cost}(h(x^{(i)}, y^{(i)}))$
  - 최적의 parameter $w^{*}=\min_{w}J(w)$
  - $h(x)$의 값을 계산해 $0.5$보다 큰지 작은지 비교 후 $\text{predicted}\;y$를 구해볼 수 있음


- parameter $w$를 update 하는 방법
  - 학습 상수 $\alpha$에 대해
  - $w_{j}:=w_{j}-\alpha \nabla J(w)$
