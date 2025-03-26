# Logistic-Regression
Plot decision boundary using logistic regression and gradient descent

Class KMOOC:모두를위한머신러닝

<br/><br/>

# Regression Training

<br/>

- Hypothsis Function

$m$개의 examples $${(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \cdots, (x^{(m)}, y^{(m)})}$$ 가 주어졌을 때,

$$h(x)=\frac{1}{1+e^{-w^{T}x}}$$

<br/>

- Linear Regression과 다르게 Cost Function을 사용하는 차이점 존재

$$
\text{cost}(h(x), y) =
\begin{cases}
-\log(h(x)) & \text{if } y = 0 \\
-\log(1 - h(x)) & \text{if } y = 1
\end{cases}
$$

<br/>

- 비용함수 $J(w)$

$$J(w)=-\frac{1}{m}\sum_{i=1}^{m}\text{cost}(h(x^{(i)}, y^{(i)}))$$

<br/>

- 최적의 parameter

$$w^{*}=\min_{w}J(w)$$

<br/>

- $w^{*}$를 통해 $h(x)$의 값을 계산해 $0.5$보다 큰지 작은지 비교 후 예상되는 $y$를 구해볼 수 있음
- parameter $w$를 update 하는 방법 : 학습 상수 $\alpha$에 대해

$$w_{j}:=w_{j}-\alpha \nabla J(w)$$

<br/><br/>

# Code execution result

<br/>

- 초기 데이터 ($\text{iterations} = 0$)

![Image](https://github.com/user-attachments/assets/b3249417-0e20-491f-812a-0df5be201a1e)

<br/>

- 100번 업데이트 ($\text{iterations} = 100 \cdot 8$)
  
![Image](https://github.com/user-attachments/assets/892cd20c-3f27-435f-95bb-9e76391cb764)
