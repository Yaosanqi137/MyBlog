---
icon: wheelchair-move
date: 2025-05-03
category: 知识
tag:
  - 加速度
  - 运动学
  - 物理
star: true
sticky: true
---

# 运动学基础

::: tip

最好自己上完课了再来看，这样体验更佳

:::

## 位矢

**定义**: 从原点O到P点的有向线段 $\vec{OP}$ = $\bf{r}$ 来表示，矢量$\bf{r}$称为**位置矢量**

在直角坐标系中，位矢$\bf{r}$可以表示成

::: center

$\bf{r} = x\bf{i} + y\bf{j} + z\bf{k}$

:::

其中，$\bf{i}、\bf{j}、\bf{k}$ 分别是 x y z 轴正方向的单位矢量，位矢$\bf{r}$的大小为

::: center

$|\bf{r}| = \sqrt{x^2 + y^2 +z^2}$

:::

方向余弦为

::: center

$\cos{α} = \frac{x}{r}, \cos{β} = \frac{y}{r}, \cos{γ} = \frac{z}{r}$

:::

运动方程为

::: center

x = x(t), y = y(t), z = z(t)

$\bf{r} = \bf{r}(t)$

:::

其中变量 t 就是运动时间啦，描述每个时刻，质点的位置，将变量 t 通过变换消去即可得到轨迹方程

::: note 例子

我现在有一个运动方程 $x = 3\sin{\frac{\pi}{6}}t$， $x = 3\cos{\frac{\pi}{6}}t$， $z = 0$

显然，他的轨迹是一个圆，但是如何得到一个圆的方程呢？

注意到 $\cos^2 + \sin^2 = 1$

那么不难发现，$x^2 + y^2 = 9$，这显然就是一个半径为 3 的圆的方程，并且 z 在这里没有任何作用，始终为 0，但是**千万别漏写上了**

完整的轨迹方程为

::: center

$x^2 + y^2 = 9$，$z = 0$

:::

## 位移

在 Δt 的时间内，质点由 A 点移动到 B 点，A、B 两点的位矢分别记为$\bf{r_1}，\bf{r_2}$，则位矢增量为

::: center

$Δ\bf{r} = \bf{r_2} - \bf{r_1}$

:::

位移大小**只能**记为 $|Δ\bf{r}|$，而不能是 Δr，因为 Δr 一般表示位矢**大小**的增量，也就是 Δr = $|\bf{r_2}| - |\bf{r_1}|$

通常，$|Δ\bf{r}| \ne Δr$，在 $Δt \to 0$ 时，才有 $|Δ\bf{r}| = ds$，但是仍没有 $|d\bf{r}| = dr$
 
![](/assets/images/kinematics/位移.png)

位移的模为

::: center

$|\bf{r}| = \sqrt{Δx^2 + Δy^2 + Δz^2}$

:::

## 速度

$Δ\bf{r}$ 与 Δt 的比值称为质点在时刻 t 附近 Δt 时间内的平均速度

$\overline{\bf{v}} = \frac{\vec{AB}}{Δt} = \frac{Δ\bf{r}}{Δt}$

若变为微分形式，即变成了瞬时速度

$\bf{v} = \lim\limits_{Δt \to 0}\frac{Δ\bf{r}}{Δt} = \frac{d\bf{r}}{dt}$

即速度是位矢对时间的一阶导

速度也可以表示成

::: center

$\bf{v} = v_x\bf{i} + v_y\bf{j} + v_z\bf{k}$

:::

模为

::: center

$|\bf{v}| = \sqrt{v_x^2 + v_y^2 + v_z^2}$

:::

## 加速度

有了上面的基础，那么加速度在数学上就很好解释了，不就是速度的变化率吗，也就是速度对时间的一阶导，或者位矢对时间的二阶导

::: center

$\bf{a} = \frac{d\bf{v}}{dt} = \frac{d^2\bf{r}}{dt^2}$

::: 

也可以表示成

::: center

$\bf{a} = a_x\bf{i} + a_y\bf{j} + a_z\bf{k}$

:::

模为

::: center

$|\bf{a}| = \sqrt{a_x^2 + a_y^2 + a_z^2}$

:::

