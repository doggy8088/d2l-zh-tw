# 凸性
:label:`sec_convexity`

*凸性*（convexity）在最佳化演算法的設計中起到至關重要的作用，
這主要是由於在這種情況下對演算法進行分析和測試要容易。
換言之，如果演算法在凸性條件設定下的效果很差，
那通常我們很難在其他條件下看到好的結果。
此外，即使深度學習中的最佳化問題通常是非凸的，
它們也經常在區域性極小值附近表現出一些凸性。
這可能會產生一些像 :cite:`Izmailov.Podoprikhin.Garipov.ea.2018`這樣比較有意思的新優化變體。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
from mpl_toolkits import mplot3d
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
from mpl_toolkits import mplot3d
import tensorflow as tf
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from mpl_toolkits import mplot3d
import paddle
```

## 定義

在進行凸分析之前，我們需要定義*凸集*（convex sets）和*凸函式*（convex functions）。

### 凸集

*凸集*（convex set）是凸性的基礎。
簡單地說，如果對於任何$a, b \in \mathcal{X}$，連線$a$和$b$的線段也位於$\mathcal{X}$中，則向量空間中的一個集合$\mathcal{X}$是*凸*（convex）的。
在數學術語上，這意味著對於所有$\lambda \in [0, 1]$，我們得到

$$\lambda  a + (1-\lambda)  b \in \mathcal{X} \text{ 當 } a, b \in \mathcal{X}.$$

這聽起來有點抽象，那我們來看一下 :numref:`fig_pacman`裡的例子。
第一組存在不包含在集合內部的線段，所以該集合是非凸的，而另外兩組則沒有這樣的問題。

![第一組是非凸的，另外兩組是凸的。](../img/pacman.svg)
:label:`fig_pacman`

接下來來看一下交集 :numref:`fig_convex_intersect`。
假設$\mathcal{X}$和$\mathcal{Y}$是凸集，那麼$\mathcal {X} \cap \mathcal{Y}$也是凸集的。
現在考慮任意$a, b \in \mathcal{X} \cap \mathcal{Y}$，
因為$\mathcal{X}$和$\mathcal{Y}$是凸集，
所以連線$a$和$b$的線段包含在$\mathcal{X}$和$\mathcal{Y}$中。
鑑於此，它們也需要包含在$\mathcal {X} \cap \mathcal{Y}$中，從而證明我們的定理。

![兩個凸集的交集是凸的。](../img/convex-intersect.svg)
:label:`fig_convex_intersect`

我們可以毫不費力地進一步得到這樣的結果：
給定凸集$\mathcal{X}_i$，它們的交集$\cap_{i} \mathcal{X}_i$是凸的。
但是反向是不正確的，考慮兩個不相交的集合$\mathcal{X} \cap \mathcal{Y} = \emptyset$，
取$a \in \mathcal{X}$和$b \in \mathcal{Y}$。
因為我們假設$\mathcal{X} \cap \mathcal{Y} = \emptyset$，
在 :numref:`fig_nonconvex`中連線$a$和$b$的線段需要包含一部分既不在$\mathcal{X}$也不在$\mathcal{Y}$中。
因此線段也不在$\mathcal{X} \cup \mathcal{Y}$中，因此證明了凸集的並集不一定是凸的，即*非凸*（nonconvex）的。

![兩個凸集的並集不一定是凸的。](../img/nonconvex.svg)
:label:`fig_nonconvex`

通常，深度學習中的問題是在凸集上定義的。
例如，$\mathbb{R}^d$，即實數的$d$-維向量的集合是凸集（畢竟$\mathbb{R}^d$中任意兩點之間的線存在$\mathbb{R}^d$）中。
在某些情況下，我們使用有界長度的變數，例如球的半徑定義為$\{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ 且 } \| \mathbf{x} \| \leq r\}$。

### 凸函式

現在我們有了凸集，我們可以引入*凸函式*（convex function）$f$。
給定一個凸集$\mathcal{X}$，如果對於所有$x, x' \in \mathcal{X}$和所有$\lambda \in [0, 1]$，函式$f: \mathcal{X} \to \mathbb{R}$是凸的，我們可以得到

$$\lambda f(x) + (1-\lambda) f(x') \geq f(\lambda x + (1-\lambda) x').$$

為了說明這一點，讓我們繪製一些函式並檢查哪些函式滿足要求。
下面我們定義一些函式，包括凸函式和非凸函式。

```{.python .input}
#@tab mxnet, pytorch, tensorflow
f = lambda x: 0.5 * x**2  # 凸函式
g = lambda x: d2l.cos(np.pi * x)  # 非凸函式
h = lambda x: d2l.exp(0.5 * x)  # 凸函式

x, segment = d2l.arange(-2, 2, 0.01), d2l.tensor([-1.5, 1])
d2l.use_svg_display()
_, axes = d2l.plt.subplots(1, 3, figsize=(9, 3))
for ax, func in zip(axes, [f, g, h]):
    d2l.plot([x, segment], [func(x), func(segment)], axes=ax)
```

```{.python .input}
#@tab paddle
f = lambda x: 0.5 * x**2  # 凸函式
g = lambda x: d2l.cos(np.pi * x)  # 非凸函式
h = lambda x: d2l.exp(0.5 * x)  # 凸函式

x, segment = d2l.arange(-2, 2, 0.01, dtype='float32'), d2l.tensor([-1.5, 1])
d2l.use_svg_display()
_, axes = d2l.plt.subplots(1, 3, figsize=(9, 3))
for ax, func in zip(axes, [f, g, h]):
    d2l.plot([x, segment], [func(x), func(segment)], axes=ax)
```

不出所料，餘弦函式為非凸的，而拋物線函式和指數函式為凸的。
請注意，為使該條件有意義，$\mathcal{X}$是凸集的要求是必要的。
否則可能無法很好地界定$f(\lambda x + (1-\lambda) x')$的結果。

### 詹森不等式

給定一個凸函式$f$，最有用的數學工具之一就是*詹森不等式*（Jensen's inequality）。
它是凸性定義的一種推廣：

$$\sum_i \alpha_i f(x_i)  \geq f\left(\sum_i \alpha_i x_i\right) \text{ and } E_X[f(X)] \geq f\left(E_X[X]\right),$$
:eqlabel:`eq_jensens-inequality`

其中$\alpha_i$是滿足$\sum_i \alpha_i = 1$的非負實數，$X$是隨機變數。
換句話說，凸函式的期望不小於期望的凸函式，其中後者通常是一個更簡單的表示式。
為了證明第一個不等式，我們多次將凸性的定義應用於一次求和中的一項。

詹森不等式的一個常見應用：用一個較簡單的表示式約束一個較複雜的表示式。
例如，它可以應用於部分觀察到的隨機變數的對數似然。
具體地說，由於$\int P(Y) P(X \mid Y) dY = P(X)$，所以

$$E_{Y \sim P(Y)}[-\log P(X \mid Y)] \geq -\log P(X),$$

這裡，$Y$是典型的未觀察到的隨機變數，$P(Y)$是它可能如何分佈的最佳猜測，$P(X)$是將$Y$積分後的分佈。
例如，在聚類別中$Y$可能是簇標籤，而在應用簇標籤時，$P(X \mid Y)$是生成模型。

## 性質

下面我們來看一下凸函式一些有趣的性質。

### 區域性極小值是全域極小值

首先凸函式的區域性極小值也是全域極小值。
下面我們用反證法給出證明。

假設$x^{\ast} \in \mathcal{X}$是一個區域性最小值，則存在一個很小的正值$p$，使得當$x \in \mathcal{X}$滿足$0 < |x - x^{\ast}| \leq p$時，有$f(x^{\ast}) < f(x)$。

現在假設區域性極小值$x^{\ast}$不是$f$的全域極小值：存在$x' \in \mathcal{X}$使得$f(x') < f(x^{\ast})$。
則存在
$\lambda \in [0, 1)$，比如$\lambda = 1 - \frac{p}{|x^{\ast} - x'|}$，使得
$0 < |\lambda x^{\ast} + (1-\lambda) x' - x^{\ast}| \leq p$。

然而，根據凸性的性質，有

$$\begin{aligned}
    f(\lambda x^{\ast} + (1-\lambda) x') &\leq \lambda f(x^{\ast}) + (1-\lambda) f(x') \\
    &< \lambda f(x^{\ast}) + (1-\lambda) f(x^{\ast}) \\
    &= f(x^{\ast}), \\
\end{aligned}$$

這與$x^{\ast}$是區域性最小值相矛盾。
因此，不存在$x' \in \mathcal{X}$滿足$f(x') < f(x^{\ast})$。
綜上所述，區域性最小值$x^{\ast}$也是全域最小值。

例如，對於凸函式$f(x) = (x-1)^2$，有一個區域性最小值$x=1$，它也是全域最小值。

```{.python .input}
#@tab all
f = lambda x: (x - 1) ** 2
d2l.set_figsize()
d2l.plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')
```

凸函式的區域性極小值同時也是全域極小值這一性質是很方便的。
這意味著如果我們最小化函式，我們就不會『卡住』。
但是請注意，這並不意味著不能有多個全域最小值，或者可能不存在一個全域最小值。
例如，函式$f(x) = \mathrm{max}(|x|-1, 0)$在$[-1,1]$區間上都是最小值。
相反，函式$f(x) = \exp(x)$在$\mathbb{R}$上沒有取得最小值。對於$x \to -\infty$，它趨近於$0$，但是沒有$f(x) = 0$的$x$。

### 凸函式的下水平集是凸的

我們可以方便地透過凸函式的*下水平集*（below sets）定義凸集。
具體來說，給定一個定義在凸集$\mathcal{X}$上的凸函式$f$，其任意一個下水平集

$$\mathcal{S}_b := \{x | x \in \mathcal{X} \text{ and } f(x) \leq b\}$$

是凸的。

讓我們快速證明一下。
對於任何$x, x' \in \mathcal{S}_b$，我們需要證明：當$\lambda \in [0, 1]$時，$\lambda x + (1-\lambda) x' \in \mathcal{S}_b$。
因為$f(x) \leq b$且$f(x') \leq b$，所以

$$f(\lambda x + (1-\lambda) x') \leq \lambda f(x) + (1-\lambda) f(x') \leq b.$$

### 凸性和二階導數

當一個函式的二階導數$f: \mathbb{R}^n \rightarrow \mathbb{R}$存在時，我們很容易檢查這個函式的凸性。
我們需要做的就是檢查$\nabla^2f \succeq 0$，
即對於所有$\mathbf{x} \in \mathbb{R}^n$，$\mathbf{x}^\top \mathbf{H} \mathbf{x} \geq 0$.
例如，函式$f(\mathbf{x}) = \frac{1}{2} \|\mathbf{x}\|^2$是凸的，因為$\nabla^2 f = \mathbf{1}$，即其導數是單位矩陣。

更正式地講，$f$為凸函式，若且唯若任意二次可微一維函式$f: \mathbb{R}^n \rightarrow \mathbb{R}$是凸的。
對於任意二次可微多維函式$f: \mathbb{R}^{n} \rightarrow \mathbb{R}$，
它是凸的當且僅當它的Hessian$\nabla^2f\succeq 0$。

首先，我們來證明一下一維情況。
為了證明凸函式的$f''(x) \geq 0$，我們使用：

$$\frac{1}{2} f(x + \epsilon) + \frac{1}{2} f(x - \epsilon) \geq f\left(\frac{x + \epsilon}{2} + \frac{x - \epsilon}{2}\right) = f(x).$$

因為二階導數是由有限差分的極限給出的，所以遵循

$$f''(x) = \lim_{\epsilon \to 0} \frac{f(x+\epsilon) + f(x - \epsilon) - 2f(x)}{\epsilon^2} \geq 0.$$

為了證明$f'' \geq 0$可以推導$f$是凸的，
我們使用這樣一個事實：$f'' \geq 0$意味著$f'$是一個單調的非遞減函式。
假設$a < x < b$是$\mathbb{R}$中的三個點，
其中，$x = (1-\lambda)a + \lambda b$且$\lambda \in (0, 1)$.
根據中值定理，存在$\alpha \in [a, x]$，$\beta \in [x, b]$，使得
$$f'(\alpha) = \frac{f(x) - f(a)}{x-a} \text{ 且 } f'(\beta) = \frac{f(b) - f(x)}{b-x}.$$

透過單調性$f'(\beta) \geq f'(\alpha)$，因此

$$\frac{x-a}{b-a}f(b) + \frac{b-x}{b-a}f(a) \geq f(x).$$

由於$x = (1-\lambda)a + \lambda b$，所以

$$\lambda f(b) + (1-\lambda)f(a) \geq f((1-\lambda)a + \lambda b),$$

從而證明了凸性。

第二，我們需要一個引理證明多維情況：
$f: \mathbb{R}^n \rightarrow \mathbb{R}$
是凸的當且僅當對於所有$\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$

$$g(z) \stackrel{\mathrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y}) \text{ where } z \in [0,1]$$ 

是凸的。

為了證明$f$的凸性意味著$g$是凸的，
我們可以證明，對於所有的$a，b，\lambda \in[0，1]$（這樣有$0 \leq \lambda a + (1-\lambda) b \leq 1$），

$$\begin{aligned} &g(\lambda a + (1-\lambda) b)\\
=&f\left(\left(\lambda a + (1-\lambda) b\right)\mathbf{x} + \left(1-\lambda a - (1-\lambda) b\right)\mathbf{y} \right)\\
=&f\left(\lambda \left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) \left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \right)\\
\leq& \lambda f\left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) f\left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \\
=& \lambda g(a) + (1-\lambda) g(b).
\end{aligned}$$

為了證明這一點，我們可以證明對
$[0，1]$中所有的$\lambda$：

$$\begin{aligned} &f(\lambda \mathbf{x} + (1-\lambda) \mathbf{y})\\
=&g(\lambda \cdot 1 + (1-\lambda) \cdot 0)\\
\leq& \lambda g(1)  + (1-\lambda) g(0) \\
=& \lambda f(\mathbf{x}) + (1-\lambda) f(\mathbf{y}).
\end{aligned}$$

最後，利用上面的引理和一維情況的結果，我們可以證明多維情況：
多維函式$f:\mathbb{R}^n\rightarrow\mathbb{R}$是凸函式，若且唯若$g(z) \stackrel{\mathrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y})$是凸的，這裡$z \in [0,1]$，$\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$。
根據一維情況，
此條成立的條件為，若且唯若對於所有$\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$，
$g'' = (\mathbf{x} - \mathbf{y})^\top \mathbf{H}(\mathbf{x} - \mathbf{y}) \geq 0$（$\mathbf{H} \stackrel{\mathrm{def}}{=} \nabla^2f$）。
這相當於根據半正定矩陣的定義，$\mathbf{H} \succeq 0$。

## 約束

凸最佳化的一個很好的屬性是能夠讓我們有效地處理*約束*（constraints）。
即它使我們能夠解決以下形式的*約束最佳化*（constrained optimization）問題：

$$\begin{aligned} \mathop{\mathrm{minimize~}}_{\mathbf{x}} & f(\mathbf{x}) \\
    \text{ subject to } & c_i(\mathbf{x}) \leq 0 \text{ for all } i \in \{1, \ldots, N\}.
\end{aligned}$$

這裡$f$是目標函式，$c_i$是約束函式。
例如第一個約束$c_1(\mathbf{x}) = \|\mathbf{x}\|_2 - 1$，則引數$\mathbf{x}$被限制為單位球。
如果第二個約束$c_2(\mathbf{x}) = \mathbf{v}^\top \mathbf{x} + b$，那麼這對應於半空間上所有的$\mathbf{x}$。
同時滿足這兩個約束等於選擇一個球的切片作為約束集。

### 拉格朗日函式

通常，求解一個有約束的最佳化問題是困難的，解決這個問題的一種方法來自物理中相當簡單的直覺。
想象一個球在一個盒子裡，球會滾到最低的地方，重力將與盒子兩側對球施加的力平衡。
簡而言之，目標函式（即重力）的梯度將被約束函式的梯度所抵消（由於牆壁的『推回』作用，需要保持在盒子內）。
請注意，任何不起作用的約束（即球不接觸壁）都將無法對球施加任何力。

這裡我們簡略拉格朗日函式$L$的推導，上述推理可以透過以下鞍點最佳化問題來表示：

$$L(\mathbf{x}, \alpha_1, \ldots, \alpha_n) = f(\mathbf{x}) + \sum_{i=1}^n \alpha_i c_i(\mathbf{x}) \text{ where } \alpha_i \geq 0.$$

這裡的變數$\alpha_i$（$i=1,\ldots,n$）是所謂的*拉格朗日乘數*（Lagrange multipliers），它確保約束被正確地執行。
選擇它們的大小足以確保所有$i$的$c_i(\mathbf{x}) \leq 0$。
例如，對於$c_i(\mathbf{x}) < 0$中任意$\mathbf{x}$，我們最終會選擇$\alpha_i = 0$。
此外，這是一個*鞍點*（saddlepoint）最佳化問題。
在這個問題中，我們想要使$L$相對於$\alpha_i$*最大化*（maximize），同時使它相對於$\mathbf{x}$*最小化*（minimize）。
有大量的文獻解釋如何得出函式$L(\mathbf{x}, \alpha_1, \ldots, \alpha_n)$。
我們這裡只需要知道$L$的鞍點是原始約束最佳化問題的最優解就足夠了。

### 懲罰

一種至少近似地滿足約束最佳化問題的方法是採用拉格朗日函式$L$。除了滿足$c_i(\mathbf{x}) \leq 0$之外，我們只需將$\alpha_i c_i(\mathbf{x})$新增到目標函式$f(x)$。
這樣可以確保不會嚴重違反約束。

事實上，我們一直在使用這個技巧。
比如權重衰減 :numref:`sec_weight_decay`，在目標函式中加入$\frac{\lambda}{2} |\mathbf{w}|^2$，以確保$\mathbf{w}$不會增長太大。
使用約束最佳化的觀點，我們可以看到，對於若干半徑$r$，這將確保$|\mathbf{w}|^2 - r^2 \leq 0$。
透過調整$\lambda$的值，我們可以改變$\mathbf{w}$的大小。

通常，新增懲罰是確保近似滿足約束的一種好方法。
在實踐中，這被證明比精確的滿意度更可靠。
此外，對於非凸問題，許多使精確方法在凸情況下的性質（例如，可求最優解）不再成立。

### 投影

滿足約束條件的另一種策略是*投影*（projections）。
同樣，我們之前也遇到過，例如在 :numref:`sec_rnn_scratch`中處理梯度截斷時，我們透過

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \mathrm{min}(1, \theta/\|\mathbf{g}\|),$$

確保梯度的長度以$\theta$為界限。

這就是$\mathbf{g}$在半徑為$\theta$的球上的*投影*（projection）。
更泛化地說，在凸集$\mathcal{X}$上的投影被定義為

$$\mathrm{Proj}_\mathcal{X}(\mathbf{x}) = \mathop{\mathrm{argmin}}_{\mathbf{x}' \in \mathcal{X}} \|\mathbf{x} - \mathbf{x}'\|.$$

它是$\mathcal{X}$中離$\mathbf{X}$最近的點。

![Convex Projections.](../img/projections.svg)
:label:`fig_projections`

投影的數學定義聽起來可能有點抽象，為了解釋得更清楚一些，請看 :numref:`fig_projections`。
圖中有兩個凸集，一個圓和一個菱形。
兩個集合內的點（黃色）在投影期間保持不變。
兩個集合（黑色）之外的點投影到集合中接近原始點（黑色）的點（紅色）。
雖然對$L_2$的球面來說，方向保持不變，但一般情況下不需要這樣。

凸投影的一個用途是計算稀疏權重向量。
在本例中，我們將權重向量投影到一個$L_1$的球上，
這是 :numref:`fig_projections`中菱形例子的一個廣義版本。

## 小結

在深度學習的背景下，凸函式的主要目的是幫助我們詳細瞭解最佳化演算法。
我們由此得出梯度下降法和隨機梯度下降法是如何相應推導出來的。

* 凸集的交點是凸的，並集不是。
* 根據詹森不等式，『一個多變量凸函式的總期望值』大於或等於『用每個變數的期望值計算這個函式的總值『。
* 一個二次可微函式是凸函式，若且唯若其Hessian（二階導數矩陣）是半正定的。
* 凸約束可以透過拉格朗日函式來新增。在實踐中，只需在目標函式中加上一個懲罰就可以了。
* 投影對映到凸集中最接近原始點的點。

## 練習 

1. 假設我們想要透過繪製集合內點之間的所有直線並檢查這些直線是否包含來驗證集合的凸性。i.證明只檢查邊界上的點是充分的。ii.證明只檢查集合的頂點是充分的。

2. 用$p$-範數表示半徑為$r$的球，證明$\mathcal{B}_p[r] := \{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ and } \|\mathbf{x}\|_p \leq r\}$，$\mathcal{B}_p[r]$對於所有$p \geq 1$是凸的。

3. 已知凸函式$f$和$g$表明$\mathrm{max}(f, g)$也是凸函式。證明$\mathrm{min}(f, g)$是非凸的。

4. 證明Softmax函式的規範化是凸的，即$f(x) = \log \sum_i \exp(x_i)$的凸性。

5. 證明線性子空間$\mathcal{X} = \{\mathbf{x} | \mathbf{W} \mathbf{x} = \mathbf{b}\}$是凸集。

6. 證明線上性子空間$\mathbf{b} = \mathbf{0}$的情況下，對於矩陣$\mathbf{M}$的投影$\mathrm {Proj} \mathcal{X}$可以寫成$\mathbf{M} \mathbf{X}$。

7. 證明對於凸二次可微函式$f$，對於$\xi \in [0, \epsilon]$，我們可以寫成$f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{1}{2} \epsilon^2 f''(x + \xi)$。

8. 給定一個凸集$\mathcal{X}$和兩個向量$\mathbf{x}$和$\mathbf{y}$證明了投影不會增加距離，即$\|\mathbf{x} - \mathbf{y}\| \geq \|\mathrm{Proj}_\mathcal{X}(\mathbf{x}) - \mathrm{Proj}_\mathcal{X}(\mathbf{y})\|$。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/3814)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/3815)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3816)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11847)
:end_tab:
