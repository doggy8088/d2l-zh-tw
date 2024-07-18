# 隨機梯度下降
:label:`sec_sgd`

在前面的章節中，我們一直在訓練過程中使用隨機梯度下降，但沒有解釋它為什麼起作用。為了澄清這一點，我們剛在 :numref:`sec_gd`中描述了梯度下降的基本原則。本節繼續更詳細地說明*隨機梯度下降*（stochastic gradient descent）。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import math
import paddle
```

## 隨機梯度更新

在深度學習中，目標函式通常是訓練資料集中每個樣本的損失函式的平均值。給定$n$個樣本的訓練資料集，我們假設$f_i(\mathbf{x})$是關於索引$i$的訓練樣本的損失函式，其中$\mathbf{x}$是引數向量。然後我們得到目標函式

$$f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\mathbf{x}).$$

$\mathbf{x}$的目標函式的梯度計算為

$$\nabla f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}).$$

如果使用梯度下降法，則每個自變量迭代的計算代價為$\mathcal{O}(n)$，它隨$n$線性增長。因此，當訓練資料集較大時，每次迭代的梯度下降計算代價將較高。

隨機梯度下降（SGD）可降低每次迭代時的計算代價。在隨機梯度下降的每次迭代中，我們對資料樣本隨機均勻取樣一個索引$i$，其中$i\in\{1,\ldots, n\}$，並計算梯度$\nabla f_i(\mathbf{x})$以更新$\mathbf{x}$：

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x}),$$

其中$\eta$是學習率。我們可以看到，每次迭代的計算代價從梯度下降的$\mathcal{O}(n)$降至常數$\mathcal{O}(1)$。此外，我們要強調，隨機梯度$\nabla f_i(\mathbf{x})$是對完整梯度$\nabla f(\mathbf{x})$的無偏估計，因為

$$\mathbb{E}_i \nabla f_i(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}).$$

這意味著，平均而言，隨機梯度是對梯度的良好估計。

現在，我們將把它與梯度下降進行比較，方法是向梯度新增均值為0、方差為1的隨機噪聲，以模擬隨機梯度下降。

```{.python .input}
#@tab all
def f(x1, x2):  # 目標函式
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):  # 目標函式的梯度
    return 2 * x1, 4 * x2
```

```{.python .input}
#@tab mxnet, pytorch, paddle
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # 模擬有噪聲的梯度
    g1 += d2l.normal(0.0, 1, (1,)).item()
    g2 += d2l.normal(0.0, 1, (1,)).item()
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab tensorflow
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # 模擬有噪聲的梯度
    g1 += d2l.normal([1], 0.0, 1)
    g2 += d2l.normal([1], 0.0, 1)
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab all
def constant_lr():
    return 1

eta = 0.1
lr = constant_lr  # 常數學習速度
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

正如我們所看到的，隨機梯度下降中變量的軌跡比我們在 :numref:`sec_gd`中觀察到的梯度下降中觀察到的軌跡嘈雜得多。這是由於梯度的隨機性質。也就是說，即使我們接近最小值，我們仍然受到透過$\eta \nabla f_i(\mathbf{x})$的瞬間梯度所注入的不確定性的影響。即使經過50次迭代，質量仍然不那麼好。更糟糕的是，經過額外的步驟，它不會得到改善。這給我們留下了唯一的選擇：改變學習率$\eta$。但是，如果我們選擇的學習率太小，我們一開始就不會取得任何有意義的進展。另一方面，如果我們選擇的學習率太大，我們將無法獲得一個好的解決方案，如上所示。解決這些相互衝突的目標的唯一方法是在最佳化過程中*動態*降低學習率。

這也是在`sgd`步長函式中新增學習率函式`lr`的原因。在上面的範例中，學習率排程的任何功能都處於休眠狀態，因為我們將相關的`lr`函式設定為常數。

## 動態學習率

用與時間相關的學習率$\eta(t)$取代$\eta$增加了控制最佳化演算法收斂的複雜性。特別是，我們需要弄清$\eta$的衰減速度。如果太快，我們將過早停止最佳化。如果減少的太慢，我們會在最佳化上浪費太多時間。以下是隨著時間推移調整$\eta$時使用的一些基本策略（稍後我們將討論更高階的策略）：

$$
\begin{aligned}
    \eta(t) & = \eta_i \text{ if } t_i \leq t \leq t_{i+1}  && \text{分段常數} \\
    \eta(t) & = \eta_0 \cdot e^{-\lambda t} && \text{指數衰減} \\
    \eta(t) & = \eta_0 \cdot (\beta t + 1)^{-\alpha} && \text{多項式衰減}
\end{aligned}
$$

在第一個*分段常數*（piecewise constant）場景中，我們會降低學習率，例如，每當最佳化進度停頓時。這是訓練深度網路的常見策略。或者，我們可以透過*指數衰減*（exponential decay）來更積極地減低它。不幸的是，這往往會導致演算法收斂之前過早停止。一個受歡迎的選擇是$\alpha = 0.5$的*多項式衰減*（polynomial decay）。在凸最佳化的情況下，有許多證據表明這種速率表現良好。

讓我們看看指數衰減在實踐中是什麼樣子。

```{.python .input}
#@tab all
def exponential_lr():
    # 在函式外部定義，而在內部更新的全域變數
    global t
    t += 1
    return math.exp(-0.1 * t)

t = 1
lr = exponential_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000, f_grad=f_grad))
```

正如預期的那樣，引數的方差大大減少。但是，這是以未能收斂到最優解$\mathbf{x} = (0, 0)$為代價的。即使經過1000個迭代步驟，我們仍然離最優解很遠。事實上，該演算法根本無法收斂。另一方面，如果我們使用多項式衰減，其中學習率隨迭代次數的平方根倒數衰減，那麼僅在50次迭代之後，收斂就會更好。

```{.python .input}
#@tab all
def polynomial_lr():
    # 在函式外部定義，而在內部更新的全域變數
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)

t = 1
lr = polynomial_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

關於如何設定學習率，還有更多的選擇。例如，我們可以從較小的學習率開始，然後使其迅速上漲，再讓它降低，儘管這會更慢。我們甚至可以在較小和較大的學習率之間切換。現在，讓我們專注於可以進行全面理論分析的學習率計劃，即凸環境下的學習率。對一般的非凸問題，很難獲得有意義的收斂保證，因為總的來說，最大限度地減少非線性非凸問題是NP困難的。有關的研究調查，請參閱例如2015年Tibshirani的優秀[講義筆記](https://www.stat.cmu.edu/~ryantibs/convexopt-F15/lectures/26-nonconvex.pdf)。

## 凸目標的收斂性分析

以下對凸目標函式的隨機梯度下降的收斂性分析是可選讀的，主要用於傳達對問題的更多直覺。我們只限於最簡單的證明之一 :cite:`Nesterov.Vial.2000`。存在著明顯更先進的證明技術，例如，當目標函式表現特別好時。

假設所有$\boldsymbol{\xi}$的目標函式$f(\boldsymbol{\xi}, \mathbf{x})$在$\mathbf{x}$中都是凸的。更具體地說，我們考慮隨機梯度下降更新：

$$\mathbf{x}_{t+1} = \mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}),$$

其中$f(\boldsymbol{\xi}_t, \mathbf{x})$是訓練樣本$f(\boldsymbol{\xi}_t, \mathbf{x})$的目標函式：$\boldsymbol{\xi}_t$從第$t$步的某個分佈中提取，$\mathbf{x}$是模型引數。用

$$R(\mathbf{x}) = E_{\boldsymbol{\xi}}[f(\boldsymbol{\xi}, \mathbf{x})]$$

表示期望風險，$R^*$表示對於$\mathbf{x}$的最低風險。最後讓$\mathbf{x}^*$表示最小值（我們假設它存在於定義$\mathbf{x}$的域中）。在這種情況下，我們可以跟蹤時間$t$處的當前引數$\mathbf{x}_t$和風險最小化器$\mathbf{x}^*$之間的距離，看看它是否隨著時間的推移而改善：

$$\begin{aligned}    &\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \\ =& \|\mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}) - \mathbf{x}^*\|^2 \\    =& \|\mathbf{x}_{t} - \mathbf{x}^*\|^2 + \eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 - 2 \eta_t    \left\langle \mathbf{x}_t - \mathbf{x}^*, \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\right\rangle.   \end{aligned}$$
:eqlabel:`eq_sgd-xt+1-xstar`

我們假設隨機梯度$\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})$的$L_2$範數受到某個常數$L$的限制，因此我們有

$$\eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 \leq \eta_t^2 L^2.$$
:eqlabel:`eq_sgd-L`

我們最感興趣的是$\mathbf{x}_t$和$\mathbf{x}^*$之間的距離如何變化的*期望*。事實上，對於任何具體的步驟序列，距離可能會增加，這取決於我們遇到的$\boldsymbol{\xi}_t$。因此我們需要點積的邊界。因為對於任何凸函式$f$，所有$\mathbf{x}$和$\mathbf{y}$都滿足$f(\mathbf{y}) \geq f(\mathbf{x}) + \langle f'(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle$，按凸性我們有

$$f(\boldsymbol{\xi}_t, \mathbf{x}^*) \geq f(\boldsymbol{\xi}_t, \mathbf{x}_t) + \left\langle \mathbf{x}^* - \mathbf{x}_t, \partial_{\mathbf{x}} f(\boldsymbol{\xi}_t, \mathbf{x}_t) \right\rangle.$$
:eqlabel:`eq_sgd-f-xi-xstar`

將不等式 :eqref:`eq_sgd-L`和 :eqref:`eq_sgd-f-xi-xstar`代入 :eqref:`eq_sgd-xt+1-xstar`我們在時間$t+1$時獲得引數之間距離的邊界，如下所示：

$$\|\mathbf{x}_{t} - \mathbf{x}^*\|^2 - \|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \geq 2 \eta_t (f(\boldsymbol{\xi}_t, \mathbf{x}_t) - f(\boldsymbol{\xi}_t, \mathbf{x}^*)) - \eta_t^2 L^2.$$
:eqlabel:`eqref_sgd-xt-diff`

這意味著，只要當前損失和最優損失之間的差異超過$\eta_t L^2/2$，我們就會取得進展。由於這種差異必然會收斂到零，因此學習率$\eta_t$也需要*消失*。

接下來，我們根據 :eqref:`eqref_sgd-xt-diff`取期望。得到

$$E\left[\|\mathbf{x}_{t} - \mathbf{x}^*\|^2\right] - E\left[\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2\right] \geq 2 \eta_t [E[R(\mathbf{x}_t)] - R^*] -  \eta_t^2 L^2.$$

最後一步是對$t \in \{1, \ldots, T\}$的不等式求和。在求和過程中抵消中間項，然後捨去低階項，可以得到

$$\|\mathbf{x}_1 - \mathbf{x}^*\|^2 \geq 2 \left (\sum_{t=1}^T   \eta_t \right) [E[R(\mathbf{x}_t)] - R^*] - L^2 \sum_{t=1}^T \eta_t^2.$$
:eqlabel:`eq_sgd-x1-xstar`

請注意，我們利用了給定的$\mathbf{x}_1$，因而可以去掉期望。最後定義

$$\bar{\mathbf{x}} \stackrel{\mathrm{def}}{=} \frac{\sum_{t=1}^T \eta_t \mathbf{x}_t}{\sum_{t=1}^T \eta_t}.$$

因為有

$$E\left(\frac{\sum_{t=1}^T \eta_t R(\mathbf{x}_t)}{\sum_{t=1}^T \eta_t}\right) = \frac{\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)]}{\sum_{t=1}^T \eta_t} = E[R(\mathbf{x}_t)],$$

根據詹森不等式（令 :eqref:`eq_jensens-inequality`中$i=t$，$\alpha_i = \eta_t/\sum_{t=1}^T \eta_t$）和$R$的凸性使其滿足的$E[R(\mathbf{x}_t)] \geq E[R(\bar{\mathbf{x}})]$，因此，

$$\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)] \geq \sum_{t=1}^T \eta_t  E\left[R(\bar{\mathbf{x}})\right].$$

將其代入不等式 :eqref:`eq_sgd-x1-xstar`得到邊界

$$
\left[E[\bar{\mathbf{x}}]\right] - R^* \leq \frac{r^2 + L^2 \sum_{t=1}^T \eta_t^2}{2 \sum_{t=1}^T \eta_t},
$$

其中$r^2 \stackrel{\mathrm{def}}{=} \|\mathbf{x}_1 - \mathbf{x}^*\|^2$是初始選擇引數與最終結果之間距離的邊界。簡而言之，收斂速度取決於隨機梯度標準的限制方式（$L$）以及初始引數值與最優結果的距離（$r$）。請注意，邊界由$\bar{\mathbf{x}}$而不是$\mathbf{x}_T$表示。因為$\bar{\mathbf{x}}$是最佳化路徑的平滑版本。只要知道$r, L$和$T$，我們就可以選擇學習率$\eta = r/(L \sqrt{T})$。這個就是上界$rL/\sqrt{T}$。也就是說，我們將按照速度$\mathcal{O}(1/\sqrt{T})$收斂到最優解。

## 隨機梯度和有限樣本

到目前為止，在談論隨機梯度下降時，我們進行得有點快而鬆散。我們假設從分佈$p(x, y)$中取樣得到樣本$x_i$（通常帶有標籤$y_i$），並且用它來以某種方式更新模型引數。特別是，對於有限的樣本數量，我們僅僅討論了由某些允許我們在其上執行隨機梯度下降的函式$\delta_{x_i}$和$\delta_{y_i}$組成的離散分佈$p(x, y) = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}(x) \delta_{y_i}(y)$。

但是，這不是我們真正做的。在本節的簡單範例中，我們只是將噪聲新增到其他非隨機梯度上，也就是說，我們假裝有成對的$(x_i, y_i)$。事實證明，這種做法在這裡是合理的（有關詳細討論，請參閱練習）。更麻煩的是，在以前的所有討論中，我們顯然沒有這樣做。相反，我們遍歷了所有實例*恰好一次*。要了解為什麼這更可取，可以反向考慮一下，即我們*有替換地*從離散分佈中取樣$n$個觀測值。隨機選擇一個元素$i$的機率是$1/n$。因此選擇它*至少*一次就是

$$P(\mathrm{choose~} i) = 1 - P(\mathrm{omit~} i) = 1 - (1-1/n)^n \approx 1-e^{-1} \approx 0.63.$$

類似的推理表明，挑選一些樣本（即訓練範例）*恰好一次*的機率是

$${n \choose 1} \frac{1}{n} \left(1-\frac{1}{n}\right)^{n-1} = \frac{n}{n-1} \left(1-\frac{1}{n}\right)^{n} \approx e^{-1} \approx 0.37.$$

這導致與*無替換*取樣相比，方差增加並且資料效率降低。因此，在實踐中我們執行後者（這是本書中的預設選擇）。最後一點注意，重複採用訓練資料集的時候，會以*不同的*隨機順序遍歷它。

## 小結

* 對於凸問題，我們可以證明，對於廣泛的學習率選擇，隨機梯度下降將收斂到最優解。
* 對於深度學習而言，情況通常並非如此。但是，對凸問題的分析使我們能夠深入瞭解如何進行最佳化，即逐步降低學習率，儘管不是太快。
* 如果學習率太小或太大，就會出現問題。實際上，通常只有經過多次實驗後才能找到合適的學習率。
* 當訓練資料集中有更多樣本時，計算梯度下降的每次迭代的代價更高，因此在這些情況下，首選隨機梯度下降。
* 隨機梯度下降的最優性保證在非凸情況下一般不可用，因為需要檢查的區域性最小值的數量可能是指數級的。

## 練習

1. 嘗試不同的隨機梯度下降學習率計劃和不同的迭代次數進行實驗。特別是，根據迭代次數的函式來繪製與最優解$(0, 0)$的距離。
1. 證明對於函式$f(x_1, x_2) = x_1^2 + 2 x_2^2$而言，向梯度新增正態噪聲等同於最小化損失函式$f(\mathbf{x}, \mathbf{w}) = (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$，其中$\mathbf{x}$是從常態分佈中提取的。
1. 從$\{(x_1, y_1), \ldots, (x_n, y_n)\}$分別使用替換方法以及不替換方法進行取樣時，比較隨機梯度下降的收斂性。
1. 如果某些梯度（或者更確切地說與之相關的某些坐標）始終比所有其他梯度都大，將如何更改隨機梯度下降求解器？
1. 假設$f(x) = x^2 (1 + \sin x)$。$f$有多少區域性最小值？請試著改變$f$以儘量減少它需要評估所有區域性最小值的方式。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/3837)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/3838)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3839)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11849)
:end_tab: