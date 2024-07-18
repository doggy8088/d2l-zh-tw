# 最佳化和深度學習

本節將討論最佳化與深度學習之間的關係以及在深度學習中使用最佳化的挑戰。對於深度學習問題，我們通常會先定義*損失函式*。一旦我們有了損失函式，我們就可以使用最佳化演算法來嘗試最小化損失。在最佳化中，損失函式通常被稱為最佳化問題的*目標函式*。按照傳統慣例，大多數最佳化演算法都關注的是*最小化*。如果我們需要最大化目標，那麼有一個簡單的解決方案：在目標函式前加負號即可。

## 最佳化的目標

儘管最佳化提供了一種最大限度地減少深度學習損失函式的方法，但本質上，最佳化和深度學習的目標是根本不同的。前者主要關注的是最小化目標，後者則關注在給定有限資料量的情況下尋找合適的模型。在 :numref:`sec_model_selection`中，我們詳細討論了這兩個目標之間的區別。例如，訓練誤差和泛化誤差通常不同：由於最佳化演算法的目標函式通常是基於訓練資料集的損失函式，因此最佳化的目標是減少訓練誤差。但是，深度學習（或更廣義地說，統計推斷）的目標是減少泛化誤差。為了實現後者，除了使用最佳化演算法來減少訓練誤差之外，我們還需要注意過擬合。

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

為了說明上述不同的目標，引入兩個概念*風險*和*經驗風險*。如 :numref:`subsec_empirical-risk-and-risk`所述，經驗風險是訓練資料集的平均損失，而風險則是整個資料群的預期損失。下面我們定義了兩個函式：風險函式`f`和經驗風險函式`g`。假設我們只有有限的訓練資料。因此，這裡的`g`不如`f`平滑。

```{.python .input}
#@tab all
def f(x):
    return x * d2l.cos(np.pi * x)

def g(x):
    return f(x) + 0.2 * d2l.cos(5 * np.pi * x)
```

下圖說明，訓練資料集的最低經驗風險可能與最低風險（泛化誤差）不同。

```{.python .input}
#@tab mxnet, pytorch, tensorflow
def annotate(text, xy, xytext):  #@save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))

x = d2l.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
```

```{.python .input}
#@tab paddle
def annotate(text, xy, xytext):  #@save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))

x = d2l.arange(0.5, 1.5, 0.01, dtype='float32')
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
```

## 深度學習中的最佳化挑戰

本章將關注最佳化演算法在最小化目標函式方面的效能，而不是模型的泛化誤差。在 :numref:`sec_linear_regression`中，我們區分了最佳化問題中的解析解和數值解。在深度學習中，大多數目標函式都很複雜，沒有解析解。相反，我們必須使用數值最佳化演算法。本章中的最佳化演算法都屬於此類別。

深度學習最佳化存在許多挑戰。其中最令人煩惱的是區域性最小值、鞍點和梯度消失。

### 區域性最小值

對於任何目標函式$f(x)$，如果在$x$處對應的$f(x)$值小於在$x$附近任意其他點的$f(x)$值，那麼$f(x)$可能是區域性最小值。如果$f(x)$在$x$處的值是整個域中目標函式的最小值，那麼$f(x)$是全域最小值。

例如，給定函式

$$f(x) = x \cdot \text{cos}(\pi x) \text{ for } -1.0 \leq x \leq 2.0,$$

我們可以近似該函式的區域性最小值和全域最小值。

```{.python .input}
#@tab mxnet, pytorch, tensorflow
x = d2l.arange(-1.0, 2.0, 0.01)
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
```

```{.python .input}
#@tab paddle
x = d2l.arange(-1.0, 2.0, 0.01, dtype='float32')
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
```

深度學習模型的目標函式通常有許多區域性最優解。當最佳化問題的數值解接近區域性最優值時，隨著目標函式解的梯度接近或變為零，透過最終迭代獲得的數值解可能僅使目標函式*區域性*最優，而不是*全域*最優。只有一定程度的噪聲可能會使引數跳出區域性最小值。事實上，這是小批次隨機梯度下降的有利屬性之一。在這種情況下，小批次上梯度的自然變化能夠將引數從區域性極小值中跳出。

### 鞍點

除了區域性最小值之外，鞍點是梯度消失的另一個原因。*鞍點*（saddle point）是指函式的所有梯度都消失但既不是全域最小值也不是區域性最小值的任何位置。考慮這個函式$f(x) = x^3$。它的一階和二階導數在$x=0$時消失。這時最佳化可能會停止，儘管它不是最小值。

```{.python .input}
#@tab mxnet, pytorch, tensorflow
x = d2l.arange(-2.0, 2.0, 0.01)
d2l.plot(x, [x**3], 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
```

```{.python .input}
#@tab paddle
x = d2l.arange(-2.0, 2.0, 0.01, dtype='float32')
d2l.plot(x, [x**3], 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
```

如下例所示，較高維度的鞍點甚至更加隱蔽。考慮這個函式$f(x, y) = x^2 - y^2$。它的鞍點為$(0, 0)$。這是關於$y$的最大值，也是關於$x$的最小值。此外，它看起來像個馬鞍，這就是鞍點的名字由來。

```{.python .input}
#@tab mxnet
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), z.asnumpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

```{.python .input}
#@tab pytorch, tensorflow, paddle
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

我們假設函式的輸入是$k$維向量，其輸出是標量，因此其Hessian矩陣（也稱黑塞矩陣）將有$k$個特徵值（參考[特徵分解的線上附錄](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/eigendecomposition.html))。函式的解可能是區域性最小值、區域性最大值或函式梯度為零位置處的鞍點：

* 當函式在零梯度位置處的Hessian矩陣的特徵值全部為正值時，我們有該函式的區域性最小值；
* 當函式在零梯度位置處的Hessian矩陣的特徵值全部為負值時，我們有該函式的區域性最大值；
* 當函式在零梯度位置處的Hessian矩陣的特徵值為負值和正值時，我們有該函式的一個鞍點。

對於高維度問題，至少*部分*特徵值為負的可能性相當高。這使得鞍點比區域性最小值更有可能出現。我們將在下一節介紹凸性時討論這種情況的一些例外。簡而言之，凸函式是Hessian函式的特徵值永遠不為負值的函式。不幸的是，大多數深度學習問題並不屬於這一類。儘管如此，它還是研究最佳化演算法的一個很好的工具。

### 梯度消失

可能遇到的最隱蔽問題是梯度消失。回想一下我們在 :numref:`subsec_activation_functions`中常用的啟用函式及其衍生函式。例如，假設我們想最小化函式$f(x) = \tanh(x)$，然後我們恰好從$x = 4$開始。正如我們所看到的那樣，$f$的梯度接近零。更具體地說，$f'(x) = 1 - \tanh^2(x)$，因此是$f'(4) = 0.0013$。因此，在我們取得進展之前，最佳化將會停滯很長一段時間。事實證明，這是在引入ReLU啟用函式之前訓練深度學習模型相當棘手的原因之一。

```{.python .input}
#@tab mxnet, pytorch, tensorflow
x = d2l.arange(-2.0, 5.0, 0.01)
d2l.plot(x, [d2l.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))
```

```{.python .input}
#@tab paddle
x = d2l.arange(-2.0, 5.0, 0.01, dtype='float32')
d2l.plot(x, [d2l.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))
```

正如我們所看到的那樣，深度學習的最佳化充滿挑戰。幸運的是，有一系列強大的演算法表現良好，即使對於初學者也很容易使用。此外，沒有必要找到最優解。區域性最優解或其近似解仍然非常有用。

## 小結

* 最小化訓練誤差並*不能*保證我們找到最佳的引數集來最小化泛化誤差。
* 最佳化問題可能有許多區域性最小值。
* 一個問題可能有很多的鞍點，因為問題通常不是凸的。
* 梯度消失可能會導致最佳化停滯，重引數化通常會有所幫助。對引數進行良好的初始化也可能是有益的。

## 練習

1. 考慮一個簡單的MLP，它有一個隱藏層，比如，隱藏層中維度為$d$和一個輸出。證明對於任何區域性最小值，至少有$d！$個等效方案。
1. 假設我們有一個對稱隨機矩陣$\mathbf{M}$，其中條目$M_{ij} = M_{ji}$各自從某種機率分佈$p_{ij}$中抽取。此外，假設$p_{ij}(x) = p_{ij}(-x)$，即分佈是對稱的（詳情請參見 :cite:`Wigner.1958`）。
    1. 證明特徵值的分佈也是對稱的。也就是說，對於任何特徵向量$\mathbf{v}$，關聯的特徵值$\lambda$滿足$P(\lambda > 0) = P(\lambda < 0)$的機率為$P(\lambda > 0) = P(\lambda < 0)$。
    1. 為什麼以上*沒有*暗示$P(\lambda > 0) = 0.5$？
1. 你能想到深度學習最佳化還涉及哪些其他挑戰？
1. 假設你想在（真實的）鞍上平衡一個（真實的）球。
    1. 為什麼這很難？
    1. 能利用這種效應來最佳化演算法嗎？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/3840)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/3841)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3842)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11846)
:end_tab: