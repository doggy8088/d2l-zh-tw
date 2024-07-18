# 微積分
:label:`sec_calculus`

在2500年前，古希臘人把一個多邊形分成三角形，並把它們的面積相加，才找到計算多邊形面積的方法。
為了求出曲線形狀（比如圓）的面積，古希臘人在這樣的形狀上刻內接多邊形。
如 :numref:`fig_circle_area`所示，內接多邊形的等長邊越多，就越接近圓。
這個過程也被稱為*逼近法*（method of exhaustion）。

![用逼近法求圓的面積](../img/polygon-circle.svg)
:label:`fig_circle_area`

事實上，逼近法就是*積分*（integral calculus）的起源。
2000多年後，微積分的另一支，*微分*（differential calculus）被髮明出來。
在微分學最重要的應用是最佳化問題，即考慮如何把事情做到最好。
正如在 :numref:`subsec_norms_and_objectives`中討論的那樣，
這種問題在深度學習中是無處不在的。

在深度學習中，我們『訓練』模型，不斷更新它們，使它們在看到越來越多的資料時變得越來越好。
通常情況下，變得更好意味著最小化一個*損失函式*（loss function），
即一個衡量『模型有多糟糕』這個問題的分數。
最終，我們真正關心的是生成一個模型，它能夠在從未見過的資料上表現良好。
但『訓練』模型只能將模型與我們實際能看到的資料相擬合。
因此，我們可以將擬合模型的任務分解為兩個關鍵問題：

* *最佳化*（optimization）：用模型擬合觀測資料的過程；
* *泛化*（generalization）：數學原理和實踐者的智慧，能夠指導我們生成出有效性超出用於訓練的資料集本身的模型。

為了幫助讀者在後麵的章節中更好地理解最佳化問題和方法，
本節提供了一個非常簡短的入門教程，幫助讀者快速掌握深度學習中常用的微分知識。

## 導數和微分

我們首先討論導數的計算，這是幾乎所有深度學習最佳化演算法的關鍵步驟。
在深度學習中，我們通常選擇對於模型引數可微的損失函式。
簡而言之，對於每個引數，
如果我們把這個引數*增加*或*減少*一個無窮小的量，可以知道損失會以多快的速度增加或減少，

假設我們有一個函式$f: \mathbb{R} \rightarrow \mathbb{R}$，其輸入和輸出都是標量。
(**如果$f$的*導數*存在，這個極限被定義為**)

(**$$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}.$$**)
:eqlabel:`eq_derivative`

如果$f'(a)$存在，則稱$f$在$a$處是*可微*（differentiable）的。
如果$f$在一個區間內的每個數上都是可微的，則此函式在此區間中是可微的。
我們可以將 :eqref:`eq_derivative`中的導數$f'(x)$解釋為$f(x)$相對於$x$的*瞬時*（instantaneous）變化率。
所謂的瞬時變化率是基於$x$中的變化$h$，且$h$接近$0$。

為了更好地解釋導數，讓我們做一個實驗。
(**定義$u=f(x)=3x^2-4x$**)如下：

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from matplotlib_inline import backend_inline
from mxnet import np, npx
npx.set_np()

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from matplotlib_inline import backend_inline
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from matplotlib_inline import backend_inline
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
from matplotlib_inline import backend_inline
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

[**透過令$x=1$並讓$h$接近$0$，**] :eqref:`eq_derivative`中(**$\frac{f(x+h)-f(x)}{h}$的數值結果接近$2$**)。
雖然這個實驗不是一個數學證明，但稍後會看到，當$x=1$時，導數$u'$是$2$。

```{.python .input}
#@tab all
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1
```

讓我們熟悉一下導數的幾個等價符號。
給定$y=f(x)$，其中$x$和$y$分別是函式$f$的自變量和因變量。以下表示式是等價的：

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$

其中符號$\frac{d}{dx}$和$D$是*微分運算符*，表示*微分*操作。
我們可以使用以下規則來對常見函式求微分：

* $DC = 0$（$C$是一個常數）
* $Dx^n = nx^{n-1}$（*冪律*（power rule），$n$是任意實數）
* $De^x = e^x$
* $D\ln(x) = 1/x$

為了微分一個由一些常見函式組成的函式，下面的一些法則方便使用。
假設函式$f$和$g$都是可微的，$C$是一個常數，則：

*常數相乘法則*
$$\frac{d}{dx} [Cf(x)] = C \frac{d}{dx} f(x),$$

*加法法則*

$$\frac{d}{dx} [f(x) + g(x)] = \frac{d}{dx} f(x) + \frac{d}{dx} g(x),$$

*乘法法則*

$$\frac{d}{dx} [f(x)g(x)] = f(x) \frac{d}{dx} [g(x)] + g(x) \frac{d}{dx} [f(x)],$$

*除法法則*

$$\frac{d}{dx} \left[\frac{f(x)}{g(x)}\right] = \frac{g(x) \frac{d}{dx} [f(x)] - f(x) \frac{d}{dx} [g(x)]}{[g(x)]^2}.$$

現在我們可以應用上述幾個法則來計算$u'=f'(x)=3\frac{d}{dx}x^2-4\frac{d}{dx}x=6x-4$。
令$x=1$，我們有$u'=2$：在這個實驗中，數值結果接近$2$，
這一點得到了在本節前面的實驗的支援。
當$x=1$時，此導數也是曲線$u=f(x)$切線的斜率。

[**為了對導數的這種解釋進行視覺化，我們將使用`matplotlib`**]，
這是一個Python中流行的繪圖函式庫。
要設定`matplotlib`生成圖形的屬性，我們需要(**定義幾個函式**)。
在下面，`use_svg_display`函式指定`matplotlib`軟體套件輸出svg圖表以獲得更清晰的影象。

注意，註解`#@save`是一個特殊的標記，會將對應的函式、類或語句儲存在`d2l`套件中。
因此，以後無須重新定義就可以直接呼叫它們（例如，`d2l.use_svg_display()`）。

```{.python .input}
#@tab all
def use_svg_display():  #@save
    """使用svg格式在Jupyter中顯示繪圖"""
    backend_inline.set_matplotlib_formats('svg')
```

我們定義`set_figsize`函式來設定圖表大小。
注意，這裡可以直接使用`d2l.plt`，因為匯入語句
`from matplotlib import pyplot as plt`已標記為儲存到`d2l`套件中。

```{.python .input}
#@tab all
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """設定matplotlib的圖表大小"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
```

下面的`set_axes`函式用於設定由`matplotlib`生成圖表的軸的屬性。

```{.python .input}
#@tab all
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """設定matplotlib的軸"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
```

透過這三個用於圖形設定的函式，定義一個`plot`函式來簡潔地繪製多條曲線，
因為我們需要在整個書中視覺化許多曲線。

```{.python .input}
#@tab all
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """繪製資料點"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # 如果X有一個軸，輸出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
```

現在我們可以[**繪製函式$u=f(x)$及其在$x=1$處的切線$y=2x-3$**]，
其中係數$2$是切線的斜率。

```{.python .input}
#@tab all
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```

## 偏導數

到目前為止，我們只討論了僅含一個變數的函式的微分。
在深度學習中，函式通常依賴於許多變數。
因此，我們需要將微分的思想推廣到*多元函式*（multivariate function）上。

設$y = f(x_1, x_2, \ldots, x_n)$是一個具有$n$個變數的函式。
$y$關於第$i$個引數$x_i$的*偏導數*（partial derivative）為：

$$ \frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$

為了計算$\frac{\partial y}{\partial x_i}$，
我們可以簡單地將$x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$看作常數，
並計算$y$關於$x_i$的導數。
對於偏導數的表示，以下是等價的：

$$\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = f_{x_i} = f_i = D_i f = D_{x_i} f.$$

## 梯度
:label:`subsec_calculus-grad`

我們可以連結一個多元函式對其所有變數的偏導數，以得到該函式的*梯度*（gradient）向量。
具體而言，設函式$f:\mathbb{R}^n\rightarrow\mathbb{R}$的輸入是
一個$n$維向量$\mathbf{x}=[x_1,x_2,\ldots,x_n]^\top$，並且輸出是一個標量。
函式$f(\mathbf{x})$相對於$\mathbf{x}$的梯度是一個包含$n$個偏導數的向量:

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_n}\bigg]^\top,$$

其中$\nabla_{\mathbf{x}} f(\mathbf{x})$通常在沒有歧義時被$\nabla f(\mathbf{x})$取代。

假設$\mathbf{x}$為$n$維向量，在微分多元函式時經常使用以下規則:

* 對於所有$\mathbf{A} \in \mathbb{R}^{m \times n}$，都有$\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$
* 對於所有$\mathbf{A} \in \mathbb{R}^{n \times m}$，都有$\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$
* 對於所有$\mathbf{A} \in \mathbb{R}^{n \times n}$，都有$\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$
* $\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$

同樣，對於任何矩陣$\mathbf{X}$，都有$\nabla_{\mathbf{X}} \|\mathbf{X} \|_F^2 = 2\mathbf{X}$。
正如我們之後將看到的，梯度對於設計深度學習中的最佳化演算法有很大用處。

## 鏈式法則

然而，上面方法可能很難找到梯度。
這是因為在深度學習中，多元函式通常是*複合*（composite）的，
所以難以應用上述任何規則來微分這些函式。
幸運的是，鏈式法則可以被用來微分複合函式。

讓我們先考慮單變數函式。假設函式$y=f(u)$和$u=g(x)$都是可微的，根據鏈式法則：

$$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.$$

現在考慮一個更一般的場景，即函式具有任意數量的變數的情況。
假設可微分函式$y$有變數$u_1, u_2, \ldots, u_m$，其中每個可微分函式$u_i$都有變數$x_1, x_2, \ldots, x_n$。
注意，$y$是$x_1, x_2， \ldots, x_n$的函式。
對於任意$i = 1, 2, \ldots, n$，鏈式法則給出：

$$\frac{\partial y}{\partial x_i} = \frac{\partial y}{\partial u_1} \frac{\partial u_1}{\partial x_i} + \frac{\partial y}{\partial u_2} \frac{\partial u_2}{\partial x_i} + \cdots + \frac{\partial y}{\partial u_m} \frac{\partial u_m}{\partial x_i}$$

## 小結

* 微分和積分是微積分的兩個分支，前者可以應用於深度學習中的最佳化問題。
* 導數可以被解釋為函式相對於其變數的瞬時變化率，它也是函式曲線的切線的斜率。
* 梯度是一個向量，其分量是多變量函式相對於其所有變數的偏導數。
* 鏈式法則可以用來微分複合函式。

## 練習

1. 繪製函式$y = f(x) = x^3 - \frac{1}{x}$和其在$x = 1$處切線的影象。
1. 求函式$f(\mathbf{x}) = 3x_1^2 + 5e^{x_2}$的梯度。
1. 函式$f(\mathbf{x}) = \|\mathbf{x}\|_2$的梯度是什麼？
1. 嘗試寫出函式$u = f(x, y, z)$，其中$x = x(a, b)$，$y = y(a, b)$，$z = z(a, b)$的鏈式法則。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1755)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1756)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1754)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11684)
:end_tab:
