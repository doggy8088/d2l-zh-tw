# 梯度下降
:label:`sec_gd`

儘管*梯度下降*（gradient descent）很少直接用於深度學習，
但瞭解它是理解下一節隨機梯度下降演算法的關鍵。
例如，由於學習率過大，最佳化問題可能會發散，這種現象早已在梯度下降中出現。
同樣地，*預處理*（preconditioning）是梯度下降中的一種常用技術，
還被沿用到更高階的演算法中。
讓我們從簡單的一維梯度下降開始。

## 一維梯度下降

為什麼梯度下降演算法可以最佳化目標函式？
一維中的梯度下降給我們很好的啟發。
考慮一類連續可微實值函式$f: \mathbb{R} \rightarrow \mathbb{R}$，
利用泰勒展開，我們可以得到

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \mathcal{O}(\epsilon^2).$$
:eqlabel:`gd-taylor`

即在一階近似中，$f(x+\epsilon)$可透過$x$處的函式值$f(x)$和一階導數$f'(x)$得出。
我們可以假設在負梯度方向上移動的$\epsilon$會減少$f$。
為了簡單起見，我們選擇固定步長$\eta > 0$，然後取$\epsilon = -\eta f'(x)$。
將其代入泰勒展開式我們可以得到

$$f(x - \eta f'(x)) = f(x) - \eta f'^2(x) + \mathcal{O}(\eta^2 f'^2(x)).$$
:eqlabel:`gd-taylor-2`

如果其導數$f'(x) \neq 0$沒有消失，我們就能繼續展開，這是因為$\eta f'^2(x)>0$。
此外，我們總是可以令$\eta$小到足以使高階項變得不相關。
因此，

$$f(x - \eta f'(x)) \lessapprox f(x).$$

這意味著，如果我們使用

$$x \leftarrow x - \eta f'(x)$$

來迭代$x$，函式$f(x)$的值可能會下降。
因此，在梯度下降中，我們首先選擇初始值$x$和常數$\eta > 0$，
然後使用它們連續迭代$x$，直到停止條件達成。
例如，當梯度$|f'(x)|$的幅度足夠小或迭代次數達到某個值時。

下面我們來展示如何實現梯度下降。為了簡單起見，我們選用目標函式$f(x)=x^2$。
儘管我們知道$x=0$時$f(x)$能取得最小值，
但我們仍然使用這個簡單的函式來觀察$x$的變化。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import paddle
```

```{.python .input}
#@tab all
def f(x):  # 目標函式
    return x ** 2

def f_grad(x):  # 目標函式的梯度(導數)
    return 2 * x
```

接下來，我們使用$x=10$作為初始值，並假設$\eta=0.2$。
使用梯度下降法迭代$x$共10次，我們可以看到，$x$的值最終將接近最優解。

```{.python .input}
#@tab mxnet, pytorch, tensorflow
def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f'epoch 10, x: {x:f}')
    return results

results = gd(0.2, f_grad)
```

```{.python .input}
#@tab paddle
def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f'epoch 10, x: {float(x):f}')
    return results

results = gd(0.2, f_grad)
```

對進行$x$最佳化的過程可以繪製如下。

```{.python .input}
#@tab mxnet, pytorch, tensorflow
def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = d2l.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

show_trace(results, f)
```

```{.python .input}
#@tab paddle
def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = d2l.arange(-n, n, 0.01, dtype='float32')
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

show_trace(results, f)
```

### 學習率
:label:`subsec_gd-learningrate`

*學習率*（learning rate）決定目標函式能否收斂到區域性最小值，以及何時收斂到最小值。
學習率$\eta$可由演算法設計者設定。
請注意，如果我們使用的學習率太小，將導致$x$的更新非常緩慢，需要更多的迭代。
例如，考慮同一最佳化問題中$\eta = 0.05$的進度。
如下所示，儘管經過了10個步驟，我們仍然離最優解很遠。

```{.python .input}
#@tab all
show_trace(gd(0.05, f_grad), f)
```

相反，如果我們使用過高的學習率，$\left|\eta f'(x)\right|$對於一階泰勒展開式可能太大。
也就是說， :eqref:`gd-taylor`中的$\mathcal{O}(\eta^2 f'^2(x))$可能變得顯著了。
在這種情況下，$x$的迭代不能保證降低$f(x)$的值。
例如，當學習率為$\eta=1.1$時，$x$超出了最優解$x=0$並逐漸發散。

```{.python .input}
#@tab all
show_trace(gd(1.1, f_grad), f)
```

### 區域性最小值

為了示範非凸函式的梯度下降，考慮函式$f(x) = x \cdot \cos(cx)$，其中$c$為某常數。
這個函式有無窮多個區域性最小值。
根據我們選擇的學習率，我們最終可能只會得到許多解的一個。
下面的例子說明了（不切實際的）高學習率如何導致較差的區域性最小值。

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # 目標函式
    return x * d2l.cos(c * x)

def f_grad(x):  # 目標函式的梯度
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

show_trace(gd(2, f_grad), f)
```

## 多元梯度下降

現在我們對單變數的情況有了更好的理解，讓我們考慮一下$\mathbf{x} = [x_1, x_2, \ldots, x_d]^\top$的情況。
即目標函式$f: \mathbb{R}^d \to \mathbb{R}$將向量對映成標量。
相應地，它的梯度也是多元的，它是一個由$d$個偏導數組成的向量：

$$\nabla f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\bigg]^\top.$$

梯度中的每個偏導數元素$\partial f(\mathbf{x})/\partial x_i$代表了當輸入$x_i$時$f$在$\mathbf{x}$處的變化率。
和先前單變數的情況一樣，我們可以對多變量函式使用相應的泰勒近似來思考。
具體來說，

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \mathbf{\boldsymbol{\epsilon}}^\top \nabla f(\mathbf{x}) + \mathcal{O}(\|\boldsymbol{\epsilon}\|^2).$$
:eqlabel:`gd-multi-taylor`

換句話說，在$\boldsymbol{\epsilon}$的二階項中，
最陡下降的方向由負梯度$-\nabla f(\mathbf{x})$得出。
選擇合適的學習率$\eta > 0$來生成典型的梯度下降演算法：

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x}).$$

這個演算法在實踐中的表現如何呢？
我們構造一個目標函式$f(\mathbf{x})=x_1^2+2x_2^2$，
並有二維向量$\mathbf{x} = [x_1, x_2]^\top$作為輸入，
標量作為輸出。
梯度由$\nabla f(\mathbf{x}) = [2x_1, 4x_2]^\top$給出。
我們將從初始位置$[-5, -2]$透過梯度下降觀察$\mathbf{x}$的軌跡。

我們還需要兩個輔助函式：
第一個是update函式，並將其應用於初始值20次；
第二個函式會顯示$\mathbf{x}$的軌跡。

```{.python .input}
#@tab mxnet, pytorch, tensorflow
def train_2d(trainer, steps=20, f_grad=None):  #@save
    """用定製的訓練機最佳化2D目標函式"""
    # s1和s2是稍後將使用的內部狀態變數
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results
```

```{.python .input}
#@tab mxnet, tensorflow
def show_trace_2d(f, results):  #@save
    """顯示最佳化過程中2D變數的軌跡"""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

```{.python .input}
#@tab pytorch
def show_trace_2d(f, results):  #@save
    """顯示最佳化過程中2D變數的軌跡"""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1), indexing='ij')
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

```{.python .input}
#@tab paddle
def train_2d(trainer, steps=20, f_grad=None):  #@save
    """用定製的訓練機最佳化2D目標函式"""
    # s1和s2是稍後將使用的內部狀態變數
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results

def show_trace_2d(f, results):  #@save
    """顯示最佳化過程中2D變數的軌跡"""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1, dtype='float32'),
                          d2l.arange(-3.0, 1.0, 0.1, dtype='float32'))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

接下來，我們觀察學習率$\eta = 0.1$時最佳化變數$\mathbf{x}$的軌跡。
可以看到，經過20步之後，$\mathbf{x}$的值接近其位於$[0, 0]$的最小值。
雖然進展相當順利，但相當緩慢。

```{.python .input}
#@tab all
def f_2d(x1, x2):  # 目標函式
    return x1 ** 2 + 2 * x2 ** 2

def f_2d_grad(x1, x2):  # 目標函式的梯度
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

eta = 0.1
show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))
```

## 自適應方法

正如我們在 :numref:`subsec_gd-learningrate`中所看到的，選擇『恰到好處』的學習率$\eta$是很棘手的。
如果我們把它選得太小，就沒有什麼進展；如果太大，得到的解就會振盪，甚至可能發散。
如果我們可以自動確定$\eta$，或者完全不必選擇學習率，會怎麼樣？
除了考慮目標函式的值和梯度、還考慮它的曲率的二階方法可以幫我們解決這個問題。
雖然由於計算代價的原因，這些方法不能直接應用於深度學習，但它們為如何設計高階最佳化演算法提供了有用的思維直覺，這些演算法可以模擬下面概述的演算法的許多理想屬性。

### 牛頓法

回顧一些函式$f: \mathbb{R}^d \rightarrow \mathbb{R}$的泰勒展開式，事實上我們可以把它寫成

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \nabla^2 f(\mathbf{x}) \boldsymbol{\epsilon} + \mathcal{O}(\|\boldsymbol{\epsilon}\|^3).$$
:eqlabel:`gd-hot-taylor`

為了避免繁瑣的符號，我們將$\mathbf{H} \stackrel{\mathrm{def}}{=} \nabla^2 f(\mathbf{x})$定義為$f$的Hessian，是$d \times d$矩陣。
當$d$的值很小且問題很簡單時，$\mathbf{H}$很容易計算。
但是對於深度神經網路而言，考慮到$\mathbf{H}$可能非常大，
$\mathcal{O}(d^2)$個條目的儲存代價會很高，
此外透過反向傳播進行計算可能雪上加霜。
然而，我們姑且先忽略這些考量，看看會得到什麼演算法。

畢竟，$f$的最小值滿足$\nabla f = 0$。
遵循 :numref:`sec_calculus`中的微積分規則，
透過取$\boldsymbol{\epsilon}$對 :eqref:`gd-hot-taylor`的導數，
再忽略不重要的高階項，我們便得到

$$\nabla f(\mathbf{x}) + \mathbf{H} \boldsymbol{\epsilon} = 0 \text{ and hence }
\boldsymbol{\epsilon} = -\mathbf{H}^{-1} \nabla f(\mathbf{x}).$$

也就是說，作為最佳化問題的一部分，我們需要將Hessian矩陣$\mathbf{H}$求逆。

舉一個簡單的例子，對於$f(x) = \frac{1}{2} x^2$，我們有$\nabla f(x) = x$和$\mathbf{H} = 1$。
因此，對於任何$x$，我們可以獲得$\epsilon = -x$。
換言之，單單一步就足以完美地收斂，而無須任何調整。
我們在這裡比較幸運：泰勒展開式是確切的，因為$f(x+\epsilon)= \frac{1}{2} x^2 + \epsilon x + \frac{1}{2} \epsilon^2$。

讓我們看看其他問題。
給定一個凸雙曲餘弦函式$c$，其中$c$為某些常數，
我們可以看到經過幾次迭代後，得到了$x=0$處的全域最小值。

```{.python .input}
#@tab all
c = d2l.tensor(0.5)

def f(x):  # O目標函式
    return d2l.cosh(c * x)

def f_grad(x):  # 目標函式的梯度
    return c * d2l.sinh(c * x)

def f_hess(x):  # 目標函式的Hessian
    return c**2 * d2l.cosh(c * x)

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    print('epoch 10, x:', x)
    return results

show_trace(newton(), f)
```

現在讓我們考慮一個非凸函式，比如$f(x) = x \cos(c x)$，$c$為某些常數。
請注意在牛頓法中，我們最終將除以Hessian。
這意味著如果二階導數是負的，$f$的值可能會趨於增加。
這是這個演算法的致命缺陷！
讓我們看看實踐中會發生什麼。

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # 目標函式
    return x * d2l.cos(c * x)

def f_grad(x):  # 目標函式的梯度
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

def f_hess(x):  # 目標函式的Hessian
    return - 2 * c * d2l.sin(c * x) - x * c**2 * d2l.cos(c * x)

show_trace(newton(), f)
```

這發生了驚人的錯誤。我們怎樣才能修正它？
一種方法是用取Hessian的絕對值來修正，另一個策略是重新引入學習率。
這似乎違背了初衷，但不完全是——擁有二階訊息可以使我們在曲率較大時保持謹慎，而在目標函式較平坦時則採用較大的學習率。
讓我們看看在學習率稍小的情況下它是如何生效的，比如$\eta = 0.5$。
如我們所見，我們有了一個相當高效的演算法。

```{.python .input}
#@tab all
show_trace(newton(0.5), f)
```

### 收斂性分析

在此，我們以部分目標凸函式$f$為例，分析它們的牛頓法收斂速度。
這些目標凸函式三次可微，而且二階導數不為零，即$f'' > 0$。
由於多變量情況下的證明是對以下一維引數情況證明的直接拓展，對我們理解這個問題不能提供更多幫助，因此我們省略了多變量情況的證明。

用$x^{(k)}$表示$x$在第$k^\mathrm{th}$次迭代時的值，
令$e^{(k)} \stackrel{\mathrm{def}}{=} x^{(k)} - x^*$表示$k^\mathrm{th}$迭代時與最優性的距離。
透過泰勒展開，我們得到條件$f'(x^*) = 0$可以寫成

$$0 = f'(x^{(k)} - e^{(k)}) = f'(x^{(k)}) - e^{(k)} f''(x^{(k)}) + \frac{1}{2} (e^{(k)})^2 f'''(\xi^{(k)}),$$

這對某些$\xi^{(k)} \in [x^{(k)} - e^{(k)}, x^{(k)}]$成立。
將上述展開除以$f''(x^{(k)})$得到

$$e^{(k)} - \frac{f'(x^{(k)})}{f''(x^{(k)})} = \frac{1}{2} (e^{(k)})^2 \frac{f'''(\xi^{(k)})}{f''(x^{(k)})}.$$

回想之前的方程$x^{(k+1)} = x^{(k)} - f'(x^{(k)}) / f''(x^{(k)})$。
代入這個更新方程，取兩邊的絕對值，我們得到

$$\left|e^{(k+1)}\right| = \frac{1}{2}(e^{(k)})^2 \frac{\left|f'''(\xi^{(k)})\right|}{f''(x^{(k)})}.$$

因此，每當我們處於有界區域$\left|f'''(\xi^{(k)})\right| / (2f''(x^{(k)})) \leq c$，
我們就有一個二次遞減誤差

$$\left|e^{(k+1)}\right| \leq c (e^{(k)})^2.$$

另一方面，最佳化研究人員稱之為『線性』收斂，而將$\left|e^{(k+1)}\right| \leq \alpha \left|e^{(k)}\right|$這樣的條件稱為『恆定』收斂速度。
請注意，我們無法估計整體收斂的速度，但是一旦我們接近極小值，收斂將變得非常快。
另外，這種分析要求$f$在高階導數上表現良好，即確保$f$在如何變化它的值方面沒有任何『超常』的屬性。

### 預處理

計算和儲存完整的Hessian非常昂貴，而改善這個問題的一種方法是『預處理』。
它回避了計算整個Hessian，而只計算『對角線』項，即如下的演算法更新：

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \mathrm{diag}(\mathbf{H})^{-1} \nabla f(\mathbf{x}).$$

雖然這不如完整的牛頓法精確，但它仍然比不使用要好得多。
為什麼預處理有效呢？
假設一個變數以毫米表示高度，另一個變數以公里表示高度的情況。
假設這兩種自然尺度都以米為單位，那麼我們的引數化就出現了嚴重的不對應。
幸運的是，使用預處理可以消除這種情況。
梯度下降的有效預處理相當於為每個變數選擇不同的學習率（向量$\mathbf{x}$的坐標）。
我們將在後面一節看到，預處理推動了隨機梯度下降最佳化演算法的一些創新。

### 梯度下降和線搜尋

梯度下降的一個關鍵問題是我們可能會超過目標或進展不足，
解決這一問題的簡單方法是結合使用線搜尋和梯度下降。
也就是說，我們使用$\nabla f(\mathbf{x})$給出的方向，
然後進行二分搜尋，以確定哪個學習率$\eta$使$f(\mathbf{x} - \eta \nabla f(\mathbf{x}))$取最小值。

有關分析和證明，此演算法收斂迅速（請參見 :cite:`Boyd.Vandenberghe.2004`）。
然而，對深度學習而言，這不太可行。
因為線搜尋的每一步都需要評估整個資料集上的目標函式，實現它的方式太昂貴了。

## 小結

* 學習率的大小很重要：學習率太大會使模型發散，學習率太小會沒有進展。
* 梯度下降會可能陷入區域性極小值，而得不到全域最小值。
* 在高維模型中，調整學習率是很複雜的。
* 預處理有助於調節比例。
* 牛頓法在凸問題中一旦開始正常工作，速度就會快得多。
* 對於非凸問題，不要不作任何調整就使用牛頓法。

## 練習

1. 用不同的學習率和目標函式進行梯度下降實驗。
1. 在區間$[a, b]$中實現線搜尋以最小化凸函式。
    1. 是否需要導數來進行二分搜尋，即決定選擇$[a, (a+b)/2]$還是$[(a+b)/2, b]$。
    1. 演算法的收斂速度有多快？
    1. 實現該演算法，並將其應用於求$\log (\exp(x) + \exp(-2x -3))$的最小值。
1. 設計一個定義在$\mathbb{R}^2$上的目標函式，它的梯度下降非常緩慢。提示：不同坐標的縮放方式不同。
1. 使用預處理實現牛頓方法的輕量版本。
    1. 使用對角Hessian作為預條件子。
    1. 使用它的絕對值，而不是實際值（可能有符號）。
    1. 將此應用於上述問題。
1. 將上述演算法應用於多個目標函式（凸或非凸）。如果把坐標旋轉$45$度會怎麼樣？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/3834)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/3836)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3835)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11848)
:end_tab: