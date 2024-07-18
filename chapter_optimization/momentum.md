# 動量法
:label:`sec_momentum`

在 :numref:`sec_sgd`一節中，我們詳述瞭如何執行隨機梯度下降，即在只有嘈雜的梯度可用的情況下執行最佳化時會發生什麼。
對於嘈雜的梯度，我們在選擇學習率需要格外謹慎。
如果衰減速度太快，收斂就會停滯。
相反，如果太寬鬆，我們可能無法收斂到最優解。

## 基礎

本節將探討更有效的最佳化演算法，尤其是針對實驗中常見的某些型別的最佳化問題。

### 洩漏平均值

上一節中我們討論了小批次隨機梯度下降作為加速計算的手段。
它也有很好的副作用，即平均梯度減小了方差。
小批次隨機梯度下降可以透過以下方式計算：

$$\mathbf{g}_{t, t-1} = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}_{t-1}) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \mathbf{h}_{i, t-1}.
$$

為了保持記法簡單，在這裡我們使用$\mathbf{h}_{i, t-1} = \partial_{\mathbf{w}} f(\mathbf{x}_i, \mathbf{w}_{t-1})$作為樣本$i$的隨機梯度下降，使用時間$t-1$時更新的權重$t-1$。
如果我們能夠從方差減少的影響中受益，甚至超過小批次上的梯度平均值，那很不錯。
完成這項任務的一種選擇是用*洩漏平均值*（leaky average）取代梯度計算：

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}$$

其中$\beta \in (0, 1)$。
這有效地將瞬時梯度替換為多個『過去』梯度的平均值。
$\mathbf{v}$被稱為*動量*（momentum），
它累加了過去的梯度。
為了更詳細地解釋，讓我們遞迴地將$\mathbf{v}_t$擴充套件到

$$\begin{aligned}
\mathbf{v}_t = \beta^2 \mathbf{v}_{t-2} + \beta \mathbf{g}_{t-1, t-2} + \mathbf{g}_{t, t-1}
= \ldots, = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}.
\end{aligned}$$

其中，較大的$\beta$相當於長期平均值，而較小的$\beta$相對於梯度法只是略有修正。
新的梯度替換不再指向特定實例下降最陡的方向，而是指向過去梯度的加權平均值的方向。
這使我們能夠實現對單批次計算平均值的大部分好處，而不產生實際計算其梯度的代價。

上述推理構成了"加速"梯度方法的基礎，例如具有動量的梯度。
在最佳化問題條件不佳的情況下（例如，有些方向的進展比其他方向慢得多，類似狹窄的峽谷），"加速"梯度還額外享受更有效的好處。
此外，它們允許我們對隨後的梯度計算平均值，以獲得更穩定的下降方向。
誠然，即使是對於無噪聲凸問題，加速度這方面也是動量如此起效的關鍵原因之一。

正如人們所期望的，由於其功效，動量是深度學習及其後最佳化中一個深入研究的主題。
例如，請參閱[文章](https://distill.pub/2017/momentum/）（作者是 :cite:`Goh.2017`)，觀看深入分析和互動動畫。
動量是由 :cite:`Polyak.1964`提出的。
 :cite:`Nesterov.2018`在凸最佳化的背景下進行了詳細的理論討論。
長期以來，深度學習的動量一直被認為是有益的。
有關實例的詳細訊息，請參閱 :cite:`Sutskever.Martens.Dahl.ea.2013`的討論。

### 條件不佳的問題

為了更好地瞭解動量法的幾何屬性，我們複習一下梯度下降，儘管它的目標函式明顯不那麼令人愉快。
回想我們在 :numref:`sec_gd`中使用了$f(\mathbf{x}) = x_1^2 + 2 x_2^2$，即中度扭曲的橢球目標。
我們透過向$x_1$方向伸展它來進一步扭曲這個函式

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

與之前一樣，$f$在$(0, 0)$有最小值，
該函式在$x_1$的方向上*非常*平坦。
讓我們看看在這個新函式上執行梯度下降時會發生什麼。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

從構造來看，$x_2$方向的梯度比水平$x_1$方向的梯度大得多，變化也快得多。
因此，我們陷入兩難：如果選擇較小的學習率，我們會確保解不會在$x_2$方向發散，但要承受在$x_1$方向的緩慢收斂。相反，如果學習率較高，我們在$x_1$方向上進展很快，但在$x_2$方向將會發散。
下面的例子說明了即使學習率從$0.4$略微提高到$0.6$，也會發生變化。
$x_1$方向上的收斂有所改善，但整體來看解的質量更差了。

```{.python .input}
#@tab all
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

### 動量法

*動量法*（momentum）使我們能夠解決上面描述的梯度下降問題。
觀察上面的最佳化軌跡，我們可能會直覺到計算過去的平均梯度效果會很好。
畢竟，在$x_1$方向上，這將聚合非常對齊的梯度，從而增加我們在每一步中覆蓋的距離。
相反，在梯度振盪的$x_2$方向，由於相互抵消了對方的振盪，聚合梯度將減小步長大小。
使用$\mathbf{v}_t$而不是梯度$\mathbf{g}_t$可以生成以下更新等式：

$$
\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}, \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta_t \mathbf{v}_t.
\end{aligned}
$$

請注意，對於$\beta = 0$，我們恢復常規的梯度下降。
在深入研究它的數學屬性之前，讓我們快速看一下演算法在實驗中的表現如何。

```{.python .input}
#@tab all
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

正如所見，儘管學習率與我們以前使用的相同，動量法仍然很好地收斂了。
讓我們看看當降低動量引數時會發生什麼。
將其減半至$\beta = 0.25$會導致一條幾乎沒有收斂的軌跡。
儘管如此，它比沒有動量時解將會發散要好得多。

```{.python .input}
#@tab all
eta, beta = 0.6, 0.25
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

請注意，我們可以將動量法與隨機梯度下降，特別是小批次隨機梯度下降結合起來。
唯一的變化是，在這種情況下，我們將梯度$\mathbf{g}_{t, t-1}$替換為$\mathbf{g}_t$。
為了方便起見，我們在時間$t=0$初始化$\mathbf{v}_0 = 0$。

### 有效樣本權重

回想一下$\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}$。
極限條件下，$\sum_{\tau=0}^\infty \beta^\tau = \frac{1}{1-\beta}$。
換句話說，不同於在梯度下降或者隨機梯度下降中取步長$\eta$，我們選取步長$\frac{\eta}{1-\beta}$，同時處理潛在表現可能會更好的下降方向。
這是集兩種好處於一身的做法。
為了說明$\beta$的不同選擇的權重效果如何，請參考下面的圖表。

```{.python .input}
#@tab all
d2l.set_figsize()
betas = [0.95, 0.9, 0.6, 0]
for beta in betas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

## 實際實驗

讓我們來看看動量法在實驗中是如何運作的。
為此，我們需要一個更加可擴充套件的實現。

### 從零開始實現

相比於小批次隨機梯度下降，動量方法需要維護一組輔助變數，即速度。
它與梯度以及最佳化問題的變數具有相同的形狀。
在下面的實現中，我們稱這些變數為`states`。

```{.python .input}
#@tab mxnet, pytorch
def init_momentum_states(feature_dim):
    v_w = d2l.zeros((feature_dim, 1))
    v_b = d2l.zeros(1)
    return (v_w, v_b)
```

```{.python .input}
#@tab tensorflow
def init_momentum_states(features_dim):
    v_w = tf.Variable(d2l.zeros((features_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    return (v_w, v_b)
```

```{.python .input}
#@tab paddle
def init_momentum_states(feature_dim):
    v_w = d2l.zeros((feature_dim, 1))
    v_b = d2l.zeros([1])
    return (v_w, v_b)
```

```{.python .input}
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams['momentum'] * v + p.grad
        p[:] -= hyperparams['lr'] * v
```

```{.python .input}
#@tab pytorch
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd_momentum(params, grads, states, hyperparams):
    for p, v, g in zip(params, states, grads):
            v[:].assign(hyperparams['momentum'] * v + g)
            p[:].assign(p - hyperparams['lr'] * v)
```

```{.python .input}
#@tab paddle
def sgd_momentum(params, states, hyperparams):
    a = []
    for p, v in zip(params, states):
        with paddle.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.zero_()
        a.append(p)
    return a
```

讓我們看看它在實驗中是如何運作的。

```{.python .input}
#@tab all
def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
train_momentum(0.02, 0.5)
```

當我們將動量超引數`momentum`增加到0.9時，它相當於有效樣本數量增加到$\frac{1}{1 - 0.9} = 10$。
我們將學習率略微降至$0.01$，以確保可控。

```{.python .input}
#@tab all
train_momentum(0.01, 0.9)
```

降低學習率進一步解決了任何非平滑最佳化問題的困難，將其設定為$0.005$會產生良好的收斂性能。

```{.python .input}
#@tab all
train_momentum(0.005, 0.9)
```

### 簡潔實現

由於深度學習框架中的最佳化求解器早已建立了動量法，設定對應引數會產生非常類似的軌跡。

```{.python .input}
d2l.train_concise_ch11('sgd', {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD
d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD
d2l.train_concise_ch11(trainer, {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

```{.python .input}
#@tab paddle
trainer = paddle.optimizer.Momentum
d2l.train_concise_ch11(trainer, {'learning_rate': 0.005, 'momentum': 0.9}, data_iter)
```

## 理論分析

$f(x) = 0.1 x_1^2 + 2 x_2^2$的2D範例似乎相當牽強。
下面我們將看到，它在實際生活中非常具有代表性，至少最小化凸二次目標函式的情況下是如此。

### 二次凸函式

考慮這個函式

$$h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b.$$

這是一個普通的二次函式。
對於正定矩陣$\mathbf{Q} \succ 0$，即對於具有正特徵值的矩陣，有最小化器為$\mathbf{x}^* = -\mathbf{Q}^{-1} \mathbf{c}$，最小值為$b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$。
因此我們可以將$h$重寫為

$$h(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})^\top \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c}) + b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}.$$

梯度由$\partial_{\mathbf{x}} f(\mathbf{x}) = \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$給出。
也就是說，它是由$\mathbf{x}$和最小化器之間的距離乘以$\mathbf{Q}$所得出的。
因此，動量法還是$\mathbf{Q} (\mathbf{x}_t - \mathbf{Q}^{-1} \mathbf{c})$的線性組合。

由於$\mathbf{Q}$是正定的，因此可以透過$\mathbf{Q} = \mathbf{O}^\top \boldsymbol{\Lambda} \mathbf{O}$分解為正交（旋轉）矩陣$\mathbf{O}$和正特徵值的對角矩陣$\boldsymbol{\Lambda}$。
這使我們能夠將變數從$\mathbf{x}$更改為$\mathbf{z} := \mathbf{O} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$，以獲得一個非常簡化的表示式：

$$h(\mathbf{z}) = \frac{1}{2} \mathbf{z}^\top \boldsymbol{\Lambda} \mathbf{z} + b'.$$

這裡$b' = b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$。
由於$\mathbf{O}$只是一個正交矩陣，因此不會真正意義上擾動梯度。
以$\mathbf{z}$表示的梯度下降變成

$$\mathbf{z}_t = \mathbf{z}_{t-1} - \boldsymbol{\Lambda} \mathbf{z}_{t-1} = (\mathbf{I} - \boldsymbol{\Lambda}) \mathbf{z}_{t-1}.$$

這個表示式中的重要事實是梯度下降在不同的特徵空間之間不會混合。
也就是說，如果用$\mathbf{Q}$的特徵系統來表示，最佳化問題是以逐坐標順序的方式進行的。
這在動量法中也適用。

$$\begin{aligned}
\mathbf{v}_t & = \beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1} \\
\mathbf{z}_t & = \mathbf{z}_{t-1} - \eta \left(\beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1}\right) \\
    & = (\mathbf{I} - \eta \boldsymbol{\Lambda}) \mathbf{z}_{t-1} - \eta \beta \mathbf{v}_{t-1}.
\end{aligned}$$

在這樣做的過程中，我們只是證明了以下定理：帶有和帶有不凸二次函式動量的梯度下降，可以分解為朝二次矩陣特徵向量方向坐標順序的最佳化。

### 標量函式

鑑於上述結果，讓我們看看當我們最小化函式$f(x) = \frac{\lambda}{2} x^2$時會發生什麼。
對於梯度下降我們有

$$x_{t+1} = x_t - \eta \lambda x_t = (1 - \eta \lambda) x_t.$$

每$|1 - \eta \lambda| < 1$時，這種最佳化以指數速度收斂，因為在$t$步之後我們可以得到$x_t = (1 - \eta \lambda)^t x_0$。
這顯示了在我們將學習率$\eta$提高到$\eta \lambda = 1$之前，收斂率最初是如何提高的。
超過該數值之後，梯度開始發散，對於$\eta \lambda > 2$而言，最佳化問題將會發散。

```{.python .input}
#@tab all
lambdas = [0.1, 1, 10, 19]
eta = 0.1
d2l.set_figsize((6, 4))
for lam in lambdas:
    t = d2l.numpy(d2l.arange(20))
    d2l.plt.plot(t, (1 - eta * lam) ** t, label=f'lambda = {lam:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

為了分析動量的收斂情況，我們首先用兩個標量重寫更新方程：一個用於$x$，另一個用於動量$v$。這產生了：

$$
\begin{bmatrix} v_{t+1} \\ x_{t+1} \end{bmatrix} =
\begin{bmatrix} \beta & \lambda \\ -\eta \beta & (1 - \eta \lambda) \end{bmatrix}
\begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix} = \mathbf{R}(\beta, \eta, \lambda) \begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix}.
$$

我們用$\mathbf{R}$來表示$2 \times 2$管理的收斂表現。
在$t$步之後，最初的值$[v_0, x_0]$變為$\mathbf{R}(\beta, \eta, \lambda)^t [v_0, x_0]$。
因此，收斂速度是由$\mathbf{R}$的特徵值決定的。
請參閱[文章](https://distill.pub/2017/momentum/) :cite:`Goh.2017`瞭解精彩動畫。
請參閱 :cite:`Flammarion.Bach.2015`瞭解詳細分析。
簡而言之，當$0 < \eta \lambda < 2 + 2 \beta$時動量收斂。
與梯度下降的$0 < \eta \lambda < 2$相比，這是更大範圍的可行引數。
另外，一般而言較大值的$\beta$是可取的。

## 小結

* 動量法用過去梯度的平均值來替換梯度，這大大加快了收斂速度。
* 對於無噪聲梯度下降和嘈雜隨機梯度下降，動量法都是可取的。
* 動量法可以防止在隨機梯度下降的最佳化過程停滯的問題。
* 由於對過去的資料進行了指數降權，有效梯度數為$\frac{1}{1-\beta}$。
* 在凸二次問題中，可以對動量法進行明確而詳細的分析。
* 動量法的實現非常簡單，但它需要我們儲存額外的狀態向量（動量$\mathbf{v}$）。

## 練習

1. 使用動量超引數和學習率的其他組合，觀察和分析不同的實驗結果。
1. 試試梯度下降和動量法來解決一個二次問題，其中有多個特徵值，即$f(x) = \frac{1}{2} \sum_i \lambda_i x_i^2$，例如$\lambda_i = 2^{-i}$。繪製出$x$的值在初始化$x_i = 1$時如何下降。
1. 推導$h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b$的最小值和最小化器。
1. 當我們執行帶動量法的隨機梯度下降時會有什麼變化？當我們使用帶動量法的小批次隨機梯度下降時會發生什麼？試驗引數如何？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/4327)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/4328)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/4329)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11851)
:end_tab: