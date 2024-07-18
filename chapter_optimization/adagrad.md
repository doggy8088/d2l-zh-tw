# AdaGrad演算法
:label:`sec_adagrad`

我們從有關特徵學習中並不常見的問題入手。

## 稀疏特徵和學習率

假設我們正在訓練一個語言模型。
為了獲得良好的準確性，我們大多希望在訓練的過程中降低學習率，速度通常為$\mathcal{O}(t^{-\frac{1}{2}})$或更低。
現在討論關於稀疏特徵（即只在偶爾出現的特徵）的模型訓練，這對自然語言來說很常見。
例如，我們看到『預先條件』這個詞比『學習』這個詞的可能性要小得多。
但是，它在計算廣告學和個性化協同過濾等其他領域也很常見。

只有在這些不常見的特徵出現時，與其相關的引數才會得到有意義的更新。
鑑於學習率下降，我們可能最終會面臨這樣的情況：常見特徵的引數相當迅速地收斂到最佳值，而對於不常見的特徵，我們仍缺乏足夠的觀測以確定其最佳值。
換句話說，學習率要麼對於常見特徵而言降低太慢，要麼對於不常見特徵而言降低太快。

解決此問題的一個方法是記錄我們看到特定特徵的次數，然後將其用作調整學習率。
即我們可以使用大小為$\eta_i = \frac{\eta_0}{\sqrt{s(i, t) + c}}$的學習率，而不是$\eta = \frac{\eta_0}{\sqrt{t + c}}$。
在這裡$s(i, t)$計下了我們截至$t$時觀察到功能$i$的次數。
這其實很容易實施且不產生額外損耗。

AdaGrad演算法 :cite:`Duchi.Hazan.Singer.2011`透過將粗略的計數器$s(i, t)$替換為先前觀察所得梯度的平方之和來解決這個問題。
它使用$s(i, t+1) = s(i, t) + \left(\partial_i f(\mathbf{x})\right)^2$來調整學習率。
這有兩個好處：首先，我們不再需要決定梯度何時算足夠大。
其次，它會隨梯度的大小自動變化。通常對應於較大梯度的坐標會顯著縮小，而其他梯度較小的坐標則會得到更平滑的處理。
在實際應用中，它促成了計算廣告學及其相關問題中非常有效的最佳化程式。
但是，它遮蓋了AdaGrad固有的一些額外優勢，這些優勢在預處理環境中很容易被理解。

## 預處理

凸最佳化問題有助於分析演算法的特點。
畢竟對大多數非凸問題來說，獲得有意義的理論保證很難，但是直覺和洞察往往會延續。
讓我們來看看最小化$f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x} + b$這一問題。

正如在 :numref:`sec_momentum`中那樣，我們可以根據其特徵分解$\mathbf{Q} = \mathbf{U}^\top \boldsymbol{\Lambda} \mathbf{U}$重寫這個問題，來得到一個簡化得多的問題，使每個坐標都可以單獨解出：

$$f(\mathbf{x}) = \bar{f}(\bar{\mathbf{x}}) = \frac{1}{2} \bar{\mathbf{x}}^\top \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}}^\top \bar{\mathbf{x}} + b.$$

在這裡我們使用了$\mathbf{x} = \mathbf{U} \mathbf{x}$，且因此$\mathbf{c} = \mathbf{U} \mathbf{c}$。
修改後最佳化器為$\bar{\mathbf{x}} = -\boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}}$且最小值為$-\frac{1}{2} \bar{\mathbf{c}}^\top \boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}} + b$。
這樣更容易計算，因為$\boldsymbol{\Lambda}$是一個包含$\mathbf{Q}$特徵值的對角矩陣。

如果稍微擾動$\mathbf{c}$，我們會期望在$f$的最小化器中只產生微小的變化。
遺憾的是，情況並非如此。
雖然$\mathbf{c}$的微小變化導致了$\bar{\mathbf{c}}$同樣的微小變化，但$f$的（以及$\bar{f}$的）最小化器並非如此。
每當特徵值$\boldsymbol{\Lambda}_i$很大時，我們只會看到$\bar{x}_i$和$\bar{f}$的最小值發聲微小變化。
相反，對小的$\boldsymbol{\Lambda}_i$來說，$\bar{x}_i$的變化可能是劇烈的。
最大和最小的特徵值之比稱為最佳化問題的*條件數*（condition number）。

$$\kappa = \frac{\boldsymbol{\Lambda}_1}{\boldsymbol{\Lambda}_d}.$$

如果條件編號$\kappa$很大，準確解決最佳化問題就會很難。
我們需要確保在獲取大量動態的特徵值範圍時足夠謹慎：難道我們不能簡單地透過扭曲空間來『修復』這個問題，從而使所有特徵值都是$1$？
理論上這很容易：我們只需要$\mathbf{Q}$的特徵值和特徵向量即可將問題從$\mathbf{x}$整理到$\mathbf{z} := \boldsymbol{\Lambda}^{\frac{1}{2}} \mathbf{U} \mathbf{x}$中的一個。
在新的座標系中，$\mathbf{x}^\top \mathbf{Q} \mathbf{x}$可以被簡化為$\|\mathbf{z}\|^2$。
可惜，這是一個相當不切實際的想法。
一般而言，計算特徵值和特徵向量要比解決實際問題『貴』得多。

雖然準確計算特徵值可能會很昂貴，但即便只是大致猜測並計算它們，也可能已經比不做任何事情好得多。
特別是，我們可以使用$\mathbf{Q}$的對角線條目並相應地重新縮放它。
這比計算特徵值開銷小的多。

$$\tilde{\mathbf{Q}} = \mathrm{diag}^{-\frac{1}{2}}(\mathbf{Q}) \mathbf{Q} \mathrm{diag}^{-\frac{1}{2}}(\mathbf{Q}).$$

在這種情況下，我們得到了$\tilde{\mathbf{Q}}_{ij} = \mathbf{Q}_{ij} / \sqrt{\mathbf{Q}_{ii} \mathbf{Q}_{jj}}$，特別注意對於所有$i$，$\tilde{\mathbf{Q}}_{ii} = 1$。
在大多數情況下，這大大簡化了條件數。
例如我們之前討論的案例，它將完全消除眼下的問題，因為問題是軸對齊的。

遺憾的是，我們還面臨另一個問題：在深度學習中，我們通常情況甚至無法計算目標函式的二階導數：對於$\mathbf{x} \in \mathbb{R}^d$，即使只在小批次上，二階導數可能也需要$\mathcal{O}(d^2)$空間來計算，導致幾乎不可行。
AdaGrad演算法巧妙的思路是，使用一個代理來表示黑塞矩陣（Hessian）的對角線，既相對易於計算又高效。

為了瞭解它是如何生效的，讓我們來看看$\bar{f}(\bar{\mathbf{x}})$。
我們有

$$\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}}) = \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}} = \boldsymbol{\Lambda} \left(\bar{\mathbf{x}} - \bar{\mathbf{x}}_0\right),$$

其中$\bar{\mathbf{x}}_0$是$\bar{f}$的最佳化器。
因此，梯度的大小取決於$\boldsymbol{\Lambda}$和與最佳值的差值。
如果$\bar{\mathbf{x}} - \bar{\mathbf{x}}_0$沒有改變，那這就是我們所求的。
畢竟在這種情況下，梯度$\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}})$的大小就足夠了。
由於AdaGrad演算法是一種隨機梯度下降演算法，所以即使是在最佳值中，我們也會看到具有非零方差的梯度。
因此，我們可以放心地使用梯度的方差作為黑塞矩陣比例的廉價替代。
詳盡的分析（要花幾頁解釋）超出了本節的範圍，請讀者參考 :cite:`Duchi.Hazan.Singer.2011`。

## 演算法

讓我們接著上面正式開始討論。
我們使用變數$\mathbf{s}_t$來累加過去的梯度方差，如下所示：

$$\begin{aligned}
    \mathbf{g}_t & = \partial_{\mathbf{w}} l(y_t, f(\mathbf{x}_t, \mathbf{w})), \\
    \mathbf{s}_t & = \mathbf{s}_{t-1} + \mathbf{g}_t^2, \\
    \mathbf{w}_t & = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \cdot \mathbf{g}_t.
\end{aligned}$$

在這裡，操作是按照坐標順序應用。
也就是說，$\mathbf{v}^2$有條目$v_i^2$。
同樣，$\frac{1}{\sqrt{v}}$有條目$\frac{1}{\sqrt{v_i}}$，
並且$\mathbf{u} \cdot \mathbf{v}$有條目$u_i v_i$。
與之前一樣，$\eta$是學習率，$\epsilon$是一個為維持數值穩定性而新增的常數，用來確保我們不會除以$0$。
最後，我們初始化$\mathbf{s}_0 = \mathbf{0}$。

就像在動量法中我們需要跟蹤一個輔助變數一樣，在AdaGrad演算法中，我們允許每個坐標有單獨的學習率。
與SGD演算法相比，這並沒有明顯增加AdaGrad的計算代價，因為主要計算用在$l(y_t, f(\mathbf{x}_t, \mathbf{w}))$及其導數。

請注意，在$\mathbf{s}_t$中累加平方梯度意味著$\mathbf{s}_t$基本上以線性速率增長（由於梯度從最初開始衰減，實際上比線性慢一些）。
這產生了一個學習率$\mathcal{O}(t^{-\frac{1}{2}})$，但是在單個坐標的層面上進行了調整。
對於凸問題，這完全足夠了。
然而，在深度學習中，我們可能希望更慢地降低學習率。
這引出了許多AdaGrad演算法的變體，我們將在後續章節中討論它們。
眼下讓我們先看看它在二次凸問題中的表現如何。
我們仍然以同一函式為例：

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

我們將使用與之前相同的學習率來實現AdaGrad演算法，即$\eta = 0.4$。
可以看到，自變量的迭代軌跡較平滑。
但由於$\boldsymbol{s}_t$的累加效果使學習率不斷衰減，自變量在迭代後期的移動幅度較小。

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

```{.python .input}
#@tab all
def adagrad_2d(x1, x2, s1, s2):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

我們將學習率提高到$2$，可以看到更好的表現。
這已經表明，即使在無噪聲的情況下，學習率的降低可能相當劇烈，我們需要確保引數能夠適當地收斂。

```{.python .input}
#@tab all
eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

## 從零開始實現

同動量法一樣，AdaGrad演算法需要對每個自變量維護同它一樣形狀的狀態變數。

```{.python .input}
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] += np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def init_adagrad_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)

def adagrad(params, grads, states, hyperparams):
    eps = 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(s + tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

```{.python .input}
#@tab paddle
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(shape=(1, ))
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    a = []
    eps = 1e-6
    for p, s in zip(params, states):
        with paddle.no_grad():
            s[:] += paddle.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / paddle.sqrt(s + eps)
        p.grad.zero_()
        a.append(p)
    return a
```

與 :numref:`sec_minibatch_sgd`一節中的實驗相比，這裡使用更大的學習率來訓練模型。

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adagrad, init_adagrad_states(feature_dim),
               {'lr': 0.1}, data_iter, feature_dim);
```

## 簡潔實現

我們可直接使用深度學習框架中提供的AdaGrad演算法來訓練模型。

```{.python .input}
d2l.train_concise_ch11('adagrad', {'learning_rate': 0.1}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adagrad
d2l.train_concise_ch11(trainer, {'lr': 0.1}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adagrad
d2l.train_concise_ch11(trainer, {'learning_rate' : 0.1}, data_iter)
```

```{.python .input}
#@tab paddle
trainer = paddle.optimizer.Adagrad
d2l.train_concise_ch11(trainer, {'learning_rate': 0.1}, data_iter)
```

## 小結

* AdaGrad演算法會在單個坐標層面動態降低學習率。
* AdaGrad演算法利用梯度的大小作為調整進度速率的手段：用較小的學習率來補償帶有較大梯度的坐標。
* 在深度學習問題中，由於記憶體和計算限制，計算準確的二階導數通常是不可行的。梯度可以作為一個有效的代理。
* 如果最佳化問題的結構相當不均勻，AdaGrad演算法可以幫助緩解扭曲。
* AdaGrad演算法對於稀疏特徵特別有效，在此情況下由於不常出現的問題，學習率需要更慢地降低。
* 在深度學習問題上，AdaGrad演算法有時在降低學習率方面可能過於劇烈。我們將在 :numref:`sec_adam`一節討論緩解這種情況的策略。

## 練習

1. 證明對於正交矩陣$\mathbf{U}$和向量$\mathbf{c}$，以下等式成立：$\|\mathbf{c} - \mathbf{\delta}\|_2 = \|\mathbf{U} \mathbf{c} - \mathbf{U} \mathbf{\delta}\|_2$。為什麼這意味著在變數的正交變化之後，擾動的程度不會改變？
1. 嘗試對函式$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$、以及它旋轉45度後的函式即$f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$使用AdaGrad演算法。它的表現會不同嗎？
1. 證明[格什戈林圓盤定理](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem)，其中提到，矩陣$\mathbf{M}$的特徵值$\lambda_i$在至少一個$j$的選項中滿足$|\lambda_i - \mathbf{M}_{jj}| \leq \sum_{k \neq j} |\mathbf{M}_{jk}|$的要求。
1. 關於對角線預處理矩陣$\mathrm{diag}^{-\frac{1}{2}}(\mathbf{M}) \mathbf{M} \mathrm{diag}^{-\frac{1}{2}}(\mathbf{M})$的特徵值，格什戈林的定理告訴了我們什麼？
1. 嘗試對適當的深度網路使用AdaGrad演算法，例如，:numref:`sec_lenet`中應用於Fashion-MNIST的深度網路。
1. 要如何修改AdaGrad演算法，才能使其在學習率方面的衰減不那麼激進？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/4318)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/4319)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/4320)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11852)
:end_tab: