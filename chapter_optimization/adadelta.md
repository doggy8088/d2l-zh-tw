# Adadelta
:label:`sec_adadelta`


Adadelta是AdaGrad的另一種變體（ :numref:`sec_adagrad`），
主要區別在於前者減少了學習率適應坐標的數量。
此外，廣義上Adadelta被稱為沒有學習率，因為它使用變化量本身作為未來變化的校準。
Adadelta演算法是在 :cite:`Zeiler.2012`中提出的。

## Adadelta演算法

簡而言之，Adadelta使用兩個狀態變數，$\mathbf{s}_t$用於儲存梯度二階導數的洩露平均值，$\Delta\mathbf{x}_t$用於儲存模型本身中引數變化二階導數的洩露平均值。請注意，為了與其他出版物和實現的相容性，我們使用作者的原始符號和命名（沒有其它真正理由讓大家使用不同的希臘變數來表示在動量法、AdaGrad、RMSProp和Adadelta中用於相同用途的引數）。

以下是Adadelta的技術細節。鑑於引數du jour是$\rho$，我們獲得了與 :numref:`sec_rmsprop`類似的以下洩漏更新：

$$\begin{aligned}
    \mathbf{s}_t & = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2.
\end{aligned}$$

與 :numref:`sec_rmsprop`的區別在於，我們使用重新縮放的梯度$\mathbf{g}_t'$執行更新，即

$$\begin{aligned}
    \mathbf{x}_t  & = \mathbf{x}_{t-1} - \mathbf{g}_t'. \\
\end{aligned}$$

那麼，調整後的梯度$\mathbf{g}_t'$是什麼？我們可以按如下方式計算它：

$$\begin{aligned}
    \mathbf{g}_t' & = \frac{\sqrt{\Delta\mathbf{x}_{t-1} + \epsilon}}{\sqrt{{\mathbf{s}_t + \epsilon}}} \odot \mathbf{g}_t, \\
\end{aligned}$$

其中$\Delta \mathbf{x}_{t-1}$是重新縮放梯度的平方$\mathbf{g}_t'$的洩漏平均值。我們將$\Delta \mathbf{x}_{0}$初始化為$0$，然後在每個步驟中使用$\mathbf{g}_t'$更新它，即

$$\begin{aligned}
    \Delta \mathbf{x}_t & = \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) {\mathbf{g}_t'}^2,
\end{aligned}$$

和$\epsilon$（例如$10^{-5}$這樣的小值）是為了保持數字穩定性而加入的。

## 程式碼實現

Adadelta需要為每個變數維護兩個狀態變數，即$\mathbf{s}_t$和$\Delta\mathbf{x}_t$。這將產生以下實現。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        # In-placeupdatesvia[:]
        s[:] = rho * s + (1 - rho) * np.square(p.grad)
        g = (np.sqrt(delta + eps) / np.sqrt(s + eps)) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # In-placeupdatesvia[:]
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adadelta_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    delta_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    delta_b = tf.Variable(d2l.zeros(1))
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, grads, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta), grad in zip(params, states, grads):
        s[:].assign(rho * s + (1 - rho) * tf.math.square(grad))
        g = (tf.math.sqrt(delta + eps) / tf.math.sqrt(s + eps)) * grad
        p[:].assign(p - g)
        delta[:].assign(rho * delta + (1 - rho) * g * g)
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros(shape=(feature_dim, 1)), d2l.zeros(shape=(1, )) 
    delta_w, delta_b = d2l.zeros(shape=(feature_dim, 1)), d2l.zeros(shape=(1, )) 
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    a = []
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with paddle.no_grad():
            # In-placeupdatesvia[:]
            s[:] = rho * s + (1 - rho) * paddle.square(p.grad)
            g = (paddle.sqrt(delta + eps) / paddle.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.zero_()
        a.append(p)
    return a
```

對於每次引數更新，選擇$\rho = 0.9$相當於10個半衰期。由此我們得到：

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);
```

為了簡潔實現，我們只需使用高階API中的Adadelta演算法。

```{.python .input}
d2l.train_concise_ch11('adadelta', {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
# adadeltaisnotconvergingatdefaultlearningrate
# butit'sconvergingatlr=5.0
trainer = tf.keras.optimizers.Adadelta
d2l.train_concise_ch11(trainer, {'learning_rate':5.0, 'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab paddle
trainer = paddle.optimizer.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
```

## 小結

* Adadelta沒有學習率引數。相反，它使用引數本身的變化率來調整學習率。
* Adadelta需要兩個狀態變數來儲存梯度的二階導數和引數的變化。
* Adadelta使用洩漏的平均值來保持對適當統計資料的執行估計。

## 練習

1. 調整$\rho$的值，會發生什麼？
1. 展示如何在不使用$\mathbf{g}_t'$的情況下實現演算法。為什麼這是個好主意？
1. Adadelta真的是學習率為0嗎？能找到Adadelta無法解決的最佳化問題嗎？
1. 將Adadelta的收斂行為與AdaGrad和RMSProp進行比較。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/5771)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/5772)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/5773)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11854)
:end_tab:
