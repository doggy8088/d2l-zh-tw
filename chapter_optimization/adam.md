# Adam演算法
:label:`sec_adam`

本章我們已經學習了許多有效最佳化的技術。
在本節討論之前，我們先詳細回顧一下這些技術：

* 在 :numref:`sec_sgd`中，我們學習了：隨機梯度下降在解決最佳化問題時比梯度下降更有效。
* 在 :numref:`sec_minibatch_sgd`中，我們學習了：在一個小批次中使用更大的觀測值集，可以透過向量化提供額外效率。這是高效的多機、多GPU和整體並行處理的關鍵。
* 在 :numref:`sec_momentum`中我們新增了一種機制，用於彙總過去梯度的歷史以加速收斂。
* 在 :numref:`sec_adagrad`中，我們透過對每個坐標縮放來實現高效計算的預處理器。
* 在 :numref:`sec_rmsprop`中，我們透過學習率的調整來分離每個坐標的縮放。

Adam演算法 :cite:`Kingma.Ba.2014`將所有這些技術彙總到一個高效的學習演算法中。
不出預料，作為深度學習中使用的更強大和有效的最佳化演算法之一，它非常受歡迎。
但是它並非沒有問題，尤其是 :cite:`Reddi.Kale.Kumar.2019`表明，有時Adam演算法可能由於方差控制不良而發散。
在完善工作中， :cite:`Zaheer.Reddi.Sachan.ea.2018`給Adam演算法提供了一個稱為Yogi的熱補丁來解決這些問題。
下面我們瞭解一下Adam演算法。

## 演算法

Adam演算法的關鍵組成部分之一是：它使用指數加權移動平均值來估算梯度的動量和二次矩，即它使用狀態變數

$$\begin{aligned}
    \mathbf{v}_t & \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \\
    \mathbf{s}_t & \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2.
\end{aligned}$$

這裡$\beta_1$和$\beta_2$是非負加權引數。
常將它們設定為$\beta_1 = 0.9$和$\beta_2 = 0.999$。
也就是說，方差估計的移動遠遠慢於動量估計的移動。
注意，如果我們初始化$\mathbf{v}_0 = \mathbf{s}_0 = 0$，就會獲得一個相當大的初始偏差。
我們可以透過使用$\sum_{i=0}^t \beta^i = \frac{1 - \beta^t}{1 - \beta}$來解決這個問題。
相應地，標準化狀態變數由下式獲得

$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t} \text{ and } \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}.$$

有了正確的估計，我們現在可以寫出更新方程。
首先，我們以非常類似於RMSProp演算法的方式重新縮放梯度以獲得

$$\mathbf{g}_t' = \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}.$$

與RMSProp不同，我們的更新使用動量$\hat{\mathbf{v}}_t$而不是梯度本身。
此外，由於使用$\frac{1}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$而不是$\frac{1}{\sqrt{\hat{\mathbf{s}}_t + \epsilon}}$進行縮放，兩者會略有差異。
前者在實踐中效果略好一些，因此與RMSProp演算法有所區分。
通常，我們選擇$\epsilon = 10^{-6}$，這是為了在數值穩定性和逼真度之間取得良好的平衡。

最後，我們簡單更新：

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \mathbf{g}_t'.$$

回顧Adam演算法，它的設計靈感很清楚：
首先，動量和規模在狀態變數中清晰可見，
它們相當獨特的定義使我們移除偏項（這可以透過稍微不同的初始化和更新條件來修正）。
其次，RMSProp演算法中兩項的組合都非常簡單。
最後，明確的學習率$\eta$使我們能夠控制步長來解決收斂問題。

## 實現

從頭開始實現Adam演算法並不難。
為方便起見，我們將時間步$t$儲存在`hyperparams`字典中。
除此之外，一切都很簡單。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adam_states(feature_dim):
    v_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return ((v_w, s_w), (v_b, s_b))

def adam(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(beta2 * s + (1 - beta2) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr
                    / tf.math.sqrt(s_bias_corr) + eps)
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros((1, ))
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros((1, ))
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    a = []
    for p, (v, s) in zip(params, states):
        with paddle.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * paddle.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (paddle.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.zero_()
        a.append(p)
    hyperparams['t'] += 1
    return a
```

現在，我們用以上Adam演算法來訓練模型，這裡我們使用$\eta = 0.01$的學習率。

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

此外，我們可以用深度學習框架自帶演算法應用Adam演算法，這裡我們只需要傳遞設定引數。

```{.python .input}
d2l.train_concise_ch11('adam', {'learning_rate': 0.01}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adam
d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adam
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01}, data_iter)
```

```{.python .input}
#@tab paddle
trainer = paddle.optimizer.Adam
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01}, data_iter)
```

## Yogi

Adam演算法也存在一些問題：
即使在凸環境下，當$\mathbf{s}_t$的二次矩估計值爆炸時，它可能無法收斂。
 :cite:`Zaheer.Reddi.Sachan.ea.2018`為$\mathbf{s}_t$提出了的改進更新和引數初始化。
論文中建議我們重寫Adam演算法更新如下：

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \left(\mathbf{g}_t^2 - \mathbf{s}_{t-1}\right).$$

每當$\mathbf{g}_t^2$具有值很大的變數或更新很稀疏時，$\mathbf{s}_t$可能會太快地『忘記』過去的值。
一個有效的解決方法是將$\mathbf{g}_t^2 - \mathbf{s}_{t-1}$替換為$\mathbf{g}_t^2 \odot \mathop{\mathrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1})$。
這就是Yogi更新，現在更新的規模不再取決於偏差的量。

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \odot \mathop{\mathrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1}).$$

論文中，作者還進一步建議用更大的初始批次來初始化動量，而不僅僅是初始的逐點估計。

```{.python .input}
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = s + (1 - beta2) * np.sign(
            np.square(p.grad) - s) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab pytorch
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = s + (1 - beta2) * torch.sign(
                torch.square(p.grad) - s) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab tensorflow
def yogi(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(s + (1 - beta2) * tf.math.sign(
                   tf.math.square(grad) - s) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr
                    / tf.math.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab paddle
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    a=[]
    for p, (v, s) in zip(params, states):
        with paddle.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = s + (1 - beta2) * paddle.sign(
                paddle.square(p.grad) - s) * paddle.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (paddle.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.zero_()
        a.append(p)
    hyperparams['t'] += 1
    return a

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

## 小結

* Adam演算法將許多最佳化演算法的功能結合到了相當強大的更新規則中。
* Adam演算法在RMSProp演算法基礎上建立的，還在小批次的隨機梯度上使用EWMA。
* 在估計動量和二次矩時，Adam演算法使用偏差校正來調整緩慢的啟動速度。
* 對於具有顯著差異的梯度，我們可能會遇到收斂性問題。我們可以透過使用更大的小批次或者切換到改進的估計值$\mathbf{s}_t$來修正它們。Yogi提供了這樣的替代方案。

## 練習

1. 調節學習率，觀察並分析實驗結果。
1. 試著重寫動量和二次矩更新，從而使其不需要偏差校正。
1. 收斂時為什麼需要降低學習率$\eta$？
1. 嘗試構造一個使用Adam演算法會發散而Yogi會收斂的例子。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/4330)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/4331)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/4332)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11855)
:end_tab:
