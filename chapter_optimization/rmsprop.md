# RMSProp演算法
:label:`sec_rmsprop`

 :numref:`sec_adagrad`中的關鍵問題之一，是學習率按預定時間表$\mathcal{O}(t^{-\frac{1}{2}})$顯著降低。
雖然這通常適用於凸問題，但對於深度學習中遇到的非凸問題，可能並不理想。
但是，作為一個預處理器，Adagrad演算法按坐標順序的適應性是非常可取的。

 :cite:`Tieleman.Hinton.2012`建議以RMSProp演算法作為將速率排程與坐標自適應學習率分離的簡單修復方法。
問題在於，Adagrad演算法將梯度$\mathbf{g}_t$的平方累加成狀態向量$\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$。
因此，由於缺乏規範化，沒有約束力，$\mathbf{s}_t$持續增長，幾乎上是在演算法收斂時呈線性遞增。

解決此問題的一種方法是使用$\mathbf{s}_t / t$。
對$\mathbf{g}_t$的合理分佈來說，它將收斂。
遺憾的是，限制行為生效可能需要很長時間，因為該流程記住了值的完整軌跡。
另一種方法是按動量法中的方式使用洩漏平均值，即$\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$，其中引數$\gamma > 0$。
保持所有其它部分不變就產生了RMSProp演算法。

## 演算法

讓我們詳細寫出這些方程式。

$$\begin{aligned}
    \mathbf{s}_t & \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^2, \\
    \mathbf{x}_t & \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t.
\end{aligned}$$

常數$\epsilon > 0$通常設定為$10^{-6}$，以確保我們不會因除以零或步長過大而受到影響。
鑑於這種擴充套件，我們現在可以自由控制學習率$\eta$，而不考慮基於每個坐標應用的縮放。
就洩漏平均值而言，我們可以採用與之前在動量法中適用的相同推理。
擴充套件$\mathbf{s}_t$定義可獲得

$$
\begin{aligned}
\mathbf{s}_t & = (1 - \gamma) \mathbf{g}_t^2 + \gamma \mathbf{s}_{t-1} \\
& = (1 - \gamma) \left(\mathbf{g}_t^2 + \gamma \mathbf{g}_{t-1}^2 + \gamma^2 \mathbf{g}_{t-2} + \ldots, \right).
\end{aligned}
$$

同之前在 :numref:`sec_momentum`小節一樣，我們使用$1 + \gamma + \gamma^2 + \ldots, = \frac{1}{1-\gamma}$。
因此，權重總和標準化為$1$且觀測值的半衰期為$\gamma^{-1}$。
讓我們影象化各種數值的$\gamma$在過去40個時間步長的權重。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import math
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import math
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
import math
```

```{.python .input}
#@tab all
d2l.set_figsize()
gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, (1-gamma) * gamma ** x, label=f'gamma = {gamma:.2f}')
d2l.plt.xlabel('time');
```

## 從零開始實現

和之前一樣，我們使用二次函式$f(\mathbf{x})=0.1x_1^2+2x_2^2$來觀察RMSProp演算法的軌跡。
回想在 :numref:`sec_adagrad`一節中，當我們使用學習率為0.4的Adagrad演算法時，變數在演算法的後期階段移動非常緩慢，因為學習率衰減太快。
RMSProp演算法中不會發生這種情況，因為$\eta$是單獨控制的。

```{.python .input}
#@tab all
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
```

接下來，我們在深度網路中實現RMSProp演算法。

```{.python .input}
#@tab mxnet, pytorch
def init_rmsprop_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)
```

```{.python .input}
#@tab paddle
def init_rmsprop_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros([1])
    return (s_w, s_b)
```

```{.python .input}
#@tab tensorflow
def init_rmsprop_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)
```

```{.python .input}
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s[:] = gamma * s + (1 - gamma) * np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def rmsprop(params, grads, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(gamma * s + (1 - gamma) * tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

```{.python .input}
#@tab paddle
def rmsprop(params, states, hyperparams):
    a = []
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with paddle.no_grad():
            s[:] = gamma * s + (1 - gamma) * paddle.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / paddle.sqrt(s + eps)
        p.grad.zero_()
        a.append(p)
    return a 
```

我們將初始學習率設定為0.01，加權項$\gamma$設定為0.9。
也就是說，$\mathbf{s}$累加了過去的$1/(1-\gamma) = 10$次平方梯度觀測值的平均值。

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim);
```

## 簡潔實現

我們可直接使用深度學習框架中提供的RMSProp演算法來訓練模型。

```{.python .input}
d2l.train_concise_ch11('rmsprop', {'learning_rate': 0.01, 'gamma1': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.RMSprop
d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9},
                       data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.RMSprop
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01, 'rho': 0.9},
                       data_iter)
```

```{.python .input}
#@tab paddle
trainer = paddle.optimizer.RMSProp
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01, 'rho': 0.9},
                       data_iter)
```

## 小結

* RMSProp演算法與Adagrad演算法非常相似，因為兩者都使用梯度的平方來縮放係數。
* RMSProp演算法與動量法都使用洩漏平均值。但是，RMSProp演算法使用該技術來調整按係數順序的預處理器。
* 在實驗中，學習率需要由實驗者排程。
* 係數$\gamma$決定了在調整每坐標比例時歷史記錄的時長。

## 練習

1. 如果我們設定$\gamma = 1$，實驗會發生什麼？為什麼？
1. 旋轉最佳化問題以最小化$f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$。收斂會發生什麼？
1. 試試在真正的機器學習問題上應用RMSProp演算法會發生什麼，例如在Fashion-MNIST上的訓練。試驗不同的取值來調整學習率。
1. 隨著最佳化的進展，需要調整$\gamma$嗎？RMSProp演算法對此有多敏感？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/4321)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/4322)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/4323)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11853)
:end_tab: