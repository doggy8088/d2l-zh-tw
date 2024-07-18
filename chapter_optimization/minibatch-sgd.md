# 小批次隨機梯度下降
:label:`sec_minibatch_sgd`

到目前為止，我們在基於梯度的學習方法中遇到了兩個極端情況：
 :numref:`sec_gd`中使用完整資料集來計算梯度並更新引數，
 :numref:`sec_sgd`中一次處理一個訓練樣本來取得進展。
二者各有利弊：每當資料非常相似時，梯度下降並不是非常『資料高效』。
而由於CPU和GPU無法充分利用向量化，隨機梯度下降並不特別『計算高效』。
這暗示了兩者之間可能有折中方案，這便涉及到*小批次隨機梯度下降*（minibatch gradient descent）。

## 向量化和緩存

使用小批次的決策的核心是計算效率。
當考慮與多個GPU和多台伺服器並行處理時，這一點最容易被理解。在這種情況下，我們需要向每個GPU傳送至少一張影象。
有了每台伺服器8個GPU和16台伺服器，我們就能得到大小為128的小批次。

當涉及到單個GPU甚至CPU時，事情會更微妙一些：
這些裝置有多種型別的記憶體、通常情況下多種型別的計算單元以及在它們之間不同的頻寬限制。
例如，一個CPU有少量暫存器（register），L1和L2快取，以及L3快取（在不同的處理器核心之間共享）。
隨著快取的大小的增加，它們的延遲也在增加，同時頻寬在減少。
可以說，處理器能夠執行的操作遠比主記憶體介面所能提供的多得多。

首先，具有16個核心和AVX-512向量化的2GHz CPU每秒可處理高達$2 \cdot 10^9 \cdot 16 \cdot 32 = 10^{12}$個位元組。
同時，GPU的效能很容易超過該數字100倍。
而另一方面，中端伺服器處理器的頻寬可能不超過100Gb/s，即不到處理器滿負荷所需的十分之一。
更糟糕的是，並非所有的記憶體入口都是相等的：記憶體介面通常為64位或更寬（例如，在最多384位的GPU上）。
因此讀取單個位元組會導致由於更寬的存取而產生的代價。

其次，第一次存取的額外開銷很大，而按序存取（sequential access）或突發讀取（burst read）相對開銷較小。
有關更深入的討論，請參閱此[維基百科文章](https://en.wikipedia.org/wiki/Cache_hierarchy)。

減輕這些限制的方法是使用足夠快的CPU快取層次結構來為處理器提供資料。
這是深度學習中批次處理背後的推動力。
舉一個簡單的例子：矩陣-矩陣乘法。
比如$\mathbf{A} = \mathbf{B}\mathbf{C}$，我們有很多方法來計算$\mathbf{A}$。例如，我們可以嘗試以下方法：

1. 我們可以計算$\mathbf{A}_{ij} = \mathbf{B}_{i,:} \mathbf{C}_{:,j}^\top$，也就是說，我們可以透過點積進行逐元素計算。
1. 我們可以計算$\mathbf{A}_{:,j} = \mathbf{B} \mathbf{C}_{:,j}^\top$，也就是說，我們可以一次計算一列。同樣，我們可以一次計算$\mathbf{A}$一行$\mathbf{A}_{i,:}$。
1. 我們可以簡單地計算$\mathbf{A} = \mathbf{B} \mathbf{C}$。
1. 我們可以將$\mathbf{B}$和$\mathbf{C}$分成較小的區塊矩陣，然後一次計算$\mathbf{A}$的一個區塊。

如果我們使用第一個選擇，每次我們計算一個元素$\mathbf{A}_{ij}$時，都需要將一行和一列向量複製到CPU中。
更糟糕的是，由於矩陣元素是按順序對齊的，因此當從記憶體中讀取它們時，我們需要訪問兩個向量中許多不相交的位置。
第二種選擇相對更有利：我們能夠在遍歷$\mathbf{B}$的同時，將列向量$\mathbf{C}_{:,j}$保留在CPU快取中。
它將記憶體頻寬需求減半，相應地提高了訪問速度。
第三種選擇表面上是最可取的，然而大多數矩陣可能不能完全放入快取中。
第四種選擇提供了一個實踐上很有用的方案：我們可以將矩陣的區塊移到快取中然後在本地將它們相乘。
讓我們來看看這些操作在實踐中的效率如何。

除了計算效率之外，Python和深度學習框架本身帶來的額外開銷也是相當大的。
回想一下，每次我們執行程式碼時，Python直譯器都會向深度學習框架傳送一個指令，要求將其插入到計算圖中並在排程過程中處理它。
這樣的額外開銷可能是非常不利的。
總而言之，我們最好用向量化（和矩陣）。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

timer = d2l.Timer()
A = np.zeros((256, 256))
B = np.random.normal(0, 1, (256, 256))
C = np.random.normal(0, 1, (256, 256))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
import numpy as np

timer = d2l.Timer()
A = torch.zeros(256, 256)
B = torch.randn(256, 256)
C = torch.randn(256, 256)
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np

timer = d2l.Timer()
A = tf.Variable(d2l.zeros((256, 256)))
B = tf.Variable(d2l.normal([256, 256], 0, 1))
C = tf.Variable(d2l.normal([256, 256], 0, 1))
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
import numpy as np

timer = d2l.Timer()
A = d2l.zeros((256, 256))
B = d2l.randn((256, 256))
C = d2l.randn((256, 256))
```

按元素分配只需遍歷分別為$\mathbf{B}$和$\mathbf{C}$的所有行和列，即可將該值分配給$\mathbf{A}$。

```{.python .input}
# 逐元素計算A=BC
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = np.dot(B[i, :], C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# 逐元素計算A=BC
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = torch.dot(B[i, :], C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
# 逐元素計算A=BC
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j].assign(tf.tensordot(B[i, :], C[:, j], axes=1))
timer.stop()
```

```{.python .input}
#@tab paddle
# 逐元素計算A=BC
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = paddle.dot(B[i, :], C[:, j])
timer.stop()
```

更快的策略是執行按列分配。

```{.python .input}
# 逐列計算A=BC
timer.start()
for j in range(256):
    A[:, j] = np.dot(B, C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# 逐列計算A=BC
timer.start()
for j in range(256):
    A[:, j] = torch.mv(B, C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(256):
    A[:, j].assign(tf.tensordot(B, C[:, j], axes=1))
timer.stop()
```

```{.python .input}
#@tab paddle
# 逐列計算A=BC
timer.start()
for j in range(256):
    A[:, j] = paddle.mv(B, C[:, j])
timer.stop()
```

最有效的方法是在一個區塊中執行整個操作。讓我們看看它們各自的操作速度是多少。

```{.python .input}
# 一次性計算A=BC
timer.start()
A = np.dot(B, C)
A.wait_to_read()
timer.stop()

# 乘法和加法作為單獨的操作（在實踐中融合）
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab pytorch
# 一次性計算A=BC
timer.start()
A = torch.mm(B, C)
timer.stop()

# 乘法和加法作為單獨的操作（在實踐中融合）
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab tensorflow
# 一次性計算A=BC
timer.start()
A.assign(tf.tensordot(B, C, axes=1))
timer.stop()

# 乘法和加法作為單獨的操作（在實踐中融合）
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab paddle
# 一次性計算A=BC
timer.start()
A = paddle.mm(B, C)
timer.stop()

# 乘法和加法作為單獨的操作（在實踐中融合）
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

## 小批次

:label:`sec_minibatches`

之前我們會理所當然地讀取資料的*小批次*，而不是觀測單個資料來更新引數，現在簡要解釋一下原因。
處理單個觀測值需要我們執行許多單一矩陣-向量（甚至向量-向量）乘法，這耗費相當大，而且對應深度學習框架也要巨大的開銷。
這既適用於計算梯度以更新引數時，也適用於用神經網路預測。
也就是說，每當我們執行$\mathbf{w} \leftarrow \mathbf{w} - \eta_t \mathbf{g}_t$時，消耗巨大。其中

$$\mathbf{g}_t = \partial_{\mathbf{w}} f(\mathbf{x}_{t}, \mathbf{w}).$$

我們可以透過將其應用於一個小批次觀測值來提高此操作的*計算*效率。
也就是說，我們將梯度$\mathbf{g}_t$替換為一個小批次而不是單個觀測值

$$\mathbf{g}_t = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}).$$

讓我們看看這對$\mathbf{g}_t$的統計屬性有什麼影響：由於$\mathbf{x}_t$和小批次$\mathcal{B}_t$的所有元素都是從訓練集中隨機抽出的，因此梯度的期望保持不變。
另一方面，方差顯著降低。
由於小批次梯度由正在被平均計算的$b := |\mathcal{B}_t|$個獨立梯度組成，其標準差降低了$b^{-\frac{1}{2}}$。
這本身就是一件好事，因為這意味著更新與完整的梯度更接近了。

直觀來說，這表明選擇大型的小批次$\mathcal{B}_t$將是普遍可行的。
然而，經過一段時間後，與計算代價的線性增長相比，標準差的額外減少是微乎其微的。
在實踐中我們選擇一個足夠大的小批次，它可以提供良好的計算效率同時仍適合GPU的記憶體。
下面，我們來看看這些高效的程式碼。
在裡面我們執行相同的矩陣-矩陣乘法，但是這次我們將其一次性分為64列的『小批次』。

```{.python .input}
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = np.dot(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab pytorch
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64].assign(tf.tensordot(B, C[:, j:j+64], axes=1))
timer.stop()
print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab paddle
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = paddle.mm(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')
```

顯而易見，小批次上的計算基本上與完整矩陣一樣有效。
需要注意的是，在 :numref:`sec_batch_norm`中，我們使用了一種在很大程度上取決於小批次中的方差的正則化。
隨著後者增加，方差會減少，隨之而來的是批次規範化帶來的噪聲注入的好處。
關於實例，請參閱 :cite:`Ioffe.2017`，瞭解有關如何重新縮放並計算適當專案。

## 讀取資料集

讓我們來看看如何從資料中有效地生成小批次。
下面我們使用NASA開發的測試機翼的資料集[不同飛行器產生的噪聲](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise)來比較這些最佳化演算法。
為方便起見，我們只使用前$1,500$樣本。
資料已作預處理：我們移除了均值並將方差重新縮放到每個坐標為$1$。

```{.python .input}
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array(
        (data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab pytorch
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab tensorflow
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab paddle
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = d2l.tensor((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

## 從零開始實現

 :numref:`sec_linear_scratch`一節中已經實現過小批次隨機梯度下降演算法。
我們在這裡將它的輸入引數變得更加通用，主要是為了方便本章後面介紹的其他最佳化演算法也可以使用同樣的輸入。
具體來說，我們新增了一個狀態輸入`states`並將超引數放在字典`hyperparams`中。
此外，我們將在訓練函式裡對各個小批次樣本的損失求平均，因此最佳化演算法中的梯度不需要除以批次大小。

```{.python .input}
def sgd(params, states, hyperparams):
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad
```

```{.python .input}
#@tab pytorch
def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, states, hyperparams):
    for param, grad in zip(params, grads):
        param.assign_sub(hyperparams['lr']*grad)
```

```{.python .input}
#@tab paddle
def sgd(params, states, hyperparams):
    a = []
    with paddle.no_grad():
        for p in params:
            p = p - hyperparams['lr'] * p.grad
            p.stop_gradient = False
            a.append(p)
        return a
```

下面實現一個通用的訓練函式，以方便本章後面介紹的其他最佳化演算法使用。
它初始化了一個線性迴歸模型，然後可以使用小批次隨機梯度下降以及後續小節介紹的其他演算法來訓練模型。

```{.python .input}
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # 初始化模型
    w = np.random.normal(scale=0.01, size=(feature_dim, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # 訓練模型
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab pytorch
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # 初始化模型
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # 訓練模型
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab tensorflow
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # 初始化模型
    w = tf.Variable(tf.random.normal(shape=(feature_dim, 1),
                                   mean=0, stddev=0.01),trainable=True)
    b = tf.Variable(tf.zeros(1), trainable=True)

    # 訓練模型
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()

    for _ in range(num_epochs):
        for X, y in data_iter:
          with tf.GradientTape() as g:
            l = tf.math.reduce_mean(loss(net(X), y))

          dw, db = g.gradient(l, [w, b])
          trainer_fn([w, b], [dw, db], states, hyperparams)
          n += X.shape[0]
          if n % 200 == 0:
              timer.stop()
              p = n/X.shape[0]
              q = p/tf.data.experimental.cardinality(data_iter).numpy()
              r = (d2l.evaluate_loss(net, data_iter, loss),)
              animator.add(q, r)
              timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab paddle
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # 初始化模型
    w = d2l.tensor(d2l.normal(mean=0.0, std=0.01, shape=(feature_dim, 1)), stop_gradient=False)
    b = d2l.tensor(d2l.zeros((1,)), stop_gradient=False)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # 訓練模型
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            w, b = trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

讓我們來看看批次梯度下降的最佳化是如何進行的。
這可以透過將小批次設定為1500（即樣本總數）來實現。
因此，模型引數每個迭代輪數只迭代一次。

```{.python .input}
#@tab all
def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

gd_res = train_sgd(1, 1500, 10)
```

當批次大小為1時，最佳化使用的是隨機梯度下降。
為了簡化實現，我們選擇了很小的學習率。
在隨機梯度下降的實驗中，每當一個樣本被處理，模型引數都會更新。
在這個例子中，這相當於每個迭代輪數有1500次更新。
可以看到，目標函式值的下降在1個迭代輪數後就變得較為平緩。
儘管兩個例子在一個迭代輪數內都處理了1500個樣本，但實驗中隨機梯度下降的一個迭代輪數耗時更多。
這是因為隨機梯度下降更頻繁地更新了引數，而且一次處理單個觀測值效率較低。

```{.python .input}
#@tab all
sgd_res = train_sgd(0.005, 1)
```

最後，當批次大小等於100時，我們使用小批次隨機梯度下降進行最佳化。
每個迭代輪數所需的時間比隨機梯度下降和批次梯度下降所需的時間短。

```{.python .input}
#@tab all
mini1_res = train_sgd(.4, 100)
```

將批次大小減少到10，每個迭代輪數的時間都會增加，因為每批工作負載的執行效率變得更低。

```{.python .input}
#@tab all
mini2_res = train_sgd(.05, 10)
```

現在我們可以比較前四個實驗的時間與損失。
可以看出，儘管在處理的樣本數方面，隨機梯度下降的收斂速度快於梯度下降，但與梯度下降相比，它需要更多的時間來達到同樣的損失，因為逐個樣本來計算梯度並不那麼有效。
小批次隨機梯度下降能夠平衡收斂速度和計算效率。
大小為10的小批次比隨機梯度下降更有效；
大小為100的小批次在執行時間上甚至優於梯度下降。

```{.python .input}
#@tab all
d2l.set_figsize([6, 3])
d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'time (sec)', 'loss', xlim=[1e-2, 10],
         legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
d2l.plt.gca().set_xscale('log')
```

## 簡潔實現

下面用深度學習框架自帶演算法實現一個通用的訓練函式，我們將在本章中其它小節使用它。

```{.python .input}
#@save
def train_concise_ch11(tr_name, hyperparams, data_iter, num_epochs=2):
    # 初始化模型
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    trainer = gluon.Trainer(net.collect_params(), tr_name, hyperparams)
    loss = gluon.loss.L2Loss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(X.shape[0])
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
```

```{.python .input}
#@tab pytorch
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    # 初始化模型
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)

    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                # MSELoss計算平方誤差時不帶係數1/2
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
```

```{.python .input}
#@tab tensorflow
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=2):
    # 初始化模型
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    optimizer = trainer_fn(**hyperparams)
    loss = tf.keras.losses.MeanSquaredError()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with tf.GradientTape() as g:
                out = net(X)
                l = loss(y, out)
                params = net.trainable_variables
                grads = g.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                p = n/X.shape[0]
                q = p/tf.data.experimental.cardinality(data_iter).numpy()
                # MeanSquaredError計算平方誤差時不帶係數1/2
                r = (d2l.evaluate_loss(net, data_iter, loss) / 2,)
                animator.add(q, r)
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
```

```{.python .input}
#@tab paddle
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    # 初始化模型
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(m):
        if type(m) == nn.Linear:
            paddle.nn.initializer.Normal(m.weight, std=0.01)

    net.apply(init_weights)

    optimizer = trainer_fn(parameters=net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.clear_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                # MSELoss計算平方誤差時不帶係數1/2
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
```

下面使用這個訓練函式，復現之前的實驗。

```{.python .input}
data_iter, _ = get_data_ch11(10)
train_concise_ch11('sgd', {'learning_rate': 0.05}, data_iter)
```

```{.python .input}
#@tab pytorch
data_iter, _ = get_data_ch11(10)
trainer = torch.optim.SGD
train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
data_iter, _ = get_data_ch11(10)
trainer = tf.keras.optimizers.SGD
train_concise_ch11(trainer, {'learning_rate': 0.05}, data_iter)
```

```{.python .input}
#@tab paddle
data_iter, _ = get_data_ch11(10)
trainer = paddle.optimizer.SGD
train_concise_ch11(trainer, {'learning_rate': 0.01}, data_iter)
```

## 小結

* 由於減少了深度學習框架的額外開銷，使用更好的記憶體定位以及CPU和GPU上的快取，向量化使程式碼更加高效。
* 隨機梯度下降的『統計效率』與大批量一次處理資料的『計算效率』之間存在權衡。小批次隨機梯度下降提供了兩全其美的答案：計算和統計效率。
* 在小批次隨機梯度下降中，我們處理透過訓練資料的隨機排列獲得的批次資料（即每個觀測值只處理一次，但按隨機順序）。
* 在訓練期間降低學習率有助於訓練。
* 一般來說，小批次隨機梯度下降比隨機梯度下降和梯度下降的速度快，收斂風險較小。

## 練習

1. 修改批次大小和學習率，並觀察目標函式值的下降率以及每個迭代輪數消耗的時間。
1. 將小批次隨機梯度下降與實際從訓練集中*取樣替換*的變體進行比較。會看出什麼？
1. 一個邪惡的精靈在沒通知你的情況下複製了你的資料集（即每個觀測發生兩次，資料集增加到原始大小的兩倍，但沒有人告訴你）。隨機梯度下降、小批次隨機梯度下降和梯度下降的表現將如何變化？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/4324)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/4325)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/4326)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11850)
:end_tab: