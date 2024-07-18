# 迴圈神經網路的從零開始實現
:label:`sec_rnn_scratch`

本節將根據 :numref:`sec_rnn`中的描述，
從頭開始基於迴圈神經網路實現字元級語言模型。
這樣的模型將在H.G.Wells的時光機器資料集上訓練。
和前面 :numref:`sec_language_model`中介紹過的一樣，
我們先讀取資料集。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
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
from paddle import nn
from paddle.nn import functional as F
```

```{.python .input}
#@tab all
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab tensorflow
train_random_iter, vocab_random_iter = d2l.load_data_time_machine(
    batch_size, num_steps, use_random_iter=True)
```

## [**獨熱編碼**]

回想一下，在`train_iter`中，每個詞元都表示為一個數字索引，
將這些索引直接輸入神經網路可能會使學習變得困難。
我們通常將每個詞元表示為更具表現力的特徵向量。
最簡單的表示稱為*獨熱編碼*（one-hot encoding），
它在 :numref:`subsec_classification-problem`中介紹過。

簡言之，將每個索引對映為相互不同的單位向量：
假設詞表中不同詞元的數目為$N$（即`len(vocab)`），
詞元索引的範圍為$0$到$N-1$。
如果詞元的索引是整數$i$，
那麼我們將建立一個長度為$N$的全$0$向量，
並將第$i$處的元素設定為$1$。
此向量是原始詞元的一個獨熱向量。
索引為$0$和$2$的獨熱向量如下所示：

```{.python .input}
npx.one_hot(np.array([0, 2]), len(vocab))
```

```{.python .input}
#@tab pytorch
F.one_hot(torch.tensor([0, 2]), len(vocab))
```

```{.python .input}
#@tab tensorflow
tf.one_hot(tf.constant([0, 2]), len(vocab))
```

```{.python .input}
#@tab paddle
F.one_hot(paddle.to_tensor([0, 2]), len(vocab))
```

我們每次取樣的(**小批次資料形狀是二維張量：
（批次大小，時間步數）。**)
`one_hot`函式將這樣一個小批次資料轉換成三維張量，
張量的最後一個維度等於詞表大小（`len(vocab)`）。
我們經常轉換輸入的維度，以便獲得形狀為
（時間步數，批次大小，詞表大小）的輸出。
這將使我們能夠更方便地透過最外層的維度，
一步一步地更新小批次資料的隱狀態。

```{.python .input}
X = d2l.reshape(d2l.arange(10), (2, 5))
npx.one_hot(X.T, 28).shape
```

```{.python .input}
#@tab pytorch
X = d2l.reshape(d2l.arange(10), (2, 5))
F.one_hot(X.T, 28).shape
```

```{.python .input}
#@tab tensorflow
X = d2l.reshape(d2l.arange(10), (2, 5))
tf.one_hot(tf.transpose(X), 28).shape
```

```{.python .input}
#@tab paddle
X = paddle.arange(10).reshape((2, 5))
F.one_hot(X.T, 28).shape
```

## 初始化模型引數

接下來，我們[**初始化迴圈神經網路模型的模型引數**]。
隱藏單元數`num_hiddens`是一個可調的超引數。
當訓練語言模型時，輸入和輸出來自相同的詞表。
因此，它們具有相同的維度，即詞表的大小。

```{.python .input}
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape, ctx=device)

    # 隱藏層引數
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = d2l.zeros(num_hiddens, ctx=device)
    # 輸出層引數
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, ctx=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params
```

```{.python .input}
#@tab pytorch
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隱藏層引數
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = d2l.zeros(num_hiddens, device=device)
    # 輸出層引數
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

```{.python .input}
#@tab tensorflow
def get_params(vocab_size, num_hiddens):
    num_inputs = num_outputs = vocab_size
    
    def normal(shape):
        return d2l.normal(shape=shape,stddev=0.01,mean=0,dtype=tf.float32)

    # 隱藏層引數
    W_xh = tf.Variable(normal((num_inputs, num_hiddens)), dtype=tf.float32)
    W_hh = tf.Variable(normal((num_hiddens, num_hiddens)), dtype=tf.float32)
    b_h = tf.Variable(d2l.zeros(num_hiddens), dtype=tf.float32)
    # 輸出層引數
    W_hq = tf.Variable(normal((num_hiddens, num_outputs)), dtype=tf.float32)
    b_q = tf.Variable(d2l.zeros(num_outputs), dtype=tf.float32)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    return params
```

```{.python .input}
#@tab paddle
def get_params(vocab_size, num_hiddens):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return paddle.randn(shape=shape)* 0.01

    # 隱藏層引數
    W_xh = normal([num_inputs, num_hiddens])
    W_hh = normal([num_hiddens, num_hiddens])
    b_h = d2l.zeros(shape=[num_hiddens])
    # 輸出層引數
    W_hq = normal([num_hiddens, num_outputs])
    b_q = d2l.zeros(shape=[num_outputs])
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.stop_gradient=False
    return params
```

## 迴圈神經網路模型

為了定義迴圈神經網路模型，
我們首先需要[**一個`init_rnn_state`函式在初始化時回傳隱狀態**]。
這個函式的回傳是一個張量，張量全用0填充，
形狀為（批次大小，隱藏單元數）。
在後麵的章節中我們將會遇到隱狀態包含多個變數的情況，
而使用元組可以更容易地處理些。

```{.python .input}
def init_rnn_state(batch_size, num_hiddens, device):
    return (d2l.zeros((batch_size, num_hiddens), ctx=device), )
```

```{.python .input}
#@tab pytorch
def init_rnn_state(batch_size, num_hiddens, device):
    return (d2l.zeros((batch_size, num_hiddens), device=device), )
```

```{.python .input}
#@tab tensorflow
def init_rnn_state(batch_size, num_hiddens):
    return (d2l.zeros((batch_size, num_hiddens)), )
```

```{.python .input}
#@tab paddle
def init_rnn_state(batch_size, num_hiddens):
    return (paddle.zeros(shape=[batch_size, num_hiddens]), )
```

[**下面的`rnn`函式定義瞭如何在一個時間步內計算隱狀態和輸出。**]
迴圈神經網路模型透過`inputs`最外層的維度實現迴圈，
以便逐時間步更新小批次資料的隱狀態`H`。
此外，這裡使用$\tanh$函式作為啟用函式。
如 :numref:`sec_mlp`所述，
當元素在實數上滿足均勻分佈時，$\tanh$函式的平均值為0。

```{.python .input}
def rnn(inputs, state, params):
    # inputs的形狀：(時間步數量，批次大小，詞表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形狀：(批次大小，詞表大小)
    for X in inputs:
        H = np.tanh(np.dot(X, W_xh) + np.dot(H, W_hh) + b_h)
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H,)
```

```{.python .input}
#@tab pytorch
def rnn(inputs, state, params):
    # inputs的形狀：(時間步數量，批次大小，詞表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形狀：(批次大小，詞表大小)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
```

```{.python .input}
#@tab tensorflow
def rnn(inputs, state, params):
    # inputs的形狀：(時間步數量，批次大小，詞表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形狀：(批次大小，詞表大小)
    for X in inputs:
        X = tf.reshape(X,[-1,W_xh.shape[0]])
        H = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(H, W_hh) + b_h)
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return d2l.concat(outputs, axis=0), (H,)
```

```{.python .input}
#@tab paddle
def rnn(inputs, state, params):
    # inputs的形狀：(時間步數量，批次大小，詞表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形狀：(批次大小，詞表大小)
    for X in inputs:
        H = paddle.tanh(paddle.mm(X, W_xh) + paddle.mm(H, W_hh) + b_h)
        Y = paddle.mm(H, W_hq) + b_q
        outputs.append(Y)
    return paddle.concat(x=outputs, axis=0), (H,)
```

定義了所有需要的函式之後，接下來我們[**建立一個類來包裝這些函式**]，
並存儲從零開始實現的迴圈神經網路模型的引數。

```{.python .input}
class RNNModelScratch:  #@save
    """從零開始實現的迴圈神經網路模型"""
    def __init__(self, vocab_size, num_hiddens, device, get_params,
                 init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = npx.one_hot(X.T, self.vocab_size)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, ctx):
        return self.init_state(batch_size, self.num_hiddens, ctx)
```

```{.python .input}
#@tab pytorch
class RNNModelScratch: #@save
    """從零開始實現的迴圈神經網路模型"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)
```

```{.python .input}
#@tab tensorflow
class RNNModelScratch: #@save
    """從零開始實現的迴圈神經網路模型"""
    def __init__(self, vocab_size, num_hiddens,
                 init_state, forward_fn, get_params):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.init_state, self.forward_fn = init_state, forward_fn
        self.trainable_variables = get_params(vocab_size, num_hiddens)

    def __call__(self, X, state):
        X = tf.one_hot(tf.transpose(X), self.vocab_size)
        X = tf.cast(X, tf.float32)
        return self.forward_fn(X, state, self.trainable_variables)

    def begin_state(self, batch_size, *args, **kwargs):
        return self.init_state(batch_size, self.num_hiddens)
```

```{.python .input}
#@tab paddle
class RNNModelScratch: #@save
    """從零開始實現的迴圈神經網路模型"""
    def __init__(self, vocab_size, num_hiddens,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size):
        return self.init_state(batch_size, self.num_hiddens)
```

讓我們[**檢查輸出是否具有正確的形狀**]。
例如，隱狀態的維數是否保持不變。

```{.python .input}
#@tab mxnet
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.as_in_context(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape
```

```{.python .input}
#@tab pytorch
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape
```

```{.python .input}
#@tab tensorflow
# 定義tensorflow訓練策略
device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)

num_hiddens = 512
with strategy.scope():
    net = RNNModelScratch(len(vocab), num_hiddens, init_rnn_state, rnn,
                          get_params)
state = net.begin_state(X.shape[0])
Y, new_state = net(X, state)
Y.shape, len(new_state), new_state[0].shape
```

```{.python .input}
#@tab paddle
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0])
Y, new_state = net(X, state)
Y.shape, len(new_state), new_state[0].shape
```

我們可以看到輸出形狀是（時間步數$\times$批次大小，詞表大小），
而隱狀態形狀保持不變，即（批次大小，隱藏單元數）。

## 預測

讓我們[**首先定義預測函式來生成`prefix`之後的新字元**]，
其中的`prefix`是一個使用者提供的包含多個字元的字串。
在迴圈遍歷`prefix`中的開始字元時，
我們不斷地將隱狀態傳遞到下一個時間步，但是不生成任何輸出。
這被稱為*預熱*（warm-up）期，
因為在此期間模型會自我更新（例如，更新隱狀態），
但不會進行預測。
預熱期結束後，隱狀態的值通常比剛開始的初始值更適合預測，
從而預測字元並輸出它們。

```{.python .input}
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """在prefix後面生成新字元"""
    state = net.begin_state(batch_size=1, ctx=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(
        d2l.tensor([outputs[-1]], ctx=device), (1, 1))
    for y in prefix[1:]:  # 預熱期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 預測num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input}
#@tab pytorch
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """在prefix後面生成新字元"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(d2l.tensor(
        [outputs[-1]], device=device), (1, 1))
    for y in prefix[1:]:  # 預熱期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 預測num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input}
#@tab tensorflow
def predict_ch8(prefix, num_preds, net, vocab):  #@save
    """在prefix後面生成新字元"""
    state = net.begin_state(batch_size=1, dtype=tf.float32)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(d2l.tensor([outputs[-1]]), 
                                    (1, 1)).numpy()
    for y in prefix[1:]:  # 預熱期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 預測num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.numpy().argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input}
#@tab paddle
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """在prefix後面生成新字元"""
    state = net.begin_state(batch_size=1)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(d2l.tensor(outputs[-1], place=device), (1, 1))
    for y in prefix[1:]:  # 預熱期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 預測num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(paddle.reshape(paddle.argmax(y,axis=1),shape=[1])))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

現在我們可以測試`predict_ch8`函式。
我們將字首指定為`time traveller `，
並基於這個字首生成10個後續字元。
鑑於我們還沒有訓練網路，它會生成荒謬的預測結果。

```{.python .input}
#@tab mxnet,pytorch, paddle
predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())
```

```{.python .input}
#@tab tensorflow
predict_ch8('time traveller ', 10, net, vocab)
```

## [**梯度裁剪**]

對於長度為$T$的序列，我們在迭代中計算這$T$個時間步上的梯度，
將會在反向傳播過程中產生長度為$\mathcal{O}(T)$的矩陣乘法鏈。
如 :numref:`sec_numerical_stability`所述，
當$T$較大時，它可能導致數值不穩定，
例如可能導致梯度爆炸或梯度消失。
因此，迴圈神經網路模型往往需要額外的方式來支援穩定訓練。

一般來說，當解決最佳化問題時，我們對模型引數採用更新步驟。
假定在向量形式的$\mathbf{x}$中，
或者在小批次資料的負梯度$\mathbf{g}$方向上。
例如，使用$\eta > 0$作為學習率時，在一次迭代中，
我們將$\mathbf{x}$更新為$\mathbf{x} - \eta \mathbf{g}$。
如果我們進一步假設目標函式$f$表現良好，
即函式$f$在常數$L$下是*利普希茨連續的*（Lipschitz continuous）。
也就是說，對於任意$\mathbf{x}$和$\mathbf{y}$我們有：

$$|f(\mathbf{x}) - f(\mathbf{y})| \leq L \|\mathbf{x} - \mathbf{y}\|.$$

在這種情況下，我們可以安全地假設：
如果我們透過$\eta \mathbf{g}$更新引數向量，則

$$|f(\mathbf{x}) - f(\mathbf{x} - \eta\mathbf{g})| \leq L \eta\|\mathbf{g}\|,$$

這意味著我們不會觀察到超過$L \eta \|\mathbf{g}\|$的變化。
這既是壞事也是好事。
壞的方面，它限制了取得進展的速度；
好的方面，它限制了事情變糟的程度，尤其當我們朝著錯誤的方向前進時。

有時梯度可能很大，從而最佳化演算法可能無法收斂。
我們可以透過降低$\eta$的學習率來解決這個問題。
但是如果我們很少得到大的梯度呢？
在這種情況下，這種做法似乎毫無道理。
一個流行的替代方案是透過將梯度$\mathbf{g}$投影回給定半徑
（例如$\theta$）的球來裁剪梯度$\mathbf{g}$。
如下式：

(**$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$**)

透過這樣做，我們知道梯度範數永遠不會超過$\theta$，
並且更新後的梯度完全與$\mathbf{g}$的原始方向對齊。
它還有一個值得擁有的副作用，
即限制任何給定的小批次資料（以及其中任何給定的樣本）對引數向量的影響，
這賦予了模型一定程度的穩定性。
梯度裁剪提供了一個快速修復梯度爆炸的方法，
雖然它並不能完全解決問題，但它是眾多有效的技術之一。

下面我們定義一個函式來裁剪模型的梯度，
模型是從零開始實現的模型或由高階API建立的模型。
我們在此計算了所有模型引數的梯度的範數。

```{.python .input}
def grad_clipping(net, theta):  #@save
    """裁剪梯度"""
    if isinstance(net, gluon.Block):
        params = [p.data() for p in net.collect_params().values()]
    else:
        params = net.params
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

```{.python .input}
#@tab pytorch
def grad_clipping(net, theta):  #@save
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

```{.python .input}
#@tab tensorflow
def grad_clipping(grads, theta):  #@save
    """裁剪梯度"""
    theta = tf.constant(theta, dtype=tf.float32)
    new_grad = []
    for grad in grads:
        if isinstance(grad, tf.IndexedSlices):
            new_grad.append(tf.convert_to_tensor(grad))
        else:
            new_grad.append(grad)
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)).numpy()
                        for grad in new_grad))
    norm = tf.cast(norm, tf.float32)
    if tf.greater(norm, theta):
        for i, grad in enumerate(new_grad):
            new_grad[i] = grad * theta / norm
    else:
        new_grad = new_grad
    return new_grad
```

```{.python .input}
#@tab paddle
def grad_clipping(net, theta):  #@save
    """裁剪梯度"""
    if isinstance(net, nn.Layer):
        params = [p for p in net.parameters() if not p.stop_gradient]
    else:
        params = net.params
    norm = paddle.sqrt(sum(paddle.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        with paddle.no_grad():
            for param in params:
                param.grad.set_value(param.grad * theta / norm)
```

## 訓練

在訓練模型之前，讓我們[**定義一個函式在一個迭代週期內訓練模型**]。
它與我們訓練 :numref:`sec_softmax_scratch`模型的方式有三個不同之處。

1. 序列資料的不同取樣方法（隨機取樣和順序分割槽）將導致隱狀態初始化的差異。
1. 我們在更新模型引數之前裁剪梯度。
   這樣的操作的目的是，即使訓練過程中某個點上發生了梯度爆炸，也能保證模型不會發散。
1. 我們用困惑度來評價模型。如 :numref:`subsec_perplexity`所述，
   這樣的度量確保了不同長度的序列具有可比性。

具體來說，當使用順序分割槽時，
我們只在每個迭代週期的開始位置初始化隱狀態。
由於下一個小批次資料中的第$i$個子序列樣本
與當前第$i$個子序列樣本相鄰，
因此當前小批次資料最後一個樣本的隱狀態，
將用於初始化下一個小批次資料第一個樣本的隱狀態。
這樣，儲存在隱狀態中的序列的歷史訊息
可以在一個迭代週期內流經相鄰的子序列。
然而，在任何一點隱狀態的計算，
都依賴於同一迭代週期中前面所有的小批次資料，
這使得梯度計算變得複雜。
為了降低計算量，在處理任何一個小批次資料之前，
我們先分離梯度，使得隱狀態的梯度計算總是限制在一個小批次資料的時間步內。

當使用隨機抽樣時，因為每個樣本都是在一個隨機位置抽樣的，
因此需要為每個迭代週期重新初始化隱狀態。
與 :numref:`sec_softmax_scratch`中的
`train_epoch_ch3`函式相同，
`updater`是更新模型引數的常用函式。
它既可以是從頭開始實現的`d2l.sgd`函式，
也可以是深度學習框架中內建的最佳化函式。

```{.python .input}
#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """訓練模型一個迭代週期（定義見第8章）"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 訓練損失之和,詞元數量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用隨機抽樣時初始化state
            state = net.begin_state(batch_size=X.shape[0], ctx=device)
        else:
            for s in state:
                s.detach()
        y = Y.T.reshape(-1)
        X, y = X.as_in_ctx(device), y.as_in_ctx(device)
        with autograd.record():
            y_hat, state = net(X, state)
            l = loss(y_hat, y).mean()
        l.backward()
        grad_clipping(net, 1)
        updater(batch_size=1)  # 因為已經呼叫了mean函式
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

```{.python .input}
#@tab pytorch
#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """訓練網路一個迭代週期（定義見第8章）"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 訓練損失之和,詞元數量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用隨機抽樣時初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state對於nn.GRU是個張量
                state.detach_()
            else:
                # state對於nn.LSTM或對於我們從零開始實現的模型是個張量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因為已經呼叫了mean函式
            updater(batch_size=1)
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

```{.python .input}
#@tab tensorflow
#@save
def train_epoch_ch8(net, train_iter, loss, updater, use_random_iter):
    """訓練模型一個迭代週期（定義見第8章）"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 訓練損失之和,詞元數量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用隨機抽樣時初始化state
            state = net.begin_state(batch_size=X.shape[0], dtype=tf.float32)
        with tf.GradientTape(persistent=True) as g:
            y_hat, state = net(X, state)
            y = d2l.reshape(tf.transpose(Y), (-1))
            l = loss(y, y_hat)
        params = net.trainable_variables
        grads = g.gradient(l, params)
        grads = grad_clipping(grads, 1)
        updater.apply_gradients(zip(grads, params))
        # Keras預設回傳一個批次中的平均損失
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

```{.python .input}
#@tab paddle
#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """訓練網路一個迭代週期（定義見第8章)"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 訓練損失之和,詞元數量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用隨機抽樣時初始化state
            state = net.begin_state(batch_size=X.shape[0])
        else:
            if isinstance(net, nn.Layer) and not isinstance(state, tuple):
                # state對於nn.GRU是個張量
                state.stop_gradient=True
            else:
                # state對於nn.LSTM或對於我們從零開始實現的模型是個張量
                for s in state:
                    s.stop_gradient=True
        y = paddle.reshape(Y.T,shape=[-1])
        X = paddle.to_tensor(X, place=device)
        y = paddle.to_tensor(y, place=device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y).mean()
        if isinstance(updater, paddle.optimizer.Optimizer):
            updater.clear_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因為已經呼叫了mean函式
            updater(batch_size=1)
        
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

[**迴圈神經網路模型的訓練函式既支援從零開始實現，
也可以使用高階API來實現。**]

```{.python .input}
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,  #@save
              use_random_iter=False):
    """訓練模型（定義見第8章）"""
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, gluon.Block):
        net.initialize(ctx=device, force_reinit=True,
                         init=init.Normal(0.01))
        trainer = gluon.Trainer(net.collect_params(),
                                'sgd', {'learning_rate': lr})
        updater = lambda batch_size: trainer.step(batch_size)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 訓練和預測
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 詞元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

```{.python .input}
#@tab pytorch
#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """訓練模型（定義見第8章）"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 訓練和預測
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 詞元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

```{.python .input}
#@tab tensorflow
#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, strategy,
              use_random_iter=False):
    """訓練模型（定義見第8章）"""
    with strategy.scope():
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        updater = tf.keras.optimizers.SGD(lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab)
    # 訓練和預測
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater,
                                     use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    device = d2l.try_gpu()._device_name
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 詞元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

```{.python .input}
#@tab paddle
#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """訓練模型（定義見第8章）"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Layer):
        updater = paddle.optimizer.SGD(
                learning_rate=lr, parameters=net.parameters())
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 訓練和預測
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 詞元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

[**現在，我們訓練迴圈神經網路模型。**]
因為我們在資料集中只使用了10000個詞元，
所以模型需要更多的迭代週期來更好地收斂。

```{.python .input}
#@tab mxnet,pytorch, paddle
num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
```

```{.python .input}
#@tab tensorflow
num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, strategy)
```

[**最後，讓我們檢查一下使用隨機抽樣方法的結果。**]

```{.python .input}
#@tab mxnet,pytorch
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True)
```

```{.python .input}
#@tab tensorflow
with strategy.scope():
    net = RNNModelScratch(len(vocab), num_hiddens, init_rnn_state, rnn,
                          get_params)
train_ch8(net, train_iter, vocab_random_iter, lr, num_epochs, strategy,
          use_random_iter=True)
```

```{.python .input}
#@tab paddle
net = RNNModelScratch(len(vocab), num_hiddens, get_params,
                      init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True)
```

從零開始實現上述迴圈神經網路模型，
雖然有指導意義，但是並不方便。
在下一節中，我們將學習如何改進迴圈神經網路模型。
例如，如何使其實現地更容易，且執行速度更快。

## 小結

* 我們可以訓練一個基於迴圈神經網路的字元級語言模型，根據使用者提供的文字的字首生成後續文字。
* 一個簡單的迴圈神經網絡語言模型包括輸入編碼、迴圈神經網路模型和輸出生成。
* 迴圈神經網路模型在訓練以前需要初始化狀態，不過隨機抽樣和順序劃分使用初始化方法不同。
* 當使用順序劃分時，我們需要分離梯度以減少計算量。
* 在進行任何預測之前，模型透過預熱期進行自我更新（例如，獲得比初始值更好的隱狀態）。
* 梯度裁剪可以防止梯度爆炸，但不能應對梯度消失。

## 練習

1. 嘗試說明獨熱編碼等價於為每個物件選擇不同的嵌入表示。
1. 透過調整超引數（如迭代週期數、隱藏單元數、小批次資料的時間步數、學習率等）來改善困惑度。
    * 困惑度可以降到多少？
    * 用可學習的嵌入表示替換獨熱編碼，是否會帶來更好的表現？
    * 如果用H.G.Wells的其他書作為資料集時效果如何，
      例如[*世界大戰*](http://www.gutenberg.org/ebooks/36)？
1. 修改預測函式，例如使用取樣，而不是選擇最有可能的下一個字元。
    * 會發生什麼？
    * 調整模型使之偏向更可能的輸出，例如，當$\alpha > 1$，從$q(x_t \mid x_{t-1}, \ldots, x_1) \propto P(x_t \mid x_{t-1}, \ldots, x_1)^\alpha$中取樣。
1. 在不裁剪梯度的情況下執行本節中的程式碼會發生什麼？
1. 更改順序劃分，使其不會從計算圖中分離隱狀態。執行時間會有變化嗎？困惑度呢？
1. 用ReLU替換本節中使用的啟用函式，並重複本節中的實驗。我們還需要梯度裁剪嗎？為什麼？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2102)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2103)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/2104)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11799)
:end_tab:
