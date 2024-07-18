# 門控迴圈單元（GRU）
:label:`sec_gru`

在 :numref:`sec_bptt`中，
我們討論瞭如何在迴圈神經網路中計算梯度，
以及矩陣連續乘積可以導致梯度消失或梯度爆炸的問題。
下面我們簡單思考一下這種梯度異常在實踐中的意義：

* 我們可能會遇到這樣的情況：早期觀測值對預測所有未來觀測值具有非常重要的意義。
  考慮一個極端情況，其中第一個觀測值包含一個校驗和，
  目標是在序列的末尾辨別校驗和是否正確。
  在這種情況下，第一個詞元的影響至關重要。
  我們希望有某些機制能夠在一個記憶元裡儲存重要的早期訊息。
  如果沒有這樣的機制，我們將不得不給這個觀測值指定一個非常大的梯度，
  因為它會影響所有後續的觀測值。
* 我們可能會遇到這樣的情況：一些詞元沒有相關的觀測值。
  例如，在對網頁內容進行情感分析時，
  可能有一些輔助HTML程式碼與網頁傳達的情緒無關。
  我們希望有一些機制來*跳過*隱狀態表示中的此類詞元。
* 我們可能會遇到這樣的情況：序列的各個部分之間存在邏輯中斷。
  例如，書的章節之間可能會有轉場存在，
  或者證券的熊市和牛市之間可能會有轉場存在。
  在這種情況下，最好有一種方法來*重置*我們的內部狀態表示。

在學術界已經提出了許多方法來解決這類問題。
其中最早的方法是"長短期記憶"（long-short-term memory，LSTM）
 :cite:`Hochreiter.Schmidhuber.1997`，
我們將在 :numref:`sec_lstm`中討論。
門控迴圈單元（gated recurrent unit，GRU）
 :cite:`Cho.Van-Merrienboer.Bahdanau.ea.2014`
是一個稍微簡化的變體，通常能夠提供同等的效果，
並且計算 :cite:`Chung.Gulcehre.Cho.ea.2014`的速度明顯更快。
由於門控迴圈單元更簡單，我們從它開始解讀。

## 門控隱狀態

門控迴圈單元與普通的迴圈神經網路之間的關鍵區別在於：
前者支援隱狀態的門控。
這意味著模型有專門的機制來確定應該何時更新隱狀態，
以及應該何時重置隱狀態。
這些機制是可學習的，並且能夠解決了上面列出的問題。
例如，如果第一個詞元非常重要，
模型將學會在第一次觀測之後不更新隱狀態。
同樣，模型也可以學會跳過不相關的臨時觀測。
最後，模型還將學會在需要的時候重置隱狀態。
下面我們將詳細討論各類門控。

### 重置門和更新門

我們首先介紹*重置門*（reset gate）和*更新門*（update gate）。
我們把它們設計成$(0, 1)$區間中的向量，
這樣我們就可以進行凸組合。
重置門允許我們控制『可能還想記住』的過去狀態的數量；
更新門將允許我們控制新狀態中有多少個是舊狀態的副本。

我們從構造這些門控開始。 :numref:`fig_gru_1`
描述了門控迴圈單元中的重置門和更新門的輸入，
輸入是由當前時間步的輸入和前一時間步的隱狀態給出。
兩個門的輸出是由使用sigmoid啟用函式的兩個全連線層給出。

![在門控迴圈單元模型中計算重置門和更新門](../img/gru-1.svg)
:label:`fig_gru_1`

我們來看一下門控迴圈單元的數學表達。
對於給定的時間步$t$，假設輸入是一個小批次
$\mathbf{X}_t \in \mathbb{R}^{n \times d}$
（樣本個數$n$，輸入個數$d$），
上一個時間步的隱狀態是
$\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$
（隱藏單元個數$h$）。
那麼，重置門$\mathbf{R}_t \in \mathbb{R}^{n \times h}$和
更新門$\mathbf{Z}_t \in \mathbb{R}^{n \times h}$的計算如下所示：

$$
\begin{aligned}
\mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xr} + \mathbf{H}_{t-1} \mathbf{W}_{hr} + \mathbf{b}_r),\\
\mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xz} + \mathbf{H}_{t-1} \mathbf{W}_{hz} + \mathbf{b}_z),
\end{aligned}
$$

其中$\mathbf{W}_{xr}, \mathbf{W}_{xz} \in \mathbb{R}^{d \times h}$
和$\mathbf{W}_{hr}, \mathbf{W}_{hz} \in \mathbb{R}^{h \times h}$是權重引數，
$\mathbf{b}_r, \mathbf{b}_z \in \mathbb{R}^{1 \times h}$是偏置引數。
請注意，在求和過程中會觸發廣播機制
（請參閱 :numref:`subsec_broadcasting`）。
我們使用sigmoid函式（如 :numref:`sec_mlp`中介紹的）
將輸入值轉換到區間$(0, 1)$。

### 候選隱狀態

接下來，讓我們將重置門$\mathbf{R}_t$
與 :eqref:`rnn_h_with_state`
中的常規隱狀態更新機制整合，
得到在時間步$t$的*候選隱狀態*（candidate hidden state）
$\tilde{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$。

$$\tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{xh} + \left(\mathbf{R}_t \odot \mathbf{H}_{t-1}\right) \mathbf{W}_{hh} + \mathbf{b}_h),$$
:eqlabel:`gru_tilde_H`

其中$\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}$
和$\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$是權重引數，
$\mathbf{b}_h \in \mathbb{R}^{1 \times h}$是偏置項，
符號$\odot$是Hadamard積（按元素乘積）運算符。
在這裡，我們使用tanh非線性啟用函式來確保候選隱狀態中的值保持在區間$(-1, 1)$中。

與 :eqref:`rnn_h_with_state`相比，
 :eqref:`gru_tilde_H`中的$\mathbf{R}_t$和$\mathbf{H}_{t-1}$
的元素相乘可以減少以往狀態的影響。
每當重置門$\mathbf{R}_t$中的項接近$1$時，
我們恢復一個如 :eqref:`rnn_h_with_state`中的普通的迴圈神經網路。
對於重置門$\mathbf{R}_t$中所有接近$0$的項，
候選隱狀態是以$\mathbf{X}_t$作為輸入的多層感知機的結果。
因此，任何預先存在的隱狀態都會被*重置*為預設值。

 :numref:`fig_gru_2`說明了應用重置門之後的計算流程。

![在門控迴圈單元模型中計算候選隱狀態](../img/gru-2.svg)
:label:`fig_gru_2`

### 隱狀態

上述的計算結果只是候選隱狀態，我們仍然需要結合更新門$\mathbf{Z}_t$的效果。
這一步確定新的隱狀態$\mathbf{H}_t \in \mathbb{R}^{n \times h}$
在多大程度上來自舊的狀態$\mathbf{H}_{t-1}$和
新的候選狀態$\tilde{\mathbf{H}}_t$。
更新門$\mathbf{Z}_t$僅需要在
$\mathbf{H}_{t-1}$和$\tilde{\mathbf{H}}_t$
之間進行按元素的凸組合就可以實現這個目標。
這就得出了門控迴圈單元的最終更新公式：

$$\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1}  + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t.$$

每當更新門$\mathbf{Z}_t$接近$1$時，模型就傾向只保留舊狀態。
此時，來自$\mathbf{X}_t$的訊息基本上被忽略，
從而有效地跳過了依賴鏈條中的時間步$t$。
相反，當$\mathbf{Z}_t$接近$0$時，
新的隱狀態$\mathbf{H}_t$就會接近候選隱狀態$\tilde{\mathbf{H}}_t$。
這些設計可以幫助我們處理迴圈神經網路中的梯度消失問題，
並更好地捕獲時間步距離很長的序列的依賴關係。
例如，如果整個子序列的所有時間步的更新門都接近於$1$，
則無論序列的長度如何，在序列起始時間步的舊隱狀態都將很容易保留並傳遞到序列結束。

 :numref:`fig_gru_3`說明了更新門起作用後的計算流。

![計算門控迴圈單元模型中的隱狀態](../img/gru-3.svg)
:label:`fig_gru_3`

總之，門控迴圈單元具有以下兩個顯著特徵：

* 重置門有助於捕獲序列中的短期依賴關係；
* 更新門有助於捕獲序列中的長期依賴關係。

## 從零開始實現

為了更好地理解門控迴圈單元模型，我們從零開始實現它。
首先，我們讀取 :numref:`sec_rnn_scratch`中使用的時間機器資料集：

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import rnn
npx.set_np()

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
import paddle.nn.functional as F
from paddle import nn

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

### [**初始化模型引數**]

下一步是初始化模型引數。
我們從標準差為$0.01$的高斯分佈中提取權重，
並將偏置項設為$0$，超引數`num_hiddens`定義隱藏單元的數量，
實例化與更新門、重置門、候選隱狀態和輸出層相關的所有權重和偏置。

```{.python .input}
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape, ctx=device)

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                np.zeros(num_hiddens, ctx=device))

    W_xz, W_hz, b_z = three()  # 更新門引數
    W_xr, W_hr, b_r = three()  # 重置門引數
    W_xh, W_hh, b_h = three()  # 候選隱狀態參數
    # 輸出層引數
    W_hq = normal((num_hiddens, num_outputs))
    b_q = np.zeros(num_outputs, ctx=device)
    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params
```

```{.python .input}
#@tab pytorch
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                d2l.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # 更新門引數
    W_xr, W_hr, b_r = three()  # 重置門引數
    W_xh, W_hh, b_h = three()  # 候選隱狀態參數
    # 輸出層引數
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
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

    def three():
        return (tf.Variable(normal((num_inputs, num_hiddens)), dtype=tf.float32),
                tf.Variable(normal((num_hiddens, num_hiddens)), dtype=tf.float32),
                tf.Variable(d2l.zeros(num_hiddens), dtype=tf.float32))

    W_xz, W_hz, b_z = three()  # 更新門引數
    W_xr, W_hr, b_r = three()  # 重置門引數
    W_xh, W_hh, b_h = three()  # 候選隱狀態參數
    # 輸出層引數
    W_hq = tf.Variable(normal((num_hiddens, num_outputs)), dtype=tf.float32)
    b_q = tf.Variable(d2l.zeros(num_outputs), dtype=tf.float32)
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    return params
```

```{.python .input}
#@tab paddle
def get_params(vocab_size, num_hiddens):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return paddle.randn(shape=shape)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                paddle.zeros([num_hiddens]))

    W_xz, W_hz, b_z = three()  # 更新門引數
    W_xr, W_hr, b_r = three()  # 重置門引數
    W_xh, W_hh, b_h = three()  # 候選隱狀態參數
    # 輸出層引數
    W_hq = normal((num_hiddens, num_outputs))
    b_q = paddle.zeros([num_outputs])
    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.stop_gradient = False
    return params
```

### 定義模型

現在我們將[**定義隱狀態的初始化函式**]`init_gru_state`。
與 :numref:`sec_rnn_scratch`中定義的`init_rnn_state`函式一樣，
此函式回傳一個形狀為（批次大小，隱藏單元個數）的張量，張量的值全部為零。

```{.python .input}
def init_gru_state(batch_size, num_hiddens, device):
    return (np.zeros(shape=(batch_size, num_hiddens), ctx=device), )
```

```{.python .input}
#@tab pytorch
def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )
```

```{.python .input}
#@tab tensorflow
def init_gru_state(batch_size, num_hiddens):
    return (d2l.zeros((batch_size, num_hiddens)), )
```

```{.python .input}
#@tab paddle
def init_gru_state(batch_size, num_hiddens):
    return (paddle.zeros([batch_size, num_hiddens]), )
```

現在我們準備[**定義門控迴圈單元模型**]，
模型的架構與基本的迴圈神經網路單元是相同的，
只是權重更新公式更為複雜。

```{.python .input}
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = npx.sigmoid(np.dot(X, W_xz) + np.dot(H, W_hz) + b_z)
        R = npx.sigmoid(np.dot(X, W_xr) + np.dot(H, W_hr) + b_r)
        H_tilda = np.tanh(np.dot(X, W_xh) + np.dot(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H,)
```

```{.python .input}
#@tab pytorch
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
```

```{.python .input}
#@tab tensorflow
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        X = tf.reshape(X,[-1,W_xh.shape[0]])
        Z = tf.sigmoid(tf.matmul(X, W_xz) + tf.matmul(H, W_hz) + b_z)
        R = tf.sigmoid(tf.matmul(X, W_xr) + tf.matmul(H, W_hr) + b_r)
        H_tilda = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return tf.concat(outputs, axis=0), (H,)
```

```{.python .input}
#@tab paddle
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H,*_ = state
    outputs = []
    for X in inputs:
        Z = F.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = F.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = paddle.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return paddle.concat(outputs, axis=0), (H,*_)
```

### [**訓練**]與預測

訓練和預測的工作方式與 :numref:`sec_rnn_scratch`完全相同。
訓練結束後，我們分別列印輸出訓練集的困惑度，
以及字首『time traveler』和『traveler』的預測序列上的困惑度。

```{.python .input}
#@tab mxnet, pytorch
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                            init_gru_state, gru)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab tensorflow
vocab_size, num_hiddens, device_name = len(vocab), 256, d2l.try_gpu()._device_name
# 定義訓練策略
strategy = tf.distribute.OneDeviceStrategy(device_name)
num_epochs, lr = 500, 1
with strategy.scope():
    model = d2l.RNNModelScratch(len(vocab), num_hiddens, init_gru_state, gru, get_params)

d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, strategy)
```

```{.python .input}
#@tab paddle
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1.0
model = d2l.RNNModelScratch(len(vocab), num_hiddens, get_params,
                            init_gru_state, gru)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

## [**簡潔實現**]

高階API包含了前文介紹的所有設定細節，
所以我們可以直接實例化門控迴圈單元模型。
這段程式碼的執行速度要快得多，
因為它使用的是編譯好的運算符而不是Python來處理之前闡述的許多細節。

```{.python .input}
gru_layer = rnn.GRU(num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab pytorch
num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab tensorflow
gru_cell = tf.keras.layers.GRUCell(num_hiddens,
    kernel_initializer='glorot_uniform')
gru_layer = tf.keras.layers.RNN(gru_cell, time_major=True,
    return_sequences=True, return_state=True)

device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)
with strategy.scope():
    model = d2l.RNNModel(gru_layer, vocab_size=len(vocab))

d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, strategy)
```

```{.python .input}
#@tab paddle
num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens, time_major=True)
model = d2l.RNNModel(gru_layer, len(vocab))
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

## 小結

* 門控迴圈神經網路可以更好地捕獲時間步距離很長的序列上的依賴關係。
* 重置門有助於捕獲序列中的短期依賴關係。
* 更新門有助於捕獲序列中的長期依賴關係。
* 重置門開啟時，門控迴圈單元包含基本迴圈神經網路；更新門開啟時，門控迴圈單元可以跳過子序列。

## 練習

1. 假設我們只想使用時間步$t'$的輸入來預測時間步$t > t'$的輸出。對於每個時間步，重置門和更新門的最佳值是什麼？
1. 調整和分析超引數對執行時間、困惑度和輸出順序的影響。
1. 比較`rnn.RNN`和`rnn.GRU`的不同實現對執行時間、困惑度和輸出字串的影響。
1. 如果僅僅實現門控迴圈單元的一部分，例如，只有一個重置門或一個更新門會怎樣？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2764)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2763)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11812)
:end_tab: