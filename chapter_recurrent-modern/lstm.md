# 長短期記憶網路（LSTM）
:label:`sec_lstm`

長期以來，隱變數模型存在著長期訊息儲存和短期輸入缺失的問題。
解決這一問題的最早方法之一是長短期儲存器（long short-term memory，LSTM）
 :cite:`Hochreiter.Schmidhuber.1997`。
它有許多與門控迴圈單元（ :numref:`sec_gru`）一樣的屬性。
有趣的是，長短期記憶網路的設計比門控迴圈單元稍微複雜一些，
卻比門控迴圈單元早誕生了近20年。

## 門控記憶元

可以說，長短期記憶網路的設計靈感來自於電腦的邏輯門。
長短期記憶網路引入了*記憶元*（memory cell），或簡稱為*單元*（cell）。
有些文獻認為記憶元是隱狀態的一種特殊型別，
它們與隱狀態具有相同的形狀，其設計目的是用於記錄附加的訊息。
為了控制記憶元，我們需要許多門。
其中一個門用來從單元中輸出條目，我們將其稱為*輸出門*（output gate）。
另外一個門用來決定何時將資料讀入單元，我們將其稱為*輸入門*（input gate）。
我們還需要一種機制來重置單元的內容，由*遺忘門*（forget gate）來管理，
這種設計的動機與門控迴圈單元相同，
能夠透過專用機制決定什麼時候記憶或忽略隱狀態中的輸入。
讓我們看看這在實踐中是如何運作的。

### 輸入門、忘記門和輸出門

就如在門控迴圈單元中一樣，
當前時間步的輸入和前一個時間步的隱狀態
作為資料送入長短期記憶網路的門中，
如 :numref:`lstm_0`所示。
它們由三個具有sigmoid啟用函式的全連線層處理，
以計算輸入門、遺忘門和輸出門的值。
因此，這三個門的值都在$(0, 1)$的範圍內。

![長短期記憶模型中的輸入門、遺忘門和輸出門](../img/lstm-0.svg)
:label:`lstm_0`

我們來細化一下長短期記憶網路的數學表達。
假設有$h$個隱藏單元，批次大小為$n$，輸入數為$d$。
因此，輸入為$\mathbf{X}_t \in \mathbb{R}^{n \times d}$，
前一時間步的隱狀態為$\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$。
相應地，時間步$t$的門被定義如下：
輸入門是$\mathbf{I}_t \in \mathbb{R}^{n \times h}$，
遺忘門是$\mathbf{F}_t \in \mathbb{R}^{n \times h}$，
輸出門是$\mathbf{O}_t \in \mathbb{R}^{n \times h}$。
它們的計算方法如下：

$$
\begin{aligned}
\mathbf{I}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xi} + \mathbf{H}_{t-1} \mathbf{W}_{hi} + \mathbf{b}_i),\\
\mathbf{F}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xf} + \mathbf{H}_{t-1} \mathbf{W}_{hf} + \mathbf{b}_f),\\
\mathbf{O}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xo} + \mathbf{H}_{t-1} \mathbf{W}_{ho} + \mathbf{b}_o),
\end{aligned}
$$

其中$\mathbf{W}_{xi}, \mathbf{W}_{xf}, \mathbf{W}_{xo} \in \mathbb{R}^{d \times h}$
和$\mathbf{W}_{hi}, \mathbf{W}_{hf}, \mathbf{W}_{ho} \in \mathbb{R}^{h \times h}$是權重引數，
$\mathbf{b}_i, \mathbf{b}_f, \mathbf{b}_o \in \mathbb{R}^{1 \times h}$是偏置引數。

### 候選記憶元

由於還沒有指定各種門的操作，所以先介紹*候選記憶元*（candidate memory cell）
$\tilde{\mathbf{C}}_t \in \mathbb{R}^{n \times h}$。
它的計算與上面描述的三個門的計算類似，
但是使用$\tanh$函式作為啟用函式，函式的值範圍為$(-1, 1)$。
下面匯出在時間步$t$處的方程：

$$\tilde{\mathbf{C}}_t = \text{tanh}(\mathbf{X}_t \mathbf{W}_{xc} + \mathbf{H}_{t-1} \mathbf{W}_{hc} + \mathbf{b}_c),$$

其中$\mathbf{W}_{xc} \in \mathbb{R}^{d \times h}$和
$\mathbf{W}_{hc} \in \mathbb{R}^{h \times h}$是權重引數，
$\mathbf{b}_c \in \mathbb{R}^{1 \times h}$是偏置引數。

候選記憶元的如 :numref:`lstm_1`所示。

![長短期記憶模型中的候選記憶元](../img/lstm-1.svg)
:label:`lstm_1`

### 記憶元

在門控迴圈單元中，有一種機制來控制輸入和遺忘（或跳過）。
類似地，在長短期記憶網路中，也有兩個門用於這樣的目的：
輸入門$\mathbf{I}_t$控制採用多少來自$\tilde{\mathbf{C}}_t$的新資料，
而遺忘門$\mathbf{F}_t$控制保留多少過去的
記憶元$\mathbf{C}_{t-1} \in \mathbb{R}^{n \times h}$的內容。
使用按元素乘法，得出：

$$\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t.$$

如果遺忘門始終為$1$且輸入門始終為$0$，
則過去的記憶元$\mathbf{C}_{t-1}$
將隨時間被儲存並傳遞到當前時間步。
引入這種設計是為了緩解梯度消失問題，
並更好地捕獲序列中的長距離依賴關係。

這樣我們就得到了計算記憶元的流程圖，如 :numref:`lstm_2`。

![在長短期記憶網路模型中計算記憶元](../img/lstm-2.svg)

:label:`lstm_2`

### 隱狀態

最後，我們需要定義如何計算隱狀態
$\mathbf{H}_t \in \mathbb{R}^{n \times h}$，
這就是輸出門發揮作用的地方。
在長短期記憶網路中，它僅僅是記憶元的$\tanh$的門控版本。
這就確保了$\mathbf{H}_t$的值始終在區間$(-1, 1)$內：

$$\mathbf{H}_t = \mathbf{O}_t \odot \tanh(\mathbf{C}_t).$$

只要輸出門接近$1$，我們就能夠有效地將所有記憶訊息傳遞給預測部分，
而對於輸出門接近$0$，我們只保留記憶元內的所有訊息，而不需要更新隱狀態。

 :numref:`lstm_3`提供了資料流的圖形化示範。

![在長短期記憶模型中計算隱狀態](../img/lstm-3.svg)
:label:`lstm_3`

## 從零開始實現

現在，我們從零開始實現長短期記憶網路。
與 :numref:`sec_rnn_scratch`中的實驗相同，
我們首先載入時光機器資料集。

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
from paddle import nn
import paddle.nn.functional as Function

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

### [**初始化模型引數**]

接下來，我們需要定義和初始化模型引數。
如前所述，超引數`num_hiddens`定義隱藏單元的數量。
我們按照標準差$0.01$的高斯分佈初始化權重，並將偏置項設為$0$。

```{.python .input}
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape, ctx=device)

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                np.zeros(num_hiddens, ctx=device))

    W_xi, W_hi, b_i = three()  # 輸入門引數
    W_xf, W_hf, b_f = three()  # 遺忘門引數
    W_xo, W_ho, b_o = three()  # 輸出門引數
    W_xc, W_hc, b_c = three()  # 候選記憶元引數
    # 輸出層引數
    W_hq = normal((num_hiddens, num_outputs))
    b_q = np.zeros(num_outputs, ctx=device)
    # 附加梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params
```

```{.python .input}
#@tab pytorch
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                d2l.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = three()  # 輸入門引數
    W_xf, W_hf, b_f = three()  # 遺忘門引數
    W_xo, W_ho, b_o = three()  # 輸出門引數
    W_xc, W_hc, b_c = three()  # 候選記憶元引數
    # 輸出層引數
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

```{.python .input}
#@tab tensorflow
def get_lstm_params(vocab_size, num_hiddens):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return tf.Variable(tf.random.normal(shape=shape, stddev=0.01,
                                            mean=0, dtype=tf.float32))
    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                tf.Variable(tf.zeros(num_hiddens), dtype=tf.float32))

    W_xi, W_hi, b_i = three()  # 輸入門引數
    W_xf, W_hf, b_f = three()  # 遺忘門引數
    W_xo, W_ho, b_o = three()  # 輸出門引數
    W_xc, W_hc, b_c = three()  # 候選記憶元引數
    # 輸出層引數
    W_hq = normal((num_hiddens, num_outputs))
    b_q = tf.Variable(tf.zeros(num_outputs), dtype=tf.float32)
    # 附加梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    return params
```

```{.python .input}
#@tab paddle
def get_lstm_params(vocab_size, num_hiddens):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return paddle.randn(shape=shape)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                d2l.zeros([num_hiddens]))

    W_xi, W_hi, b_i = three()  # 輸入門引數
    W_xf, W_hf, b_f = three()  # 遺忘門引數
    W_xo, W_ho, b_o = three()  # 輸出門引數
    W_xc, W_hc, b_c = three()  # 候選記憶元引數
    # 輸出層引數
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros([num_outputs])
    # 附加梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.stop_gradient = False
    return params
```

### 定義模型

在[**初始化函式**]中，
長短期記憶網路的隱狀態需要回傳一個*額外*的記憶元，
單元的值為0，形狀為（批次大小，隱藏單元數）。
因此，我們得到以下的狀態初始化。

```{.python .input}
def init_lstm_state(batch_size, num_hiddens, device):
    return (np.zeros((batch_size, num_hiddens), ctx=device),
            np.zeros((batch_size, num_hiddens), ctx=device))
```

```{.python .input}
#@tab pytorch
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))
```

```{.python .input}
#@tab tensorflow
def init_lstm_state(batch_size, num_hiddens):
    return (tf.zeros(shape=(batch_size, num_hiddens)),
            tf.zeros(shape=(batch_size, num_hiddens)))
```

```{.python .input}
#@tab paddle
def init_lstm_state(batch_size, num_hiddens):
    return (paddle.zeros([batch_size, num_hiddens]),
            paddle.zeros([batch_size, num_hiddens]))
```

[**實際模型**]的定義與我們前面討論的一樣：
提供三個門和一個額外的記憶元。
請注意，只有隱狀態才會傳遞到輸出層，
而記憶元$\mathbf{C}_t$不直接參與輸出計算。

```{.python .input}
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = npx.sigmoid(np.dot(X, W_xi) + np.dot(H, W_hi) + b_i)
        F = npx.sigmoid(np.dot(X, W_xf) + np.dot(H, W_hf) + b_f)
        O = npx.sigmoid(np.dot(X, W_xo) + np.dot(H, W_ho) + b_o)
        C_tilda = np.tanh(np.dot(X, W_xc) + np.dot(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * np.tanh(C)
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H, C)
```

```{.python .input}
#@tab pytorch
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)
```

```{.python .input}
#@tab tensorflow
def lstm(inputs, state, params):
    W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = params
    (H, C) = state
    outputs = []
    for X in inputs:
        X=tf.reshape(X,[-1,W_xi.shape[0]])
        I = tf.sigmoid(tf.matmul(X, W_xi) + tf.matmul(H, W_hi) + b_i)
        F = tf.sigmoid(tf.matmul(X, W_xf) + tf.matmul(H, W_hf) + b_f)
        O = tf.sigmoid(tf.matmul(X, W_xo) + tf.matmul(H, W_ho) + b_o)
        C_tilda = tf.tanh(tf.matmul(X, W_xc) + tf.matmul(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * tf.tanh(C)
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return tf.concat(outputs, axis=0), (H,C)
```

```{.python .input}
#@tab paddle
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = Function.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = Function.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = Function.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = paddle.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * paddle.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return paddle.concat(outputs, axis=0), (H, C)
```

### [**訓練**]和預測

讓我們透過實例化 :numref:`sec_rnn_scratch`中
引入的`RNNModelScratch`類來訓練一個長短期記憶網路，
就如我們在 :numref:`sec_gru`中所做的一樣。

```{.python .input}
#@tab mxnet, pytorch
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
                            init_lstm_state, lstm)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab tensorflow
vocab_size, num_hiddens, device_name = len(vocab), 256, d2l.try_gpu()._device_name
num_epochs, lr = 500, 1
strategy = tf.distribute.OneDeviceStrategy(device_name)
with strategy.scope():
    model = d2l.RNNModelScratch(len(vocab), num_hiddens, init_lstm_state, lstm, get_lstm_params)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, strategy)
```

```{.python .input}
#@tab paddle
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1.0
model = d2l.RNNModelScratch(len(vocab), num_hiddens, get_lstm_params,
                            init_lstm_state, lstm)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

## [**簡潔實現**]

使用高階API，我們可以直接實例化`LSTM`模型。
高階API封裝了前文介紹的所有設定細節。
這段程式碼的執行速度要快得多，
因為它使用的是編譯好的運算符而不是Python來處理之前闡述的許多細節。

```{.python .input}
lstm_layer = rnn.LSTM(num_hiddens)
model = d2l.RNNModel(lstm_layer, len(vocab))
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab pytorch
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab tensorflow
lstm_cell = tf.keras.layers.LSTMCell(num_hiddens,
    kernel_initializer='glorot_uniform')
lstm_layer = tf.keras.layers.RNN(lstm_cell, time_major=True,
    return_sequences=True, return_state=True)
device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)
with strategy.scope():
    model = d2l.RNNModel(lstm_layer, vocab_size=len(vocab))
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, strategy)
```

```{.python .input}
#@tab paddle
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens, time_major=True)
model = d2l.RNNModel(lstm_layer, len(vocab))
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

長短期記憶網路是典型的具有重要狀態控制的隱變數自迴歸模型。
多年來已經提出了其許多變體，例如，多層、殘差連線、不同型別的正則化。
然而，由於序列的長距離依賴性，訓練長短期記憶網路
和其他序列模型（例如門控迴圈單元）的成本是相當高的。
在後麵的內容中，我們將講述更高階的替代模型，如Transformer。

## 小結

* 長短期記憶網路有三種型別的門：輸入門、遺忘門和輸出門。
* 長短期記憶網路的隱藏層輸出包括『隱狀態』和『記憶元』。只有隱狀態會傳遞到輸出層，而記憶元完全屬於內部訊息。
* 長短期記憶網路可以緩解梯度消失和梯度爆炸。


## 練習

1. 調整和分析超引數對執行時間、困惑度和輸出順序的影響。
1. 如何更改模型以生成適當的單詞，而不是字元序列？
1. 在給定隱藏層維度的情況下，比較門控迴圈單元、長短期記憶網路和常規迴圈神經網路的計算成本。要特別注意訓練和推斷成本。
1. 既然候選記憶元透過使用$\tanh$函式來確保值範圍在$(-1,1)$之間，那麼為什麼隱狀態需要再次使用$\tanh$函式來確保輸出值範圍在$(-1,1)$之間呢？
1. 實現一個能夠基於時間序列進行預測而不是基於字元序列進行預測的長短期記憶網路模型。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2766)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2768)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11833)
:end_tab: