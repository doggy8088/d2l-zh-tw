# 迴圈神經網路的簡潔實現
:label:`sec_rnn-concise`

雖然 :numref:`sec_rnn_scratch`
對了解迴圈神經網路的實現方式具有指導意義，但並不方便。
本節將展示如何使用深度學習框架的高階API提供的函式更有效地實現相同的語言模型。
我們仍然從讀取時光機器資料集開始。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn, rnn
npx.set_np()

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

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
from paddle.nn import functional as F

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

## [**定義模型**]

高階API提供了迴圈神經網路的實現。
我們構造一個具有256個隱藏單元的單隱藏層的迴圈神經網路層`rnn_layer`。
事實上，我們還沒有討論多層迴圈神經網路的意義（這將在 :numref:`sec_deep_rnn`中介紹）。
現在僅需要將多層理解為一層迴圈神經網路的輸出被用作下一層迴圈神經網路的輸入就足夠了。

```{.python .input}
num_hiddens = 256
rnn_layer = rnn.RNN(num_hiddens)
rnn_layer.initialize()
```

```{.python .input}
#@tab pytorch
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)
```

```{.python .input}
#@tab tensorflow
num_hiddens = 256
rnn_cell = tf.keras.layers.SimpleRNNCell(num_hiddens,
    kernel_initializer='glorot_uniform')
rnn_layer = tf.keras.layers.RNN(rnn_cell, time_major=True,
    return_sequences=True, return_state=True)
```

```{.python .input}
#@tab paddle
num_hiddens = 256
rnn_layer = nn.SimpleRNN(len(vocab), num_hiddens, time_major=True)
```

:begin_tab:`mxnet`
初始化隱狀態是簡單的，只需要呼叫成員函式`begin_state`即可。
函式將回傳一個清單（`state`），清單中包含了初始隱狀態用於小批次資料中的每個樣本，
其形狀為（隱藏層數，批次大小，隱藏單元數）。
對於以後要介紹的一些模型（例如長-短期記憶網路），這樣的清單還會包含其他訊息。
:end_tab:

:begin_tab:`pytorch`
我們(**使用張量來初始化隱狀態**)，它的形狀是（隱藏層數，批次大小，隱藏單元數）。
:end_tab:

```{.python .input}
state = rnn_layer.begin_state(batch_size=batch_size)
len(state), state[0].shape
```

```{.python .input}
#@tab pytorch
state = torch.zeros((1, batch_size, num_hiddens))
state.shape
```

```{.python .input}
#@tab tensorflow
state = rnn_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
state.shape
```

```{.python .input}
#@tab paddle
state = paddle.zeros(shape=[1, batch_size, num_hiddens])
state.shape
```

[**透過一個隱狀態和一個輸入，我們就可以用更新後的隱狀態計算輸出。**]
需要強調的是，`rnn_layer`的『輸出』（`Y`）不涉及輸出層的計算：
它是指每個時間步的隱狀態，這些隱狀態可以用作後續輸出層的輸入。

:begin_tab:`mxnet`
此外，`rnn_layer`回傳的更新後的隱狀態（`state_new`）
是指小批次資料的最後時間步的隱狀態。
這個隱狀態可以用來初始化順序分割槽中一個迭代週期內下一個小批次資料的隱狀態。
對於多個隱藏層，每一層的隱狀態將儲存在（`state_new`）變數中。
至於稍後要介紹的某些模型（例如，長－短期記憶），此變數還包含其他訊息。
:end_tab:

```{.python .input}
X = np.random.uniform(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, len(state_new), state_new[0].shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, state_new.shape
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, len(state_new), state_new[0].shape
```

```{.python .input}
#@tab paddle
X = paddle.rand(shape=[num_steps, batch_size, len(vocab)])
Y, state_new = rnn_layer(X, state)
Y.shape, state_new.shape
```

與 :numref:`sec_rnn_scratch`類似，
[**我們為一個完整的迴圈神經網路模型定義了一個`RNNModel`類**]。
注意，`rnn_layer`只包含隱藏的迴圈層，我們還需要建立一個單獨的輸出層。

```{.python .input}
#@save
class RNNModel(nn.Block):
    """迴圈神經網路模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        X = npx.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        # 全連線層首先將Y的形狀改為(時間步數*批次大小,隱藏單元數)
        # 它的輸出形狀是(時間步數*批次大小,詞表大小)
        output = self.dense(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```

```{.python .input}
#@tab pytorch
#@save
class RNNModel(nn.Module):
    """迴圈神經網路模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是雙向的（之後將介紹），num_directions應該是2，否則應該是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全連線層首先將Y的形狀改為(時間步數*批次大小,隱藏單元數)
        # 它的輸出形狀是(時間步數*批次大小,詞表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以張量作為隱狀態
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens), 
                                device=device)
        else:
            # nn.LSTM以元組作為隱狀態
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))
```

```{.python .input}
#@tab tensorflow
#@save
class RNNModel(tf.keras.layers.Layer):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, state):
        X = tf.one_hot(tf.transpose(inputs), self.vocab_size)
        # rnn回傳兩個以上的值
        Y, *state = self.rnn(X, state)
        output = self.dense(tf.reshape(Y, (-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.cell.get_initial_state(*args, **kwargs)
```

```{.python .input}
#@tab paddle
#@save
class RNNModel(nn.Layer):   
    """迴圈神經網路模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是雙向的（之後將介紹），num_directions應該是2，否則應該是1
        if self.rnn.num_directions==1:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T, self.vocab_size) 
        Y, state = self.rnn(X, state)
        # 全連線層首先將Y的形狀改為(時間步數*批次大小,隱藏單元數)
        # 它的輸出形狀是(時間步數*批次大小,詞表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以張量作為隱狀態
            return  paddle.zeros(shape=[self.num_directions * self.rnn.num_layers,
                                                           batch_size, self.num_hiddens])
        else:
            # nn.LSTM以元組作為隱狀態
            return (paddle.zeros(
                shape=[self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens]),
                    paddle.zeros(
                        shape=[self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens]))
```

## 訓練與預測

在訓練模型之前，讓我們[**基於一個具有隨機權重的模型進行預測**]。

```{.python .input}
device = d2l.try_gpu()
net = RNNModel(rnn_layer, len(vocab))
net.initialize(force_reinit=True, ctx=device)
d2l.predict_ch8('time traveller', 10, net, vocab, device)
```

```{.python .input}
#@tab pytorch
device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
d2l.predict_ch8('time traveller', 10, net, vocab, device)
```

```{.python .input}
#@tab tensorflow
device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)
with strategy.scope():
    net = RNNModel(rnn_layer, vocab_size=len(vocab))

d2l.predict_ch8('time traveller', 10, net, vocab)
```

```{.python .input}
#@tab paddle
device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
d2l.predict_ch8('time traveller', 10, net, vocab, device)
```

很明顯，這種模型根本不能輸出好的結果。
接下來，我們使用 :numref:`sec_rnn_scratch`中
定義的超引數呼叫`train_ch8`，並且[**使用高階API訓練模型**]。

```{.python .input}
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab pytorch
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab tensorflow
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, strategy)
```

```{.python .input}
#@tab paddle
num_epochs, lr = 500, 1.0
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
```

與上一節相比，由於深度學習框架的高階API對程式碼進行了更多的最佳化，
該模型在較短的時間內達到了較低的困惑度。

## 小結

* 深度學習框架的高階API提供了迴圈神經網路層的實現。
* 高階API的迴圈神經網路層回傳一個輸出和一個更新後的隱狀態，我們還需要計算整個模型的輸出層。
* 相比從零開始實現的迴圈神經網路，使用高階API實現可以加速訓練。

## 練習

1. 嘗試使用高階API，能使迴圈神經網路模型過擬合嗎？
1. 如果在迴圈神經網路模型中增加隱藏層的數量會發生什麼？能使模型正常工作嗎？
1. 嘗試使用迴圈神經網路實現 :numref:`sec_sequence`的自迴歸模型。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2105)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2106)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/5766)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11800)
:end_tab:
