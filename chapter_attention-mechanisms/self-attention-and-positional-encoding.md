# 自注意力和位置編碼
:label:`sec_self-attention-and-positional-encoding`

在深度學習中，經常使用卷積神經網路（CNN）或迴圈神經網路（RNN）對序列進行編碼。
想象一下，有了注意力機制之後，我們將詞元序列輸入注意力池化中，
以便同一組詞元同時充當查詢、鍵和值。
具體來說，每個查詢都會關注所有的鍵－值對並生成一個注意力輸出。
由於查詢、鍵和值來自同一組輸入，因此被稱為
*自注意力*（self-attention）
 :cite:`Lin.Feng.Santos.ea.2017,Vaswani.Shazeer.Parmar.ea.2017`，
也被稱為*內部注意力*（intra-attention） :cite:`Cheng.Dong.Lapata.2016,Parikh.Tackstrom.Das.ea.2016,Paulus.Xiong.Socher.2017`。
本節將使用自注意力進行序列編碼，以及如何使用序列的順序作為補充訊息。

```{.python .input}
from d2l import mxnet as d2l
import math
from mxnet import autograd, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import math
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
```

## [**自注意力**]

給定一個由詞元組成的輸入序列$\mathbf{x}_1, \ldots, \mathbf{x}_n$，
其中任意$\mathbf{x}_i \in \mathbb{R}^d$（$1 \leq i \leq n$）。
該序列的自注意力輸出為一個長度相同的序列
$\mathbf{y}_1, \ldots, \mathbf{y}_n$，其中：

$$\mathbf{y}_i = f(\mathbf{x}_i, (\mathbf{x}_1, \mathbf{x}_1), \ldots, (\mathbf{x}_n, \mathbf{x}_n)) \in \mathbb{R}^d$$

根據 :eqref:`eq_attn-pooling`中定義的注意力彙聚函式$f$。
下面的程式碼片段是基於多頭注意力對一個張量完成自注意力的計算，
張量的形狀為（批次大小，時間步的數目或詞元序列的長度，$d$）。
輸出與輸入的張量形狀相同。

```{.python .input}
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
attention.initialize()
```

```{.python .input}
#@tab pytorch
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
attention.eval()
```

```{.python .input}
#@tab tensorflow
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
```

```{.python .input}
#@tab paddle
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
attention.eval()
```

```{.python .input}
#@tab mxnet, pytorch, paddle
batch_size, num_queries, valid_lens = 2, 4, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
attention(X, X, X, valid_lens).shape
```

```{.python .input}
#@tab tensorflow
batch_size, num_queries, valid_lens = 2, 4, tf.constant([3, 2])
X = tf.ones((batch_size, num_queries, num_hiddens))
attention(X, X, X, valid_lens, training=False).shape
```

## 比較卷積神經網路、迴圈神經網路和自注意力
:label:`subsec_cnn-rnn-self-attention`

接下來比較下面幾個架構，目標都是將由$n$個詞元組成的序列對映到另一個長度相等的序列，其中的每個輸入詞元或輸出詞元都由$d$維向量表示。具體來說，將比較的是卷積神經網路、迴圈神經網路和自注意力這幾個架構的計算複雜性、順序操作和最大路徑長度。請注意，順序操作會妨礙並行計算，而任意的序列位置組合之間的路徑越短，則能更輕鬆地學習序列中的遠距離依賴關係 :cite:`Hochreiter.Bengio.Frasconi.ea.2001`。

![比較卷積神經網路（填充詞元被忽略）、迴圈神經網路和自注意力三種架構](../img/cnn-rnn-self-attention.svg)
:label:`fig_cnn-rnn-self-attention`

考慮一個卷積核大小為$k$的卷積層。
在後麵的章節將提供關於使用卷積神經網路處理序列的更多詳細訊息。
目前只需要知道的是，由於序列長度是$n$，輸入和輸出的通道數量都是$d$，
所以卷積層的計算複雜度為$\mathcal{O}(knd^2)$。
如 :numref:`fig_cnn-rnn-self-attention`所示，
卷積神經網路是分層的，因此為有$\mathcal{O}(1)$個順序操作，
最大路徑長度為$\mathcal{O}(n/k)$。
例如，$\mathbf{x}_1$和$\mathbf{x}_5$處於
 :numref:`fig_cnn-rnn-self-attention`中卷積核大小為3的雙層卷積神經網路的感受野內。

當更新迴圈神經網路的隱狀態時，
$d \times d$權重矩陣和$d$維隱狀態的乘法計算複雜度為$\mathcal{O}(d^2)$。
由於序列長度為$n$，因此迴圈神經網路層的計算複雜度為$\mathcal{O}(nd^2)$。
根據 :numref:`fig_cnn-rnn-self-attention`，
有$\mathcal{O}(n)$個順序操作無法並行化，最大路徑長度也是$\mathcal{O}(n)$。

在自注意力中，查詢、鍵和值都是$n \times d$矩陣。
考慮 :eqref:`eq_softmax_QK_V`中縮放的』點－積『注意力，
其中$n \times d$矩陣乘以$d \times n$矩陣。
之後輸出的$n \times n$矩陣乘以$n \times d$矩陣。
因此，自注意力具有$\mathcal{O}(n^2d)$計算複雜性。
正如在 :numref:`fig_cnn-rnn-self-attention`中所講，
每個詞元都透過自注意力直接連線到任何其他詞元。
因此，有$\mathcal{O}(1)$個順序操作可以並行計算，
最大路徑長度也是$\mathcal{O}(1)$。

總而言之，卷積神經網路和自注意力都擁有並行計算的優勢，
而且自注意力的最大路徑長度最短。
但是因為其計算複雜度是關於序列長度的二次方，所以在很長的序列中計算會非常慢。

## [**位置編碼**]
:label:`subsec_positional-encoding`

在處理詞元序列時，迴圈神經網路是逐個的重複地處理詞元的，
而自注意力則因為並行計算而放棄了順序操作。
為了使用序列的順序訊息，透過在輸入表示中新增
*位置編碼*（positional encoding）來注入絕對的或相對的位置訊息。
位置編碼可以透過學習得到也可以直接固定得到。
接下來描述的是基於正弦函式和餘弦函式的固定位置編碼
 :cite:`Vaswani.Shazeer.Parmar.ea.2017`。

假設輸入表示$\mathbf{X} \in \mathbb{R}^{n \times d}$
包含一個序列中$n$個詞元的$d$維嵌入表示。
位置編碼使用相同形狀的位置嵌入矩陣
$\mathbf{P} \in \mathbb{R}^{n \times d}$輸出$\mathbf{X} + \mathbf{P}$，
矩陣第$i$行、第$2j$列和$2j+1$列上的元素為：

$$\begin{aligned} p_{i, 2j} &= \sin\left(\frac{i}{10000^{2j/d}}\right),\\p_{i, 2j+1} &= \cos\left(\frac{i}{10000^{2j/d}}\right).\end{aligned}$$
:eqlabel:`eq_positional-encoding-def`

乍一看，這種基於三角函數的設計看起來很奇怪。
在解釋這個設計之前，讓我們先在下面的`PositionalEncoding`類別中實現它。

```{.python .input}
#@save
class PositionalEncoding(nn.Block):
    """位置編碼"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 建立一個足夠長的P
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len).reshape(-1, 1) / np.power(
            10000, np.arange(0, num_hiddens, 2) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].as_in_ctx(X.ctx)
        return self.dropout(X)
```

```{.python .input}
#@tab pytorch
#@save
class PositionalEncoding(nn.Module):
    """位置編碼"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 建立一個足夠長的P
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
```

```{.python .input}
#@tab tensorflow
#@save
class PositionalEncoding(tf.keras.layers.Layer):
    """位置編碼"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        # 建立一個足夠長的P
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(max_len, dtype=np.float32).reshape(
            -1,1)/np.power(10000, np.arange(
            0, num_hiddens, 2, dtype=np.float32) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)
        
    def call(self, X, **kwargs):
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X, **kwargs)
```

```{.python .input}
#@tab paddle
#@save
class PositionalEncoding(nn.Layer):
    """位置編碼"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 建立一個足夠長的P
        self.P = paddle.zeros((1, max_len, num_hiddens))
        X = paddle.arange(max_len, dtype=paddle.float32).reshape(
            (-1, 1)) / paddle.pow(paddle.to_tensor([10000.0]), paddle.arange(
            0, num_hiddens, 2, dtype=paddle.float32) / num_hiddens)
        self.P[:, :, 0::2] = paddle.sin(X)
        self.P[:, :, 1::2] = paddle.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X)
```

在位置嵌入矩陣$\mathbf{P}$中，
[**行代表詞元在序列中的位置，列代表位置編碼的不同維度**]。
從下面的例子中可以看到位置嵌入矩陣的第$6$列和第$7$列的頻率高於第$8$列和第$9$列。
第$6$列和第$7$列之間的偏移量（第$8$列和第$9$列相同）是由於正弦函式和餘弦函式的交替。

```{.python .input}
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.initialize()
X = pos_encoding(np.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

```{.python .input}
#@tab pytorch
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()
X = pos_encoding(d2l.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

```{.python .input}
#@tab tensorflow
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
X = pos_encoding(tf.zeros((1, num_steps, encoding_dim)), training=False)
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(np.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in np.arange(6, 10)])
```

```{.python .input}
#@tab paddle
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()
X = pos_encoding(paddle.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(paddle.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in paddle.arange(6, 10)])
```

### 絕對位置訊息

為了明白沿著編碼維度單調降低的頻率與絕對位置訊息的關係，
讓我們列印出$0, 1, \ldots, 7$的[**二進位表示**]形式。
正如所看到的，每個數字、每兩個數字和每四個數字上的比特值
在第一個最低位、第二個最低位和第三個最低位上分別交替。

```{.python .input}
#@tab all
for i in range(8):
    print(f'{i}的二進位是：{i:>03b}')
```

在二進位表示中，較高比特位的交替頻率低於較低比特位，
與下面的熱圖所示相似，只是位置編碼透過使用三角函數[**在編碼維度上降低頻率**]。
由於輸出是浮點數，因此此類連續表示比二進位表示法更節省空間。

```{.python .input}
P = np.expand_dims(np.expand_dims(P[0, :, :], 0), 0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
#@tab pytorch
P = P[0, :, :].unsqueeze(0).unsqueeze(0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
#@tab tensorflow
P = tf.expand_dims(tf.expand_dims(P[0, :, :], axis=0), axis=0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
#@tab paddle
P = P[0, :, :].unsqueeze(0).unsqueeze(0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

### 相對位置訊息

除了捕獲絕對位置訊息之外，上述的位置編碼還允許模型學習得到輸入序列中相對位置訊息。
這是因為對於任何確定的位置偏移$\delta$，位置$i + \delta$處
的位置編碼可以線性投影位置$i$處的位置編碼來表示。

這種投影的數學解釋是，令$\omega_j = 1/10000^{2j/d}$，
對於任何確定的位置偏移$\delta$，
 :eqref:`eq_positional-encoding-def`中的任何一對
$(p_{i, 2j}, p_{i, 2j+1})$都可以線性投影到
$(p_{i+\delta, 2j}, p_{i+\delta, 2j+1})$：

$$\begin{aligned}
&\begin{bmatrix} \cos(\delta \omega_j) & \sin(\delta \omega_j) \\  -\sin(\delta \omega_j) & \cos(\delta \omega_j) \\ \end{bmatrix}
\begin{bmatrix} p_{i, 2j} \\  p_{i, 2j+1} \\ \end{bmatrix}\\
=&\begin{bmatrix} \cos(\delta \omega_j) \sin(i \omega_j) + \sin(\delta \omega_j) \cos(i \omega_j) \\  -\sin(\delta \omega_j) \sin(i \omega_j) + \cos(\delta \omega_j) \cos(i \omega_j) \\ \end{bmatrix}\\
=&\begin{bmatrix} \sin\left((i+\delta) \omega_j\right) \\  \cos\left((i+\delta) \omega_j\right) \\ \end{bmatrix}\\
=& 
\begin{bmatrix} p_{i+\delta, 2j} \\  p_{i+\delta, 2j+1} \\ \end{bmatrix},
\end{aligned}$$

$2\times 2$投影矩陣不依賴於任何位置的索引$i$。

## 小結

* 在自注意力中，查詢、鍵和值都來自同一組輸入。
* 卷積神經網路和自注意力都擁有並行計算的優勢，而且自注意力的最大路徑長度最短。但是因為其計算複雜度是關於序列長度的二次方，所以在很長的序列中計算會非常慢。
* 為了使用序列的順序訊息，可以透過在輸入表示中新增位置編碼，來注入絕對的或相對的位置訊息。

## 練習

1. 假設設計一個深度架構，透過堆疊基於位置編碼的自注意力層來表示序列。可能會存在什麼問題？
1. 請設計一種可學習的位置編碼方法。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/5761)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/5762)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11844)
:end_tab: