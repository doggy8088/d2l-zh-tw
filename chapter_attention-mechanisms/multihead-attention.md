# 多頭注意力
:label:`sec_multihead-attention`

在實踐中，當給定相同的查詢、鍵和值的集合時，
我們希望模型可以基於相同的注意力機制學習到不同的行為，
然後將不同的行為作為知識組合起來，
捕獲序列內各種範圍的依賴關係
（例如，短距離依賴和長距離依賴關係）。
因此，允許注意力機制組合使用查詢、鍵和值的不同
*子空間表示*（representation subspaces）可能是有益的。

為此，與其只使用單獨一個注意力彙聚，
我們可以用獨立學習得到的$h$組不同的
*線性投影*（linear projections）來變換查詢、鍵和值。
然後，這$h$組變換後的查詢、鍵和值將並行地送到注意力彙聚中。
最後，將這$h$個注意力彙聚的輸出拼接在一起，
並且透過另一個可以學習的線性投影進行變換，
以產生最終輸出。
這種設計被稱為*多頭注意力*（multihead attention）
 :cite:`Vaswani.Shazeer.Parmar.ea.2017`。
對於$h$個注意力彙聚輸出，每一個注意力彙聚都被稱作一個*頭*（head）。
 :numref:`fig_multi-head-attention`
展示了使用全連線層來實現可學習的線性變換的多頭注意力。

![多頭注意力：多個頭連結然後線性變換](../img/multi-head-attention.svg)
:label:`fig_multi-head-attention`

## 模型

在實現多頭注意力之前，讓我們用數學語言將這個模型形式化地描述出來。
給定查詢$\mathbf{q} \in \mathbb{R}^{d_q}$、
鍵$\mathbf{k} \in \mathbb{R}^{d_k}$和
值$\mathbf{v} \in \mathbb{R}^{d_v}$，
每個注意力頭$\mathbf{h}_i$（$i = 1, \ldots, h$）的計算方法為：

$$\mathbf{h}_i = f(\mathbf W_i^{(q)}\mathbf q, \mathbf W_i^{(k)}\mathbf k,\mathbf W_i^{(v)}\mathbf v) \in \mathbb R^{p_v},$$

其中，可學習的引數包括
$\mathbf W_i^{(q)}\in\mathbb R^{p_q\times d_q}$、
$\mathbf W_i^{(k)}\in\mathbb R^{p_k\times d_k}$和
$\mathbf W_i^{(v)}\in\mathbb R^{p_v\times d_v}$，
以及代表注意力彙聚的函式$f$。
$f$可以是 :numref:`sec_attention-scoring-functions`中的
加性注意力和縮放點積注意力。
多頭注意力的輸出需要經過另一個線性轉換，
它對應著$h$個頭連結後的結果，因此其可學習引數是
$\mathbf W_o\in\mathbb R^{p_o\times h p_v}$：

$$\mathbf W_o \begin{bmatrix}\mathbf h_1\\\vdots\\\mathbf h_h\end{bmatrix} \in \mathbb{R}^{p_o}.$$

基於這種設計，每個頭都可能會關注輸入的不同部分，
可以表示比簡單加權平均值更復雜的函式。

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
import tensorflow as tf
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import math
import paddle
from paddle import nn
```

## 實現

在實現過程中通常[**選擇縮放點積注意力作為每一個注意力頭**]。
為了避免計算代價和引數代價的大幅增長，
我們設定$p_q = p_k = p_v = p_o / h$。
值得注意的是，如果將查詢、鍵和值的線性變換的輸出數量設定為
$p_q h = p_k h = p_v h = p_o$，
則可以並行計算$h$個頭。
在下面的實現中，$p_o$是透過引數`num_hiddens`指定的。

```{.python .input}
#@save
class MultiHeadAttention(nn.Block):
    """多頭注意力"""
    def __init__(self, num_hiddens, num_heads, dropout, use_bias=False,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_k = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_v = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_o = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形狀:
        # (batch_size，查詢或者『鍵－值』對的個數，num_hiddens)
        # valid_lens　的形狀:
        # (batch_size，)或(batch_size，查詢的個數)
        # 經過變換後，輸出的queries，keys，values　的形狀:
        # (batch_size*num_heads，查詢或者『鍵－值』對的個數，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在軸0，將第一項（標量或者向量）複製num_heads次，
            # 然後如此複製第二項，然後諸如此類。
            valid_lens = valid_lens.repeat(self.num_heads, axis=0)

        # output的形狀:(batch_size*num_heads，查詢的個數，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        
        # output_concat的形狀:(batch_size，查詢的個數，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
```

```{.python .input}
#@tab pytorch
#@save
class MultiHeadAttention(nn.Module):
    """多頭注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形狀:
        # (batch_size，查詢或者『鍵－值』對的個數，num_hiddens)
        # valid_lens　的形狀:
        # (batch_size，)或(batch_size，查詢的個數)
        # 經過變換後，輸出的queries，keys，values　的形狀:
        # (batch_size*num_heads，查詢或者『鍵－值』對的個數，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在軸0，將第一項（標量或者向量）複製num_heads次，
            # 然後如此複製第二項，然後諸如此類。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # output的形狀:(batch_size*num_heads，查詢的個數，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形狀:(batch_size，查詢的個數，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
```

```{.python .input}
#@tab tensorflow
#@save
class MultiHeadAttention(tf.keras.layers.Layer):
    """多頭注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_v = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_o = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
    
    def call(self, queries, keys, values, valid_lens, **kwargs):
        # queries，keys，values的形狀:
        # (batch_size，查詢或者『鍵－值』對的個數，num_hiddens)
        # valid_lens　的形狀:
        # (batch_size，)或(batch_size，查詢的個數)
        # 經過變換後，輸出的queries，keys，values　的形狀:
        # (batch_size*num_heads，查詢或者『鍵－值』對的個數，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        
        if valid_lens is not None:
            # 在軸0，將第一項（標量或者向量）複製num_heads次，
            # 然後如此複製第二項，然後諸如此類。
            valid_lens = tf.repeat(valid_lens, repeats=self.num_heads, axis=0)
            
        # output的形狀:(batch_size*num_heads，查詢的個數，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens, **kwargs)
        
        # output_concat的形狀:(batch_size，查詢的個數，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
```

```{.python .input}
#@tab paddle
#@save
class MultiHeadAttention(nn.Layer):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias_attr=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias_attr=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias_attr=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias_attr=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形狀:
        # (batch_size，查詢或者『鍵－值』對的個數，num_hiddens)
        # valid_lens　的形狀:
        # (batch_size，)或(batch_size，查詢的個數)
        # 經過變換後，輸出的queries，keys，values　的形狀:
        # (batch_size*num_heads，查詢或者『鍵－值』對的個數，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None:
            # 在軸0，將第一項（標量或者向量）複製num_heads次，
            # 然後如此複製第二項，然後諸如此類。
            valid_lens = paddle.repeat_interleave(
                valid_lens, repeats=self.num_heads, axis=0)

        # output的形狀:(batch_size*num_heads，查詢的個數，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形狀:(batch_size，查詢的個數，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
```

為了能夠[**使多個頭並行計算**]，
上面的`MultiHeadAttention`類將使用下面定義的兩個轉置函式。
具體來說，`transpose_output`函式反轉了`transpose_qkv`函式的操作。

```{.python .input}
#@save
def transpose_qkv(X, num_heads):
    """為了多注意力頭的並行計算而變換形狀"""
    # 輸入X的形狀:(batch_size，查詢或者『鍵－值』對的個數，num_hiddens)
    # 輸出X的形狀:(batch_size，查詢或者『鍵－值』對的個數，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 輸出X的形狀:(batch_size，num_heads，查詢或者『鍵－值』對的個數,
    # num_hiddens/num_heads)
    X = X.transpose(0, 2, 1, 3)

    # 最終輸出的形狀:(batch_size*num_heads,查詢或者『鍵－值』對的個數,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


#@save
def transpose_output(X, num_heads):
    """逆轉transpose_qkv函式的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.transpose(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
```

```{.python .input}
#@tab pytorch
#@save
def transpose_qkv(X, num_heads):
    """為了多注意力頭的並行計算而變換形狀"""
    # 輸入X的形狀:(batch_size，查詢或者『鍵－值』對的個數，num_hiddens)
    # 輸出X的形狀:(batch_size，查詢或者『鍵－值』對的個數，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 輸出X的形狀:(batch_size，num_heads，查詢或者『鍵－值』對的個數,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最終輸出的形狀:(batch_size*num_heads,查詢或者『鍵－值』對的個數,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


#@save
def transpose_output(X, num_heads):
    """逆轉transpose_qkv函式的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
```

```{.python .input}
#@tab tensorflow
#@save
def transpose_qkv(X, num_heads):
    """為了多注意力頭的並行計算而變換形狀"""
    # 輸入X的形狀:(batch_size，查詢或者『鍵－值』對的個數，num_hiddens)
    # 輸出X的形狀:(batch_size，查詢或者『鍵－值』對的個數，num_heads，
    # num_hiddens/num_heads)
    X = tf.reshape(X, shape=(X.shape[0], X.shape[1], num_heads, -1))

    # 輸出X的形狀:(batch_size，num_heads，查詢或者『鍵－值』對的個數,
    # num_hiddens/num_heads)
    X = tf.transpose(X, perm=(0, 2, 1, 3))

    # 最終輸出的形狀:(batch_size*num_heads,查詢或者『鍵－值』對的個數,
    # num_hiddens/num_heads)
    return tf.reshape(X, shape=(-1, X.shape[2], X.shape[3]))


#@save
def transpose_output(X, num_heads):
    """逆轉transpose_qkv函式的操作"""
    X = tf.reshape(X, shape=(-1, num_heads, X.shape[1], X.shape[2]))
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    return tf.reshape(X, shape=(X.shape[0], X.shape[1], -1))
```

```{.python .input}
#@tab paddle
#@save
def transpose_qkv(X, num_heads):
    """為了多注意力頭的並行計算而變換形狀"""
    # 輸入X的形狀:(batch_size，查詢或者『鍵－值』對的個數，num_hiddens)
    # 輸出X的形狀:(batch_size，查詢或者『鍵－值』對的個數，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape((X.shape[0], X.shape[1], num_heads, -1))

    # 輸出X的形狀:(batch_size，num_heads，查詢或者『鍵－值』對的個數,
    # num_hiddens/num_heads)
    X = X.transpose((0, 2, 1, 3))

    # 最終輸出的形狀:(batch_size*num_heads,查詢或者『鍵－值』對的個數,
    # num_hiddens/num_heads)
    return X.reshape((-1, X.shape[2], X.shape[3]))


#@save
def transpose_output(X, num_heads):
    """逆轉transpose_qkv函式的操作"""
    X = X.reshape((-1, num_heads, X.shape[1], X.shape[2]))
    X = X.transpose((0, 2, 1, 3))
    return X.reshape((X.shape[0], X.shape[1], -1))
```

下面使用鍵和值相同的小例子來[**測試**]我們編寫的`MultiHeadAttention`類。
多頭注意力輸出的形狀是（`batch_size`，`num_queries`，`num_hiddens`）。

```{.python .input}
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_heads, 0.5)
attention.initialize()
```

```{.python .input}
#@tab pytorch
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                               num_hiddens, num_heads, 0.5)
attention.eval()
```

```{.python .input}
#@tab tensorflow
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                               num_hiddens, num_heads, 0.5)
```

```{.python .input}
#@tab paddle
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                               num_hiddens, num_heads, 0.5)
attention.eval()
```

```{.python .input}
#@tab mxnet, pytorch, paddle
batch_size, num_queries = 2, 4
num_kvpairs, valid_lens =  6, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
Y = d2l.ones((batch_size, num_kvpairs, num_hiddens))
attention(X, Y, Y, valid_lens).shape
```

```{.python .input}
#@tab tensorflow
batch_size, num_queries = 2, 4
num_kvpairs, valid_lens = 6, d2l.tensor([3, 2])
X = tf.ones((batch_size, num_queries, num_hiddens))
Y = tf.ones((batch_size, num_kvpairs, num_hiddens))
attention(X, Y, Y, valid_lens, training=False).shape
```

## 小結

* 多頭注意力融合了來自於多個注意力彙聚的不同知識，這些知識的不同來源於相同的查詢、鍵和值的不同的子空間表示。
* 基於適當的張量操作，可以實現多頭注意力的並行計算。

## 練習

1. 分別視覺化這個實驗中的多個頭的注意力權重。
1. 假設有一個完成訓練的基於多頭注意力的模型，現在希望修剪最不重要的注意力頭以提高預測速度。如何設計實驗來衡量注意力頭的重要性呢？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/5757)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/5758)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11843)
:end_tab: