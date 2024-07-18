# 注意力評分函式
:label:`sec_attention-scoring-functions`

 :numref:`sec_nadaraya-watson`使用了高斯核來對查詢和鍵之間的關係建模。
 :eqref:`eq_nadaraya-watson-gaussian`中的
高斯核指數部分可以視為*注意力評分函式*（attention scoring function），
簡稱*評分函式*（scoring function），
然後把這個函式的輸出結果輸入到softmax函式中進行運算。
透過上述步驟，將得到與鍵對應的值的機率分佈（即注意力權重）。
最後，注意力彙聚的輸出就是基於這些注意力權重的值的加權和。

從宏觀來看，上述演算法可以用來實現
 :numref:`fig_qkv`中的注意力機制框架。
 :numref:`fig_attention_output`說明了
如何將注意力彙聚的輸出計算成為值的加權和，
其中$a$表示注意力評分函式。
由於注意力權重是機率分佈，
因此加權和其本質上是加權平均值。

![計算注意力彙聚的輸出為值的加權和](../img/attention-output.svg)
:label:`fig_attention_output`

用數學語言描述，假設有一個查詢
$\mathbf{q} \in \mathbb{R}^q$和
$m$個『鍵－值』對
$(\mathbf{k}_1, \mathbf{v}_1), \ldots, (\mathbf{k}_m, \mathbf{v}_m)$，
其中$\mathbf{k}_i \in \mathbb{R}^k$，$\mathbf{v}_i \in \mathbb{R}^v$。
注意力彙聚函式$f$就被表示成值的加權和：

$$f(\mathbf{q}, (\mathbf{k}_1, \mathbf{v}_1), \ldots, (\mathbf{k}_m, \mathbf{v}_m)) = \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i \in \mathbb{R}^v,$$
:eqlabel:`eq_attn-pooling`

其中查詢$\mathbf{q}$和鍵$\mathbf{k}_i$的注意力權重（標量）
是透過注意力評分函式$a$將兩個向量對映成標量，
再經過softmax運算得到的：

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \mathrm{softmax}(a(\mathbf{q}, \mathbf{k}_i)) = \frac{\exp(a(\mathbf{q}, \mathbf{k}_i))}{\sum_{j=1}^m \exp(a(\mathbf{q}, \mathbf{k}_j))} \in \mathbb{R}.$$
:eqlabel:`eq_attn-scoring-alpha`

正如上圖所示，選擇不同的注意力評分函式$a$會導致不同的注意力彙聚操作。
本節將介紹兩個流行的評分函式，稍後將用他們來實現更復雜的注意力機制。

```{.python .input}
import math
from d2l import mxnet as d2l
from mxnet import np, npx
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
import math
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
```

## [**掩蔽softmax操作**]

正如上面提到的，softmax操作用於輸出一個機率分佈作為注意力權重。
在某些情況下，並非所有的值都應該被納入到注意力彙聚中。
例如，為了在 :numref:`sec_machine_translation`中高效處理小批次資料集，
某些文字序列被填充了沒有意義的特殊詞元。
為了僅將有意義的詞元作為值來獲取注意力彙聚，
可以指定一個有效序列長度（即詞元的個數），
以便在計算softmax時過濾掉超出指定範圍的位置。
下面的`masked_softmax`函式
實現了這樣的*掩蔽softmax操作*（masked softmax operation），
其中任何超出有效長度的位置都被掩蔽並置為0。

```{.python .input}
#@save
def masked_softmax(X, valid_lens):
    """透過在最後一個軸上掩蔽元素來執行softmax操作"""
    # X:3D張量，valid_lens:1D或2D張量
    if valid_lens is None:
        return npx.softmax(X)
    else:
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = valid_lens.repeat(shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最後一軸上被掩蔽的元素使用一個非常大的負值替換，從而其softmax輸出為0
        X = npx.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, True,
                              value=-1e6, axis=1)
        return npx.softmax(X).reshape(shape)
```

```{.python .input}
#@tab pytorch
#@save
def masked_softmax(X, valid_lens):
    """透過在最後一個軸上掩蔽元素來執行softmax操作"""
    # X:3D張量，valid_lens:1D或2D張量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最後一軸上被掩蔽的元素使用一個非常大的負值替換，從而其softmax輸出為0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
```

```{.python .input}
#@tab tensorflow
#@save
def masked_softmax(X, valid_lens):
    """透過在最後一個軸上掩蔽元素來執行softmax操作"""
    # X:3D張量，valid_lens:1D或2D張量
    if valid_lens is None:
        return tf.nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if len(valid_lens.shape) == 1:
            valid_lens = tf.repeat(valid_lens, repeats=shape[1])
            
        else:
            valid_lens = tf.reshape(valid_lens, shape=-1)
        # 最後一軸上被掩蔽的元素使用一個非常大的負值替換，從而其softmax輸出為0
        X = d2l.sequence_mask(tf.reshape(X, shape=(-1, shape[-1])), 
                              valid_lens, value=-1e6)    
        return tf.nn.softmax(tf.reshape(X, shape=shape), axis=-1)
```

```{.python .input}
#@tab paddle
#@save
def masked_softmax(X, valid_lens):
    """透過在最後一個軸上掩蔽元素來執行softmax操作"""
    # X:3D張量，valid_lens:1D或2D張量
    if valid_lens is None:
        return nn.functional.softmax(X, axis=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = paddle.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape((-1,))
        # 最後一軸上被掩蔽的元素使用一個非常大的負值替換，從而其softmax輸出為0
        X = d2l.sequence_mask(X.reshape((-1, shape[-1])), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), axis=-1)
```

為了[**示範此函式是如何工作**]的，
考慮由兩個$2 \times 4$矩陣表示的樣本，
這兩個樣本的有效長度分別為$2$和$3$。
經過掩蔽softmax操作，超出有效長度的值都被掩蔽為0。

```{.python .input}
masked_softmax(np.random.uniform(size=(2, 2, 4)), d2l.tensor([2, 3]))
```

```{.python .input}
#@tab pytorch
masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
```

```{.python .input}
#@tab tensorflow
masked_softmax(tf.random.uniform(shape=(2, 2, 4)), tf.constant([2, 3]))
```

```{.python .input}
#@tab paddle
masked_softmax(paddle.rand((2, 2, 4)), paddle.to_tensor([2, 3]))
```

同樣，也可以使用二維張量，為矩陣樣本中的每一行指定有效長度。

```{.python .input}
masked_softmax(np.random.uniform(size=(2, 2, 4)),
               d2l.tensor([[1, 3], [2, 4]]))
```

```{.python .input}
#@tab pytorch
masked_softmax(torch.rand(2, 2, 4), d2l.tensor([[1, 3], [2, 4]]))
```

```{.python .input}
#@tab tensorflow
masked_softmax(tf.random.uniform(shape=(2, 2, 4)), tf.constant([[1, 3], [2, 4]]))
```

```{.python .input}
#@tab paddle
masked_softmax(paddle.rand((2, 2, 4)), paddle.to_tensor([[1, 3], [2, 4]]))
```

## [**加性注意力**]
:label:`subsec_additive-attention`

一般來說，當查詢和鍵是不同長度的向量時，可以使用加性注意力作為評分函式。
給定查詢$\mathbf{q} \in \mathbb{R}^q$和
鍵$\mathbf{k} \in \mathbb{R}^k$，
*加性注意力*（additive attention）的評分函式為

$$a(\mathbf q, \mathbf k) = \mathbf w_v^\top \text{tanh}(\mathbf W_q\mathbf q + \mathbf W_k \mathbf k) \in \mathbb{R},$$
:eqlabel:`eq_additive-attn`

其中可學習的引數是$\mathbf W_q\in\mathbb R^{h\times q}$、
$\mathbf W_k\in\mathbb R^{h\times k}$和
$\mathbf w_v\in\mathbb R^{h}$。
如 :eqref:`eq_additive-attn`所示，
將查詢和鍵連結起來後輸入到一個多層感知機（MLP）中，
感知機包含一個隱藏層，其隱藏單元數是一個超引數$h$。
透過使用$\tanh$作為啟用函式，並且停用偏置項。

下面來實現加性注意力。

```{.python .input}
#@save
class AdditiveAttention(nn.Block):
    """加性注意力"""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        # 使用'flatten=False'只轉換最後一個軸，以便其他軸的形狀保持不變
        self.W_k = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.W_q = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.w_v = nn.Dense(1, use_bias=False, flatten=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在維度擴充套件後，
        # queries的形狀：(batch_size，查詢的個數，1，num_hidden)
        # key的形狀：(batch_size，1，『鍵－值』對的個數，num_hiddens)
        # 使用廣播的方式進行求和
        features = np.expand_dims(queries, axis=2) + np.expand_dims(
            keys, axis=1)
        features = np.tanh(features)
        # self.w_v僅有一個輸出，因此從形狀中移除最後那個維度。
        # scores的形狀：(batch_size，查詢的個數，『鍵-值』對的個數)
        scores = np.squeeze(self.w_v(features), axis=-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形狀：(batch_size，『鍵－值』對的個數，值的維度)
        return npx.batch_dot(self.dropout(self.attention_weights), values)
```

```{.python .input}
#@tab pytorch
#@save
class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在維度擴充套件後，
        # queries的形狀：(batch_size，查詢的個數，1，num_hidden)
        # key的形狀：(batch_size，1，『鍵－值』對的個數，num_hiddens)
        # 使用廣播方式進行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v僅有一個輸出，因此從形狀中移除最後那個維度。
        # scores的形狀：(batch_size，查詢的個數，『鍵-值』對的個數)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形狀：(batch_size，『鍵－值』對的個數，值的維度)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

```{.python .input}
#@tab tensorflow
#@save
class AdditiveAttention(tf.keras.layers.Layer):
    """Additiveattention."""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=False)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=False)
        self.w_v = tf.keras.layers.Dense(1, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    def call(self, queries, keys, values, valid_lens, **kwargs):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在維度擴充套件後，
        # queries的形狀：(batch_size，查詢的個數，1，num_hidden)
        # key的形狀：(batch_size，1，『鍵－值』對的個數，num_hiddens)
        # 使用廣播方式進行求和
        features = tf.expand_dims(queries, axis=2) + tf.expand_dims(
            keys, axis=1)
        features = tf.nn.tanh(features)
        # self.w_v僅有一個輸出，因此從形狀中移除最後那個維度。
        # scores的形狀：(batch_size，查詢的個數，『鍵-值』對的個數)
        scores = tf.squeeze(self.w_v(features), axis=-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形狀：(batch_size，『鍵－值』對的個數，值的維度)
        return tf.matmul(self.dropout(
            self.attention_weights, **kwargs), values)
```

```{.python .input}
#@tab paddle
#@save
class AdditiveAttention(nn.Layer):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias_attr=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias_attr=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias_attr=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在維度擴充套件後，
        # queries的形狀：(batch_size，查詢的個數，1，num_hidden)
        # key的形狀：(batch_size，1，『鍵－值』對的個數，num_hiddens)
        # 使用廣播方式進行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = paddle.tanh(features)
        # self.w_v僅有一個輸出，因此從形狀中移除最後那個維度。
        # scores的形狀：(batch_size，查詢的個數，『鍵-值』對的個數)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形狀：(batch_size，『鍵－值』對的個數，值的維度)
        return paddle.bmm(self.dropout(self.attention_weights), values)
```

用一個小例子來[**示範上面的`AdditiveAttention`類**]，
其中查詢、鍵和值的形狀為（批次大小，步數或詞元序列長度，特徵大小），
實際輸出為$(2,1,20)$、$(2,10,2)$和$(2,10,4)$。
注意力彙聚輸出的形狀為（批次大小，查詢的步數，值的維度）。

```{.python .input}
queries, keys = d2l.normal(0, 1, (2, 1, 20)), d2l.ones((2, 10, 2))
# values的小批次資料集中，兩個值矩陣是相同的
values = np.arange(40).reshape(1, 10, 4).repeat(2, axis=0)
valid_lens = d2l.tensor([2, 6])

attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
attention.initialize()
attention(queries, keys, values, valid_lens)
```

```{.python .input}
#@tab pytorch
queries, keys = d2l.normal(0, 1, (2, 1, 20)), d2l.ones((2, 10, 2))
# values的小批次，兩個值矩陣是相同的
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
    2, 1, 1)
valid_lens = d2l.tensor([2, 6])

attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                              dropout=0.1)
attention.eval()
attention(queries, keys, values, valid_lens)
```

```{.python .input}
#@tab tensorflow
queries, keys = tf.random.normal(shape=(2, 1, 20)), tf.ones((2, 10, 2))
# values的小批次，兩個值矩陣是相同的
values = tf.repeat(tf.reshape(
    tf.range(40, dtype=tf.float32), shape=(1, 10, 4)), repeats=2, axis=0)
valid_lens = tf.constant([2, 6])

attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                              dropout=0.1)
attention(queries, keys, values, valid_lens, training=False)
```

```{.python .input}
#@tab paddle
queries, keys = paddle.normal(0, 1, (2, 1, 20)), paddle.ones((2, 10, 2))
# values的小批次，兩個值矩陣是相同的
values = paddle.arange(40, dtype=paddle.float32).reshape((1, 10, 4)).tile(
    [2, 1, 1])
valid_lens = paddle.to_tensor([2, 6])

attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                              dropout=0.1)
attention.eval()
attention(queries, keys, values, valid_lens)
```

儘管加性注意力包含了可學習的引數，但由於本例子中每個鍵都是相同的，
所以[**注意力權重**]是均勻的，由指定的有效長度決定。

```{.python .input}
#@tab all
d2l.show_heatmaps(d2l.reshape(attention.attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

## [**縮放點積注意力**]

使用點積可以得到計算效率更高的評分函式，
但是點積操作要求查詢和鍵具有相同的長度$d$。
假設查詢和鍵的所有元素都是獨立的隨機變數，
並且都滿足零均值和單位方差，
那麼兩個向量的點積的均值為$0$，方差為$d$。
為確保無論向量長度如何，
點積的方差在不考慮向量長度的情況下仍然是$1$，
我們再將點積除以$\sqrt{d}$，
則*縮放點積注意力*（scaled dot-product attention）評分函式為：

$$a(\mathbf q, \mathbf k) = \mathbf{q}^\top \mathbf{k}  /\sqrt{d}.$$

在實踐中，我們通常從小批次的角度來考慮提高效率，
例如基於$n$個查詢和$m$個鍵－值對計算注意力，
其中查詢和鍵的長度為$d$，值的長度為$v$。
查詢$\mathbf Q\in\mathbb R^{n\times d}$、
鍵$\mathbf K\in\mathbb R^{m\times d}$和
值$\mathbf V\in\mathbb R^{m\times v}$的縮放點積注意力是：

$$ \mathrm{softmax}\left(\frac{\mathbf Q \mathbf K^\top }{\sqrt{d}}\right) \mathbf V \in \mathbb{R}^{n\times v}.$$
:eqlabel:`eq_softmax_QK_V`

下面的縮放點積注意力的實現使用了暫退法進行模型正則化。

```{.python .input}
#@save
class DotProductAttention(nn.Block):
    """縮放點積注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形狀：(batch_size，查詢的個數，d)
    # keys的形狀：(batch_size，『鍵－值』對的個數，d)
    # values的形狀：(batch_size，『鍵－值』對的個數，值的維度)
    # valid_lens的形狀:(batch_size，)或者(batch_size，查詢的個數)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 設定transpose_b=True為了交換keys的最後兩個維度
        scores = npx.batch_dot(queries, keys, transpose_b=True) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return npx.batch_dot(self.dropout(self.attention_weights), values)
```

```{.python .input}
#@tab pytorch
#@save
class DotProductAttention(nn.Module):
    """縮放點積注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形狀：(batch_size，查詢的個數，d)
    # keys的形狀：(batch_size，『鍵－值』對的個數，d)
    # values的形狀：(batch_size，『鍵－值』對的個數，值的維度)
    # valid_lens的形狀:(batch_size，)或者(batch_size，查詢的個數)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 設定transpose_b=True為了交換keys的最後兩個維度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

```{.python .input}
#@tab tensorflow
#@save
class DotProductAttention(tf.keras.layers.Layer):
    """Scaleddotproductattention."""
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    # queries的形狀：(batch_size，查詢的個數，d)
    # keys的形狀：(batch_size，『鍵－值』對的個數，d)
    # values的形狀：(batch_size，『鍵－值』對的個數，值的維度)
    # valid_lens的形狀:(batch_size，)或者(batch_size，查詢的個數)
    def call(self, queries, keys, values, valid_lens, **kwargs):
        d = queries.shape[-1]
        scores = tf.matmul(queries, keys, transpose_b=True)/tf.math.sqrt(
            tf.cast(d, dtype=tf.float32))
        self.attention_weights = masked_softmax(scores, valid_lens)
        return tf.matmul(self.dropout(self.attention_weights, **kwargs), values)
```

```{.python .input}
#@tab paddle
#@save
class DotProductAttention(nn.Layer):
    """縮放點積注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形狀：(batch_size，查詢的個數，d)
    # keys的形狀：(batch_size，『鍵－值』對的個數，d)
    # values的形狀：(batch_size，『鍵－值』對的個數，值的維度)
    # valid_lens的形狀:(batch_size，)或者(batch_size，查詢的個數)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 設定transpose_b=True為了交換keys的最後兩個維度
        scores = paddle.bmm(queries, keys.transpose((0,2,1))) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return paddle.bmm(self.dropout(self.attention_weights), values)
```

為了[**示範上述的`DotProductAttention`類**]，
我們使用與先前加性注意力例子中相同的鍵、值和有效長度。
對於點積操作，我們令查詢的特徵維度與鍵的特徵維度大小相同。

```{.python .input}
queries = d2l.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention.initialize()
attention(queries, keys, values, valid_lens)
```

```{.python .input}
#@tab pytorch
queries = d2l.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention.eval()
attention(queries, keys, values, valid_lens)
```

```{.python .input}
#@tab tensorflow
queries = tf.random.normal(shape=(2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention(queries, keys, values, valid_lens, training=False)
```

```{.python .input}
#@tab paddle
queries = paddle.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention.eval()
attention(queries, keys, values, valid_lens)
```

與加性注意力示範相同，由於鍵包含的是相同的元素，
而這些元素無法透過任何查詢進行區分，因此獲得了[**均勻的注意力權重**]。

```{.python .input}
#@tab all
d2l.show_heatmaps(d2l.reshape(attention.attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

## 小結

* 將注意力彙聚的輸出計算可以作為值的加權平均，選擇不同的注意力評分函式會帶來不同的注意力彙聚操作。
* 當查詢和鍵是不同長度的向量時，可以使用可加性注意力評分函式。當它們的長度相同時，使用縮放的『點－積』注意力評分函式的計算效率更高。

## 練習

1. 修改小例子中的鍵，並且視覺化注意力權重。可加性注意力和縮放的『點－積』注意力是否仍然產生相同的結果？為什麼？
1. 只使用矩陣乘法，能否為具有不同向量長度的查詢和鍵設計新的評分函式？
1. 當查詢和鍵具有相同的向量長度時，向量求和作為評分函式是否比『點－積』更好？為什麼？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/5751)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/5752)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11841)
:end_tab: