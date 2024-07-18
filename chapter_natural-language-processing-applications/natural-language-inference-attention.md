# 自然語言推斷：使用注意力
:label:`sec_natural-language-inference-attention`

我們在 :numref:`sec_natural-language-inference-and-dataset`中介紹了自然語言推斷任務和SNLI資料集。鑑於許多模型都是基於複雜而深度的架構，Parikh等人提出用注意力機制解決自然語言推斷問題，並稱之為『可分解注意力模型』 :cite:`Parikh.Tackstrom.Das.ea.2016`。這使得模型沒有迴圈層或卷積層，在SNLI資料集上以更少的引數實現了當時的最佳結果。本節將描述並實現這種基於注意力的自然語言推斷方法（使用MLP），如 :numref:`fig_nlp-map-nli-attention`中所述。

![將預訓練GloVe送入基於注意力和MLP的自然語言推斷架構](../img/nlp-map-nli-attention.svg)
:label:`fig_nlp-map-nli-attention`

## 模型

與保留前提和假設中詞元的順序相比，我們可以將一個文字序列中的詞元與另一個文字序列中的每個詞元對齊，然後比較和聚合這些訊息，以預測前提和假設之間的邏輯關係。與機器翻譯中源句和目標句之間的詞元對齊類似，前提和假設之間的詞元對齊可以透過注意力機制靈活地完成。

![利用注意力機制進行自然語言推斷](../img/nli-attention.svg)
:label:`fig_nli_attention`

 :numref:`fig_nli_attention`描述了使用注意力機制的自然語言推斷方法。從高層次上講，它由三個聯合訓練的步驟組成：對齊、比較和彙總。我們將在下面一步一步地對它們進行說明。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
from paddle.nn import functional as F
```

### 注意（Attending）

第一步是將一個文字序列中的詞元與另一個序列中的每個詞元對齊。假設前提是『我確實需要睡眠』，假設是『我累了』。由於語義上的相似性，我們不妨將假設中的『我』與前提中的『我』對齊，將假設中的『累』與前提中的『睡眠』對齊。同樣，我們可能希望將前提中的『我』與假設中的『我』對齊，將前提中的『需要』和『睡眠』與假設中的『累』對齊。請注意，這種對齊是使用加權平均的『軟』對齊，其中理想情況下較大的權重與要對齊的詞元相關聯。為了便於示範， :numref:`fig_nli_attention`以『硬』對齊的方式顯示了這種對齊方式。

現在，我們更詳細地描述使用注意力機制的軟對齊。用$\mathbf{A} = (\mathbf{a}_1, \ldots, \mathbf{a}_m)$和$\mathbf{B} = (\mathbf{b}_1, \ldots, \mathbf{b}_n)$表示前提和假設，其詞元數量分別為$m$和$n$，其中$\mathbf{a}_i, \mathbf{b}_j \in \mathbb{R}^{d}$（$i = 1, \ldots, m, j = 1, \ldots, n$）是$d$維的詞向量。對於軟對齊，我們將注意力權重$e_{ij} \in \mathbb{R}$計算為：

$$e_{ij} = f(\mathbf{a}_i)^\top f(\mathbf{b}_j),$$
:eqlabel:`eq_nli_e`

其中函式$f$是在下面的`mlp`函式中定義的多層感知機。輸出維度$f$由`mlp`的`num_hiddens`引數指定。

```{.python .input}
def mlp(num_hiddens, flatten):
    net = nn.Sequential()
    net.add(nn.Dropout(0.2))
    net.add(nn.Dense(num_hiddens, activation='relu', flatten=flatten))
    net.add(nn.Dropout(0.2))
    net.add(nn.Dense(num_hiddens, activation='relu', flatten=flatten))
    return net
```

```{.python .input}
#@tab pytorch
def mlp(num_inputs, num_hiddens, flatten):
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*net)
```

```{.python .input}
#@tab paddle
def mlp(num_inputs, num_hiddens, flatten):
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_axis=1)) 
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_axis=1))
    return nn.Sequential(*net)
```

值得注意的是，在 :eqref:`eq_nli_e`中，$f$分別輸入$\mathbf{a}_i$和$\mathbf{b}_j$，而不是將它們一對放在一起作為輸入。這種*分解*技巧導致$f$只有$m + n$個次計算（線性複雜度），而不是$mn$次計算（二次複雜度）

對 :eqref:`eq_nli_e`中的注意力權重進行規範化，我們計算假設中所有詞元向量的加權平均值，以獲得假設的表示，該假設與前提中索引$i$的詞元進行軟對齊：

$$
\boldsymbol{\beta}_i = \sum_{j=1}^{n}\frac{\exp(e_{ij})}{ \sum_{k=1}^{n} \exp(e_{ik})} \mathbf{b}_j.
$$

同樣，我們計算假設中索引為$j$的每個詞元與前提詞元的軟對齊：

$$
\boldsymbol{\alpha}_j = \sum_{i=1}^{m}\frac{\exp(e_{ij})}{ \sum_{k=1}^{m} \exp(e_{kj})} \mathbf{a}_i.
$$

下面，我們定義`Attend`類來計算假設（`beta`）與輸入前提`A`的軟對齊以及前提（`alpha`）與輸入假設`B`的軟對齊。

```{.python .input}
class Attend(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_hiddens=num_hiddens, flatten=False)

    def forward(self, A, B):
        # A/B的形狀：（批次大小，序列A/B的詞元數，embed_size）
        # f_A/f_B的形狀：（批次大小，序列A/B的詞元數，num_hiddens）
        f_A = self.f(A)
        f_B = self.f(B)
        # e的形狀：（批次大小，序列A的詞元數，序列B的詞元數）
        e = npx.batch_dot(f_A, f_B, transpose_b=True)
        # beta的形狀：（批次大小，序列A的詞元數，embed_size），
        # 意味著序列B被軟對齊到序列A的每個詞元(beta的第1個維度)
        beta = npx.batch_dot(npx.softmax(e), B)
        # alpha的形狀：（批次大小，序列B的詞元數，embed_size），
        # 意味著序列A被軟對齊到序列B的每個詞元(alpha的第1個維度)
        alpha = npx.batch_dot(npx.softmax(e.transpose(0, 2, 1)), A)
        return beta, alpha
```

```{.python .input}
#@tab pytorch
class Attend(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B):
        # A/B的形狀：（批次大小，序列A/B的詞元數，embed_size）
        # f_A/f_B的形狀：（批次大小，序列A/B的詞元數，num_hiddens）
        f_A = self.f(A)
        f_B = self.f(B)
        # e的形狀：（批次大小，序列A的詞元數，序列B的詞元數）
        e = torch.bmm(f_A, f_B.permute(0, 2, 1))
        # beta的形狀：（批次大小，序列A的詞元數，embed_size），
        # 意味著序列B被軟對齊到序列A的每個詞元(beta的第1個維度)
        beta = torch.bmm(F.softmax(e, dim=-1), B)
        # beta的形狀：（批次大小，序列B的詞元數，embed_size），
        # 意味著序列A被軟對齊到序列B的每個詞元(alpha的第1個維度)
        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
        return beta, alpha
```

```{.python .input}
#@tab paddle
class Attend(nn.Layer):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B):
        # A/B的形狀：（批次大小，序列A/B的詞元數，embed_size）
        # f_A/f_B的形狀：（批次大小，序列A/B的詞元數，num_hiddens）
        f_A = self.f(A)
        f_B = self.f(B)
        # e的形狀：（批次大小，序列A的詞元數，序列B的詞元數）
        e = paddle.bmm(f_A, f_B.transpose([0, 2, 1]))
        # beta的形狀：（批次大小，序列A的詞元數，embed_size），
        # 意味著序列B被軟對齊到序列A的每個詞元(beta的第1個維度)
        beta = paddle.bmm(F.softmax(e, axis=-1), B)
        # beta的形狀：（批次大小，序列B的詞元數，embed_size），
        # 意味著序列A被軟對齊到序列B的每個詞元(alpha的第1個維度)
        alpha = paddle.bmm(F.softmax(e.transpose([0, 2, 1]), axis=-1), A)
        return beta, alpha
```

### 比較

在下一步中，我們將一個序列中的詞元與與該詞元軟對齊的另一個序列進行比較。請注意，在軟對齊中，一個序列中的所有詞元（儘管可能具有不同的注意力權重）將與另一個序列中的詞元進行比較。為便於示範， :numref:`fig_nli_attention`對詞元以*硬*的方式對齊。例如，上述的*注意*（attending）步驟確定前提中的『need』和『sleep』都與假設中的『tired』對齊，則將對『疲倦-需要睡眠』進行比較。

在比較步驟中，我們將來自一個序列的詞元的連結（運算符$[\cdot, \cdot]$）和來自另一序列的對齊的詞元送入函式$g$（一個多層感知機）：

$$\mathbf{v}_{A,i} = g([\mathbf{a}_i, \boldsymbol{\beta}_i]), i = 1, \ldots, m\\ \mathbf{v}_{B,j} = g([\mathbf{b}_j, \boldsymbol{\alpha}_j]), j = 1, \ldots, n.$$

:eqlabel:`eq_nli_v_ab`

在 :eqref:`eq_nli_v_ab`中，$\mathbf{v}_{A,i}$是指，所有假設中的詞元與前提中詞元$i$軟對齊，再與詞元$i$的比較；而$\mathbf{v}_{B,j}$是指，所有前提中的詞元與假設中詞元$i$軟對齊，再與詞元$i$的比較。下面的`Compare`個類別定義了比較步驟。

```{.python .input}
class Compare(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_hiddens=num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(np.concatenate([A, beta], axis=2))
        V_B = self.g(np.concatenate([B, alpha], axis=2))
        return V_A, V_B
```

```{.python .input}
#@tab pytorch
class Compare(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(torch.cat([A, beta], dim=2))
        V_B = self.g(torch.cat([B, alpha], dim=2))
        return V_A, V_B
```

```{.python .input}
#@tab paddle
class Compare(nn.Layer):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(paddle.concat([A, beta], axis=2))
        V_B = self.g(paddle.concat([B, alpha], axis=2))
        return V_A, V_B
```

### 聚合

現在我們有兩組比較向量$\mathbf{v}_{A,i}$（$i = 1, \ldots, m$）和$\mathbf{v}_{B,j}$（$j = 1, \ldots, n$）。在最後一步中，我們將聚合這些訊息以推斷邏輯關係。我們首先求和這兩組比較向量：

$$
\mathbf{v}_A = \sum_{i=1}^{m} \mathbf{v}_{A,i}, \quad \mathbf{v}_B = \sum_{j=1}^{n}\mathbf{v}_{B,j}.
$$

接下來，我們將兩個求和結果的連結提供給函式$h$（一個多層感知機），以獲得邏輯關係的分類結果：

$$
\hat{\mathbf{y}} = h([\mathbf{v}_A, \mathbf{v}_B]).
$$

聚合步驟在以下`Aggregate`類別中定義。

```{.python .input}
class Aggregate(nn.Block):
    def __init__(self, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_hiddens=num_hiddens, flatten=True)
        self.h.add(nn.Dense(num_outputs))

    def forward(self, V_A, V_B):
        # 對兩組比較向量分別求和
        V_A = V_A.sum(axis=1)
        V_B = V_B.sum(axis=1)
        # 將兩個求和結果的連結送到多層感知機中
        Y_hat = self.h(np.concatenate([V_A, V_B], axis=1))
        return Y_hat
```

```{.python .input}
#@tab pytorch
class Aggregate(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_inputs, num_hiddens, flatten=True)
        self.linear = nn.Linear(num_hiddens, num_outputs)

    def forward(self, V_A, V_B):
        # 對兩組比較向量分別求和
        V_A = V_A.sum(dim=1)
        V_B = V_B.sum(dim=1)
        # 將兩個求和結果的連結送到多層感知機中
        Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
        return Y_hat
```

```{.python .input}
#@tab paddle
class Aggregate(nn.Layer):
    def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_inputs, num_hiddens, flatten=True)
        self.linear = nn.Linear(num_hiddens, num_outputs)

    def forward(self, V_A, V_B):
        # 對兩組比較向量分別求和
        V_A = V_A.sum(axis=1)
        V_B = V_B.sum(axis=1)
        # 將兩個求和結果的連結送到多層感知機中
        Y_hat = self.linear(self.h(paddle.concat([V_A, V_B], axis=1)))
        return Y_hat
```

### 整合程式碼

透過將注意步驟、比較步驟和聚合步驟組合在一起，我們定義了可分解注意力模型來聯合訓練這三個步驟。

```{.python .input}
class DecomposableAttention(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_hiddens)
        self.compare = Compare(num_hiddens)
        # 有3種可能的輸出：蘊涵、矛盾和中性
        self.aggregate = Aggregate(num_hiddens, 3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
```

```{.python .input}
#@tab pytorch
class DecomposableAttention(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_inputs_attend=100,
                 num_inputs_compare=200, num_inputs_agg=400, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_inputs_attend, num_hiddens)
        self.compare = Compare(num_inputs_compare, num_hiddens)
        # 有3種可能的輸出：蘊涵、矛盾和中性
        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
```

```{.python .input}
#@tab paddle
class DecomposableAttention(nn.Layer):
    def __init__(self, vocab, embed_size, num_hiddens, num_inputs_attend=100,
                 num_inputs_compare=200, num_inputs_agg=400, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_inputs_attend, num_hiddens)
        self.compare = Compare(num_inputs_compare, num_hiddens)
        # 有3種可能的輸出：蘊涵、矛盾和中性
        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
```

## 訓練和評估模型

現在，我們將在SNLI資料集上對定義好的可分解注意力模型進行訓練和評估。我們從讀取資料集開始。

### 讀取資料集

我們使用 :numref:`sec_natural-language-inference-and-dataset`中定義的函式下載並讀取SNLI資料集。批次大小和序列長度分別設定為$256$和$50$。

```{.python .input}
#@tab mxnet, pytorch
batch_size, num_steps = 256, 50
train_iter, test_iter, vocab = d2l.load_data_snli(batch_size, num_steps)
```

```{.python .input}
#@tab paddle
def load_data_snli(batch_size, num_steps=50):
    """下載SNLI資料集並回傳資料迭代器和詞表

    Defined in :numref:`sec_natural-language-inference-and-dataset`"""
    data_dir = d2l.download_extract('SNLI')
    train_data = d2l.read_snli(data_dir, True)
    test_data = d2l.read_snli(data_dir, False)
    train_set = d2l.SNLIDataset(train_data, num_steps)
    test_set = d2l.SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = paddle.io.DataLoader(train_set,batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=0,
                                             return_list=True)

    test_iter = paddle.io.DataLoader(test_set, batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=0,
                                            return_list=True)
    return train_iter, test_iter, train_set.vocab

batch_size, num_steps = 256, 50
train_iter, test_iter, vocab = load_data_snli(batch_size, num_steps)
```

### 建立模型

我們使用預訓練好的100維GloVe嵌入來表示輸入詞元。我們將向量$\mathbf{a}_i$和$\mathbf{b}_j$在 :eqref:`eq_nli_e`中的維數預定義為100。 :eqref:`eq_nli_e`中的函式$f$和 :eqref:`eq_nli_v_ab`中的函式$g$的輸出維度被設定為200.然後我們建立一個模型實例，初始化它的引數，並載入GloVe嵌入來初始化輸入詞元的向量。

```{.python .input}
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
net.initialize(init.Xavier(), ctx=devices)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.set_data(embeds)
```

```{.python .input}
#@tab pytorch
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds);
```

```{.python .input}
#@tab paddle
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.set_value(embeds);
```

### 訓練和評估模型

與 :numref:`sec_multi_gpu`中接受單一輸入（如文字序列或影象）的`split_batch`函式不同，我們定義了一個`split_batch_multi_inputs`函式以小批次接受多個輸入，如前提和假設。

```{.python .input}
#@save
def split_batch_multi_inputs(X, y, devices):
    """將多輸入'X'和'y'拆分到多個裝置"""
    X = list(zip(*[gluon.utils.split_and_load(
        feature, devices, even_split=False) for feature in X]))
    return (X, gluon.utils.split_and_load(y, devices, even_split=False))
```

現在我們可以在SNLI資料集上訓練和評估模型。

```{.python .input}
lr, num_epochs = 0.001, 4
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, 
    devices, split_batch_multi_inputs)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.001, 4
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, 
    devices)
```

```{.python .input}
#@tab paddle
lr, num_epochs = 0.001, 4
trainer = paddle.optimizer.Adam(learning_rate=lr, parameters=net.parameters())
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
    devices[:1])
```

### 使用模型

最後，定義預測函式，輸出一對前提和假設之間的邏輯關係。

```{.python .input}
#@save
def predict_snli(net, vocab, premise, hypothesis):
    """預測前提和假設之間的邏輯關係"""
    premise = np.array(vocab[premise], ctx=d2l.try_gpu())
    hypothesis = np.array(vocab[hypothesis], ctx=d2l.try_gpu())
    label = np.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), axis=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'
```

```{.python .input}
#@tab pytorch
#@save
def predict_snli(net, vocab, premise, hypothesis):
    """預測前提和假設之間的邏輯關係"""
    net.eval()
    premise = torch.tensor(vocab[premise], device=d2l.try_gpu())
    hypothesis = torch.tensor(vocab[hypothesis], device=d2l.try_gpu())
    label = torch.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), dim=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'
```

```{.python .input}
#@tab paddle
#@save
def predict_snli(net, vocab, premise, hypothesis):
    """預測前提和假設之間的邏輯關係"""
    net.eval()
    premise = paddle.to_tensor(vocab[premise], place=d2l.try_gpu())
    hypothesis = paddle.to_tensor(vocab[hypothesis], place=d2l.try_gpu())
    label = paddle.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), axis=1)
                           
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'
```

我們可以使用訓練好的模型來獲得對範例句子的自然語言推斷結果。

```{.python .input}
#@tab all
predict_snli(net, vocab, ['he', 'is', 'good', '.'], ['he', 'is', 'bad', '.'])
```

## 小結

* 可分解注意模型包括三個步驟來預測前提和假設之間的邏輯關係：注意、比較和聚合。
* 透過注意力機制，我們可以將一個文字序列中的詞元與另一個文字序列中的每個詞元對齊，反之亦然。這種對齊是使用加權平均的軟對齊，其中理想情況下較大的權重與要對齊的詞元相關聯。
* 在計算注意力權重時，分解技巧會帶來比二次複雜度更理想的線性複雜度。
* 我們可以使用預訓練好的詞向量作為下游自然語言處理任務（如自然語言推斷）的輸入表示。

## 練習

1. 使用其他超引數組合訓練模型，能在測試集上獲得更高的準確度嗎？
1. 自然語言推斷的可分解注意模型的主要缺點是什麼？
1. 假設我們想要獲得任何一對句子的語義相似級別（例如，0～1之間的連續值）。我們應該如何收集和標註資料集？請嘗試設計一個有注意力機制的模型。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/5727)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/5728)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11829)
:end_tab:
