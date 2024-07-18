# 情感分析：使用卷積神經網路
:label:`sec_sentiment_cnn`

在 :numref:`chap_cnn`中，我們探討了使用二維卷積神經網路處理二維影象資料的機制，並將其應用於區域性特徵，如相鄰畫素。雖然卷積神經網路最初是為電腦視覺設計的，但它也被廣泛用於自然語言處理。簡單地說，只要將任何文字序列想象成一維影象即可。透過這種方式，一維卷積神經網路可以處理文字中的區域性特徵，例如$n$元語法。

本節將使用*textCNN*模型來示範如何設計一個表示單個文字 :cite:`Kim.2014`的卷積神經網路架構。與 :numref:`fig_nlp-map-sa-rnn`中使用帶有GloVe預訓練的迴圈神經網路架構進行情感分析相比， :numref:`fig_nlp-map-sa-cnn`中唯一的區別在於架構的選擇。

![將GloVe放入卷積神經網路架構進行情感分析](../img/nlp-map-sa-cnn.svg)
:label:`fig_nlp-map-sa-cnn`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

## 一維卷積

在介紹該模型之前，讓我們先看看一維卷積是如何工作的。請記住，這只是基於互相關運算的二維卷積的特例。

![一維互相關運算。陰影部分是第一個輸出元素以及用於輸出計算的輸入和核張量元素：$0\times1+1\times2=2$](../img/conv1d.svg)
:label:`fig_conv1d`

如 :numref:`fig_conv1d`中所示，在一維情況下，卷積視窗在輸入張量上從左向右滑動。在滑動期間，卷積視窗中某個位置包含的輸入子張量（例如， :numref:`fig_conv1d`中的$0$和$1$）和核張量（例如， :numref:`fig_conv1d`中的$1$和$2$）按元素相乘。這些乘法的總和在輸出張量的相應位置給出單個標量值（例如， :numref:`fig_conv1d`中的$0\times1+1\times2=2$）。

我們在下面的`corr1d`函式中實現了一維互相關。給定輸入張量`X`和核張量`K`，它回傳輸出張量`Y`。

```{.python .input}
#@tab mxnet, pytorch
def corr1d(X, K):
    w = K.shape[0]
    Y = d2l.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y
```

```{.python .input}
#@tab paddle
def corr1d(X, K):
    w = K.shape[0]
    Y = d2l.zeros([X.shape[0] - w + 1], dtype=X.dtype)
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y
```

我們可以從 :numref:`fig_conv1d`構造輸入張量`X`和核張量`K`來驗證上述一維互相關實現的輸出。

```{.python .input}
#@tab all
X, K = d2l.tensor([0, 1, 2, 3, 4, 5, 6]), d2l.tensor([1, 2])
corr1d(X, K)
```

對於任何具有多個通道的一維輸入，卷積核需要具有相同數量的輸入通道。然後，對於每個通道，對輸入的一維張量和卷積核的一維張量執行互相關運算，將所有通道上的結果相加以產生一維輸出張量。 :numref:`fig_conv1d_channel`示範了具有3個輸入通道的一維互相關操作。

![具有3個輸入通道的一維互相關運算。陰影部分是第一個輸出元素以及用於輸出計算的輸入和核張量元素：$2\times(-1)+3\times(-3)+1\times3+2\times4+0\times1+1\times2=2$](../img/conv1d-channel.svg)
:label:`fig_conv1d_channel`

我們可以實現多個輸入通道的一維互相關運算，並在 :numref:`fig_conv1d_channel`中驗證結果。

```{.python .input}
#@tab all
def corr1d_multi_in(X, K):
    # 首先，遍歷'X'和'K'的第0維（通道維）。然後，把它們加在一起
    return sum(corr1d(x, k) for x, k in zip(X, K))

X = d2l.tensor([[0, 1, 2, 3, 4, 5, 6],
              [1, 2, 3, 4, 5, 6, 7],
              [2, 3, 4, 5, 6, 7, 8]])
K = d2l.tensor([[1, 2], [3, 4], [-1, -3]])
corr1d_multi_in(X, K)
```

注意，多輸入通道的一維互相關等同於單輸入通道的二維互相關。舉例說明， :numref:`fig_conv1d_channel`中的多輸入通道一維互相關的等價形式是 :numref:`fig_conv1d_2d`中的單輸入通道二維互相關，其中卷積核的高度必須與輸入張量的高度相同。

![具有單個輸入通道的二維互相關操作。陰影部分是第一個輸出元素以及用於輸出計算的輸入和核心張量元素： $2\times(-1)+3\times(-3)+1\times3+2\times4+0\times1+1\times2=2$](../img/conv1d-2d.svg)
:label:`fig_conv1d_2d`

 :numref:`fig_conv1d`和 :numref:`fig_conv1d_channel`中的輸出都只有一個通道。與 :numref:`subsec_multi-output-channels`中描述的具有多個輸出通道的二維卷積相同，我們也可以為一維卷積指定多個輸出通道。

## 最大時間彙聚層

類似地，我們可以使用彙聚層從序列表示中提取最大值，作為跨時間步的最重要特徵。textCNN中使用的*最大時間彙聚層*的工作原理類似於一維全域彙聚 :cite:`Collobert.Weston.Bottou.ea.2011`。對於每個通道在不同時間步儲存值的多通道輸入，每個通道的輸出是該通道的最大值。請注意，最大時間彙聚允許在不同通道上使用不同數量的時間步。

## textCNN模型

使用一維卷積和最大時間彙聚，textCNN模型將單個預訓練的詞元表示作為輸入，然後獲得並轉換用於下游應用的序列表示。

對於具有由$d$維向量表示的$n$個詞元的單個文字序列，輸入張量的寬度、高度和通道數分別為$n$、$1$和$d$。textCNN模型將輸入轉換為輸出，如下所示：

1. 定義多個一維卷積核，並分別對輸入執行卷積運算。具有不同寬度的卷積核可以捕獲不同數目的相鄰詞元之間的區域性特徵。
1. 在所有輸出通道上執行最大時間彙聚層，然後將所有標量彙聚輸出連結為向量。
1. 使用全連線層將連結後的向量轉換為輸出類別。Dropout可以用來減少過擬合。

![textCNN的模型架構](../img/textcnn.svg)
:label:`fig_conv1d_textcnn`

 :numref:`fig_conv1d_textcnn`透過一個具體的例子說明了textCNN的模型架構。輸入是具有11個詞元的句子，其中每個詞元由6維向量表示。因此，我們有一個寬度為11的6通道輸入。定義兩個寬度為2和4的一維卷積核，分別具有4個和5個輸出通道。它們產生4個寬度為$11-2+1=10$的輸出通道和5個寬度為$11-4+1=8$的輸出通道。儘管這9個通道的寬度不同，但最大時間彙聚層給出了一個連結的9維向量，該向量最終被轉換為用於二元情感預測的2維輸出向量。

### 定義模型

我們在下面的類別中實現textCNN模型。與 :numref:`sec_sentiment_rnn`的雙向迴圈神經網路模型相比，除了用卷積層代替迴圈神經網路層外，我們還使用了兩個嵌入層：一個是可訓練權重，另一個是固定權重。

```{.python .input}
class TextCNN(nn.Block):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 這個嵌入層不需要訓練
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(2)
        # 最大時間彙聚層沒有引數，因此可以共享此實例
        self.pool = nn.GlobalMaxPool1D()
        # 建立多個一維卷積層
        self.convs = nn.Sequential()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.add(nn.Conv1D(c, k, activation='relu'))

    def forward(self, inputs):
        # 沿著向量維度將兩個嵌入層連結起來，
        # 每個嵌入層的輸出形狀都是（批次大小，詞元數量，詞元向量維度）連結起來
        embeddings = np.concatenate((
            self.embedding(inputs), self.constant_embedding(inputs)), axis=2)
        # 根據一維卷積層的輸入格式，重新排列張量，以便通道作為第2維
        embeddings = embeddings.transpose(0, 2, 1)
        # 每個一維卷積層在最大時間彙聚層合併後，獲得的張量形狀是（批次大小，通道數，1）
        # 刪除最後一個維度並沿通道維度連結
        encoding = np.concatenate([
            np.squeeze(self.pool(conv(embeddings)), axis=-1)
            for conv in self.convs], axis=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```

```{.python .input}
#@tab pytorch
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 這個嵌入層不需要訓練
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # 最大時間彙聚層沒有引數，因此可以共享此實例
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        # 建立多個一維卷積層
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # 沿著向量維度將兩個嵌入層連結起來，
        # 每個嵌入層的輸出形狀都是（批次大小，詞元數量，詞元向量維度）連結起來
        embeddings = torch.cat((
            self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # 根據一維卷積層的輸入格式，重新排列張量，以便通道作為第2維
        embeddings = embeddings.permute(0, 2, 1)
        # 每個一維卷積層在最大時間彙聚層合併後，獲得的張量形狀是（批次大小，通道數，1）
        # 刪除最後一個維度並沿通道維度連結
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```

```{.python .input}
#@tab paddle
class TextCNN(nn.Layer):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 這個嵌入層不需要訓練
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # 最大時間彙聚層沒有引數，因此可以共享此實例
        self.pool = nn.AdaptiveAvgPool1D(1)
        self.relu = nn.ReLU()
        # 建立多個一維卷積層
        self.convs = nn.LayerList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1D(2 * embed_size, c, k))

    def forward(self, inputs):
        # 沿著向量維度將兩個嵌入層連結起來，
        # 每個嵌入層的輸出形狀都是（批次大小，詞元數量，詞元向量維度）連結起來
        embeddings = paddle.concat((
            self.embedding(inputs), self.constant_embedding(inputs)), axis=2)
        # 根據一維卷積層的輸入格式，重新排列張量，以便通道作為第2維
        embeddings = embeddings.transpose([0, 2, 1])
        # 每個一維卷積層在最大時間彙聚層合併後，獲得的張量形狀是（批次大小，通道數，1）
        # 刪除最後一個維度並沿通道維度連結
        encoding = paddle.concat([
            paddle.squeeze(self.relu(self.pool(conv(embeddings))), axis=-1)
            for conv in self.convs], axis=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```

讓我們建立一個textCNN實例。它有3個卷積層，卷積核寬度分別為3、4和5，均有100個輸出通道。

```{.python .input}
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)
net.initialize(init.Xavier(), ctx=devices)
```

```{.python .input}
#@tab pytorch
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)

def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights);
```

```{.python .input}
#@tab paddle
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)

def init_weights(net):
    init_normal = nn.initializer.XavierUniform()
    for i in net.sublayers():
        if type(i) in [nn.Linear, nn.Conv1D]:  
            init_normal(i.weight)
            
init_weights(net)
```

### 載入預訓練詞向量

與 :numref:`sec_sentiment_rnn`相同，我們載入預訓練的100維GloVe嵌入作為初始化的詞元表示。這些詞元表示（嵌入權重）在`embedding`中將被訓練，在`constant_embedding`中將被固定。

```{.python .input}
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.set_data(embeds)
net.constant_embedding.weight.set_data(embeds)
net.constant_embedding.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.requires_grad = False
```

```{.python .input}
#@tab paddle
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.set_value(embeds)
net.constant_embedding.weight.set_value(embeds)
net.constant_embedding.weight.stop_gradient = True
```

### 訓練和評估模型

現在我們可以訓練textCNN模型進行情感分析。

```{.python .input}
lr, num_epochs = 0.001, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.001, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab paddle
lr, num_epochs = 0.001, 5
trainer = paddle.optimizer.Adam(learning_rate=lr, parameters=net.parameters())
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

下面，我們使用訓練好的模型來預測兩個簡單句子的情感。

```{.python .input}
#@tab all
d2l.predict_sentiment(net, vocab, 'this movie is so great')
```

```{.python .input}
#@tab all
d2l.predict_sentiment(net, vocab, 'this movie is so bad')
```

## 小結

* 一維卷積神經網路可以處理文字中的區域性特徵，例如$n$元語法。
* 多輸入通道的一維互相關等價於單輸入通道的二維互相關。
* 最大時間彙聚層允許在不同通道上使用不同數量的時間步長。
* textCNN模型使用一維卷積層和最大時間彙聚層將單個詞元表示轉換為下游應用輸出。

## 練習

1. 調整超引數，並比較 :numref:`sec_sentiment_rnn`中用於情感分析的架構和本節中用於情感分析的架構，例如在分類精度和計算效率方面。
1. 請試著用 :numref:`sec_sentiment_rnn`練習中介紹的方法進一步提高模型的分類精度。
1. 在輸入表示中新增位置編碼。它是否提高了分類的精度？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/5719)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/5720)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11827)
:end_tab:
