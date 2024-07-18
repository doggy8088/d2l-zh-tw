# 情感分析：使用迴圈神經網路
:label:`sec_sentiment_rnn`

與詞相似度和類比任務一樣，我們也可以將預先訓練的詞向量應用於情感分析。由於 :numref:`sec_sentiment`中的IMDb評論資料集不是很大，使用在大規模語料庫上預訓練的文字表示可以減少模型的過擬合。作為 :numref:`fig_nlp-map-sa-rnn`中所示的具體範例，我們將使用預訓練的GloVe模型來表示每個詞元，並將這些詞元表示送入多層雙向迴圈神經網路以獲得文字序列表示，該文字序列表示將被轉換為情感分析輸出 :cite:`Maas.Daly.Pham.ea.2011`。對於相同的下游應用，我們稍後將考慮不同的架構選擇。

![將GloVe送入基於迴圈神經網路的架構，用於情感分析](../img/nlp-map-sa-rnn.svg)
:label:`fig_nlp-map-sa-rnn`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn, rnn
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

## 使用迴圈神經網路表示單個文字

在文字分類任務（如情感分析）中，可變長度的文字序列將被轉換為固定長度的類別。在下面的`BiRNN`類別中，雖然文字序列的每個詞元經由嵌入層（`self.embedding`）獲得其單獨的預訓練GloVe表示，但是整個序列由雙向迴圈神經網路（`self.encoder`）編碼。更具體地說，雙向長短期記憶網路在初始和最終時間步的隱狀態（在最後一層）被連結起來作為文字序列的表示。然後，透過一個具有兩個輸出（『積極』和『消極』）的全連線層（`self.decoder`），將此單一文字表示轉換為輸出類別。

```{.python .input}
class BiRNN(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 將bidirectional設定為True以獲取雙向迴圈神經網路
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # inputs的形狀是（批次大小，時間步數）
        # 因為長短期記憶網路要求其輸入的第一個維度是時間維，
        # 所以在獲得詞元表示之前，輸入會被轉置。
        # 輸出形狀為（時間步數，批次大小，詞向量維度）
        embeddings = self.embedding(inputs.T)
        # 回傳上一個隱藏層在不同時間步的隱狀態，
        # outputs的形狀是（時間步數，批次大小，2*隱藏單元數）
        outputs = self.encoder(embeddings)
        # 連結初始和最終時間步的隱狀態，作為全連線層的輸入，
        # 其形狀為（批次大小，4*隱藏單元數）
        encoding = np.concatenate((outputs[0], outputs[-1]), axis=1)
        outs = self.decoder(encoding)
        return outs
```

```{.python .input}
#@tab pytorch
class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 將bidirectional設定為True以獲取雙向迴圈神經網路
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                                bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # inputs的形狀是（批次大小，時間步數）
        # 因為長短期記憶網路要求其輸入的第一個維度是時間維，
        # 所以在獲得詞元表示之前，輸入會被轉置。
        # 輸出形狀為（時間步數，批次大小，詞向量維度）
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # 回傳上一個隱藏層在不同時間步的隱狀態，
        # outputs的形狀是（時間步數，批次大小，2*隱藏單元數）
        outputs, _ = self.encoder(embeddings)
        # 連結初始和最終時間步的隱狀態，作為全連線層的輸入，
        # 其形狀為（批次大小，4*隱藏單元數）
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs
```

```{.python .input}
#@tab paddle
class BiRNN(nn.Layer):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 將direction設定為'bidirect'或'bidirectional'以獲取雙向迴圈神經網路
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                                direction='bidirect',time_major=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # inputs的形狀是（批次大小，時間步數）
        # 因為長短期記憶網路要求其輸入的第一個維度是時間維，
        # 所以在獲得詞元表示之前，輸入會被轉置。
        # 輸出形狀為（時間步數，批次大小，詞向量維度）
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # 回傳上一個隱藏層在不同時間步的隱狀態，
        # outputs的形狀是（時間步數，批次大小，2*隱藏單元數）
        outputs, _ = self.encoder(embeddings)
        # 連結初始和最終時間步的隱狀態，作為全連線層的輸入，
        # 其形狀為（批次大小，4*隱藏單元數）
        encoding = paddle.concat((outputs[0], outputs[-1]), axis=1)
        outs = self.decoder(encoding)
        return outs
```

讓我們構造一個具有兩個隱藏層的雙向迴圈神經網路來表示單個文字以進行情感分析。

```{.python .input}
#@tab all
embed_size, num_hiddens, num_layers = 100, 100, 2
devices = d2l.try_all_gpus()
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
```

```{.python .input}
net.initialize(init.Xavier(), ctx=devices)
```

```{.python .input}
#@tab pytorch
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])
net.apply(init_weights);
```

```{.python .input}
#@tab paddle
def init_weights(layer):
    if isinstance(layer,(nn.Linear, nn.Embedding)):
        if isinstance(layer.weight, paddle.Tensor):
            nn.initializer.XavierUniform()(layer.weight)
    if isinstance(layer, nn.LSTM):
        for n, p in layer.named_parameters():
            if "weigth" in n:
                nn.initializer.XavierUniform()(p)
net.apply(init_weights)
```

## 載入預訓練的詞向量

下面，我們為詞表中的單詞載入預訓練的100維（需要與`embed_size`一致）的GloVe嵌入。

```{.python .input}
#@tab all
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
```

列印詞表中所有詞元向量的形狀。

```{.python .input}
#@tab all
embeds = glove_embedding[vocab.idx_to_token]
embeds.shape
```

我們使用這些預訓練的詞向量來表示評論中的詞元，並且在訓練期間不要更新這些向量。

```{.python .input}
net.embedding.weight.set_data(embeds)
net.embedding.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False
```

```{.python .input}
#@tab paddle
net.embedding.weight.set_value(embeds)
net.embedding.weight.stop_gradient = False
```

## 訓練和評估模型

現在我們可以訓練雙向迴圈神經網路進行情感分析。

```{.python .input}
lr, num_epochs = 0.01, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, 
    devices)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.01, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
    devices)
```

```{.python .input}
#@tab paddle
lr, num_epochs = 0.01, 2
trainer = paddle.optimizer.Adam(learning_rate=lr,parameters=net.parameters())
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
    devices)
```

我們定義以下函式來使用訓練好的模型`net`預測文字序列的情感。

```{.python .input}
#@save
def predict_sentiment(net, vocab, sequence):
    """預測文字序列的情感"""
    sequence = np.array(vocab[sequence.split()], ctx=d2l.try_gpu())
    label = np.argmax(net(sequence.reshape(1, -1)), axis=1)
    return 'positive' if label == 1 else 'negative'
```

```{.python .input}
#@tab pytorch
#@save
def predict_sentiment(net, vocab, sequence):
    """預測文字序列的情感"""
    sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'
```

```{.python .input}
#@tab paddle
#@save
def predict_sentiment(net, vocab, sequence):
    """預測文字序列的情感"""
    sequence = paddle.to_tensor(vocab[sequence.split()], place=d2l.try_gpu())
    label = paddle.argmax(net(sequence.reshape((1, -1))), axis=1)
    return 'positive' if label == 1 else 'negative'
```

最後，讓我們使用訓練好的模型對兩個簡單的句子進行情感預測。

```{.python .input}
#@tab all
predict_sentiment(net, vocab, 'this movie is so great')
```

```{.python .input}
#@tab all
predict_sentiment(net, vocab, 'this movie is so bad')
```

## 小結

* 預訓練的詞向量可以表示文字序列中的各個詞元。
* 雙向迴圈神經網路可以表示文字序列。例如透過連結初始和最終時間步的隱狀態，可以使用全連線的層將該單個文字表示轉換為類別。

## 練習

1. 增加迭代輪數可以提高訓練和測試的準確性嗎？調優其他超引數怎麼樣？
1. 使用較大的預訓練詞向量，例如300維的GloVe嵌入。它是否提高了分類精度？
1. 是否可以透過spaCy詞元化來提高分類精度？需要安裝Spacy（`pip install spacy`）和英語語言套件（`python -m spacy download en`）。在程式碼中，首先匯入Spacy（`import spacy`）。然後，載入Spacy英語軟體套件（`spacy_en = spacy.load('en')`）。最後，定義函式`def tokenizer(text): return [tok.text for tok in spacy_en.tokenizer(text)]`並替換原來的`tokenizer`函式。請注意GloVe和spaCy中短語標記的不同形式。例如，短語標記『new york』在GloVe中的形式是『new-york』，而在spaCy詞元化之後的形式是『new york』。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/5723)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/5724)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11826)
:end_tab:
