# 文字情感分類：使用卷積神經網路（textCNN）

在『卷積神經網路』一章中我們探究瞭如何使用二維卷積神經網路來處理二維影象資料。在之前的語言模型和文字分類任務中，我們將文字資料看作只有一個維度的時間序列，並很自然地使用迴圈神經網路來表徵這樣的資料。其實，我們也可以將文字當作一維影象，從而可以用一維卷積神經網路來捕捉臨近詞之間的關聯。本節將介紹將卷積神經網路應用到文字分析的開創性工作之一：textCNN [1]。

首先匯入實驗所需的套件和模組。

```{.python .input  n=2}
import d2lzh as d2l
from d2lzh import text
from mxnet import gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
```

## 一維卷積層

在介紹模型前我們先來解釋一維卷積層的工作原理。與二維卷積層一樣，一維卷積層使用一維的互相關運算。在一維互相關運算中，卷積視窗從輸入陣列的最左方開始，按從左往右的順序，依次在輸入陣列上滑動。當卷積視窗滑動到某一位置時，視窗中的輸入子陣列與核數組按元素相乘並求和，得到輸出陣列中相應位置的元素。如圖10.4所示，輸入是一個寬為7的一維陣列，核數組的寬為2。可以看到輸出的寬度為$7-2+1=6$，且第一個元素是由輸入的最左邊的寬為2的子陣列與核數組按元素相乘後再相加得到的：$0\times1+1\times2=2$。

![一維互相關運算](../img/conv1d.svg)

下面我們將一維互相關運算實現在`corr1d`函式裡。它接受輸入陣列`X`和核陣列`K`，並輸出陣列`Y`。

```{.python .input  n=3}
def corr1d(X, K):
    w = K.shape[0]
    Y = nd.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y
```

讓我們復現圖10.4中一維互相關運算的結果。

```{.python .input  n=4}
X, K = nd.array([0, 1, 2, 3, 4, 5, 6]), nd.array([1, 2])
corr1d(X, K)
```

多輸入通道的一維互相關運算也與多輸入通道的二維互相關運算類似：在每個通道上，將核與相應的輸入做一維互相關運算，並將通道之間的結果相加得到輸出結果。圖10.5展示了含3個輸入通道的一維互相關運算，其中陰影部分為第一個輸出元素及其計算所使用的輸入和核陣列元素：$0\times1+1\times2+1\times3+2\times4+2\times(-1)+3\times(-3)=2$。

![含3個輸入通道的一維互相關運算](../img/conv1d-channel.svg)

讓我們復現圖10.5中多輸入通道的一維互相關運算的結果。

```{.python .input  n=5}
def corr1d_multi_in(X, K):
    # 首先沿著X和K的第0維（通道維）遍歷。然後使用*將結果清單變成add_n函式的位置引數
    #（positional argument）來進行相加
    return nd.add_n(*[corr1d(x, k) for x, k in zip(X, K)])

X = nd.array([[0, 1, 2, 3, 4, 5, 6],
              [1, 2, 3, 4, 5, 6, 7],
              [2, 3, 4, 5, 6, 7, 8]])
K = nd.array([[1, 2], [3, 4], [-1, -3]])
corr1d_multi_in(X, K)
```

由二維互相關運算的定義可知，多輸入通道的一維互相關運算可以看作單輸入通道的二維互相關運算。如圖10.6所示，我們也可以將圖10.5中多輸入通道的一維互相關運算以等價的單輸入通道的二維互相關運算呈現。這裡核的高等於輸入的高。圖10.6中的陰影部分為第一個輸出元素及其計算所使用的輸入和核陣列元素：$2\times(-1)+3\times(-3)+1\times3+2\times4+0\times1+1\times2=2$。

![單輸入通道的二維互相關運算](../img/conv1d-2d.svg)

圖10.4和圖10.5中的輸出都只有一個通道。我們在[『多輸入通道和多輸出通道』](../chapter_convolutional-neural-networks/channels.md)一節中介紹瞭如何在二維卷積層中指定多個輸出通道。類似地，我們也可以在一維卷積層指定多個輸出通道，從而拓展卷積層中的模型引數。


## 時序最大池化層

類似地，我們有一維池化層。textCNN中使用的時序最大池化（max-over-time pooling）層實際上對應一維全域最大池化層：假設輸入包含多個通道，各通道由不同時間步上的數值組成，各通道的輸出即該通道所有時間步中最大的數值。因此，時序最大池化層的輸入在各個通道上的時間步數可以不同。

為提升計算效能，我們常常將不同長度的時序樣本組成一個小批次，並透過在較短序列後附加特殊字元（如0）令批次中各時序樣本長度相同。這些人為新增的特殊字元當然是無意義的。由於時序最大池化的主要目的是抓取時序中最重要的特徵，它通常能使模型不受人為新增字元的影響。


## 讀取和預處理IMDb資料集

我們依然使用和上一節中相同的IMDb資料集做情感分析。以下讀取和預處理資料集的步驟與上一節中的相同。

```{.python .input  n=2}
batch_size = 64
d2l.download_imdb()
train_data, test_data = d2l.read_imdb('train'), d2l.read_imdb('test')
vocab = d2l.get_vocab_imdb(train_data)
train_iter = gdata.DataLoader(gdata.ArrayDataset(
    *d2l.preprocess_imdb(train_data, vocab)), batch_size, shuffle=True)
test_iter = gdata.DataLoader(gdata.ArrayDataset(
    *d2l.preprocess_imdb(test_data, vocab)), batch_size)
```

## textCNN模型

textCNN模型主要使用了一維卷積層和時序最大池化層。假設輸入的文字序列由$n$個片語成，每個詞用$d$維的詞向量表示。那麼輸入樣本的寬為$n$，高為1，輸入通道數為$d$。textCNN的計算主要分為以下幾步。

1. 定義多個一維卷積核，並使用這些卷積核對輸入分別做卷積計算。寬度不同的卷積核可能會捕捉到不同個數的相鄰詞的相關性。
2. 對輸出的所有通道分別做時序最大池化，再將這些通道的池化輸出值連結為向量。
3. 透過全連線層將連結後的向量變換為有關各類別的輸出。這一步可以使用丟棄層應對過擬合。

圖10.7用一個例子解釋了textCNN的設計。這裡的輸入是一個有11個詞的句子，每個詞用6維詞向量表示。因此輸入序列的寬為11，輸入通道數為6。給定2個一維卷積核，核寬分別為2和4，輸出通道數分別設為4和5。因此，一維卷積計算後，4個輸出通道的寬為$11-2+1=10$，而其他5個通道的寬為$11-4+1=8$。儘管每個通道的寬不同，我們依然可以對各個通道做時序最大池化，並將9個通道的池化輸出連結成一個9維向量。最終，使用全連線將9維向量變換為2維輸出，即正面情感和負面情感的預測。

![textCNN的設計](../img/textcnn.svg)

下面我們來實現textCNN模型。與上一節相比，除了用一維卷積層替換迴圈神經網路外，這裡我們還使用了兩個嵌入層，一個的權重固定，另一個的權重則參與訓練。

```{.python .input  n=10}
class TextCNN(nn.Block):
    def __init__(self, vocab, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # 不參與訓練的嵌入層
        self.constant_embedding = nn.Embedding(len(vocab), embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(2)
        # 時序最大池化層沒有權重，所以可以共用一個實例
        self.pool = nn.GlobalMaxPool1D()
        self.convs = nn.Sequential()  # 建立多個一維卷積層
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.add(nn.Conv1D(c, k, activation='relu'))

    def forward(self, inputs):
        # 將兩個形狀是(批次大小, 詞數, 詞向量維度)的嵌入層的輸出按詞向量連結
        embeddings = nd.concat(
            self.embedding(inputs), self.constant_embedding(inputs), dim=2)
        # 根據Conv1D要求的輸入格式，將詞向量維，即一維卷積層的通道維，變換到前一維
        embeddings = embeddings.transpose((0, 2, 1))
        # 對於每個一維卷積層，在時序最大池化後會得到一個形狀為(批次大小, 通道大小, 1)的
        # NDArray。使用flatten函式去掉最後一維，然後在通道維上連結
        encoding = nd.concat(*[nd.flatten(
            self.pool(conv(embeddings))) for conv in self.convs], dim=1)
        # 應用暫退法後使用全連線層得到輸出
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```

建立一個`TextCNN`實例。它有3個卷積層，它們的核寬分別為3、4和5，輸出通道數均為100。

```{.python .input}
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
ctx = d2l.try_all_gpus()
net = TextCNN(vocab, embed_size, kernel_sizes, nums_channels)
net.initialize(init.Xavier(), ctx=ctx)
```

### 載入預訓練的詞向量

同上一節一樣，載入預訓練的100維GloVe詞向量，並分別初始化嵌入層`embedding`和`constant_embedding`，前者權重參與訓練，而後者權重固定。

```{.python .input  n=7}
glove_embedding = text.embedding.create(
    'glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)
net.embedding.weight.set_data(glove_embedding.idx_to_vec)
net.constant_embedding.weight.set_data(glove_embedding.idx_to_vec)
net.constant_embedding.collect_params().setattr('grad_req', 'null')
```

### 訓練模型

現在就可以訓練模型了。

```{.python .input  n=30}
lr, num_epochs = 0.001, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)
```

下面，我們使用訓練好的模型對兩個簡單句子的情感進行分類。

```{.python .input}
d2l.predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great'])
```

```{.python .input}
d2l.predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad'])
```

## 小結

* 可以使用一維卷積來表徵時序資料。
* 多輸入通道的一維互相關運算可以看作單輸入通道的二維互相關運算。
* 時序最大池化層的輸入在各個通道上的時間步數可以不同。
* textCNN主要使用了一維卷積層和時序最大池化層。


## 練習

* 動手調參，從精度和執行效率比較情感分析的兩類方法：使用迴圈神經網路和使用卷積神經網路。
* 使用上一節練習中介紹的3種方法（調節超引數、使用更大的預訓練詞向量和使用spaCy分詞工具），能使模型在測試集上的精度進一步提高嗎？
* 還能將textCNN應用於自然語言處理的哪些任務中？





## 參考文獻

[1] Kim, Y. (2014). Convolutional neural networks for sentence classification. arXiv preprint arXiv:1408.5882.

## 掃碼直達[討論區](https://discuss.gluon.ai/t/topic/7762)

![](../img/qr_sentiment-analysis-cnn.svg)
