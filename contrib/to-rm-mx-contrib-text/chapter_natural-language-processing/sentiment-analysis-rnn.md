# 文字情感分類：使用迴圈神經網路

文字分類是自然語言處理的一個常見任務，它把一段不定長的文字序列變換為文字的類別。本節關注它的一個子問題：使用文字情感分類來分析文字作者的情緒。這個問題也叫情感分析（sentiment analysis），並有著廣泛的應用。例如，我們可以分析使用者對產品的評論並統計使用者的滿意度，或者分析使用者對市場行情的情緒並用以預測接下來的行情。

同求近義詞和類比詞一樣，文字分類也屬於詞嵌入的下游應用。本節將應用預訓練的詞向量和含多個隱藏層的雙向迴圈神經網路，來判斷一段不定長的文字序列中包含的是正面還是負面的情緒。

在實驗開始前，匯入所需的套件或模組。

```{.python .input  n=1}
import collections
import d2lzh as d2l
from d2lzh import text
from mxnet import gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn, utils as gutils
import os
import random
import tarfile
```

## 文字情感分類資料集

我們使用斯坦福的IMDb資料集（Stanford's Large Movie Review Dataset）作為文字情感分類的資料集 [1]。這個資料集分為訓練和測試用的兩個資料集，分別包含25,000條從IMDb下載的關於電影的評論。在每個資料集中，標籤為『正面』和『負面』的評論數量相等。

###  讀取資料集

首先下載這個資料集到`../data`路徑下，然後解壓至`../data/aclImdb`路徑下。

```{.python .input  n=3}
# 本函式已儲存在d2lzh套件中方便以後使用
def download_imdb(data_dir='../data'):
    url = ('http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
    sha1 = '01ada507287d82875905620988597833ad4e0903'
    fname = gutils.download(url, data_dir, sha1_hash=sha1)
    with tarfile.open(fname, 'r') as f:
        f.extractall(data_dir)

download_imdb()
```

接下來，讀取訓練資料集和測試資料集。每個樣本是一條評論及其對應的標籤：1表示『正面』，0表示『負面』。

```{.python .input  n=13}
def read_imdb(folder='train'):  # 本函式已儲存在d2lzh套件中方便以後使用
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join('../data/aclImdb/', folder, label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data

train_data, test_data = read_imdb('train'), read_imdb('test')
```

### 預處理資料集

我們需要對每條評論做分詞，從而得到分好詞的評論。這裡定義的`get_tokenized_imdb`函式使用最簡單的方法：基於空格進行分詞。

```{.python .input  n=14}
def get_tokenized_imdb(data):  # 本函式已儲存在d2lzh套件中方便以後使用
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data]
```

現在，我們可以根據分好詞的訓練資料集來建立詞典了。我們在這裡過濾掉了出現次數少於5的詞。

```{.python .input  n=28}
def get_vocab_imdb(data):  # 本函式已儲存在d2lzh套件中方便以後使用
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return text.vocab.Vocabulary(counter, min_freq=5,
                                 reserved_tokens=['<pad>'])

vocab = get_vocab_imdb(train_data)
'# words in vocab:', len(vocab)
```

因為每條評論長度不一致所以不能直接組合成小批次，我們定義`preprocess_imdb`函式對每條評論進行分詞，並透過詞典轉換成詞索引，然後透過截斷或者補『&lt;pad&gt;』（padding）符號來將每條評論長度固定成500。

```{.python .input  n=44}
def preprocess_imdb(data, vocab):  # 本函式已儲存在d2lzh套件中方便以後使用
    max_l = 500  # 將每條評論透過截斷或者補'<pad>'，使得長度變成500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [
            vocab.token_to_idx['<pad>']] * (max_l - len(x))

    tokenized_data = get_tokenized_imdb(data)
    features = nd.array([pad(vocab.to_indices(x)) for x in tokenized_data])
    labels = nd.array([score for _, score in data])
    return features, labels
```

### 建立資料迭代器

現在，我們建立資料迭代器。每次迭代將回傳一個小批次的資料。

```{.python .input}
batch_size = 64
train_set = gdata.ArrayDataset(*preprocess_imdb(train_data, vocab))
test_set = gdata.ArrayDataset(*preprocess_imdb(test_data, vocab))
train_iter = gdata.DataLoader(train_set, batch_size, shuffle=True)
test_iter = gdata.DataLoader(test_set, batch_size)
```

列印第一個小批次資料的形狀以及訓練集中小批次的個數。

```{.python .input}
for X, y in train_iter:
    print('X', X.shape, 'y', y.shape)
    break
'#batches:', len(train_iter)
```

## 使用迴圈神經網路的模型

在這個模型中，每個詞先透過嵌入層得到特徵向量。然後，我們使用雙向迴圈神經網路對特徵序列進一步編碼得到序列訊息。最後，我們將編碼的序列訊息透過全連線層變換為輸出。具體來說，我們可以將雙向長短期記憶在最初時間步和最終時間步的隱狀態連結，作為特徵序列的表徵傳遞給輸出層分類。在下面實現的`BiRNN`類別中，`Embedding`實例即嵌入層，`LSTM`實例即為序列編碼的隱藏層，`Dense`實例即生成分類結果的輸出層。

```{.python .input  n=46}
class BiRNN(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # bidirectional設為True即得到雙向迴圈神經網路
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # inputs的形狀是(批次大小, 詞數)，因為LSTM需要將序列作為第一維，所以將輸入轉置後
        # 再提取詞特徵，輸出形狀為(詞數, 批次大小, 詞向量維度)
        embeddings = self.embedding(inputs.T)
        # rnn.LSTM只傳入輸入embeddings，因此只回傳最後一層的隱藏層在各時間步的隱狀態。
        # outputs形狀是(詞數, 批次大小, 2 * 隱藏單元個數)
        outputs = self.encoder(embeddings)
        # 連結初始時間步和最終時間步的隱狀態作為全連線層輸入。它的形狀為
        # (批次大小, 4 * 隱藏單元個數)。
        encoding = nd.concat(outputs[0], outputs[-1])
        outs = self.decoder(encoding)
        return outs
```

建立一個含兩個隱藏層的雙向迴圈神經網路。

```{.python .input}
embed_size, num_hiddens, num_layers, ctx = 100, 100, 2, d2l.try_all_gpus()
net = BiRNN(vocab, embed_size, num_hiddens, num_layers)
net.initialize(init.Xavier(), ctx=ctx)
```

### 載入預訓練的詞向量

由於情感分類的訓練資料集並不是很大，為應對過擬合，我們將直接使用在更大規模語料上預訓練的詞向量作為每個詞的特徵向量。這裡，我們為詞典`vocab`中的每個詞載入100維的GloVe詞向量。

```{.python .input  n=45}
glove_embedding = text.embedding.create(
    'glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)
```

然後，我們將用這些詞向量作為評論中每個詞的特徵向量。注意，預訓練詞向量的維度需要與建立的模型中的嵌入層輸出大小`embed_size`一致。此外，在訓練中我們不再更新這些詞向量。

```{.python .input  n=47}
net.embedding.weight.set_data(glove_embedding.idx_to_vec)
net.embedding.collect_params().setattr('grad_req', 'null')
```

### 訓練模型

這時候就可以開始訓練模型了。

```{.python .input  n=48}
lr, num_epochs = 0.01, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)
```

最後，定義預測函式。

```{.python .input  n=49}
# 本函式已儲存在d2lzh套件中方便以後使用
def predict_sentiment(net, vocab, sentence):
    sentence = nd.array(vocab.to_indices(sentence), ctx=d2l.try_gpu())
    label = nd.argmax(net(sentence.reshape((1, -1))), axis=1)
    return 'positive' if label.asscalar() == 1 else 'negative'
```

下面使用訓練好的模型對兩個簡單句子的情感進行分類。

```{.python .input  n=50}
predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great'])
```

```{.python .input}
predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad'])
```

## 小結

* 文字分類把一段不定長的文字序列變換為文字的類別。它屬於詞嵌入的下游應用。
* 可以應用預訓練的詞向量和迴圈神經網路對文字的情感進行分類。


## 練習

* 增加迭代週期。訓練後的模型能在訓練和測試資料集上得到怎樣的精度？再調節其他超引數試試？

* 使用更大的預訓練詞向量，如300維的GloVe詞向量，能否提升分類精度？

* 使用spaCy分詞工具，能否提升分類精度？你需要安裝spaCy（`pip install spacy`），並且安裝英文套件（`python -m spacy download en`）。在程式碼中，先導入spaCy（`import spacy`）。然後載入spaCy英文套件（`spacy_en = spacy.load('en')`）。最後定義函式`def tokenizer(text): return [tok.text for tok in spacy_en.tokenizer(text)]`並替換原來的基於空格分詞的`tokenizer`函式。需要注意的是，GloVe詞向量對於名詞片語的儲存方式是用『-』連線各個單詞，例如，片語『new york』在GloVe詞向量中的表示為『new-york』，而使用spaCy分詞之後『new york』的儲存可能是『new york』。






## 參考文獻

[1] Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011, June). Learning word vectors for sentiment analysis. In Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies-volume 1 (pp. 142-150). Association for Computational Linguistics.

## 掃碼直達[討論區](https://discuss.gluon.ai/t/topic/6155)

![](../img/qr_sentiment-analysis.svg)
