# 情感分析及資料集
:label:`sec_sentiment`

隨著線上社交媒體和評論平台的快速發展，大量評論的資料被記錄下來。這些資料具有支援決策過程的巨大潛力。
*情感分析*（sentiment analysis）研究人們在文字中
（如產品評論、部落格評論和論壇討論等）『隱藏』的情緒。
它在廣泛應用於政治（如公眾對政策的情緒分析）、
金融（如市場情緒分析）和營銷（如產品研究和品牌管理）等領域。

由於情感可以被分類為離散的極性或尺度（例如，積極的和消極的），我們可以將情感分析看作一項文字分類任務，它將可變長度的文字序列轉換為固定長度的文字類別。在本章中，我們將使用斯坦福大學的[大型電影評論資料集（large movie review dataset）](https://ai.stanford.edu/~amaas/data/sentiment/)進行情感分析。它由一個訓練集和一個測試集組成，其中包含從IMDb下載的25000個電影評論。在這兩個資料集中，『積極』和『消極』標籤的數量相同，表示不同的情感極性。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
import os
```

##  讀取資料集

首先，下載並提取路徑`../data/aclImdb`中的IMDb評論資料集。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['aclImdb'] = (
    'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
    '01ada507287d82875905620988597833ad4e0903')

data_dir = d2l.download_extract('aclImdb', 'aclImdb')
```

接下來，讀取訓練和測試資料集。每個樣本都是一個評論及其標籤：1表示『積極』，0表示『消極』。

```{.python .input}
#@tab all
#@save
def read_imdb(data_dir, is_train):
    """讀取IMDb評論資料集文字序列和標籤"""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

train_data = read_imdb(data_dir, is_train=True)
print('訓練集數目：', len(train_data[0]))
for x, y in zip(train_data[0][:3], train_data[1][:3]):
    print('標籤：', y, 'review:', x[0:60])
```

## 預處理資料集

將每個單詞作為一個詞元，過濾掉出現不到5次的單詞，我們從訓練資料集中建立一個詞表。

```{.python .input}
#@tab all
train_tokens = d2l.tokenize(train_data[0], token='word')
vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
```

在詞元化之後，讓我們繪製評論詞元長度的直方圖。

```{.python .input}
#@tab all
d2l.set_figsize()
d2l.plt.xlabel('# tokens per review')
d2l.plt.ylabel('count')
d2l.plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50));
```

正如我們所料，評論的長度各不相同。為了每次處理一小批次這樣的評論，我們透過截斷和填充將每個評論的長度設定為500。這類似於 :numref:`sec_machine_translation`中對機器翻譯資料集的預處理步驟。

```{.python .input}
#@tab all
num_steps = 500  # 序列長度
train_features = d2l.tensor([d2l.truncate_pad(
    vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
print(train_features.shape)
```

## 建立資料迭代器

現在我們可以建立資料迭代器了。在每次迭代中，都會回傳一小批次樣本。

```{.python .input}
train_iter = d2l.load_array((train_features, train_data[1]), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('小批次數目：', len(train_iter))
```

```{.python .input}
#@tab pytorch
train_iter = d2l.load_array((train_features, 
    torch.tensor(train_data[1])), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('小批次數目：', len(train_iter))
```

```{.python .input}
#@tab paddle
train_iter = d2l.load_array((train_features,
    d2l.tensor(train_data[1])), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('小批次數目：', len(train_iter))
```

## 整合程式碼

最後，我們將上述步驟封裝到`load_data_imdb`函式中。它回傳訓練和測試資料迭代器以及IMDb評論資料集的詞表。

```{.python .input}
#@save
def load_data_imdb(batch_size, num_steps=500):
    """回傳資料迭代器和IMDb評論資料集的詞表"""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, train_data[1]), batch_size)
    test_iter = d2l.load_array((test_features, test_data[1]), batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_imdb(batch_size, num_steps=500):
    """回傳資料迭代器和IMDb評論資料集的詞表"""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

```{.python .input}
#@tab paddle
#@save
def load_data_imdb(batch_size, num_steps=500):
    """回傳資料迭代器和IMDb評論資料集的詞表"""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = d2l.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = d2l.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, d2l.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, d2l.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

## 小結

* 情感分析研究人們在文字中的情感，這被認為是一個文字分類問題，它將可變長度的文字序列進行轉換轉換為固定長度的文字類別。
* 經過預處理後，我們可以使用詞表將IMDb評論資料集載入到資料迭代器中。

## 練習

1. 我們可以修改本節中的哪些超引數來加速訓練情感分析模型？
1. 請實現一個函式來將[Amazon reviews](https://snap.stanford.edu/data/web-Amazon.html)的資料集載入到資料迭代器中進行情感分析。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/5725)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/5726)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11825)
:end_tab:
