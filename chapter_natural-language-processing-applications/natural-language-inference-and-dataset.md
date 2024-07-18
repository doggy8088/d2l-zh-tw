# 自然語言推斷與資料集
:label:`sec_natural-language-inference-and-dataset`

在 :numref:`sec_sentiment`中，我們討論了情感分析問題。這個任務的目的是將單個文字序列分類到預定義的類別中，例如一組情感極性中。然而，當需要決定一個句子是否可以從另一個句子推斷出來，或者需要透過識別語義等價的句子來消除句子間冗餘時，知道如何對一個文字序列進行分類是不夠的。相反，我們需要能夠對成對的文字序列進行推斷。

## 自然語言推斷

*自然語言推斷*（natural language inference）主要研究
*假設*（hypothesis）是否可以從*前提*（premise）中推斷出來，
其中兩者都是文字序列。
換言之，自然語言推斷決定了一對文字序列之間的邏輯關係。這類關係通常分為三種型別：

* *蘊涵*（entailment）：假設可以從前提中推斷出來。
* *矛盾*（contradiction）：假設的否定可以從前提中推斷出來。
* *中性*（neutral）：所有其他情況。

自然語言推斷也被稱為識別文字蘊涵任務。
例如，下面的一個文字對將被貼上『蘊涵』的標籤，因為假設中的『表白』可以從前提中的『擁抱』中推斷出來。

>前提：兩個女人擁抱在一起。

>假設：兩個女人在示愛。

下面是一個『矛盾』的例子，因為『執行編碼範例』表示『不睡覺』，而不是『睡覺』。

>前提：一名男子正在執行Dive Into Deep Learning的編碼範例。

>假設：該男子正在睡覺。

第三個例子顯示了一種『中性』關係，因為『正在為我們表演』這一事實無法推斷出『出名』或『不出名』。

>前提：音樂家們正在為我們表演。

>假設：音樂家很有名。

自然語言推斷一直是理解自然語言的中心話題。它有著廣泛的應用，從訊息檢索到開放領域的問答。為了研究這個問題，我們將首先研究一個流行的自然語言推斷基準資料集。

## 斯坦福自然語言推斷（SNLI）資料集

[**斯坦福自然語言推斷語料庫（Stanford Natural Language Inference，SNLI）**]是由500000多個帶標籤的英語句子對組成的集合 :cite:`Bowman.Angeli.Potts.ea.2015`。我們在路徑`../data/snli_1.0`中下載並存儲提取的SNLI資料集。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import re

npx.set_np()

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
import re

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
import os
import re

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

### [**讀取資料集**]

原始的SNLI資料集包含的訊息比我們在實驗中真正需要的訊息豐富得多。因此，我們定義函式`read_snli`以僅提取資料集的一部分，然後回傳前提、假設及其標籤的清單。

```{.python .input}
#@tab all
#@save
def read_snli(data_dir, is_train):
    """將SNLI資料集解析為前提、假設和標籤"""
    def extract_text(s):
        # 刪除我們不會使用的訊息
        s = re.sub('\\(', '', s) 
        s = re.sub('\\)', '', s)
        # 用一個空格替換兩個或多個連續的空格
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] \
                in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels
```

現在讓我們[**列印前3對**]前提和假設，以及它們的標籤（『0』『1』和『2』分別對應於『蘊涵』『矛盾』和『中性』）。

```{.python .input}
#@tab all
train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('前提：', x0)
    print('假設：', x1)
    print('標籤：', y)
```

訓練集約有550000對，測試集約有10000對。下面顯示了訓練集和測試集中的三個[**標籤『蘊涵』『矛盾』和『中性』是平衡的**]。

```{.python .input}
#@tab all
test_data = read_snli(data_dir, is_train=False)
for data in [train_data, test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])
```

### [**定義用於載入資料集的類**]

下面我們來定義一個用於載入SNLI資料集的類。類建構式函式中的變數`num_steps`指定文字序列的長度，使得每個小批次序列將具有相同的形狀。換句話說，在較長序列中的前`num_steps`個標記之後的標記被截斷，而特殊標記『&lt;pad&gt;』將被附加到較短的序列後，直到它們的長度變為`num_steps`。透過實現`__getitem__`功能，我們可以任意訪問帶有索引`idx`的前提、假設和標籤。

```{.python .input}
#@save
class SNLIDataset(gluon.data.Dataset):
    """用於載入SNLI資料集的自定義資料集"""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + \
                all_hypothesis_tokens, min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = np.array(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return np.array([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

```{.python .input}
#@tab pytorch
#@save
class SNLIDataset(torch.utils.data.Dataset):
    """用於載入SNLI資料集的自定義資料集"""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + \
                all_hypothesis_tokens, min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

```{.python .input}
#@tab paddle
#@save
class SNLIDataset(paddle.io.Dataset):
    """用於載入SNLI資料集的自定義資料集"""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + \
                all_hypothesis_tokens, min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = paddle.to_tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return paddle.to_tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]
    
    def __len__(self):
        return len(self.premises)
```

### [**整合程式碼**]

現在，我們可以呼叫`read_snli`函式和`SNLIDataset`類來下載SNLI資料集，並回傳訓練集和測試集的`DataLoader`實例，以及訓練集的詞表。值得注意的是，我們必須使用從訓練集構造的詞表作為測試集的詞表。因此，在訓練集中訓練的模型將不知道來自測試集的任何新詞元。

```{.python .input}
#@save
def load_data_snli(batch_size, num_steps=50):
    """下載SNLI資料集並回傳資料迭代器和詞表"""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    test_iter = gluon.data.DataLoader(test_set, batch_size, shuffle=False,
                                      num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_snli(batch_size, num_steps=50):
    """下載SNLI資料集並回傳資料迭代器和詞表"""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

```{.python .input}
#@tab paddle
#@save
def load_data_snli(batch_size, num_steps=50):
    """下載SNLI資料集並回傳資料迭代器和詞表"""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = paddle.io.DataLoader(train_set,batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      return_list=True)
                                             
    test_iter = paddle.io.DataLoader(test_set, batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     return_list=True)
    return train_iter, test_iter, train_set.vocab
```

在這裡，我們將批次大小設定為128時，將序列長度設定為50，並呼叫`load_data_snli`函式來獲取資料迭代器和詞表。然後我們列印詞表大小。

```{.python .input}
#@tab all
train_iter, test_iter, vocab = load_data_snli(128, 50)
len(vocab)
```

現在我們列印第一個小批次的形狀。與情感分析相反，我們有分別代表前提和假設的兩個輸入`X[0]`和`X[1]`。

```{.python .input}
#@tab all
for X, Y in train_iter:
    print(X[0].shape)
    print(X[1].shape)
    print(Y.shape)
    break
```

## 小結

* 自然語言推斷研究『假設』是否可以從『前提』推斷出來，其中兩者都是文字序列。
* 在自然語言推斷中，前提和假設之間的關係包括蘊涵關係、矛盾關係和中性關係。
* 斯坦福自然語言推斷（SNLI）語料庫是一個比較流行的自然語言推斷基準資料集。

## 練習

1. 機器翻譯長期以來一直是基於翻譯輸出和翻譯真實值之間的表面$n$元語法對應來進行評估的。可以設計一種用自然語言推斷來評價機器翻譯結果的方法嗎？
1. 我們如何更改超引數以減小詞表大小？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/5721)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/5722)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11828)
:end_tab:
