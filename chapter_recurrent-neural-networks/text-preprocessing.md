# 文字預處理
:label:`sec_text_preprocessing`

對於序列資料處理問題，我們在 :numref:`sec_sequence`中
評估了所需的統計工具和預測時面臨的挑戰。
這樣的資料存在許多種形式，文字是最常見例子之一。
例如，一篇文章可以被簡單地看作一串單詞序列，甚至是一串字元序列。
本節中，我們將解析文字的常見預處理步驟。
這些步驟通常包括：

1. 將文字作為字串載入到記憶體中。
1. 將字串拆分為詞元（如單詞和字元）。
1. 建立一個詞表，將拆分的詞元對映到數字索引。
1. 將文字轉換為數字索引序列，方便模型操作。

```{.python .input}
import collections
from d2l import mxnet as d2l
import re
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import re
```

```{.python .input}
#@tab tensorflow
import collections
from d2l import tensorflow as d2l
import re
```

```{.python .input}
#@tab paddle
import collections
from d2l import paddle as d2l
import re
```

## 讀取資料集

首先，我們從H.G.Well的[時光機器](https://www.gutenberg.org/ebooks/35)中載入文字。
這是一個相當小的語料庫，只有30000多個單詞，但足夠我們小試牛刀，
而現實中的文件集合可能會包含數十億個單詞。
下面的函式(**將資料集讀取到由多條文本行組成的清單中**)，其中每條文字行都是一個字串。
為簡單起見，我們在這裡忽略了標點符號和字母大寫。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """將時間機器資料集載入到文字行的清單中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# 文字總行數: {len(lines)}')
print(lines[0])
print(lines[10])
```

## 詞元化

下面的`tokenize`函式將文字行列表（`lines`）作為輸入，
清單中的每個元素是一個文字序列（如一條文本行）。
[**每個文字序列又被拆分成一個詞元清單**]，*詞元*（token）是文字的基本單位。
最後，回傳一個由詞元清單組成的清單，其中的每個詞元都是一個字串（string）。

```{.python .input}
#@tab all
def tokenize(lines, token='word'):  #@save
    """將文字行拆分為單詞或字元詞元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('錯誤：未知詞元型別：' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])
```

## 詞表

詞元的型別是字串，而模型需要的輸入是數字，因此這種型別不方便模型使用。
現在，讓我們[**建立一個字典，通常也叫做*詞表*（vocabulary），
用來將字串型別的詞元對映到從$0$開始的數字索引中**]。
我們先將訓練集中的所有文件合併在一起，對它們的唯一詞元進行統計，
得到的統計結果稱之為*語料*（corpus）。
然後根據每個唯一詞元的出現頻率，為其分配一個數字索引。
很少出現的詞元通常被移除，這可以降低複雜性。
另外，語料庫中不存在或已刪除的任何詞元都將對映到一個特定的未知詞元『&lt;unk&gt;』。
我們可以選擇增加一個清單，用於儲存那些被保留的詞元，
例如：填充詞元（『&lt;pad&gt;』）；
序列開始詞元（『&lt;bos&gt;』）；
序列結束詞元（『&lt;eos&gt;』）。

```{.python .input}
#@tab all
class Vocab:  #@save
    """文字詞表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = [] 
        # 按出現頻率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知詞元的索引為0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
        
    @property
    def unk(self):  # 未知詞元的索引為0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):  #@save
    """統計詞元的頻率"""
    # 這裡的tokens是1D清單或2D清單
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 將詞元清單展平成一個清單
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
```

我們首先使用時光機器資料集作為語料庫來[**建立詞表**]，然後列印前幾個高頻詞元及其索引。

```{.python .input}
#@tab all
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])
```

現在，我們可以(**將每一條文字行轉換成一個數字索引清單**)。

```{.python .input}
#@tab all
for i in [0, 10]:
    print('文字:', tokens[i])
    print('索引:', vocab[tokens[i]])
```

## 整合所有功能

在使用上述函式時，我們[**將所有功能打包到`load_corpus_time_machine`函式中**]，
該函式回傳`corpus`（詞元索引清單）和`vocab`（時光機器語料庫的詞表）。
我們在這裡所做的改變是：

1. 為了簡化後面章節中的訓練，我們使用字元（而不是單詞）實現文字詞元化；
1. 時光機器資料集中的每個文字行不一定是一個句子或一個段落，還可能是一個單詞，因此回傳的`corpus`僅處理為單個清單，而不是使用多詞元清單構成的一個清單。

```{.python .input}
#@tab all
def load_corpus_time_machine(max_tokens=-1):  #@save
    """回傳時光機器資料集的詞元索引清單和詞表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因為時光機器資料集中的每個文字行不一定是一個句子或一個段落，
    # 所以將所有文字行展平到一個清單中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)
```

## 小結

* 文字是序列資料的一種最常見的形式之一。
* 為了對文字進行預處理，我們通常將文字拆分為詞元，建立詞表將詞元字串對映為數字索引，並將文字資料轉換為詞元索引以供模型操作。

## 練習

1. 詞元化是一個關鍵的預處理步驟，它因語言而異。嘗試找到另外三種常用的詞元化文字的方法。
1. 在本節的實驗中，將文字詞元為單詞和更改`Vocab`實例的`min_freq`引數。這對詞表大小有何影響？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2093)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2094)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/2095)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11796)
:end_tab:
