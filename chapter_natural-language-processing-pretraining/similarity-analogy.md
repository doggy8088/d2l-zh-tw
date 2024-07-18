# 詞的相似性和類比任務
:label:`sec_synonyms`

在 :numref:`sec_word2vec_pretraining`中，我們在一個小的資料集上訓練了一個word2vec模型，並使用它為一個輸入詞尋找語義相似的詞。實際上，在大型語料庫上預先訓練的詞向量可以應用於下游的自然語言處理任務，這將在後麵的 :numref:`chap_nlp_app`中討論。為了直觀地示範大型語料庫中預訓練詞向量的語義，讓我們將預訓練詞向量應用到詞的相似性和類比任務中。

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

## 載入預訓練詞向量

以下列出維度為50、100和300的預訓練GloVe嵌入，可從[GloVe網站](https://nlp.stanford.edu/projects/glove/)下載。預訓練的fastText嵌入有多種語言。這裡我們使用可以從[fastText網站](https://fasttext.cc/)下載300維度的英文版本（『wiki.en』）。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

#@save
d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

#@save
d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')

#@save
d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                           'c1816da3821ae9f43899be655002f6c723e91b88')
```

為了載入這些預訓練的GloVe和fastText嵌入，我們定義了以下`TokenEmbedding`類。

```{.python .input}
#@tab all
#@save
class TokenEmbedding:
    """GloVe嵌入"""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe網站：https://nlp.stanford.edu/projects/glove/
        # fastText網站：https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # 跳過標題訊息，例如fastText中的首行
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, d2l.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[d2l.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)
```

下面我們載入50維GloVe嵌入（在維基百科的子集上預訓練）。建立`TokenEmbedding`實例時，如果尚未下載指定的嵌入檔案，則必須下載該檔案。

```{.python .input}
#@tab all
glove_6b50d = TokenEmbedding('glove.6b.50d')
```

輸出詞表大小。詞表包含400000個詞（詞元）和一個特殊的未知詞元。

```{.python .input}
#@tab all
len(glove_6b50d)
```

我們可以得到詞表中一個單詞的索引，反之亦然。

```{.python .input}
#@tab all
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## 應用預訓練詞向量

使用載入的GloVe向量，我們將透過下面的詞相似性和類比任務中來展示詞向量的語義。

### 詞相似度

與 :numref:`subsec_apply-word-embed`類似，為了根據詞向量之間的餘弦相似性為輸入詞查詢語義相似的詞，我們實現了以下`knn`（$k$近鄰）函式。

```{.python .input}
def knn(W, x, k):
    # 增加1e-9以獲得數值穩定性
    cos = np.dot(W, x.reshape(-1,)) / (
        np.sqrt(np.sum(W * W, axis=1) + 1e-9) * np.sqrt((x * x).sum()))
    topk = npx.topk(cos, k=k, ret_typ='indices')
    return topk, [cos[int(i)] for i in topk]
```

```{.python .input}
#@tab pytorch
def knn(W, x, k):
    # 增加1e-9以獲得數值穩定性
    cos = torch.mv(W, x.reshape(-1,)) / (
        torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
        torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]
```

```{.python .input}
#@tab paddle
def knn(W, x, k):
    # 增加1e-9以獲得數值穩定性
    cos = paddle.mv(W, x) / (
        paddle.sqrt(paddle.sum(W * W, axis=1) + 1e-9) *
        paddle.sqrt((x * x).sum()))
    _, topk = paddle.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]
```

然後，我們使用`TokenEmbedding`的實例`embed`中預訓練好的詞向量來搜尋相似的詞。

```{.python .input}
#@tab all
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # 排除輸入詞
        print(f'{embed.idx_to_token[int(i)]}：cosine相似度={float(c):.3f}')
```

`glove_6b50d`中預訓練詞向量的詞表包含400000個詞和一個特殊的未知詞元。排除輸入詞和未知詞元后，我們在詞表中找到與『chip』一詞語義最相似的三個詞。

```{.python .input}
#@tab all
get_similar_tokens('chip', 3, glove_6b50d)
```

下面輸出與『baby』和『beautiful』相似的詞。

```{.python .input}
#@tab all
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.python .input}
#@tab all
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### 詞類比

除了找到相似的詞，我們還可以將詞向量應用到詞類比任務中。
例如，『man』 : 『woman』 :: 『son』 : 『daughter』是一個詞的類比。
『man』是對『woman』的類比，『son』是對『daughter』的類比。
具體來說，詞類比任務可以定義為：
對於單詞類比$a : b :: c : d$，給出前三個詞$a$、$b$和$c$，找到$d$。
用$\text{vec}(w)$表示詞$w$的向量，
為了完成這個類比，我們將找到一個詞，
其向量與$\text{vec}(c)+\text{vec}(b)-\text{vec}(a)$的結果最相似。

```{.python .input}
#@tab all
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # 刪除未知詞
```

讓我們使用載入的詞向量來驗證『male-female』類比。

```{.python .input}
#@tab all
get_analogy('man', 'woman', 'son', glove_6b50d)
```

下面完成一個『首都-國家』的類比：
『beijing』 : 『china』 :: 『tokyo』 : 『japan』。
這說明了預訓練詞向量中的語義。

```{.python .input}
#@tab all
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

另外，對於『bad』 : 『worst』 :: 『big』 : 『biggest』等『形容詞-形容詞最高級』的比喻，預訓練詞向量可以捕捉到句法訊息。

```{.python .input}
#@tab all
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

為了示範在預訓練詞向量中捕捉到的過去式概念，我們可以使用『現在式-過去式』的類比來測試句法：『do』 : 『did』 :: 『go』 : 『went』。

```{.python .input}
#@tab all
get_analogy('do', 'did', 'go', glove_6b50d)
```

## 小結

* 在實踐中，在大型語料庫上預先練的詞向量可以應用於下游的自然語言處理任務。
* 預訓練的詞向量可以應用於詞的相似性和類比任務。

## 練習

1. 使用`TokenEmbedding('wiki.en')`測試fastText結果。
1. 當詞表非常大時，我們怎樣才能更快地找到相似的詞或完成一個詞的類比呢？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/5745)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/5746)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11819)
:end_tab:
