# 求近義詞和類比詞

在[『word2vec的實現』](./word2vec-gluon.md)一節中，我們在小規模資料集上訓練了一個word2vec詞嵌入模型，並透過詞向量的餘弦相似度搜索近義詞。實際中，在大規模語料上預訓練的詞向量常常可以應用到下游自然語言處理任務中。本節將示範如何用這些預訓練的詞向量來求近義詞和類比詞。我們還將在後面兩節中繼續應用預訓練的詞向量。

## 使用預訓練的詞向量

MXNet的`contrib.text`套件提供了與自然語言處理相關的函式和類（更多參見GluonNLP工具包 [1]）。下面檢視它目前提供的預訓練詞嵌入的名稱。

```{.python .input}
from mxnet import nd
from d2lzh import text

text.embedding.get_pretrained_file_names().keys()
```

給定詞嵌入名稱，可以檢視該詞嵌入提供了哪些預訓練的模型。每個模型的詞向量維度可能不同，或是在不同資料集上預訓練得到的。

```{.python .input  n=35}
print(text.embedding.get_pretrained_file_names('glove'))
```

預訓練的GloVe模型的命名規範大致是『模型.（資料集.）資料集詞數.詞向量維度.txt』。更多訊息可以參考GloVe和fastText的專案網站 [2,3]。下面我們使用基於維基百科子集預訓練的50維GloVe詞向量。第一次建立預訓練詞向量實例時會自動下載相應的詞向量，因此需要聯網。

```{.python .input  n=11}
glove_6b50d = text.embedding.create(
    'glove', pretrained_file_name='glove.6B.50d.txt')
```

列印詞典大小。其中含有40萬個詞和1個特殊的未知詞符號。

```{.python .input}
len(glove_6b50d)
```

我們可以透過詞來獲取它在詞典中的索引，也可以透過索引獲取詞。

```{.python .input  n=12}
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## 應用預訓練詞向量

下面我們以GloVe模型為例，展示預訓練詞向量的應用。

### 求近義詞

這裡重新實現[『word2vec的實現』](./word2vec-gluon.md)一節中介紹過的使用餘弦相似度來搜尋近義詞的演算法。為了在求類比詞時重用其中的求$k$近鄰（$k$-nearest neighbors）的邏輯，我們將這部分邏輯單獨封裝在`knn`函式中。

```{.python .input}
def knn(W, x, k):
    # 新增的1e-9是為了數值穩定性
    cos = nd.dot(W, x.reshape((-1,))) / (
        (nd.sum(W * W, axis=1) + 1e-9).sqrt() * nd.sum(x * x).sqrt())
    topk = nd.topk(cos, k=k, ret_typ='indices').asnumpy().astype('int32')
    return topk, [cos[i].asscalar() for i in topk]
```

然後，我們透過預訓練詞向量實例`embed`來搜尋近義詞。

```{.python .input}
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec,
                    embed.get_vecs_by_tokens([query_token]), k+1)
    for i, c in zip(topk[1:], cos[1:]):  # 除去輸入詞
        print('cosine sim=%.3f: %s' % (c, (embed.idx_to_token[i])))
```

已建立的預訓練詞向量實例`glove_6b50d`的詞典中含40萬個詞和1個特殊的未知詞。除去輸入詞和未知詞，我們從中搜尋與『chip』語義最相近的3個詞。

```{.python .input}
get_similar_tokens('chip', 3, glove_6b50d)
```

接下來查詢『baby』和『beautiful』的近義詞。

```{.python .input}
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.python .input}
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### 求類比詞

除了求近義詞以外，我們還可以使用預訓練詞向量求詞與詞之間的類比關係。例如，『man』（男人）: 『woman』（女人）:: 『son』（兒子） : 『daughter』（女兒）是一個類比例子：『man』之於『woman』相當於『son』之於『daughter』。求類比詞問題可以定義為：對於類比關係中的4個詞 $a : b :: c : d$，給定前3個詞$a$、$b$和$c$，求$d$。設詞$w$的詞向量為$\text{vec}(w)$。求類比詞的思路是，搜尋與$\text{vec}(c)+\text{vec}(b)-\text{vec}(a)$的結果向量最相似的詞向量。

```{.python .input}
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed.get_vecs_by_tokens([token_a, token_b, token_c])
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[topk[0]]
```

驗證一下『男-女』類比。

```{.python .input  n=18}
get_analogy('man', 'woman', 'son', glove_6b50d)
```

『首都-國家』類比：『beijing』（北京）之於『china』（中國）相當於『tokyo』（東京）之於什麼？答案應該是『japan』（日本）。

```{.python .input  n=19}
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

『形容詞-形容詞最高級』類比：『bad』（壞的）之於『worst』（最壞的）相當於『big』（大的）之於什麼？答案應該是『biggest』（最大的）。

```{.python .input  n=20}
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

『動詞一般時-動詞過去時』類比：『do』（做）之於『did』（做過）相當於『go』（去）之於什麼？答案應該是『went』（去過）。

```{.python .input  n=21}
get_analogy('do', 'did', 'go', glove_6b50d)
```

## 小結

* 在大規模語料上預訓練的詞向量常常可以應用於下游自然語言處理任務中。
* 可以應用預訓練的詞向量求近義詞和類比詞。


## 練習

* 測試一下fastText的結果。值得一提的是，fastText有預訓練的中文詞向量（`pretrained_file_name='wiki.zh.vec'`）。
* 如果詞典特別大，如何提升近義詞或類比詞的搜尋速度？




## 參考文獻

[1] GluonNLP工具包。 https://gluon-nlp.mxnet.io/

[2] GloVe專案網站。 https://nlp.stanford.edu/projects/glove/

[3] fastText專案網站。 https://fasttext.cc/

## 掃碼直達[討論區](https://discuss.gluon.ai/t/topic/4373)

![](../img/qr_similarity-analogy.svg)
