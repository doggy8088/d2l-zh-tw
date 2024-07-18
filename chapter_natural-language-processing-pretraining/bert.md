# 來自Transformers的雙向編碼器表示（BERT）
:label:`sec_bert`

我們已經介紹了幾種用於自然語言理解的詞嵌入模型。在預訓練之後，輸出可以被認為是一個矩陣，其中每一行都是一個表示預定義詞表中詞的向量。事實上，這些詞嵌入模型都是與上下文無關的。讓我們先來說明這個性質。

## 從上下文無關到上下文敏感

回想一下 :numref:`sec_word2vec_pretraining`和 :numref:`sec_synonyms`中的實驗。例如，word2vec和GloVe都將相同的預訓練向量分配給同一個詞，而不考慮詞的上下文（如果有的話）。形式上，任何詞元$x$的上下文無關表示是函式$f(x)$，其僅將$x$作為其輸入。考慮到自然語言中豐富的多義現象和複雜的語義，上下文無關表示具有明顯的侷限性。例如，在『a crane is flying』（一隻鶴在飛）和『a crane driver came』（一名吊車司機來了）的上下文中，『crane』一詞有完全不同的含義；因此，同一個詞可以根據上下文被賦予不同的表示。

這推動了『上下文敏感』詞表示的發展，其中詞的表徵取決於它們的上下文。因此，詞元$x$的上下文敏感表示是函式$f(x, c(x))$，其取決於$x$及其上下文$c(x)$。流行的上下文敏感表示包括TagLM（language-model-augmented sequence tagger，語言模型增強的序列標記器） :cite:`Peters.Ammar.Bhagavatula.ea.2017`、CoVe（Context Vectors，上下文向量） :cite:`McCann.Bradbury.Xiong.ea.2017`和ELMo（Embeddings from Language Models，來自語言模型的嵌入） :cite:`Peters.Neumann.Iyyer.ea.2018`。

例如，透過將整個序列作為輸入，ELMo是為輸入序列中的每個單詞分配一個表示的函式。具體來說，ELMo將來自預訓練的雙向長短期記憶網路的所有中間層表示組合為輸出表示。然後，ELMo的表示將作為附加特徵新增到下游任務的現有監督模型中，例如透過將ELMo的表示和現有模型中詞元的原始表示（例如GloVe）連結起來。一方面，在加入ELMo表示後，凍結了預訓練的雙向LSTM模型中的所有權重。另一方面，現有的監督模型是專門為給定的任務定製的。利用當時不同任務的不同最佳模型，新增ELMo改進了六種自然語言處理任務的技術水平：情感分析、自然語言推斷、語義角色標註、共指消解、命名實體識別和問答。

## 從特定於任務到不可知任務

儘管ELMo顯著改進了各種自然語言處理任務的解決方案，但每個解決方案仍然依賴於一個特定於任務的架構。然而，為每一個自然語言處理任務設計一個特定的架構實際上並不是一件容易的事。GPT（Generative Pre Training，生成式預訓練）模型為上下文的敏感表示設計了通用的任務無關模型 :cite:`Radford.Narasimhan.Salimans.ea.2018`。GPT建立在Transformer解碼器的基礎上，預訓練了一個用於表示文字序列的語言模型。當將GPT應用於下游任務時，語言模型的輸出將被送到一個附加的線性輸出層，以預測任務的標籤。與ELMo凍結預訓練模型的引數不同，GPT在下游任務的監督學習過程中對預訓練Transformer解碼器中的所有引數進行微調。GPT在自然語言推斷、問答、句子相似性和分類等12項任務上進行了評估，並在對模型架構進行最小更改的情況下改善了其中9項任務的最新水平。

然而，由於語言模型的自迴歸屬性，GPT只能向前看（從左到右）。在『i went to the bank to deposit cash』（我去銀行存現金）和『i went to the bank to sit down』（我去河岸邊坐下）的上下文中，由於『bank』對其左邊的上下文敏感，GPT將回傳『bank』的相同表示，儘管它有不同的含義。

## BERT：把兩個最好的結合起來

如我們所見，ELMo對上下文進行雙向編碼，但使用特定於任務的架構；而GPT是任務無關的，但是從左到右編碼上下文。BERT（來自Transformers的雙向編碼器表示）結合了這兩個方面的優點。它對上下文進行雙向編碼，並且對於大多數的自然語言處理任務 :cite:`Devlin.Chang.Lee.ea.2018`只需要最少的架構改變。透過使用預訓練的Transformer編碼器，BERT能夠基於其雙向上下文表示任何詞元。在下游任務的監督學習過程中，BERT在兩個方面與GPT相似。首先，BERT表示將被輸入到一個新增的輸出層中，根據任務的性質對模型架構進行最小的更改，例如預測每個詞元與預測整個序列。其次，對預訓練Transformer編碼器的所有引數進行微調，而額外的輸出層將從頭開始訓練。 :numref:`fig_elmo-gpt-bert` 描述了ELMo、GPT和BERT之間的差異。

![ELMo、GPT和BERT的比較](../img/elmo-gpt-bert.svg)
:label:`fig_elmo-gpt-bert`

BERT進一步改進了11種自然語言處理任務的技術水平，這些任務分為以下幾個大類：（1）單一文字分類（如情感分析）、（2）文字對分類（如自然語言推斷）、（3）問答、（4）文字標記（如命名實體識別）。從上下文敏感的ELMo到任務不可知的GPT和BERT，它們都是在2018年提出的。概念上簡單但經驗上強大的自然語言深度表示預訓練已經徹底改變了各種自然語言處理任務的解決方案。

在本章的其餘部分，我們將深入瞭解BERT的訓練前準備。當在 :numref:`chap_nlp_app`中解釋自然語言處理應用時，我們將說明針對下游應用的BERT微調。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
```

## 輸入表示
:label:`subsec_bert_input_rep`

在自然語言處理中，有些任務（如情感分析）以單個文字作為輸入，而有些任務（如自然語言推斷）以一對文字序列作為輸入。BERT輸入序列明確地表示單個文字和文字對。當輸入為單個文字時，BERT輸入序列是特殊類別詞元『&lt;cls&gt;』、文字序列的標記、以及特殊分隔詞元『&lt;sep&gt;』的連結。當輸入為文字對時，BERT輸入序列是『&lt;cls&gt;』、第一個文字序列的標記、『&lt;sep&gt;』、第二個文字序列標記、以及『&lt;sep&gt;』的連結。我們將始終如一地將術語『BERT輸入序列』與其他型別的『序列』區分開來。例如，一個*BERT輸入序列*可以包括一個*文字序列*或兩個*文字序列*。

為了區分文字對，根據輸入序列學到的片段嵌入$\mathbf{e}_A$和$\mathbf{e}_B$分別被新增到第一序列和第二序列的詞元嵌入中。對於單文字輸入，僅使用$\mathbf{e}_A$。

下面的`get_tokens_and_segments`將一個句子或兩個句子作為輸入，然後回傳BERT輸入序列的標記及其相應的片段索引。

```{.python .input}
#@tab all
#@save
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """獲取輸入序列的詞元及其片段索引"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1分別標記片段A和B
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments
```

BERT選擇Transformer編碼器作為其雙向架構。在Transformer編碼器中常見是，位置嵌入被加入到輸入序列的每個位置。然而，與原始的Transformer編碼器不同，BERT使用*可學習的*位置嵌入。總之， 
:numref:`fig_bert-input`表明BERT輸入序列的嵌入是詞元嵌入、片段嵌入和位置嵌入的和。

![BERT輸入序列的嵌入是詞元嵌入、片段嵌入和位置嵌入的和](../img/bert-input.svg)
:label:`fig_bert-input`

下面的`BERTEncoder`類類似於 :numref:`sec_transformer`中實現的`TransformerEncoder`類。與`TransformerEncoder`不同，`BERTEncoder`使用片段嵌入和可學習的位置嵌入。

```{.python .input}
#@save
class BERTEncoder(nn.Block):
    """BERT編碼器"""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_layers, dropout, max_len=1000, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for _ in range(num_layers):
            self.blks.add(d2l.EncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, True))
        # 在BERT中，位置嵌入是可學習的，因此我們建立一個足夠長的位置嵌入引數
        self.pos_embedding = self.params.get('pos_embedding',
                                             shape=(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # 在以下程式碼段落中，X的形狀保持不變：（批次大小，最大序列長度，num_hiddens）
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data(ctx=X.ctx)[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

```{.python .input}
#@tab pytorch
#@save
class BERTEncoder(nn.Module):
    """BERT編碼器"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 在BERT中，位置嵌入是可學習的，因此我們建立一個足夠長的位置嵌入引數
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # 在以下程式碼段落中，X的形狀保持不變：（批次大小，最大序列長度，num_hiddens）
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

```{.python .input}
#@tab paddle
#@save
class BERTEncoder(nn.Layer):
    """BERT編碼器"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_sublayer(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 在BERT中，位置嵌入是可學習的，因此我們建立一個足夠長的位置嵌入引數
        x = paddle.randn([1, max_len, num_hiddens])    
        self.pos_embedding = paddle.create_parameter(shape=x.shape, dtype=str(x.numpy().dtype),
                                                     default_initializer=paddle.nn.initializer.Assign(x))

    def forward(self, tokens, segments, valid_lens):
        # 在以下程式碼段落中，X的形狀保持不變：（批次大小，最大序列長度，num_hiddens）
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

假設詞表大小為10000，為了示範`BERTEncoder`的前向推斷，讓我們建立一個實例並初始化它的引數。

```{.python .input}
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
num_layers, dropout = 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                      num_layers, dropout)
encoder.initialize()
```

```{.python .input}
#@tab pytorch, paddle
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                      ffn_num_hiddens, num_heads, num_layers, dropout)
```

我們將`tokens`定義為長度為8的2個輸入序列，其中每個詞元是詞表的索引。使用輸入`tokens`的`BERTEncoder`的前向推斷回傳編碼結果，其中每個詞元由向量表示，其長度由超引數`num_hiddens`定義。此超引數通常稱為Transformer編碼器的*隱藏大小*（隱藏單元數）。

```{.python .input}
tokens = np.random.randint(0, vocab_size, (2, 8))
segments = np.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

```{.python .input}
#@tab pytorch
tokens = torch.randint(0, vocab_size, (2, 8))
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

```{.python .input}
#@tab paddle
tokens = paddle.randint(0, vocab_size, (2, 8))
segments = paddle.to_tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

## 預訓練任務
:label:`subsec_bert_pretraining_tasks`

`BERTEncoder`的前向推斷給出了輸入文字的每個詞元和插入的特殊標記『&lt;cls&gt;』及『&lt;seq&gt;』的BERT表示。接下來，我們將使用這些表示來計算預訓練BERT的損失函式。預訓練包括以下兩個任務：掩蔽語言模型和下一句預測。

### 掩蔽語言模型（Masked Language Modeling）
:label:`subsec_mlm`

如 :numref:`sec_language_model`所示，語言模型使用左側的上下文預測詞元。為了雙向編碼上下文以表示每個詞元，BERT隨機掩蔽詞元並使用來自雙向上下文的詞元以自監督的方式預測掩蔽詞元。此任務稱為*掩蔽語言模型*。

在這個預訓練任務中，將隨機選擇15%的詞元作為預測的掩蔽詞元。要預測一個掩蔽詞元而不使用標籤作弊，一個簡單的方法是總是用一個特殊的『&lt;mask&gt;』替換輸入序列中的詞元。然而，人造特殊詞元『&lt;mask&gt;』不會出現在微調中。為了避免預訓練和微調之間的這種不對應，如果為預測而遮蔽詞元（例如，在『this movie is great』中選擇掩蔽和預測『great』），則在輸入中將其替換為：

* 80%時間為特殊的『&lt;mask&gt;『詞元（例如，『this movie is great』變為『this movie is&lt;mask&gt;』；
* 10%時間為隨機詞元（例如，『this movie is great』變為『this movie is drink』）；
* 10%時間內為不變的標籤詞元（例如，『this movie is great』變為『this movie is great』）。

請注意，在15%的時間中，有10%的時間插入了隨機詞元。這種偶然的噪聲鼓勵BERT在其雙向上下文編碼中不那麼偏向於掩蔽詞元（尤其是當標籤詞元保持不變時）。

我們實現了下面的`MaskLM`類來預測BERT預訓練的掩蔽語言模型任務中的掩蔽標記。預測使用單隱藏層的多層感知機（`self.mlp`）。在前向推斷中，它需要兩個輸入：`BERTEncoder`的編碼結果和用於預測的詞元位置。輸出是這些位置的預測結果。

```{.python .input}
#@save
class MaskLM(nn.Block):
    """BERT的掩蔽語言模型任務"""
    def __init__(self, vocab_size, num_hiddens, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential()
        self.mlp.add(
            nn.Dense(num_hiddens, flatten=False, activation='relu'))
        self.mlp.add(nn.LayerNorm())
        self.mlp.add(nn.Dense(vocab_size, flatten=False))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = np.arange(0, batch_size)
        # 假設batch_size=2，num_pred_positions=3
        # 那麼batch_idx是np.array（[0,0,0,1,1,1]）
        batch_idx = np.repeat(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

```{.python .input}
#@tab pytorch
#@save
class MaskLM(nn.Module):
    """BERT的掩蔽語言模型任務"""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # 假設batch_size=2，num_pred_positions=3
        # 那麼batch_idx是np.array（[0,0,0,1,1,1]）
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

```{.python .input}
#@tab paddle
#@save
class MaskLM(nn.Layer):
    """BERT的掩蔽語言模型任務"""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape([-1])
        batch_size = X.shape[0]
        batch_idx = paddle.arange(0, batch_size)
        # 假設batch_size=2，num_pred_positions=3
        # 那麼batch_idx是np.array（[0,0,0,1,1]）
        batch_idx = paddle.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

為了示範`MaskLM`的前向推斷，我們建立了其實例`mlm`並對其進行了初始化。回想一下，來自`BERTEncoder`的正向推斷`encoded_X`表示2個BERT輸入序列。我們將`mlm_positions`定義為在`encoded_X`的任一輸入序列中預測的3個指示。`mlm`的前向推斷回傳`encoded_X`的所有掩蔽位置`mlm_positions`處的預測結果`mlm_Y_hat`。對於每個預測，結果的大小等於詞表的大小。

```{.python .input}
mlm = MaskLM(vocab_size, num_hiddens)
mlm.initialize()
mlm_positions = np.array([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

```{.python .input}
#@tab pytorch
mlm = MaskLM(vocab_size, num_hiddens)
mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

```{.python .input}
#@tab paddle
mlm = MaskLM(vocab_size, num_hiddens)
mlm_positions = paddle.to_tensor([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

透過掩碼下的預測詞元`mlm_Y`的真實標籤`mlm_Y_hat`，我們可以計算在BERT預訓練中的遮蔽語言模型任務的交叉熵損失。

```{.python .input}
mlm_Y = np.array([[7, 8, 9], [10, 20, 30]])
loss = gluon.loss.SoftmaxCrossEntropyLoss()
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

```{.python .input}
#@tab pytorch
mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
loss = nn.CrossEntropyLoss(reduction='none')
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

```{.python .input}
#@tab paddle
mlm_Y = paddle.to_tensor([[7, 8, 9], [10, 20, 30]])
loss = nn.CrossEntropyLoss(reduction='none')
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape([-1]))
mlm_l.shape
```

### 下一句預測（Next Sentence Prediction）
:label:`subsec_nsp`

儘管掩蔽語言建模能夠編碼雙向上下文來表示單詞，但它不能顯式地建模文字對之間的邏輯關係。為了幫助理解兩個文字序列之間的關係，BERT在預訓練中考慮了一個二元分類任務——*下一句預測*。在為預訓練生成句子對時，有一半的時間它們確實是標籤為『真』的連續句子；在另一半的時間裡，第二個句子是從語料庫中隨機抽取的，標記為『假』。

下面的`NextSentencePred`類使用單隱藏層的多層感知機來預測第二個句子是否是BERT輸入序列中第一個句子的下一個句子。由於Transformer編碼器中的自注意力，特殊詞元『&lt;cls&gt;』的BERT表示已經對輸入的兩個句子進行了編碼。因此，多層感知機分類器的輸出層（`self.output`）以`X`作為輸入，其中`X`是多層感知機隱藏層的輸出，而MLP隱藏層的輸入是編碼後的『&lt;cls&gt;』詞元。

```{.python .input}
#@save
class NextSentencePred(nn.Block):
    """BERT的下一句預測任務"""
    def __init__(self, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Dense(2)

    def forward(self, X):
        # X的形狀：(batchsize，num_hiddens)
        return self.output(X)
```

```{.python .input}
#@tab pytorch
#@save
class NextSentencePred(nn.Module):
    """BERT的下一句預測任務"""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # X的形狀：(batchsize,num_hiddens)
        return self.output(X)
```

```{.python .input}
#@tab paddle
#@save
class NextSentencePred(nn.Layer):
    """BERT的下一句預測任務"""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # X的形狀：(batchsize,num_hiddens)
        return self.output(X)
```

我們可以看到，`NextSentencePred`實例的前向推斷回傳每個BERT輸入序列的二分類預測。

```{.python .input}
nsp = NextSentencePred()
nsp.initialize()
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

```{.python .input}
#@tab pytorch
encoded_X = torch.flatten(encoded_X, start_dim=1)
# NSP的輸入形狀:(batchsize，num_hiddens)
nsp = NextSentencePred(encoded_X.shape[-1])
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

```{.python .input}
#@tab paddle
encoded_X = paddle.flatten(encoded_X, start_axis=1)
# NSP的輸入形狀:(batchsize，num_hiddens)
nsp = NextSentencePred(encoded_X.shape[-1])
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

還可以計算兩個二元分類的交叉熵損失。

```{.python .input}
nsp_y = np.array([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```

```{.python .input}
#@tab pytorch
nsp_y = torch.tensor([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```

```{.python .input}
#@tab paddle
nsp_y = paddle.to_tensor([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```

值得注意的是，上述兩個預訓練任務中的所有標籤都可以從預訓練語料庫中獲得，而無需人工標註。原始的BERT已經在圖書語料庫 :cite:`Zhu.Kiros.Zemel.ea.2015`和英文維基百科的連線上進行了預訓練。這兩個文字語料庫非常龐大：它們分別有8億個單詞和25億個單詞。

## 整合程式碼

在預訓練BERT時，最終的損失函式是掩蔽語言模型損失函式和下一句預測損失函式的線性組合。現在我們可以透過實例化三個類`BERTEncoder`、`MaskLM`和`NextSentencePred`來定義`BERTModel`類。前向推斷回傳編碼後的BERT表示`encoded_X`、掩蔽語言模型預測`mlm_Y_hat`和下一句預測`nsp_Y_hat`。

```{.python .input}
#@save
class BERTModel(nn.Block):
    """BERT模型"""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_layers, dropout, max_len=1000):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens,
                                   num_heads, num_layers, dropout, max_len)
        self.hidden = nn.Dense(num_hiddens, activation='tanh')
        self.mlm = MaskLM(vocab_size, num_hiddens)
        self.nsp = NextSentencePred()

    def forward(self, tokens, segments, valid_lens=None, 
                pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用於下一句預測的多層感知機分類器的隱藏層，0是『<cls>』標記的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

```{.python .input}
#@tab pytorch
#@save
class BERTModel(nn.Module):
    """BERT模型"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, 
                pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用於下一句預測的多層感知機分類器的隱藏層，0是『<cls>』標記的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

```{.python .input}
#@tab paddle
#@save
class BERTModel(nn.Layer):
    """BERT模型"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None,
                pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用於下一句預測的多層感知機分類器的隱藏層，0是『<cls>』標記的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

## 小結

* word2vec和GloVe等詞嵌入模型與上下文無關。它們將相同的預訓練向量賦給同一個詞，而不考慮詞的上下文（如果有的話）。它們很難處理好自然語言中的一詞多義或複雜語義。
* 對於上下文敏感的詞表示，如ELMo和GPT，詞的表示依賴於它們的上下文。
* ELMo對上下文進行雙向編碼，但使用特定於任務的架構（然而，為每個自然語言處理任務設計一個特定的體系架構實際上並不容易）；而GPT是任務無關的，但是從左到右編碼上下文。
* BERT結合了這兩個方面的優點：它對上下文進行雙向編碼，並且需要對大量自然語言處理任務進行最小的架構更改。
* BERT輸入序列的嵌入是詞元嵌入、片段嵌入和位置嵌入的和。
* 預訓練包括兩個任務：掩蔽語言模型和下一句預測。前者能夠編碼雙向上下文來表示單詞，而後者則顯式地建模文字對之間的邏輯關係。

## 練習

1. 為什麼BERT成功了？
1. 在所有其他條件相同的情況下，掩蔽語言模型比從左到右的語言模型需要更多或更少的預訓練步驟來收斂嗎？為什麼？
1. 在BERT的原始實現中，`BERTEncoder`中的位置前饋網路（透過`d2l.EncoderBlock`）和`MaskLM`中的全連線層都使用高斯誤差線性單元（Gaussian error linear unit，GELU） :cite:`Hendrycks.Gimpel.2016`作為啟用函式。研究GELU與ReLU之間的差異。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/5749)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/5750)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11820)
:end_tab:
