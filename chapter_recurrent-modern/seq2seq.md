#  序列到序列學習（seq2seq）
:label:`sec_seq2seq`

正如我們在 :numref:`sec_machine_translation`中看到的，
機器翻譯中的輸入序列和輸出序列都是長度可變的。
為了解決這類問題，我們在 :numref:`sec_encoder-decoder`中
設計了一個通用的』編碼器－解碼器『架構。
本節，我們將使用兩個迴圈神經網路的編碼器和解碼器，
並將其應用於*序列到序列*（sequence to sequence，seq2seq）類別的學習任務
 :cite:`Sutskever.Vinyals.Le.2014,Cho.Van-Merrienboer.Gulcehre.ea.2014`。

遵循編碼器－解碼器架構的設計原則，
迴圈神經網路編碼器使用長度可變的序列作為輸入，
將其轉換為固定形狀的隱狀態。
換言之，輸入序列的訊息被*編碼*到迴圈神經網路編碼器的隱狀態中。
為了連續生成輸出序列的詞元，
獨立的迴圈神經網路解碼器是基於輸入序列的編碼訊息
和輸出序列已經看見的或者生成的詞元來預測下一個詞元。
 :numref:`fig_seq2seq`示範了
如何在機器翻譯中使用兩個迴圈神經網路進行序列到序列學習。

![使用迴圈神經網路編碼器和迴圈神經網路解碼器的序列到序列學習](../img/seq2seq.svg)
:label:`fig_seq2seq`

在 :numref:`fig_seq2seq`中，
特定的『&lt;eos&gt;』表示序列結束詞元。
一旦輸出序列生成此詞元，模型就會停止預測。
在迴圈神經網路解碼器的初始化時間步，有兩個特定的設計決定：
首先，特定的『&lt;bos&gt;』表示序列開始詞元，它是解碼器的輸入序列的第一個詞元。
其次，使用迴圈神經網路編碼器最終的隱狀態來初始化解碼器的隱狀態。
例如，在 :cite:`Sutskever.Vinyals.Le.2014`的設計中，
正是基於這種設計將輸入序列的編碼訊息送入到解碼器中來生成輸出序列的。
在其他一些設計中 :cite:`Cho.Van-Merrienboer.Gulcehre.ea.2014`，
如 :numref:`fig_seq2seq`所示，
編碼器最終的隱狀態在每一個時間步都作為解碼器的輸入序列的一部分。
類似於 :numref:`sec_language_model`中語言模型的訓練，
可以允許標籤成為原始的輸出序列，
從源序列詞元『&lt;bos&gt;』『Ils』『regardent』『.』
到新序列詞元
『Ils』『regardent』『.』『&lt;eos&gt;』來移動預測的位置。

下面，我們動手建立 :numref:`fig_seq2seq`的設計，
並將基於 :numref:`sec_machine_translation`中
介紹的『英－法』資料集來訓練這個機器翻譯模型。

```{.python .input}
import collections
from d2l import mxnet as d2l
import math
from mxnet import np, npx, init, gluon, autograd
from mxnet.gluon import nn, rnn
npx.set_np()
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import math
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
import collections
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input}
#@tab paddle
import collections
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import math
import paddle
from paddle import nn
```

## 編碼器

從技術上講，編碼器將長度可變的輸入序列轉換成
形狀固定的上下文變數$\mathbf{c}$，
並且將輸入序列的訊息在該上下文變數中進行編碼。
如 :numref:`fig_seq2seq`所示，可以使用迴圈神經網路來設計編碼器。

考慮由一個序列組成的樣本（批次大小是$1$）。
假設輸入序列是$x_1, \ldots, x_T$，
其中$x_t$是輸入文字序列中的第$t$個詞元。
在時間步$t$，迴圈神經網路將詞元$x_t$的輸入特徵向量
$\mathbf{x}_t$和$\mathbf{h} _{t-1}$（即上一時間步的隱狀態）
轉換為$\mathbf{h}_t$（即當前步的隱狀態）。
使用一個函式$f$來描述迴圈神經網路的迴圈層所做的變換：

$$\mathbf{h}_t = f(\mathbf{x}_t, \mathbf{h}_{t-1}). $$

總之，編碼器透過選定的函式$q$，
將所有時間步的隱狀態轉換為上下文變數：

$$\mathbf{c} =  q(\mathbf{h}_1, \ldots, \mathbf{h}_T).$$

比如，當選擇$q(\mathbf{h}_1, \ldots, \mathbf{h}_T) = \mathbf{h}_T$時
（就像 :numref:`fig_seq2seq`中一樣），
上下文變數僅僅是輸入序列在最後時間步的隱狀態$\mathbf{h}_T$。

到目前為止，我們使用的是一個單向迴圈神經網路來設計編碼器，
其中隱狀態只依賴於輸入子序列，
這個子序列是由輸入序列的開始位置到隱狀態所在的時間步的位置
（包括隱狀態所在的時間步）組成。
我們也可以使用雙向迴圈神經網路構造編碼器，
其中隱狀態依賴於兩個輸入子序列，
兩個子序列是由隱狀態所在的時間步的位置之前的序列和之後的序列
（包括隱狀態所在的時間步），
因此隱狀態對整個序列的訊息都進行了編碼。

現在，讓我們[**實現迴圈神經網路編碼器**]。
注意，我們使用了*嵌入層*（embedding layer）
來獲得輸入序列中每個詞元的特徵向量。
嵌入層的權重是一個矩陣，
其行數等於輸入詞表的大小（`vocab_size`），
其列數等於特徵向量的維度（`embed_size`）。
對於任意輸入詞元的索引$i$，
嵌入層獲取權重矩陣的第$i$行（從$0$開始）以回傳其特徵向量。
另外，本文選擇了一個多層門控迴圈單元來實現編碼器。

```{.python .input}
#@save
class Seq2SeqEncoder(d2l.Encoder):
    """用於序列到序列學習的迴圈神經網路編碼器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入層
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # 輸出'X'的形狀：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在迴圈神經網路模型中，第一個軸對應於時間步
        X = X.swapaxes(0, 1)
        state = self.rnn.begin_state(batch_size=X.shape[1], ctx=X.ctx)
        output, state = self.rnn(X, state)
        # output的形狀:(num_steps,batch_size,num_hiddens)
        # state的形狀:(num_layers,batch_size,num_hiddens)
        return output, state
```

```{.python .input}
#@tab pytorch
#@save
class Seq2SeqEncoder(d2l.Encoder):
    """用於序列到序列學習的迴圈神經網路編碼器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入層
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # 輸出'X'的形狀：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在迴圈神經網路模型中，第一個軸對應於時間步
        X = X.permute(1, 0, 2)
        # 如果未提及狀態，則預設為0
        output, state = self.rnn(X)
        # output的形狀:(num_steps,batch_size,num_hiddens)
        # state的形狀:(num_layers,batch_size,num_hiddens)
        return output, state
```

```{.python .input}
#@tab tensorflow
#@save
class Seq2SeqEncoder(d2l.Encoder):
    """用於序列到序列學習的迴圈神經網路編碼器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs): 
        super().__init__(*kwargs)
        # 嵌入層
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout)
             for _ in range(num_layers)]), return_sequences=True,
                                       return_state=True)
    
    def call(self, X, *args, **kwargs):
        # 輸入'X'的形狀：(batch_size,num_steps)
        # 輸出'X'的形狀：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        output = self.rnn(X, **kwargs)
        state = output[1:]
        return output[0], state
```

```{.python .input}
#@tab paddle
#@save
class Seq2SeqEncoder(d2l.Encoder):
    """用於序列到序列學習的迴圈神經網路編碼器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        weight_ih_attr = paddle.ParamAttr(initializer=nn.initializer.XavierUniform())
        weight_hh_attr = paddle.ParamAttr(initializer=nn.initializer.XavierUniform())
        # 嵌入層
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout,
                          time_major=True, weight_ih_attr=weight_ih_attr, weight_hh_attr=weight_hh_attr)

    def forward(self, X, *args):
        # 輸出'X'的形狀：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在迴圈神經網路模型中，第一個軸對應於時間步
        X = X.transpose([1, 0, 2])
        # 如果未提及狀態，則預設為0
        output, state = self.rnn(X)
        # PaddlePaddle的GRU層output的形狀:(batch_size,time_steps,num_directions * num_hiddens),
        # 需設定time_major=True,指定input的第一個維度為time_steps
        # state[0]的形狀:(num_layers,batch_size,num_hiddens)
        return output, state
```

迴圈層回傳變數的說明可以參考 :numref:`sec_rnn-concise`。

下面，我們實例化[**上述編碼器的實現**]：
我們使用一個兩層門控迴圈單元編碼器，其隱藏單元數為$16$。
給定一小批次的輸入序列`X`（批次大小為$4$，時間步為$7$）。
在完成所有時間步後，
最後一層的隱狀態的輸出是一個張量（`output`由編碼器的迴圈層回傳），
其形狀為（時間步數，批次大小，隱藏單元數）。

```{.python .input}
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
encoder.initialize()
X = d2l.zeros((4, 7))
output, state = encoder(X)
output.shape
```

```{.python .input}
#@tab pytorch
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
encoder.eval()
X = d2l.zeros((4, 7), dtype=torch.long)
output, state = encoder(X)
output.shape
```

```{.python .input}
#@tab tensorflow
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
X = tf.zeros((4, 7))
output, state = encoder(X, training=False)
output.shape
```

```{.python .input}
#@tab paddle
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
encoder.eval()
X = d2l.zeros((4, 7), dtype=paddle.int64)
output, state = encoder(X)
output.shape
```

由於這裡使用的是門控迴圈單元，
所以在最後一個時間步的多層隱狀態的形狀是
（隱藏層的數量，批次大小，隱藏單元的數量）。
如果使用長短期記憶網路，`state`中還將包含記憶單元訊息。

```{.python .input}
len(state), state[0].shape
```

```{.python .input}
#@tab pytorch, paddle
state.shape
```

```{.python .input}
#@tab tensorflow
len(state), [element.shape for element in state]
```

## [**解碼器**]
:label:`sec_seq2seq_decoder`

正如上文提到的，編碼器輸出的上下文變數$\mathbf{c}$
對整個輸入序列$x_1, \ldots, x_T$進行編碼。
來自訓練資料集的輸出序列$y_1, y_2, \ldots, y_{T'}$，
對於每個時間步$t'$（與輸入序列或編碼器的時間步$t$不同），
解碼器輸出$y_{t'}$的機率取決於先前的輸出子序列
$y_1, \ldots, y_{t'-1}$和上下文變數$\mathbf{c}$，
即$P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$。

為了在序列上模型化這種條件機率，
我們可以使用另一個迴圈神經網路作為解碼器。
在輸出序列上的任意時間步$t^\prime$，
迴圈神經網路將來自上一時間步的輸出$y_{t^\prime-1}$
和上下文變數$\mathbf{c}$作為其輸入，
然後在當前時間步將它們和上一隱狀態
$\mathbf{s}_{t^\prime-1}$轉換為
隱狀態$\mathbf{s}_{t^\prime}$。
因此，可以使用函式$g$來表示解碼器的隱藏層的變換：

$$\mathbf{s}_{t^\prime} = g(y_{t^\prime-1}, \mathbf{c}, \mathbf{s}_{t^\prime-1}).$$
:eqlabel:`eq_seq2seq_s_t`

在獲得解碼器的隱狀態之後，
我們可以使用輸出層和softmax操作
來計算在時間步$t^\prime$時輸出$y_{t^\prime}$的條件機率分佈
$P(y_{t^\prime} \mid y_1, \ldots, y_{t^\prime-1}, \mathbf{c})$。

根據 :numref:`fig_seq2seq`，當實現解碼器時，
我們直接使用編碼器最後一個時間步的隱狀態來初始化解碼器的隱狀態。
這就要求使用迴圈神經網路實現的編碼器和解碼器具有相同數量的層和隱藏單元。
為了進一步包含經過編碼的輸入序列的訊息，
上下文變數在所有的時間步與解碼器的輸入進行拼接（concatenate）。
為了預測輸出詞元的機率分佈，
在迴圈神經網路解碼器的最後一層使用全連線層來變換隱狀態。

```{.python .input}
class Seq2SeqDecoder(d2l.Decoder):
    """用於序列到序列學習的迴圈神經網路解碼器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # 輸出'X'的形狀：(batch_size,num_steps,embed_size)
        X = self.embedding(X).swapaxes(0, 1)
        # context的形狀:(batch_size,num_hiddens)
        context = state[0][-1]
        # 廣播context，使其具有與X相同的num_steps
        context = np.broadcast_to(context, (
            X.shape[0], context.shape[0], context.shape[1]))
        X_and_context = d2l.concat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).swapaxes(0, 1)
        # output的形狀:(batch_size,num_steps,vocab_size)
        # state的形狀:(num_layers,batch_size,num_hiddens)
        return output, state
```

```{.python .input}
#@tab pytorch
class Seq2SeqDecoder(d2l.Decoder):
    """用於序列到序列學習的迴圈神經網路解碼器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # 輸出'X'的形狀：(batch_size,num_steps,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 廣播context，使其具有與X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = d2l.concat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形狀:(batch_size,num_steps,vocab_size)
        # state的形狀:(num_layers,batch_size,num_hiddens)
        return output, state
```

```{.python .input}
#@tab tensorflow
class Seq2SeqDecoder(d2l.Decoder):
    """用於序列到序列學習的迴圈神經網路解碼器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout)
             for _ in range(num_layers)]), return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)
        
    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]
    
    def call(self, X, state, **kwargs):
        # 輸出'X'的形狀：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 廣播context，使其具有與X相同的num_steps
        context = tf.repeat(tf.expand_dims(state[-1], axis=1), repeats=X.shape[1], axis=1)
        X_and_context = tf.concat((X, context), axis=2)
        rnn_output = self.rnn(X_and_context, state, **kwargs)
        output = self.dense(rnn_output[0])
        # output的形狀:(batch_size,num_steps,vocab_size)
        # state是一個包含num_layers個元素的清單，每個元素的形狀:(batch_size,num_hiddens)
        return output, rnn_output[1:]
```

```{.python .input}
#@tab paddle
class Seq2SeqDecoder(d2l.Decoder):
    """用於序列到序列學習的迴圈神經網路解碼器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.XavierUniform())
        weight_ih_attr = paddle.ParamAttr(initializer=nn.initializer.XavierUniform())
        weight_hh_attr = paddle.ParamAttr(initializer=nn.initializer.XavierUniform())
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout,
                          time_major=True, weight_ih_attr=weight_ih_attr,weight_hh_attr=weight_hh_attr)
        self.dense = nn.Linear(num_hiddens, vocab_size,weight_attr=weight_attr)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # 輸出'X'的形狀：(batch_size,num_steps,embed_size)
        X = self.embedding(X).transpose([1, 0, 2])
        # 廣播context，使其具有與X相同的num_steps
        context = state[-1].tile([X.shape[0], 1, 1])
        X_and_context = d2l.concat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).transpose([1, 0, 2])
        # output的形狀:(batch_size,num_steps,vocab_size)
        # state[0]的形狀:(num_layers,batch_size,num_hiddens)
        return output, state
```

下面，我們用與前面提到的編碼器中相同的超引數來[**實例化解碼器**]。
如我們所見，解碼器的輸出形狀變為（批次大小，時間步數，詞表大小），
其中張量的最後一個維度儲存預測的詞元分佈。

```{.python .input}
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
decoder.initialize()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
output.shape, len(state), state[0].shape
```

```{.python .input}
#@tab pytorch
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
decoder.eval()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
output.shape, state.shape
```

```{.python .input}
#@tab tensorflow
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
state = decoder.init_state(encoder(X))
output, state = decoder(X, state, training=False)
output.shape, len(state), state[0].shape
```

```{.python .input}
#@tab paddle
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
decoder.eval()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
output.shape, state.shape
```

總之，上述迴圈神經網路『編碼器－解碼器』模型中的各層如
 :numref:`fig_seq2seq_details`所示。

![迴圈神經網路編碼器-解碼器模型中的層](../img/seq2seq-details.svg)
:label:`fig_seq2seq_details`

## 損失函式

在每個時間步，解碼器預測了輸出詞元的機率分佈。
類似於語言模型，可以使用softmax來獲得分佈，
並透過計算交叉熵損失函式來進行最佳化。
回想一下 :numref:`sec_machine_translation`中，
特定的填充詞元被新增到序列的末尾，
因此不同長度的序列可以以相同形狀的小批次載入。
但是，我們應該將填充詞元的預測排除在損失函式的計算之外。

為此，我們可以使用下面的`sequence_mask`函式
[**透過零值化遮蔽不相關的項**]，
以便後面任何不相關預測的計算都是與零的乘積，結果都等於零。
例如，如果兩個序列的有效長度（不包括填充詞元）分別為$1$和$2$，
則第一個序列的第一項和第二個序列的前兩項之後的剩餘項將被清除為零。

```{.python .input}
X = np.array([[1, 2, 3], [4, 5, 6]])
npx.sequence_mask(X, np.array([1, 2]), True, axis=1)
```

```{.python .input}
#@tab pytorch
#@save
def sequence_mask(X, valid_len, value=0):
    """在序列中遮蔽不相關的項"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

X = torch.tensor([[1, 2, 3], [4, 5, 6]])
sequence_mask(X, torch.tensor([1, 2]))
```

```{.python .input}
#@tab tensorflow
#@save
def sequence_mask(X, valid_len, value=0):
    """在序列中遮蔽不相關的項"""
    maxlen = X.shape[1]
    mask = tf.range(start=0, limit=maxlen, dtype=tf.float32)[
        None, :] < tf.cast(valid_len[:, None], dtype=tf.float32)
    
    if len(X.shape) == 3:
        return tf.where(tf.expand_dims(mask, axis=-1), X, value)
    else:
        return tf.where(mask, X, value)
    
X = tf.constant([[1, 2, 3], [4, 5, 6]])
sequence_mask(X, tf.constant([1, 2]))
```

```{.python .input}
#@tab paddle
#@save
def sequence_mask(X, valid_len, value=0):
    """在序列中遮蔽不相關的項"""
    maxlen = X.shape[1]
    mask = paddle.arange((maxlen), dtype=paddle.float32)[None, :] < valid_len[:, None]
    Xtype = X.dtype
    X = X.astype(paddle.float32)
    X[~mask] = float(value)
    return X.astype(Xtype)

X = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
sequence_mask(X, paddle.to_tensor([1, 2]))
```

(**我們還可以使用此函式遮蔽最後幾個軸上的所有項。**)如果願意，也可以使用指定的非零值來替換這些項。

```{.python .input}
X = d2l.ones((2, 3, 4))
npx.sequence_mask(X, np.array([1, 2]), True, value=-1, axis=1)
```

```{.python .input}
#@tab pytorch
X = d2l.ones(2, 3, 4)
sequence_mask(X, torch.tensor([1, 2]), value=-1)
```

```{.python .input}
#@tab tensorflow
X = tf.ones((2,3,4))
sequence_mask(X, tf.constant([1, 2]), value=-1)
```

```{.python .input}
#@tab paddle
X = d2l.ones([2, 3, 4])
sequence_mask(X, paddle.to_tensor([1, 2]), value=-1)
```

現在，我們可以[**透過擴充套件softmax交叉熵損失函式來遮蔽不相關的預測**]。
最初，所有預測詞元的掩碼都設定為1。
一旦給定了有效長度，與填充詞元對應的掩碼將被設定為0。
最後，將所有詞元的損失乘以掩碼，以過濾掉損失中填充詞元產生的不相關預測。

```{.python .input}
#@save
class MaskedSoftmaxCELoss(gluon.loss.SoftmaxCELoss):
    """帶遮蔽的softmax交叉熵損失函式"""
    # pred的形狀：(batch_size,num_steps,vocab_size)
    # label的形狀：(batch_size,num_steps)
    # valid_len的形狀：(batch_size,)
    def forward(self, pred, label, valid_len):
        # weights的形狀：(batch_size,num_steps,1)
        weights = np.expand_dims(np.ones_like(label), axis=-1)
        weights = npx.sequence_mask(weights, valid_len, True, axis=1)
        return super(MaskedSoftmaxCELoss, self).forward(pred, label, weights)
```

```{.python .input}
#@tab pytorch
#@save
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """帶遮蔽的softmax交叉熵損失函式"""
    # pred的形狀：(batch_size,num_steps,vocab_size)
    # label的形狀：(batch_size,num_steps)
    # valid_len的形狀：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss
```

```{.python .input}
#@tab tensorflow
#@save
class MaskedSoftmaxCELoss(tf.keras.losses.Loss):
    """帶遮蔽的softmax交叉熵損失函式"""
    def __init__(self, valid_len):
        super().__init__(reduction='none')
        self.valid_len = valid_len
    
    # pred的形狀：(batch_size,num_steps,vocab_size)
    # label的形狀：(batch_size,num_steps)
    # valid_len的形狀：(batch_size,)
    def call(self, label, pred):
        weights = tf.ones_like(label, dtype=tf.float32)
        weights = sequence_mask(weights, self.valid_len)
        label_one_hot = tf.one_hot(label, depth=pred.shape[-1])
        unweighted_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction='none')(label_one_hot, pred)
        weighted_loss = tf.reduce_mean((unweighted_loss*weights), axis=1)
        return weighted_loss
```

```{.python .input}
#@tab paddle
#@save
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """帶遮蔽的softmax交叉熵損失函式"""
    # pred的形狀：(batch_size,num_steps,vocab_size)
    # label的形狀：(batch_size,num_steps)
    # valid_len的形狀：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = paddle.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred, label)
        weighted_loss = (unweighted_loss * weights).mean(axis=1)
        return weighted_loss
```

我們可以建立三個相同的序列來進行[**程式碼健全性檢查**]，
然後分別指定這些序列的有效長度為$4$、$2$和$0$。
結果就是，第一個序列的損失應為第二個序列的兩倍，而第三個序列的損失應為零。

```{.python .input}
loss = MaskedSoftmaxCELoss()
loss(d2l.ones((3, 4, 10)), d2l.ones((3, 4)), np.array([4, 2, 0]))
```

```{.python .input}
#@tab pytorch
loss = MaskedSoftmaxCELoss()
loss(d2l.ones(3, 4, 10), d2l.ones((3, 4), dtype=torch.long),
     torch.tensor([4, 2, 0]))
```

```{.python .input}
#@tab tensorflow
loss = MaskedSoftmaxCELoss(tf.constant([4, 2, 0]))
loss(tf.ones((3,4), dtype = tf.int32), tf.ones((3, 4, 10))).numpy()
```

```{.python .input}
#@tab paddle
loss = MaskedSoftmaxCELoss()
loss(d2l.ones([3, 4, 10]), d2l.ones((3, 4), dtype=paddle.int64),
     paddle.to_tensor([4, 2, 0]))
```

## [**訓練**]
:label:`sec_seq2seq_training`

在下面的迴圈訓練過程中，如 :numref:`fig_seq2seq`所示，
特定的序列開始詞元（『&lt;bos&gt;』）和
原始的輸出序列（不包括序列結束詞元『&lt;eos&gt;』）
拼接在一起作為解碼器的輸入。
這被稱為*強制教學*（teacher forcing），
因為原始的輸出序列（詞元的標籤）被送入解碼器。
或者，將來自上一個時間步的*預測*得到的詞元作為解碼器的當前輸入。

```{.python .input}
#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """訓練序列到序列模型"""
    net.initialize(init.Xavier(), force_reinit=True, ctx=device)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    loss = MaskedSoftmaxCELoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 訓練損失求和，詞元數量
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [
                x.as_in_ctx(device) for x in batch]
            bos = np.array([tgt_vocab['<bos>']] * Y.shape[0], 
                       ctx=device).reshape(-1, 1)
            dec_input = np.concatenate([bos, Y[:, :-1]], 1)  # 強制教學
            with autograd.record():
                Y_hat, _ = net(X, dec_input, X_valid_len)
                l = loss(Y_hat, Y, Y_valid_len)
            l.backward()
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            trainer.step(num_tokens)
            metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
        f'tokens/sec on {str(device)}')
```

```{.python .input}
#@tab pytorch
#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """訓練序列到序列模型"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                     xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 訓練損失總和，詞元數量
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                          device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 強制教學
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()	# 損失函式的標量進行『反向傳播』
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
        f'tokens/sec on {str(device)}')
```

```{.python .input}
#@tab tensorflow
#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """訓練序列到序列模型"""
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    animator = d2l.Animator(xlabel="epoch", ylabel="loss",
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 訓練損失總和，詞元數量
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x for x in batch]
            bos = tf.reshape(tf.constant([tgt_vocab['<bos>']] * Y.shape[0]),
                             shape=(-1, 1))
            dec_input = tf.concat([bos, Y[:, :-1]], 1)  # 強制教學
            with tf.GradientTape() as tape:
                Y_hat, _ = net(X, dec_input, X_valid_len, training=True)
                l = MaskedSoftmaxCELoss(Y_valid_len)(Y, Y_hat)
            gradients = tape.gradient(l, net.trainable_variables)
            gradients = d2l.grad_clipping(gradients, 1)
            optimizer.apply_gradients(zip(gradients, net.trainable_variables))
            num_tokens = tf.reduce_sum(Y_valid_len).numpy()
            metric.add(tf.reduce_sum(l), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')
```

```{.python .input}
#@tab paddle
#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """訓練序列到序列模型"""
    optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=net.parameters())
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                     xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 訓練損失總和，詞元數量
        for batch in data_iter:
            optimizer.clear_grad()
            X, X_valid_len, Y, Y_valid_len = [paddle.to_tensor(x, place=device) for x in batch]
            bos = paddle.to_tensor([tgt_vocab['<bos>']] * Y.shape[0]).reshape([-1, 1])
            dec_input = paddle.concat([bos, Y[:, :-1]], 1)  # 強制教學
            Y_hat, _ = net(X, dec_input, X_valid_len.squeeze())
            l = loss(Y_hat, Y, Y_valid_len.squeeze())
            l.backward()	# 損失函式的標量進行『反向傳播』
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with paddle.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
        f'tokens/sec on {str(device)}')
```

現在，在機器翻譯資料集上，我們可以
[**建立和訓練一個迴圈神經網路『編碼器－解碼器』模型**]用於序列到序列的學習。

```{.python .input}
#@tab all
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                        dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                        dropout)
net = d2l.EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
```

## [**預測**]

為了採用一個接著一個詞元的方式預測輸出序列，
每個解碼器當前時間步的輸入都將來自於前一時間步的預測詞元。
與訓練類似，序列開始詞元（『&lt;bos&gt;』）
在初始時間步被輸入到解碼器中。
該預測過程如 :numref:`fig_seq2seq_predict`所示，
當輸出序列的預測遇到序列結束詞元（『&lt;eos&gt;』）時，預測就結束了。

![使用迴圈神經網路編碼器-解碼器逐詞元地預測輸出序列。](../img/seq2seq-predict.svg)
:label:`fig_seq2seq_predict`

我們將在 :numref:`sec_beam-search`中介紹不同的序列生成策略。

```{.python .input}
#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的預測"""
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = np.array([len(src_tokens)], ctx=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 新增批次軸
    enc_X = np.expand_dims(np.array(src_tokens, ctx=device), axis=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 新增批次軸
    dec_X = np.expand_dims(np.array([tgt_vocab['<bos>']], ctx=device), 
                           axis=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我們使用具有預測最高可能性的詞元，作為解碼器在下一時間步的輸入
        dec_X = Y.argmax(axis=2)
        pred = dec_X.squeeze(axis=0).astype('int32').item()
        # 儲存注意力權重（稍後討論）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列結束詞元被預測，輸出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
```

```{.python .input}
#@tab pytorch
#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的預測"""
    # 在預測時將net設定為評估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 新增批次軸
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 新增批次軸
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我們使用具有預測最高可能性的詞元，作為解碼器在下一時間步的輸入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 儲存注意力權重（稍後討論）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列結束詞元被預測，輸出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
```

```{.python .input}
#@tab tensorflow
#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    save_attention_weights=False):
    """序列到序列模型的預測"""
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = tf.constant([len(src_tokens)])
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 新增批次軸
    enc_X = tf.expand_dims(src_tokens, axis=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len, training=False)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 新增批次軸
    dec_X = tf.expand_dims(tf.constant([tgt_vocab['<bos>']]), axis=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state, training=False)
        # 我們使用具有預測最高可能性的詞元，作為解碼器在下一時間步的輸入
        dec_X = tf.argmax(Y, axis=2)
        pred = tf.squeeze(dec_X, axis=0)
        # 儲存注意力權重
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列結束詞元被預測，輸出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred.numpy())
    return ' '.join(tgt_vocab.to_tokens(tf.reshape(output_seq, 
        shape = -1).numpy().tolist())), attention_weight_seq
```

```{.python .input}
#@tab paddle
#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的預測"""
    # 在預測時將net設定為評估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = paddle.to_tensor([len(src_tokens)], place=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 新增批次軸
    enc_X = paddle.unsqueeze(
        paddle.to_tensor(src_tokens, dtype=paddle.int64, place=device), axis=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 新增批次軸
    dec_X = paddle.unsqueeze(paddle.to_tensor(
        [tgt_vocab['<bos>']], dtype=paddle.int64, place=device), axis=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我們使用具有預測最高可能性的詞元，作為解碼器在下一時間步的輸入
        dec_X = Y.argmax(axis=2)
        pred = dec_X.squeeze(axis=0).astype(paddle.int32).item()
        # 儲存注意力權重（稍後討論）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列結束詞元被預測，輸出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
```

## 預測序列的評估

我們可以透過與真實的標籤序列進行比較來評估預測序列。
雖然 :cite:`Papineni.Roukos.Ward.ea.2002`
提出的BLEU（bilingual evaluation understudy）
最先是用於評估機器翻譯的結果，
但現在它已經被廣泛用於測量許多應用的輸出序列的質量。
原則上說，對於預測序列中的任意$n$元語法（n-grams），
BLEU的評估都是這個$n$元語法是否出現在標籤序列中。

我們將BLEU定義為：

$$ \exp\left(\min\left(0, 1 - \frac{\mathrm{len}_{\text{label}}}{\mathrm{len}_{\text{pred}}}\right)\right) \prod_{n=1}^k p_n^{1/2^n},$$
:eqlabel:`eq_bleu`

其中$\mathrm{len}_{\text{label}}$表示標籤序列中的詞元數和
$\mathrm{len}_{\text{pred}}$表示預測序列中的詞元數，
$k$是用於對應的最長的$n$元語法。
另外，用$p_n$表示$n$元語法的精確度，它是兩個數量的比值：
第一個是預測序列與標籤序列中對應的$n$元語法的數量，
第二個是預測序列中$n$元語法的數量的比率。
具體地說，給定標籤序列$A$、$B$、$C$、$D$、$E$、$F$
和預測序列$A$、$B$、$B$、$C$、$D$，
我們有$p_1 = 4/5$、$p_2 = 3/4$、$p_3 = 1/3$和$p_4 = 0$。

根據 :eqref:`eq_bleu`中BLEU的定義，
當預測序列與標籤序列完全相同時，BLEU為$1$。
此外，由於$n$元語法越長則對應難度越大，
所以BLEU為更長的$n$元語法的精確度分配更大的權重。
具體來說，當$p_n$固定時，$p_n^{1/2^n}$
會隨著$n$的增長而增加（原始論文使用$p_n^{1/n}$）。
而且，由於預測的序列越短獲得的$p_n$值越高，
所以 :eqref:`eq_bleu`中乘法項之前的係數用於懲罰較短的預測序列。
例如，當$k=2$時，給定標籤序列$A$、$B$、$C$、$D$、$E$、$F$
和預測序列$A$、$B$，儘管$p_1 = p_2 = 1$，
懲罰因子$\exp(1-6/2) \approx 0.14$會降低BLEU。

[**BLEU的程式碼實現**]如下。

```{.python .input}
#@tab all
def bleu(pred_seq, label_seq, k):  #@save
    """計算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
```

最後，利用訓練好的迴圈神經網路『編碼器－解碼器』模型，
[**將幾個英語句子翻譯成法語**]，並計算BLEU的最終結果。

```{.python .input}
#@tab mxnet, pytorch, paddle
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
```

```{.python .input}
#@tab tensorflow
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
```

## 小結

* 根據『編碼器-解碼器』架構的設計，
  我們可以使用兩個迴圈神經網路來設計一個序列到序列學習的模型。
* 在實現編碼器和解碼器時，我們可以使用多層迴圈神經網路。
* 我們可以使用遮蔽來過濾不相關的計算，例如在計算損失時。
* 在『編碼器－解碼器』訓練中，強制教學方法將原始輸出序列（而非預測結果）輸入解碼器。
* BLEU是一種常用的評估方法，它透過測量預測序列和標籤序列之間的$n$元語法的對應度來評估預測。

## 練習

1. 試著透過調整超引數來改善翻譯效果。
1. 重新執行實驗並在計算損失時不使用遮蔽，可以觀察到什麼結果？為什麼會有這個結果？
1. 如果編碼器和解碼器的層數或者隱藏單元數不同，那麼如何初始化解碼器的隱狀態？
1. 在訓練中，如果用前一時間步的預測輸入到解碼器來代替強制教學，對效能有何影響？
1. 用長短期記憶網路替換門控迴圈單元重新執行實驗。
1. 有沒有其他方法來設計解碼器的輸出層？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2783)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2782)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11838)
:end_tab: