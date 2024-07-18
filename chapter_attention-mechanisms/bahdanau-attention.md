# Bahdanau 注意力
:label:`sec_seq2seq_attention`

 :numref:`sec_seq2seq`中探討了機器翻譯問題：
透過設計一個基於兩個迴圈神經網路的編碼器-解碼器架構，
用於序列到序列學習。
具體來說，迴圈神經網路編碼器將長度可變的序列轉換為固定形狀的上下文變數，
然後迴圈神經網路解碼器根據生成的詞元和上下文變數
按詞元生成輸出（目標）序列詞元。
然而，即使並非所有輸入（源）詞元都對解碼某個詞元都有用，
在每個解碼步驟中仍使用編碼*相同*的上下文變數。
有什麼方法能改變上下文變數呢？

我們試著從 :cite:`Graves.2013`中找到靈感：
在為給定文字序列生成手寫的挑戰中，
Graves設計了一種可微注意力模型，
將文字字元與更長的筆跡對齊，
其中對齊方式僅向一個方向移動。
受學習對齊想法的啟發，Bahdanau等人提出了一個沒有嚴格單向對齊限制的
可微注意力模型 :cite:`Bahdanau.Cho.Bengio.2014`。
在預測詞元時，如果不是所有輸入詞元都相關，模型將僅對齊（或參與）輸入序列中與當前預測相關的部分。這是透過將上下文變數視為注意力集中的輸出來實現的。

## 模型

下面描述的Bahdanau注意力模型
將遵循 :numref:`sec_seq2seq`中的相同符號表達。
這個新的基於注意力的模型與 :numref:`sec_seq2seq`中的模型相同，
只不過 :eqref:`eq_seq2seq_s_t`中的上下文變數$\mathbf{c}$
在任何解碼時間步$t'$都會被$\mathbf{c}_{t'}$替換。
假設輸入序列中有$T$個詞元，
解碼時間步$t'$的上下文變數是注意力集中的輸出：

$$\mathbf{c}_{t'} = \sum_{t=1}^T \alpha(\mathbf{s}_{t' - 1}, \mathbf{h}_t) \mathbf{h}_t,$$

其中，時間步$t' - 1$時的解碼器隱狀態$\mathbf{s}_{t' - 1}$是查詢，
編碼器隱狀態$\mathbf{h}_t$既是鍵，也是值，
注意力權重$\alpha$是使用 :eqref:`eq_attn-scoring-alpha`
所定義的加性注意力打分函式計算的。

與 :numref:`fig_seq2seq_details`中的迴圈神經網路編碼器-解碼器架構略有不同，
 :numref:`fig_s2s_attention_details`描述了Bahdanau注意力的架構。

![一個帶有Bahdanau注意力的迴圈神經網路編碼器-解碼器模型](../img/seq2seq-attention-details.svg)
:label:`fig_s2s_attention_details`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import rnn, nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
```

## 定義注意力解碼器

下面看看如何定義Bahdanau注意力，實現迴圈神經網路編碼器-解碼器。
其實，我們只需重新定義解碼器即可。
為了更方便地顯示學習的注意力權重，
以下`AttentionDecoder`類別定義了[**帶有注意力機制解碼器的基本介面**]。

```{.python .input}
#@tab all
#@save
class AttentionDecoder(d2l.Decoder):
    """帶有注意力機制解碼器的基本介面"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError
```

接下來，讓我們在接下來的`Seq2SeqAttentionDecoder`類別中
[**實現帶有Bahdanau注意力的迴圈神經網路解碼器**]。
首先，初始化解碼器的狀態，需要下面的輸入：

1. 編碼器在所有時間步的最終層隱狀態，將作為注意力的鍵和值；
1. 上一時間步的編碼器全層隱狀態，將作為初始化解碼器的隱狀態；
1. 編碼器有效長度（排除在注意力池中填充詞元）。

在每個解碼時間步驟中，解碼器上一個時間步的最終層隱狀態將用作查詢。
因此，注意力輸出和輸入嵌入都連結為迴圈神經網路解碼器的輸入。

```{.python .input}
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs的形狀為(num_steps，batch_size，num_hiddens)
        # hidden_state[0]的形狀為(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.swapaxes(0, 1), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # enc_outputs的形狀為(batch_size,num_steps,num_hiddens).
        # hidden_state[0]的形狀為(num_layers,batch_size,
        # num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # 輸出X的形狀為(num_steps,batch_size,embed_size)
        X = self.embedding(X).swapaxes(0, 1)
        outputs, self._attention_weights = [], []
        for x in X:
            # query的形狀為(batch_size,1,num_hiddens)
            query = np.expand_dims(hidden_state[0][-1], axis=1)
            # context的形狀為(batch_size,1,num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # 在特徵維度上連結
            x = np.concatenate((context, np.expand_dims(x, axis=1)), axis=-1)
            # 將x變形為(1,batch_size,embed_size+num_hiddens)
            out, hidden_state = self.rnn(x.swapaxes(0, 1), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 全連線層變換後，outputs的形狀為
        # (num_steps,batch_size,vocab_size)
        outputs = self.dense(np.concatenate(outputs, axis=0))
        return outputs.swapaxes(0, 1), [enc_outputs, hidden_state,
                                        enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
#@tab pytorch
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs的形狀為(batch_size，num_steps，num_hiddens).
        # hidden_state的形狀為(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # enc_outputs的形狀為(batch_size,num_steps,num_hiddens).
        # hidden_state的形狀為(num_layers,batch_size,
        # num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # 輸出X的形狀為(num_steps,batch_size,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # query的形狀為(batch_size,1,num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # context的形狀為(batch_size,1,num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # 在特徵維度上連結
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # 將x變形為(1,batch_size,embed_size+num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 全連線層變換後，outputs的形狀為
        # (num_steps,batch_size,vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]
    
    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
#@tab tensorflow
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(num_hiddens, num_hiddens,
                                               num_hiddens, dropout)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout)
             for _ in range(num_layers)]),
                                      return_sequences=True, 
                                      return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
       # outputs的形狀為(num_steps，batch_size，num_hiddens)
        # hidden_state[0]的形狀為(num_layers，batch_size，num_hiddens)

        outputs, hidden_state = enc_outputs
        return (outputs, hidden_state, enc_valid_lens)

    def call(self, X, state, **kwargs):
        # enc_outputs的形狀為(batch_size,num_steps,num_hiddens)
        # hidden_state[0]的形狀為(num_layers,batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # 輸出X的形狀為(num_steps,batch_size,embed_size)
        X = self.embedding(X) # 輸入X的形狀為(batch_size,num_steps)
        X = tf.transpose(X, perm=(1, 0, 2))
        outputs, self._attention_weights = [], []
        for x in X:
            # query的形狀為(batch_size,1,num_hiddens)
            query = tf.expand_dims(hidden_state[-1], axis=1)
            # context的形狀為(batch_size,1,num_hiddens)
            context = self.attention(query, enc_outputs, enc_outputs,
                                     enc_valid_lens, **kwargs)
            # 在特徵維度上連結
            x = tf.concat((context, tf.expand_dims(x, axis=1)), axis=-1)
            out = self.rnn(x, hidden_state, **kwargs)
            hidden_state = out[1:]
            outputs.append(out[0])
            self._attention_weights.append(self.attention.attention_weights)
        # 全連線層變換後，outputs的形狀為(num_steps,batch_size,vocab_size)
        outputs = self.dense(tf.concat(outputs, axis=1))
        return outputs, [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
#@tab paddle
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, 
                          num_layers, bias_ih_attr=True,
                          time_major=True, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs的形狀為(batch_size，num_steps，num_hiddens).
        # hidden_state的形狀為(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.transpose((1, 0, 2)), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # enc_outputs的形狀為(batch_size,num_steps,num_hiddens).
        # hidden_state的形狀為(num_layers,batch_size,num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # 輸出X的形狀為(num_steps,batch_size,embed_size)
        X = self.embedding(X).transpose((1, 0, 2))
        outputs, self._attention_weights = [], []
        for x in X:
            # query的形狀為(batch_size,1,num_hiddens)
            query = paddle.unsqueeze(hidden_state[-1], axis=1)
            # context的形狀為(batch_size,1,num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # 在特徵維度上連結
            x = paddle.concat((context, paddle.unsqueeze(x, axis=1)), axis=-1)
            # 將x變形為(1,batch_size,embed_size+num_hiddens)
            out, hidden_state = self.rnn(x.transpose((1, 0, 2)), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 全連線層變換後，outputs的形狀為
        # (num_steps,batch_size,vocab_size)
        outputs = self.dense(paddle.concat(outputs, axis=0))
        return outputs.transpose((1, 0, 2)), [enc_outputs, hidden_state, 
                                              enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

接下來，使用包含7個時間步的4個序列輸入的小批次[**測試Bahdanau注意力解碼器**]。

```{.python .input}
encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
encoder.initialize()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                  num_layers=2)
decoder.initialize()
X = d2l.zeros((4, 7))  # (batch_size,num_steps)
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape
```

```{.python .input}
#@tab pytorch
encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
encoder.eval()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                  num_layers=2)
decoder.eval()
X = d2l.zeros((4, 7), dtype=torch.long)  # (batch_size,num_steps)
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape
```

```{.python .input}
#@tab tensorflow
encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                  num_layers=2)
X = tf.zeros((4, 7))
state = decoder.init_state(encoder(X, training=False), None)
output, state = decoder(X, state, training=False)
output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape
```

```{.python .input}
#@tab paddle
encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
encoder.eval()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                  num_layers=2)
decoder.eval()
X = paddle.zeros((4, 7), dtype='int64')  # (batch_size,num_steps)
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape
```

## [**訓練**]

與 :numref:`sec_seq2seq_training`類似，
我們在這裡指定超引數，實例化一個帶有Bahdanau注意力的編碼器和解碼器，
並對這個模型進行機器翻譯訓練。
由於新增的注意力機制，訓練要比沒有注意力機制的
 :numref:`sec_seq2seq_training`慢得多。

```{.python .input}
#@tab all
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 250, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = d2l.Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
```

模型訓練後，我們用它[**將幾個英語句子翻譯成法語**]並計算它們的BLEU分數。

```{.python .input}
#@tab mxnet, pytorch, paddle
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
```

```{.python .input}
#@tab tensorflow
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
```

```{.python .input}
#@tab all
attention_weights = d2l.reshape(
    d2l.concat([step[0][0][0] for step in dec_attention_weight_seq], 0),
    (1, 1, -1, num_steps))
```

訓練結束後，下面透過[**視覺化注意力權重**]
會發現，每個查詢都會在鍵值對上分配不同的權重，這說明
在每個解碼步中，輸入序列的不同部分被選擇性地聚集在注意力池中。

```{.python .input}
# 加上一個包含序列結束詞元
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1],
    xlabel='Key positions', ylabel='Query positions')
```

```{.python .input}
#@tab pytorch, paddle
# 加上一個包含序列結束詞元
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
    xlabel='Key positions', ylabel='Query positions')
```

```{.python .input}
#@tab tensorflow
# 加上一個包含序列結束詞元
d2l.show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1],
                  xlabel='Key posistions', ylabel='Query posistions')
```

## 小結

* 在預測詞元時，如果不是所有輸入詞元都是相關的，那麼具有Bahdanau注意力的迴圈神經網路編碼器-解碼器會有選擇地統計輸入序列的不同部分。這是透過將上下文變數視為加性注意力池化的輸出來實現的。
* 在迴圈神經網路編碼器-解碼器中，Bahdanau注意力將上一時間步的解碼器隱狀態視為查詢，在所有時間步的編碼器隱狀態同時視為鍵和值。

## 練習

1. 在實驗中用LSTM替換GRU。
1. 修改實驗以將加性注意力打分函式替換為縮放點積注意力，它如何影響訓練效率？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/5753)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/5754)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11842)
:end_tab: