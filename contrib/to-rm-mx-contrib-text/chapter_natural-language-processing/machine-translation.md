# 機器翻譯

機器翻譯是指將一段文字從一種語言自動翻譯到另一種語言。因為一段文字序列在不同語言中的長度不一定相同，所以我們使用機器翻譯為例來介紹編碼器—解碼器和注意力機制的應用。

## 讀取和預處理資料集

我們先定義一些特殊符號。其中『&lt;pad&gt;』（padding）符號用來新增在較短序列後，直到每個序列等長，而『&lt;bos&gt;』和『&lt;eos&gt;』符號分別表示序列的開始和結束。

```{.python .input  n=2}
import collections
from d2lzh import text
import io
import math
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn

PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'
```

接著定義兩個輔助函式對後面讀取的資料進行預處理。

```{.python .input}
# 將一個序列中所有的詞記錄在all_tokens中以便之後構造詞典，然後在該序列後面新增PAD直到序列
# 長度變為max_seq_len，然後將序列儲存在all_seqs中
def process_one_seq(seq_tokens, all_tokens, all_seqs, max_seq_len):
    all_tokens.extend(seq_tokens)
    seq_tokens += [EOS] + [PAD] * (max_seq_len - len(seq_tokens) - 1)
    all_seqs.append(seq_tokens)

# 使用所有的詞來構造詞典。並將所有序列中的詞變換為詞索引後構造NDArray實例
def build_data(all_tokens, all_seqs):
    vocab = text.vocab.Vocabulary(collections.Counter(all_tokens),
                                  reserved_tokens=[PAD, BOS, EOS])
    indices = [vocab.to_indices(seq) for seq in all_seqs]
    return vocab, nd.array(indices)
```

為了示範方便，我們在這裡使用一個很小的法語—英語資料集。在這個資料集裡，每一行是一對法語句子和它對應的英語句子，中間使用`'\t'`隔開。在讀取資料時，我們在句末附上『&lt;eos&gt;』符號，並可能透過新增『&lt;pad&gt;』符號使每個序列的長度均為`max_seq_len`。我們為法語詞和英語詞分別建立詞典。法語詞的索引和英語詞的索引相互獨立。

```{.python .input  n=31}
def read_data(max_seq_len):
    # in和out分別是input和output的縮寫
    in_tokens, out_tokens, in_seqs, out_seqs = [], [], [], []
    with io.open('../data/fr-en-small.txt') as f:
        lines = f.readlines()
    for line in lines:
        in_seq, out_seq = line.rstrip().split('\t')
        in_seq_tokens, out_seq_tokens = in_seq.split(' '), out_seq.split(' ')
        if max(len(in_seq_tokens), len(out_seq_tokens)) > max_seq_len - 1:
            continue  # 如果加上EOS後長於max_seq_len，則忽略掉此樣本
        process_one_seq(in_seq_tokens, in_tokens, in_seqs, max_seq_len)
        process_one_seq(out_seq_tokens, out_tokens, out_seqs, max_seq_len)
    in_vocab, in_data = build_data(in_tokens, in_seqs)
    out_vocab, out_data = build_data(out_tokens, out_seqs)
    return in_vocab, out_vocab, gdata.ArrayDataset(in_data, out_data)
```

將序列的最大長度設成7，然後檢視讀取到的第一個樣本。該樣本分別包含法語詞索引序列和英語詞索引序列。

```{.python .input  n=181}
max_seq_len = 7
in_vocab, out_vocab, dataset = read_data(max_seq_len)
dataset[0]
```

## 含注意力機制的編碼器—解碼器

我們將使用含注意力機制的編碼器—解碼器來將一段簡短的法語翻譯成英語。下面我們來介紹模型的實現。

### 編碼器

在編碼器中，我們將輸入語言的詞索引透過詞嵌入層得到詞的表徵，然後輸入到一個多層門控迴圈單元中。正如我們在[『迴圈神經網路的簡潔實現』](../chapter_recurrent-neural-networks/rnn-gluon.md)一節提到的，Gluon的`rnn.GRU`實例在前向計算後也會分別回傳輸出和最終時間步的多層隱狀態。其中的輸出指的是最後一層的隱藏層在各個時間步的隱狀態，並不涉及輸出層計算。注意力機制將這些輸出作為鍵項和值項。

```{.python .input  n=165}
class Encoder(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 drop_prob=0, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob)

    def forward(self, inputs, state):
        # 輸入形狀是(批次大小, 時間步數)。將輸出互換樣本維和時間步維
        embedding = self.embedding(inputs).swapaxes(0, 1)
        return self.rnn(embedding, state)

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```

下面我們來建立一個批次大小為4、時間步數為7的小批次序列輸入。設門控迴圈單元的隱藏層個數為2，隱藏單元個數為16。編碼器對該輸入執行前向計算後回傳的輸出形狀為(時間步數, 批次大小, 隱藏單元個數)。門控迴圈單元在最終時間步的多層隱狀態的形狀為(隱藏層個數, 批次大小, 隱藏單元個數)。對門控迴圈單元來說，`state`清單中只含一個元素，即隱狀態；如果使用長短期記憶，`state`清單中還將包含另一個元素，即記憶細胞。

```{.python .input  n=166}
encoder = Encoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
encoder.initialize()
output, state = encoder(nd.zeros((4, 7)), encoder.begin_state(batch_size=4))
output.shape, state[0].shape
```

### 注意力機制

在介紹如何實現注意力機制的矢量化計算之前，我們先了解一下`Dense`實例的`flatten`選項。當輸入的維度大於2時，預設情況下，`Dense`實例會將除了第一維（樣本維）以外的維度均視作需要仿射變換的特徵維，並將輸入自動轉成行為樣本、列為特徵的二維矩陣。計算後，輸出矩陣的形狀為(樣本數, 輸出個數)。如果我們希望全連線層只對輸入的最後一維做仿射變換，而保持其他維度上的形狀不變，便需要將`Dense`實例的`flatten`選項設為`False`。在下面例子中，全連線層只對輸入的最後一維做仿射變換，因此輸出形狀中只有最後一維變為全連線層的輸出個數2。

```{.python .input}
dense = nn.Dense(2, flatten=False)
dense.initialize()
dense(nd.zeros((3, 5, 7))).shape
```

我們將實現[『注意力機制』](./attention.md)一節中定義的函式$a$：將輸入連結後透過含單隱藏層的多層感知機變換。其中隱藏層的輸入是解碼器的隱狀態與編碼器在所有時間步上隱狀態的一一連結，且使用tanh函式作為啟用函式。輸出層的輸出個數為1。兩個`Dense`實例均不使用偏置，且設`flatten=False`。其中函式$a$定義裡向量$\boldsymbol{v}$的長度是一個超引數，即`attention_size`。

```{.python .input  n=167}
def attention_model(attention_size):
    model = nn.Sequential()
    model.add(nn.Dense(attention_size, activation='tanh', use_bias=False,
                       flatten=False),
              nn.Dense(1, use_bias=False, flatten=False))
    return model
```

注意力機制的輸入包括查詢項、鍵項和值項。設編碼器和解碼器的隱藏單元個數相同。這裡的查詢項為解碼器在上一時間步的隱狀態，形狀為(批次大小, 隱藏單元個數)；鍵項和值項均為編碼器在所有時間步的隱狀態，形狀為(時間步數, 批次大小, 隱藏單元個數)。注意力機制回傳當前時間步的上下文變數，形狀為(批次大小, 隱藏單元個數)。

```{.python .input  n=168}
def attention_forward(model, enc_states, dec_state):
    # 將解碼器隱狀態廣播到和編碼器隱狀態形狀相同後進行連結
    dec_states = nd.broadcast_axis(
        dec_state.expand_dims(0), axis=0, size=enc_states.shape[0])
    enc_and_dec_states = nd.concat(enc_states, dec_states, dim=2)
    e = model(enc_and_dec_states)  # 形狀為(時間步數, 批次大小, 1)
    alpha = nd.softmax(e, axis=0)  # 在時間步維度做softmax運算
    return (alpha * enc_states).sum(axis=0)  # 回傳上下文變數
```

在下面的例子中，編碼器的時間步數為10，批次大小為4，編碼器和解碼器的隱藏單元個數均為8。注意力機制回傳一個小批次的背景向量，每個背景向量的長度等於編碼器的隱藏單元個數。因此輸出的形狀為(4, 8)。

```{.python .input  n=169}
seq_len, batch_size, num_hiddens = 10, 4, 8
model = attention_model(10)
model.initialize()
enc_states = nd.zeros((seq_len, batch_size, num_hiddens))
dec_state = nd.zeros((batch_size, num_hiddens))
attention_forward(model, enc_states, dec_state).shape
```

### 含注意力機制的解碼器

我們直接將編碼器在最終時間步的隱狀態作為解碼器的初始隱狀態。這要求編碼器和解碼器的迴圈神經網路使用相同的隱藏層個數和隱藏單元個數。

在解碼器的前向計算中，我們先透過剛剛介紹的注意力機制計算得到當前時間步的背景向量。由於解碼器的輸入來自輸出語言的詞索引，我們將輸入透過詞嵌入層得到表徵，然後和背景向量在特徵維連結。我們將連結後的結果與上一時間步的隱狀態透過門控迴圈單元計算出當前時間步的輸出與隱狀態。最後，我們將輸出透過全連線層變換為有關各個輸出詞的預測，形狀為(批次大小, 輸出詞典大小)。

```{.python .input  n=170}
class Decoder(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 attention_size, drop_prob=0, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = attention_model(attention_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob)
        self.out = nn.Dense(vocab_size, flatten=False)

    def forward(self, cur_input, state, enc_states):
        # 使用注意力機制計算背景向量
        c = attention_forward(self.attention, enc_states, state[0][-1])
        # 將嵌入後的輸入和背景向量在特徵維連結
        input_and_c = nd.concat(self.embedding(cur_input), c, dim=1)
        # 為輸入和背景向量的連結增加時間步維，時間步個數為1
        output, state = self.rnn(input_and_c.expand_dims(0), state)
        # 移除時間步維，輸出形狀為(批次大小, 輸出詞典大小)
        output = self.out(output).squeeze(axis=0)
        return output, state

    def begin_state(self, enc_state):
        # 直接將編碼器最終時間步的隱狀態作為解碼器的初始隱狀態
        return enc_state
```

## 訓練模型

我們先實現`batch_loss`函式計算一個小批次的損失。解碼器在最初時間步的輸入是特殊字元`BOS`。之後，解碼器在某時間步的輸入為樣本輸出序列在上一時間步的詞，即強制教學。此外，同[『word2vec的實現』](word2vec-gluon.md)一節中的實現一樣，我們在這裡也使用掩碼變數避免填充項對損失函式計算的影響。

```{.python .input}
def batch_loss(encoder, decoder, X, Y, loss):
    batch_size = X.shape[0]
    enc_state = encoder.begin_state(batch_size=batch_size)
    enc_outputs, enc_state = encoder(X, enc_state)
    # 初始化解碼器的隱狀態
    dec_state = decoder.begin_state(enc_state)
    # 解碼器在最初時間步的輸入是BOS
    dec_input = nd.array([out_vocab.token_to_idx[BOS]] * batch_size)
    # 我們將使用掩碼變數mask來忽略掉標籤為填充項PAD的損失
    mask, num_not_pad_tokens = nd.ones(shape=(batch_size,)), 0
    l = nd.array([0])
    for y in Y.T:
        dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)
        l = l + (mask * loss(dec_output, y)).sum()
        dec_input = y  # 使用強制教學
        num_not_pad_tokens += mask.sum().asscalar()
        # 當遇到EOS時，序列後面的詞將均為PAD，相應位置的掩碼設成0
        mask = mask * (y != out_vocab.token_to_idx[EOS])
    return l / num_not_pad_tokens
```

在訓練函式中，我們需要同時迭代編碼器和解碼器的模型引數。

```{.python .input  n=188}
def train(encoder, decoder, dataset, lr, batch_size, num_epochs):
    encoder.initialize(init.Xavier(), force_reinit=True)
    decoder.initialize(init.Xavier(), force_reinit=True)
    enc_trainer = gluon.Trainer(encoder.collect_params(), 'adam',
                                {'learning_rate': lr})
    dec_trainer = gluon.Trainer(decoder.collect_params(), 'adam',
                                {'learning_rate': lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
    for epoch in range(num_epochs):
        l_sum = 0.0
        for X, Y in data_iter:
            with autograd.record():
                l = batch_loss(encoder, decoder, X, Y, loss)
            l.backward()
            enc_trainer.step(1)
            dec_trainer.step(1)
            l_sum += l.asscalar()
        if (epoch + 1) % 10 == 0:
            print("epoch %d, loss %.3f" % (epoch + 1, l_sum / len(data_iter)))
```

接下來，建立模型實例並設定超引數。然後，我們就可以訓練模型了。

```{.python .input}
embed_size, num_hiddens, num_layers = 64, 64, 2
attention_size, drop_prob, lr, batch_size, num_epochs = 10, 0.5, 0.01, 2, 50
encoder = Encoder(len(in_vocab), embed_size, num_hiddens, num_layers,
                  drop_prob)
decoder = Decoder(len(out_vocab), embed_size, num_hiddens, num_layers,
                  attention_size, drop_prob)
train(encoder, decoder, dataset, lr, batch_size, num_epochs)
```

## 預測不定長的序列

在[『束搜尋』](beam-search.md)一節中我們介紹了3種方法來生成解碼器在每個時間步的輸出。這裡我們實現最簡單的貪婪搜尋。

```{.python .input  n=177}
def translate(encoder, decoder, input_seq, max_seq_len):
    in_tokens = input_seq.split(' ')
    in_tokens += [EOS] + [PAD] * (max_seq_len - len(in_tokens) - 1)
    enc_input = nd.array([in_vocab.to_indices(in_tokens)])
    enc_state = encoder.begin_state(batch_size=1)
    enc_output, enc_state = encoder(enc_input, enc_state)
    dec_input = nd.array([out_vocab.token_to_idx[BOS]])
    dec_state = decoder.begin_state(enc_state)
    output_tokens = []
    for _ in range(max_seq_len):
        dec_output, dec_state = decoder(dec_input, dec_state, enc_output)
        pred = dec_output.argmax(axis=1)
        pred_token = out_vocab.idx_to_token[int(pred.asscalar())]
        if pred_token == EOS:  # 當任一時間步搜尋出EOS時，輸出序列即完成
            break
        else:
            output_tokens.append(pred_token)
            dec_input = pred
    return output_tokens
```

簡單測試一下模型。輸入法語句子『ils regardent.』，翻譯後的英語句子應該是『they are watching.』。

```{.python .input}
input_seq = 'ils regardent .'
translate(encoder, decoder, input_seq, max_seq_len)
```

## 評價翻譯結果

評價機器翻譯結果通常使用BLEU（Bilingual Evaluation Understudy）[1]。對於模型預測序列中任意的子序列，BLEU考察這個子序列是否出現在標籤序列中。

具體來說，設詞數為$n$的子序列的精度為$p_n$。它是預測序列與標籤序列對應詞數為$n$的子序列的數量與預測序列中詞數為$n$的子序列的數量之比。舉個例子，假設標籤序列為$A$、$B$、$C$、$D$、$E$、$F$，預測序列為$A$、$B$、$B$、$C$、$D$，那麼$p_1 = 4/5,\ p_2 = 3/4,\ p_3 = 1/3,\ p_4 = 0$。設$len_{\text{label}}$和$len_{\text{pred}}$分別為標籤序列和預測序列的詞數，那麼，BLEU的定義為

$$ \exp\left(\min\left(0, 1 - \frac{len_{\text{label}}}{len_{\text{pred}}}\right)\right) \prod_{n=1}^k p_n^{1/2^n},$$

其中$k$是我們希望對應的子序列的最大詞數。可以看到當預測序列和標籤序列完全一致時，BLEU為1。

因為對應較長子序列比對應較短子序列更難，BLEU對對應較長子序列的精度賦予了更大權重。例如，當$p_n$固定在0.5時，隨著$n$的增大，$0.5^{1/2} \approx 0.7, 0.5^{1/4} \approx 0.84, 0.5^{1/8} \approx 0.92, 0.5^{1/16} \approx 0.96$。另外，模型預測較短序列往往會得到較高$p_n$值。因此，上式中連乘項前面的係數是為了懲罰較短的輸出而設的。舉個例子，當$k=2$時，假設標籤序列為$A$、$B$、$C$、$D$、$E$、$F$，而預測序列為$A$、$B$。雖然$p_1 = p_2 = 1$，但懲罰係數$\exp(1-6/2) \approx 0.14$，因此BLEU也接近0.14。

下面來實現BLEU的計算。

```{.python .input}
def bleu(pred_tokens, label_tokens, k):
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
```

接下來，定義一個輔助列印函式。

```{.python .input}
def score(input_seq, label_seq, k):
    pred_tokens = translate(encoder, decoder, input_seq, max_seq_len)
    label_tokens = label_seq.split(' ')
    print('bleu %.3f, predict: %s' % (bleu(pred_tokens, label_tokens, k),
                                      ' '.join(pred_tokens)))
```

預測正確則分數為1。

```{.python .input}
score('ils regardent .', 'they are watching .', k=2)
```

測試一個不在訓練集中的樣本。

```{.python .input}
score('ils sont canadiens .', 'they are canadian .', k=2)
```

## 小結

* 可以將編碼器—解碼器和注意力機制應用於機器翻譯中。
* BLEU可以用來評價翻譯結果。

## 練習

* 如果編碼器和解碼器的隱藏單元個數不同或隱藏層個數不同，該如何改進解碼器的隱狀態的初始化方法？
* 在訓練中，將強制教學替換為使用解碼器在上一時間步的輸出作為解碼器在當前時間步的輸入，結果有什麼變化嗎？
* 試著使用更大的翻譯資料集來訓練模型，如WMT [2] 和Tatoeba Project [3]。




## 參考文獻

[1] Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002, July). BLEU: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting on association for computational linguistics (pp. 311-318). Association for Computational Linguistics.

[2] WMT. http://www.statmt.org/wmt14/translation-task.html

[3] Tatoeba Project. http://www.manythings.org/anki/

## 掃碼直達[討論區](https://discuss.gluon.ai/t/topic/4689)

![](../img/qr_machine-translation.svg)
