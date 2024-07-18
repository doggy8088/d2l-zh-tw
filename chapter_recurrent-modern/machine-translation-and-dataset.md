# 機器翻譯與資料集
:label:`sec_machine_translation`

語言模型是自然語言處理的關鍵，
而*機器翻譯*是語言模型最成功的基準測試。
因為機器翻譯正是將輸入序列轉換成輸出序列的
*序列轉換模型*（sequence transduction）的核心問題。
序列轉換模型在各類現代人工智慧應用中發揮著至關重要的作用，
因此我們將其做為本章剩餘部分和 :numref:`chap_attention`的重點。
為此，本節將介紹機器翻譯問題及其後文需要使用的資料集。

*機器翻譯*（machine translation）指的是
將序列從一種語言自動翻譯成另一種語言。
事實上，這個研究領域可以追溯到數字電腦發明後不久的20世紀40年代，
特別是在第二次世界大戰中使用電腦破解語言編碼。
幾十年來，在使用神經網路進行端到端學習的興起之前，
統計學方法在這一領域一直佔據主導地位
 :cite:`Brown.Cocke.Della-Pietra.ea.1988,Brown.Cocke.Della-Pietra.ea.1990`。
因為*統計機器翻譯*（statistical machine translation）涉及了
翻譯模型和語言模型等組成部分的統計分析，
因此基於神經網路的方法通常被稱為
*神經機器翻譯*（neural machine translation），
用於將兩種翻譯模型區分開來。

本書的關注點是神經網路機器翻譯方法，強調的是端到端的學習。
與 :numref:`sec_language_model`中的語料庫
是單一語言的語言模型問題存在不同，
機器翻譯的資料集是由源語言和目標語言的文字序列對組成的。
因此，我們需要一種完全不同的方法來預處理機器翻譯資料集，
而不是複用語言模型的預處理程式。
下面，我們看一下如何將預處理後的資料載入到小批次中用於訓練。

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
import os
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import os
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
import os
```

## [**下載和預處理資料集**]

首先，下載一個由[Tatoeba專案的雙語句子對](http://www.manythings.org/anki/)
組成的『英－法』資料集，資料集中的每一行都是製表符分隔的文字序列對，
序列對由英文文字序列和翻譯後的法語文字序列組成。
請注意，每個文字序列可以是一個句子，
也可以是包含多個句子的一個段落。
在這個將英語翻譯成法語的機器翻譯問題中，
英語是*源語言*（source language），
法語是*目標語言*（target language）。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

#@save
def read_data_nmt():
    """載入『英語－法語』資料集"""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r', 
             encoding='utf-8') as f:
        return f.read()

raw_text = read_data_nmt()
print(raw_text[:75])
```

下載資料集後，原始文字資料需要經過[**幾個預處理步驟**]。
例如，我們用空格代替*不間斷空格*（non-breaking space），
使用小寫字母替換大寫字母，並在單詞和標點符號之間插入空格。

```{.python .input}
#@tab all
#@save
def preprocess_nmt(text):
    """預處理『英語－法語』資料集"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替換不間斷空格
    # 使用小寫字母替換大寫字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在單詞和標點符號之間插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

text = preprocess_nmt(raw_text)
print(text[:80])
```

## [**詞元化**]

與 :numref:`sec_language_model`中的字元級詞元化不同，
在機器翻譯中，我們更喜歡單詞級詞元化
（最先進的模型可能使用更高階的詞元化技術）。
下面的`tokenize_nmt`函式對前`num_examples`個文字序列對進行詞元，
其中每個詞元要麼是一個詞，要麼是一個標點符號。
此函式回傳兩個詞元清單：`source`和`target`：
`source[i]`是源語言（這裡是英語）第$i$個文字序列的詞元清單，
`target[i]`是目標語言（這裡是法語）第$i$個文字序列的詞元清單。

```{.python .input}
#@tab all
#@save
def tokenize_nmt(text, num_examples=None):
    """詞元化『英語－法語』資料資料集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

source, target = tokenize_nmt(text)
source[:6], target[:6]
```

讓我們[**繪製每個文字序列所包含的詞元數量的直方圖**]。
在這個簡單的『英－法』資料集中，大多數文字序列的詞元數量少於$20$個。

```{.python .input}
#@tab all
#@save
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """繪製清單長度對的直方圖"""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)

show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', source, target);
```

## [**詞表**]

由於機器翻譯資料集由語言對組成，
因此我們可以分別為源語言和目標語言建立兩個詞表。
使用單詞級詞元化時，詞表大小將明顯大於使用字元級詞元化時的詞表大小。
為了緩解這一問題，這裡我們將出現次數少於2次的低頻率詞元
視為相同的未知（『&lt;unk&gt;』）詞元。
除此之外，我們還指定了額外的特定詞元，
例如在小批次時用於將序列填充到相同長度的填充詞元（『&lt;pad&gt;』），
以及序列的開始詞元（『&lt;bos&gt;』）和結束詞元（『&lt;eos&gt;』）。
這些特殊詞元在自然語言處理任務中比較常用。

```{.python .input}
#@tab all
src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)
```

## 載入資料集
:label:`subsec_mt_data_loading`

回想一下，語言模型中的[**序列樣本都有一個固定的長度**]，
無論這個樣本是一個句子的一部分還是跨越了多個句子的一個片斷。
這個固定長度是由 :numref:`sec_language_model`中的
`num_steps`（時間步數或詞元數量）引數指定的。
在機器翻譯中，每個樣本都是由源和目標組成的文字序列對，
其中的每個文字序列可能具有不同的長度。

為了提高計算效率，我們仍然可以透過*截斷*（truncation）和
*填充*（padding）方式實現一次只處理一個小批次的文字序列。
假設同一個小批次中的每個序列都應該具有相同的長度`num_steps`，
那麼如果文字序列的詞元數目少於`num_steps`時，
我們將繼續在其末尾新增特定的『&lt;pad&gt;』詞元，
直到其長度達到`num_steps`；
反之，我們將截斷文字序列時，只取其前`num_steps` 個詞元，
並且丟棄剩餘的詞元。這樣，每個文字序列將具有相同的長度，
以便以相同形狀的小批次進行載入。

如前所述，下面的`truncate_pad`函式將(**截斷或填充文字序列**)。

```{.python .input}
#@tab all
#@save
def truncate_pad(line, num_steps, padding_token):
    """截斷或填充文字序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截斷
    return line + [padding_token] * (num_steps - len(line))  # 填充

truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
```

現在我們定義一個函式，可以將文字序列
[**轉換成小批次資料集用於訓練**]。
我們將特定的『&lt;eos&gt;』詞元新增到所有序列的末尾，
用於表示序列的結束。
當模型透過一個詞元接一個詞元地生成序列進行預測時，
生成的『&lt;eos&gt;』詞元說明完成了序列輸出工作。
此外，我們還記錄了每個文字序列的長度，
統計長度時排除了填充詞元，
在稍後將要介紹的一些模型會需要這個長度訊息。

```{.python .input}
#@tab all
#@save
def build_array_nmt(lines, vocab, num_steps):
    """將機器翻譯的文字序列轉換成小批次"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = d2l.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = d2l.reduce_sum(
        d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
    return array, valid_len
```

## [**訓練模型**]

最後，我們定義`load_data_nmt`函式來回傳資料迭代器，
以及源語言和目標語言的兩種詞表。

```{.python .input}
#@tab all
#@save
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """回傳翻譯資料集的迭代器和詞表"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab
```

下面我們[**讀出『英語－法語』資料集中的第一個小批次資料**]。

```{.python .input}
#@tab all
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', d2l.astype(X, d2l.int32))
    print('X的有效長度:', X_valid_len)
    print('Y:', d2l.astype(Y, d2l.int32))
    print('Y的有效長度:', Y_valid_len)
    break
```

## 小結

* 機器翻譯指的是將文字序列從一種語言自動翻譯成另一種語言。
* 使用單詞級詞元化時的詞表大小，將明顯大於使用字元級詞元化時的詞表大小。為了緩解這一問題，我們可以將低頻詞元視為相同的未知詞元。
* 透過截斷和填充文字序列，可以保證所有的文字序列都具有相同的長度，以便以小批次的方式載入。

## 練習

1. 在`load_data_nmt`函式中嘗試不同的`num_examples`引數值。這對源語言和目標語言的詞表大小有何影響？
1. 某些語言（例如中文和日語）的文字沒有單詞邊界指示符（例如空格）。對於這種情況，單詞級詞元化仍然是個好主意嗎？為什麼？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2777)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2776)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11836)
:end_tab:
