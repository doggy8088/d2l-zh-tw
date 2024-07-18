# 子詞嵌入
:label:`sec_fasttext`

在英語中，『helps』『helped』和『helping』等單詞都是同一個詞『help』的變形形式。『dog』和『dogs』之間的關係與『cat』和『cats』之間的關係相同，『boy』和『boyfriend』之間的關係與『girl』和『girlfriend』之間的關係相同。在法語和西班牙語等其他語言中，許多動詞有40多種變形形式，而在芬蘭語中，名詞最多可能有15種變形。在語言學中，形態學研究單詞形成和詞彙關係。但是，word2vec和GloVe都沒有對詞的內部結構進行探討。

## fastText模型

回想一下詞在word2vec中是如何表示的。在跳元模型和連續詞袋模型中，同一詞的不同變形形式直接由不同的向量表示，不需要共享引數。為了使用形態訊息，*fastText模型*提出了一種*子詞嵌入*方法，其中子詞是一個字元$n$-gram :cite:`Bojanowski.Grave.Joulin.ea.2017`。fastText可以被認為是子詞級跳元模型，而非學習詞級向量表示，其中每個*中心詞*由其子詞級向量之和表示。

讓我們來說明如何以單詞『where』為例獲得fastText中每個中心詞的子詞。首先，在詞的開頭和末尾新增特殊字元『&lt;』和『&gt;』，以將字首和字尾與其他子詞區分開來。
然後，從詞中提取字元$n$-gram。
例如，值$n=3$時，我們將獲得長度為3的所有子詞：
『&lt;wh』『whe』『her』『ere』『re&gt;』和特殊子詞『&lt;where&gt;』。

在fastText中，對於任意詞$w$，用$\mathcal{G}_w$表示其長度在3和6之間的所有子詞與其特殊子詞的並集。詞表是所有詞的子詞的集合。假設$\mathbf{z}_g$是詞典中的子詞$g$的向量，則跳元模型中作為中心詞的詞$w$的向量$\mathbf{v}_w$是其子詞向量的和：

$$\mathbf{v}_w = \sum_{g\in\mathcal{G}_w} \mathbf{z}_g.$$

fastText的其餘部分與跳元模型相同。與跳元模型相比，fastText的詞量更大，模型引數也更多。此外，為了計算一個詞的表示，它的所有子詞向量都必須求和，這導致了更高的計算複雜度。然而，由於具有相似結構的詞之間共享來自子詞的引數，罕見詞甚至詞表外的詞在fastText中可能獲得更好的向量表示。

## 位元組對編碼（Byte Pair Encoding）
:label:`subsec_Byte_Pair_Encoding`

在fastText中，所有提取的子詞都必須是指定的長度，例如$3$到$6$，因此詞表大小不能預定義。為了在固定大小的詞表中允許可變長度的子詞，我們可以應用一種稱為*位元組對編碼*（Byte Pair Encoding，BPE）的壓縮演算法來提取子詞 :cite:`Sennrich.Haddow.Birch.2015`。

位元組對編碼執行訓練資料集的統計分析，以發現單詞內的公共符號，諸如任意長度的連續字元。從長度為1的符號開始，位元組對編碼迭代地合併最頻繁的連續符號對以產生新的更長的符號。請注意，為提高效率，不考慮跨越單詞邊界的對。最後，我們可以使用像子詞這樣的符號來切分單詞。位元組對編碼及其變體已經用於諸如GPT-2 :cite:`Radford.Wu.Child.ea.2019`和RoBERTa :cite:`Liu.Ott.Goyal.ea.2019`等自然語言處理預訓練模型中的輸入表示。在下面，我們將說明位元組對編碼是如何工作的。

首先，我們將符號詞表初始化為所有英文小寫字元、特殊的詞尾符號`'_'`和特殊的未知符號`'[UNK]'`。

```{.python .input}
#@tab mxnet, pytorch
import collections

symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']
           
```

```{.python .input}
#@tab paddle
import collections

symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']
           
```

因為我們不考慮跨越詞邊界的符號對，所以我們只需要一個字典`raw_token_freqs`將詞對映到資料集中的頻率（出現次數）。注意，特殊符號`'_'`被附加到每個詞的尾部，以便我們可以容易地從輸出符號序列（例如，『a_all er_man』）恢復單詞序列（例如，『a_all er_man』）。由於我們僅從單個字元和特殊符號的詞開始合併處理，所以在每個詞（詞典`token_freqs`的鍵）內的每對連續字元之間插入空格。換句話說，空格是詞中符號之間的分隔符。

```{.python .input}
#@tab all
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
token_freqs = {}
for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]
token_freqs
```

我們定義以下`get_max_freq_pair`函式，其回傳詞內最頻繁的連續符號對，其中詞來自輸入詞典`token_freqs`的鍵。

```{.python .input}
#@tab all
def get_max_freq_pair(token_freqs):
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            # 『pairs』的鍵是兩個連續符號的元組
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get)  # 具有最大值的『pairs』鍵
```

作為基於連續符號頻率的貪心方法，位元組對編碼將使用以下`merge_symbols`函式來合併最頻繁的連續符號對以產生新符號。

```{.python .input}
#@tab all
def merge_symbols(max_freq_pair, token_freqs, symbols):
    symbols.append(''.join(max_freq_pair))
    new_token_freqs = dict()
    for token, freq in token_freqs.items():
        new_token = token.replace(' '.join(max_freq_pair),
                                  ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs
```

現在，我們對詞典`token_freqs`的鍵迭代地執行位元組對編碼演算法。在第一次迭代中，最頻繁的連續符號對是`'t'`和`'a'`，因此位元組對編碼將它們合併以產生新符號`'ta'`。在第二次迭代中，位元組對編碼繼續合併`'ta'`和`'l'`以產生另一個新符號`'tal'`。

```{.python .input}
#@tab all
num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f'合併# {i+1}:',max_freq_pair)
```

在位元組對編碼的10次迭代之後，我們可以看到清單`symbols`現在又包含10個從其他符號迭代合併而來的符號。

```{.python .input}
#@tab all
print(symbols)
```

對於在詞典`raw_token_freqs`的鍵中指定的同一資料集，作為位元組對編碼演算法的結果，資料集中的每個詞現在被子詞『fast_』『fast』『er_』『tall_』和『tall』分割。例如，單詞『fast er_』和『tall er_』分別被分割為『fast er_』和『tall er_』。

```{.python .input}
#@tab all
print(list(token_freqs.keys()))
```

請注意，位元組對編碼的結果取決於正在使用的資料集。我們還可以使用從一個資料集學習的子詞來切分另一個資料集的單詞。作為一種貪心方法，下面的`segment_BPE`函式嘗試將單詞從輸入引數`symbols`分成可能最長的子詞。

```{.python .input}
#@tab all
def segment_BPE(tokens, symbols):
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # 具有符號中可能最長子字的詞元段
        while start < len(token) and start < end:
            if token[start: end] in symbols:
                cur_output.append(token[start: end])
                start = end
                end = len(token)
            else:
                end -= 1
        if start < len(token):
            cur_output.append('[UNK]')
        outputs.append(' '.join(cur_output))
    return outputs
```

我們使用清單`symbols`中的子詞（從前面提到的資料集學習）來表示另一個資料集的`tokens`。

```{.python .input}
#@tab all
tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))
```

## 小結

* fastText模型提出了一種子詞嵌入方法：基於word2vec中的跳元模型，它將中心詞表示為其子詞向量之和。
* 位元組對編碼執行訓練資料集的統計分析，以發現詞內的公共符號。作為一種貪心方法，位元組對編碼迭代地合併最頻繁的連續符號對。
* 子詞嵌入可以提高稀有詞和詞典外詞的表示質量。

## 練習

1. 例如，英語中大約有$3\times 10^8$種可能的$6$-元組。子詞太多會有什麼問題呢？如何解決這個問題？提示:請參閱fastText論文第3.2節末尾 :cite:`Bojanowski.Grave.Joulin.ea.2017`。
1. 如何在連續詞袋模型的基礎上設計一個子詞嵌入模型？
1. 要獲得大小為$m$的詞表，當初始符號詞表大小為$n$時，需要多少合併操作？
1. 如何擴充套件位元組對編碼的思想來提取短語？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/5747)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/5748)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11818)
:end_tab:
