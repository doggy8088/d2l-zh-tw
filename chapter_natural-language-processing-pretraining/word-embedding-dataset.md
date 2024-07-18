# 用於預訓練詞嵌入的資料集
:label:`sec_word2vec_data`

現在我們已經瞭解了word2vec模型的技術細節和大致的訓練方法，讓我們來看看它們的實現。具體地說，我們將以 :numref:`sec_word2vec`的跳元模型和 :numref:`sec_approx_train`的負取樣為例。本節從用於預訓練詞嵌入模型的資料集開始：資料的原始格式將被轉換為可以在訓練期間迭代的小批次。

```{.python .input}
from d2l import mxnet as d2l
import math
from mxnet import gluon, np
import os
import random
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import torch
import os
import random
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import math
import paddle
import os
import random
```

## 讀取資料集

我們在這裡使用的資料集是[Penn Tree Bank（PTB）](https://catalog.ldc.upenn.edu/LDC99T42)。該語料庫取自『華爾街日報』的文章，分為訓練集、驗證集和測試集。在原始格式中，文字檔案的每一行表示由空格分隔的一句話。在這裡，我們將每個單詞視為一個詞元。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')

#@save
def read_ptb():
    """將PTB資料集載入到文字行的清單中"""
    data_dir = d2l.download_extract('ptb')
    # Readthetrainingset.
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

sentences = read_ptb()
f'# sentences數: {len(sentences)}'
```

在讀取訓練集之後，我們為語料庫建立了一個詞表，其中出現次數少於10次的任何單詞都將由『&lt;unk&gt;』詞元替換。請注意，原始資料集還包含表示稀有（未知）單詞的『&lt;unk&gt;』詞元。

```{.python .input}
#@tab all
vocab = d2l.Vocab(sentences, min_freq=10)
f'vocab size: {len(vocab)}'
```

## 下采樣

文字資料通常有『the』『a』和『in』等高頻詞：它們在非常大的語料庫中甚至可能出現數十億次。然而，這些詞經常在上下文視窗中與許多不同的詞共同出現，提供的有用訊息很少。例如，考慮上下文視窗中的詞『chip』：直觀地說，它與低頻單詞『intel』的共現比與高頻單詞『a』的共現在訓練中更有用。此外，大量（高頻）單詞的訓練速度很慢。因此，當訓練詞嵌入模型時，可以對高頻單詞進行*下采樣* :cite:`Mikolov.Sutskever.Chen.ea.2013`。具體地說，資料集中的每個詞$w_i$將有機率地被丟棄

$$ P(w_i) = \max\left(1 - \sqrt{\frac{t}{f(w_i)}}, 0\right),$$

其中$f(w_i)$是$w_i$的詞數與資料集中的總詞數的比率，常數$t$是超引數（在實驗中為$10^{-4}$）。我們可以看到，只有當相對比率$f(w_i) > t$時，（高頻）詞$w_i$才能被丟棄，且該詞的相對比率越高，被丟棄的機率就越大。

```{.python .input}
#@tab all
#@save
def subsample(sentences, vocab):
    """下采樣高頻詞"""
    # 排除未知詞元'<unk>'
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = d2l.count_corpus(sentences)
    num_tokens = sum(counter.values())

    # 如果在下采樣期間保留詞元，則回傳True
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

subsampled, counter = subsample(sentences, vocab)
```

下面的程式碼片段繪製了下采樣前後每句話的詞元數量的直方圖。正如預期的那樣，下采樣透過刪除高頻詞來顯著縮短句子，這將使訓練加速。

```{.python .input}
#@tab all
d2l.show_list_len_pair_hist(
    ['origin', 'subsampled'], '# tokens per sentence',
    'count', sentences, subsampled);
```

對於單個詞元，高頻詞『the』的採樣率不到1/20。

```{.python .input}
#@tab all
def compare_counts(token):
    return (f'"{token}"的數量：'
            f'之前={sum([l.count(token) for l in sentences])}, '
            f'之後={sum([l.count(token) for l in subsampled])}')

compare_counts('the')
```

相比之下，低頻詞『join』則被完全保留。

```{.python .input}
#@tab all
compare_counts('join')
```

在下采樣之後，我們將詞元對映到它們在語料庫中的索引。

```{.python .input}
#@tab all
corpus = [vocab[line] for line in subsampled]
corpus[:3]
```

## 中心詞和上下文詞的提取

下面的`get_centers_and_contexts`函式從`corpus`中提取所有中心詞及其上下文詞。它隨機取樣1到`max_window_size`之間的整數作為上下文視窗。對於任一中心詞，與其距離不超過取樣上下文視窗大小的詞為其上下文詞。

```{.python .input}
#@tab all
#@save
def get_centers_and_contexts(corpus, max_window_size):
    """回傳跳元模型中的中心詞和上下文詞"""
    centers, contexts = [], []
    for line in corpus:
        # 要形成『中心詞-上下文詞』對，每個句子至少需要有2個詞
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # 上下文視窗中間i
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # 從上下文詞中排除中心詞
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts
```

接下來，我們建立一個人工資料集，分別包含7個和3個單詞的兩個句子。設定最大上下文視窗大小為2，並列印所有中心詞及其上下文詞。

```{.python .input}
#@tab all
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('資料集', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('中心詞', center, '的上下文詞是', context)
```

在PTB資料集上進行訓練時，我們將最大上下文視窗大小設定為5。下面提取資料集中的所有中心詞及其上下文詞。

```{.python .input}
#@tab all
all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
f'# 『中心詞-上下文詞對』的數量: {sum([len(contexts) for contexts in all_contexts])}'
```

## 負取樣

我們使用負取樣進行近似訓練。為了根據預定義的分佈對噪聲詞進行取樣，我們定義以下`RandomGenerator`類，其中（可能未規範化的）取樣分佈透過變數`sampling_weights`傳遞。

```{.python .input}
#@tab all
#@save
class RandomGenerator:
    """根據n個取樣權重在{1,...,n}中隨機抽取"""
    def __init__(self, sampling_weights):
        # Exclude
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # 快取k個隨機取樣結果
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]
```

例如，我們可以在索引1、2和3中繪製10個隨機變數$X$，取樣機率為$P(X=1)=2/9, P(X=2)=3/9$和$P(X=3)=4/9$，如下所示。

```{.python .input}
#@tab all
#@save
generator = RandomGenerator([2, 3, 4])
[generator.draw() for _ in range(10)]
```

對於一對中心詞和上下文詞，我們隨機抽取了`K`個（實驗中為5個）噪聲詞。根據word2vec論文中的建議，將噪聲詞$w$的取樣機率$P(w)$設定為其在字典中的相對頻率，其冪為0.75 :cite:`Mikolov.Sutskever.Chen.ea.2013`。

```{.python .input}
#@tab all
#@save
def get_negatives(all_contexts, vocab, counter, K):
    """回傳負取樣中的噪聲詞"""
    # 索引為1、2、...（索引0是詞表中排除的未知標記）
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # 噪聲詞不能是上下文詞
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

all_negatives = get_negatives(all_contexts, vocab, counter, 5)
```

## 小批次載入訓練實例
:label:`subsec_word2vec-minibatch-loading`

在提取所有中心詞及其上下文詞和取樣噪聲詞後，將它們轉換成小批次的樣本，在訓練過程中可以迭代載入。

在小批次中，$i^\mathrm{th}$個樣本包括中心詞及其$n_i$個上下文詞和$m_i$個噪聲詞。由於上下文視窗大小不同，$n_i+m_i$對於不同的$i$是不同的。因此，對於每個樣本，我們在`contexts_negatives`個變數中將其上下文詞和噪聲詞連結起來，並填充零，直到連結長度達到$\max_i n_i+m_i$(`max_len`)。為了在計算損失時排除填充，我們定義了掩碼變數`masks`。在`masks`中的元素和`contexts_negatives`中的元素之間存在一一對應關係，其中`masks`中的0（否則為1）對應於`contexts_negatives`中的填充。

為了區分正反例，我們在`contexts_negatives`中透過一個`labels`變數將上下文詞與噪聲詞分開。類似於`masks`，在`labels`中的元素和`contexts_negatives`中的元素之間也存在一一對應關係，其中`labels`中的1（否則為0）對應於`contexts_negatives`中的上下文詞的正例。

上述思想在下面的`batchify`函式中實現。其輸入`data`是長度等於批次大小的清單，其中每個元素是由中心詞`center`、其上下文詞`context`和其噪聲詞`negative`組成的樣本。此函式回傳一個可以在訓練期間載入用於計算的小批次，例如包括掩碼變數。

```{.python .input}
#@tab all
#@save
def batchify(data):
    """回傳帶有負取樣的跳元模型的小批次樣本"""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += \
            [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (d2l.reshape(d2l.tensor(centers), (-1, 1)), d2l.tensor(
        contexts_negatives), d2l.tensor(masks), d2l.tensor(labels))
```

讓我們使用一個小批次的兩個樣本來測試此函式。

```{.python .input}
#@tab all
x_1 = (1, [2, 2], [3, 3, 3, 3])
x_2 = (1, [2, 2, 2], [3, 3])
batch = batchify((x_1, x_2))

names = ['centers', 'contexts_negatives', 'masks', 'labels']
for name, data in zip(names, batch):
    print(name, '=', data)
```

## 整合程式碼

最後，我們定義了讀取PTB資料集並回傳資料迭代器和詞表的`load_data_ptb`函式。

```{.python .input}
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """下載PTB資料集，然後將其載入到記憶體中"""
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)
    dataset = gluon.data.ArrayDataset(
        all_centers, all_contexts, all_negatives)
    data_iter = gluon.data.DataLoader(
        dataset, batch_size, shuffle=True,batchify_fn=batchify,
        num_workers=d2l.get_dataloader_workers())
    return data_iter, vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """下載PTB資料集，然後將其載入到記憶體中"""
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, 
        collate_fn=batchify, num_workers=num_workers)
    return data_iter, vocab
```

```{.python .input}
#@tab paddle
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """下載PTB資料集，然後將其載入到記憶體中"""
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(paddle.io.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = paddle.io.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, return_list=True, 
        collate_fn=batchify, num_workers=num_workers)
    return data_iter, vocab
```

讓我們列印資料迭代器的第一個小批次。

```{.python .input}
#@tab all
data_iter, vocab = load_data_ptb(512, 5, 5)
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, 'shape:', data.shape)
    break
```

## 小結

* 高頻詞在訓練中可能不是那麼有用。我們可以對他們進行下采樣，以便在訓練中加快速度。
* 為了提高計算效率，我們以小批次方式載入樣本。我們可以定義其他變數來區分填充標記和非填充標記，以及正例和負例。

## 練習

1. 如果不使用下采樣，本節中程式碼的執行時間會發生什麼變化？
1. `RandomGenerator`類快取`k`個隨機取樣結果。將`k`設定為其他值，看看它如何影響資料載入速度。
1. 本節程式碼中的哪些其他超引數可能會影響資料載入速度？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/5734)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/5735)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11816)
:end_tab:
