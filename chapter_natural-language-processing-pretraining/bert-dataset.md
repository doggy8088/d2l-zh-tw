# 用於預訓練BERT的資料集
:label:`sec_bert-dataset`

為了預訓練 :numref:`sec_bert`中實現的BERT模型，我們需要以理想的格式生成資料集，以便於兩個預訓練任務：遮蔽語言模型和下一句預測。一方面，最初的BERT模型是在兩個龐大的圖書語料庫和英語維基百科（參見 :numref:`subsec_bert_pretraining_tasks`）的合集上預訓練的，但它很難吸引這本書的大多數讀者。另一方面，現成的預訓練BERT模型可能不適合醫學等特定領域的應用。因此，在定製的資料集上對BERT進行預訓練變得越來越流行。為了方便BERT預訓練的示範，我們使用了較小的語料庫WikiText-2 :cite:`Merity.Xiong.Bradbury.ea.2016`。

與 :numref:`sec_word2vec_data`中用於預訓練word2vec的PTB資料集相比，WikiText-2（1）保留了原來的標點符號，適合於下一句預測；（2）保留了原來的大小寫和數字；（3）大了一倍以上。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import random

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import os
import random
import torch
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import os
import random
import paddle
```

在WikiText-2資料集中，每行代表一個段落，其中在任意標點符號及其前面的詞元之間插入空格。保留至少有兩句話的段落。為了簡單起見，我們僅使用句號作為分隔符來拆分句子。我們將更復雜的句子拆分技術的討論留在本節末尾的練習中。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')

#@save
def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # 大寫字母轉換為小寫字母
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs
```

## 為預訓練任務定義輔助函式

在下文中，我們首先為BERT的兩個預訓練任務實現輔助函式。這些輔助函式將在稍後將原始文字語料庫轉換為理想格式的資料集時調用，以預訓練BERT。

### 生成下一句預測任務的資料

根據 :numref:`subsec_nsp`的描述，`_get_next_sentence`函式生成二分類任務的訓練樣本。

```{.python .input}
#@tab all
#@save
def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # paragraphs是三重清單的巢狀
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next
```

下面的函式透過呼叫`_get_next_sentence`函式從輸入`paragraph`生成用於下一句預測的訓練樣本。這裡`paragraph`是句子清單，其中每個句子都是詞元清單。自變量`max_len`指定預訓練期間的BERT輸入序列的最大長度。

```{.python .input}
#@tab all
#@save
def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # 考慮1個'<cls>'詞元和2個'<sep>'詞元
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph
```

### 生成遮蔽語言模型任務的資料
:label:`subsec_prepare_mlm_data`

為了從BERT輸入序列生成遮蔽語言模型的訓練樣本，我們定義了以下`_replace_mlm_tokens`函式。在其輸入中，`tokens`是表示BERT輸入序列的詞元的清單，`candidate_pred_positions`是不包括特殊詞元的BERT輸入序列的詞元索引的清單（特殊詞元在遮蔽語言模型任務中不被預測），以及`num_mlm_preds`指示預測的數量（選擇15%要預測的隨機詞元）。在 :numref:`subsec_mlm`中定義遮蔽語言模型任務之後，在每個預測位置，輸入可以由特殊的『掩碼』詞元或隨機詞元替換，或者保持不變。最後，該函式回傳可能替換後的輸入詞元、發生預測的詞元索引和這些預測的標籤。

```{.python .input}
#@tab all
#@save
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    # 為遮蔽語言模型的輸入建立新的詞元副本，其中輸入可能包含替換的『<mask>』或隨機詞元
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # 打亂後用於在遮蔽語言模型任務中獲取15%的隨機詞元進行預測
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80%的時間：將詞替換為『<mask>』詞元
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10%的時間：保持詞不變
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10%的時間：用隨機詞替換該詞
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels
```

透過呼叫前述的`_replace_mlm_tokens`函式，以下函式將BERT輸入序列（`tokens`）作為輸入，並回傳輸入詞元的索引（在 :numref:`subsec_mlm`中描述的可能的詞元替換之後）、發生預測的詞元索引以及這些預測的標籤索引。

```{.python .input}
#@tab all
#@save
def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # tokens是一個字串清單
    for i, token in enumerate(tokens):
        # 在遮蔽語言模型任務中不會預測特殊詞元
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 遮蔽語言模型任務中預測15%的隨機詞元
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]
```

## 將文字轉換為預訓練資料集

現在我們幾乎準備好為BERT預訓練定製一個`Dataset`類。在此之前，我們仍然需要定義輔助函式`_pad_bert_inputs`來將特殊的『&lt;mask&gt;』詞元附加到輸入。它的引數`examples`包含來自兩個預訓練任務的輔助函式`_get_nsp_data_from_paragraph`和`_get_mlm_data_from_tokens`的輸出。

```{.python .input}
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(np.array(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype='int32'))
        all_segments.append(np.array(segments + [0] * (
            max_len - len(segments)), dtype='int32'))
        # valid_lens不包括'<pad>'的計數
        valid_lens.append(np.array(len(token_ids), dtype='float32'))
        all_pred_positions.append(np.array(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype='int32'))
        # 填充詞元的預測將透過乘以0權重在損失中過濾掉
        all_mlm_weights.append(
            np.array([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)), dtype='float32'))
        all_mlm_labels.append(np.array(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype='int32'))
        nsp_labels.append(np.array(is_next))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

```{.python .input}
#@tab pytorch
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        # valid_lens不包括'<pad>'的計數
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # 填充詞元的預測將透過乘以0權重在損失中過濾掉
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

```{.python .input}
#@tab paddle
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(paddle.to_tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=paddle.int64))
        all_segments.append(paddle.to_tensor(segments + [0] * (
            max_len - len(segments)), dtype=paddle.int64))
        # valid_lens不包括'<pad>'的計數
        valid_lens.append(paddle.to_tensor(len(token_ids), dtype=paddle.float32))
        all_pred_positions.append(paddle.to_tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=paddle.int64))
        # 填充詞元的預測將透過乘以0權重在損失中過濾掉
        all_mlm_weights.append(
            paddle.to_tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=paddle.float32))
        all_mlm_labels.append(paddle.to_tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=paddle.int64))
        nsp_labels.append(paddle.to_tensor(is_next, dtype=paddle.int64))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

將用於生成兩個預訓練任務的訓練樣本的輔助函式和用於填充輸入的輔助函式放在一起，我們定義以下`_WikiTextDataset`類為用於預訓練BERT的WikiText-2資料集。透過實現`__getitem__ `函式，我們可以任意訪問WikiText-2語料庫的一對句子生成的預訓練樣本（遮蔽語言模型和下一句預測）樣本。

最初的BERT模型使用詞表大小為30000的WordPiece嵌入 :cite:`Wu.Schuster.Chen.ea.2016`。WordPiece的詞元化方法是對 :numref:`subsec_Byte_Pair_Encoding`中原有的位元組對編碼演算法稍作修改。為簡單起見，我們使用`d2l.tokenize`函式進行詞元化。出現次數少於5次的不頻繁詞元將被過濾掉。

```{.python .input}
#@save
class _WikiTextDataset(gluon.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # 輸入paragraphs[i]是代表段落的句子字串清單；
        # 而輸出paragraphs[i]是代表段落的句子清單，其中每個句子都是詞元清單
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # 獲取下一句子預測任務的資料
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # 獲取遮蔽語言模型任務的資料
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # 填充輸入
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input}
#@tab pytorch
#@save
class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # 輸入paragraphs[i]是代表段落的句子字串清單；
        # 而輸出paragraphs[i]是代表段落的句子清單，其中每個句子都是詞元清單
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # 獲取下一句子預測任務的資料
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # 獲取遮蔽語言模型任務的資料
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # 填充輸入
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input}
#@tab paddle
#@save
class _WikiTextDataset(paddle.io.Dataset):
    def __init__(self, paragraphs, max_len):
        # 輸入paragraphs[i]是代表段落的句子字串清單；
        # 而輸出paragraphs[i]是代表段落的句子清單，其中每個句子都是詞元清單
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # 獲取下一句子預測任務的資料
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # 獲取遮蔽語言模型任務的資料
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # 填充輸入
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

透過使用`_read_wiki`函式和`_WikiTextDataset`類，我們定義了下面的`load_data_wiki`來下載並生成WikiText-2資料集，並從中生成預訓練樣本。

```{.python .input}
#@save
def load_data_wiki(batch_size, max_len):
    """載入WikiText-2資料集"""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    return train_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_wiki(batch_size, max_len):
    """載入WikiText-2資料集"""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab
```

```{.python .input}
#@tab paddle
#@save
def load_data_wiki(batch_size, max_len):
    """載入WikiText-2資料集"""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = paddle.io.DataLoader(dataset=train_set, batch_size=batch_size, return_list=True,
                                        shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab
```

將批次大小設定為512，將BERT輸入序列的最大長度設定為64，我們列印出小批次的BERT預訓練樣本的形狀。注意，在每個BERT輸入序列中，為遮蔽語言模型任務預測$10$（$64 \times 0.15$）個位置。

```{.python .input}
#@tab all
batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)

for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
     mlm_Y, nsp_y) in train_iter:
    print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
          pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
          nsp_y.shape)
    break
```

最後，我們來看一下詞量。即使在過濾掉不頻繁的詞元之後，它仍然比PTB資料集的大兩倍以上。

```{.python .input}
#@tab all
len(vocab)
```

## 小結

* 與PTB資料集相比，WikiText-2資料集保留了原來的標點符號、大小寫和數字，並且比PTB資料集大了兩倍多。
* 我們可以任意訪問從WikiText-2語料庫中的一對句子生成的預訓練（遮蔽語言模型和下一句預測）樣本。

## 練習

1. 為簡單起見，句號用作拆分句子的唯一分隔符。嘗試其他的句子拆分技術，比如Spacy和NLTK。以NLTK為例，需要先安裝NLTK：`pip install nltk`。在程式碼中先`import nltk`。然後下載Punkt語句詞元分析器：`nltk.download('punkt')`。要拆分句子，比如`sentences = 'This is great ! Why not ?'`，呼叫`nltk.tokenize.sent_tokenize(sentences)`將回傳兩個句子字串的清單：`['This is great !', 'Why not ?']`。
1. 如果我們不過濾出一些不常見的詞元，詞量會有多大？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/5737)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/5738)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11822)
:end_tab:
