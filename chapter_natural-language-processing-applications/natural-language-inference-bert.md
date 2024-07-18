# 自然語言推斷：微調BERT
:label:`sec_natural-language-inference-bert`

在本章的前面幾節中，我們已經為SNLI資料集（ :numref:`sec_natural-language-inference-and-dataset`）上的自然語言推斷任務設計了一個基於注意力的結構（ :numref:`sec_natural-language-inference-attention`）。現在，我們透過微調BERT來重新審視這項任務。正如在 :numref:`sec_finetuning-bert`中討論的那樣，自然語言推斷是一個序列級別的文字對分類問題，而微調BERT只需要一個額外的基於多層感知機的架構，如 :numref:`fig_nlp-map-nli-bert`中所示。

![將預訓練BERT提供給基於多層感知機的自然語言推斷架構](../img/nlp-map-nli-bert.svg)
:label:`fig_nlp-map-nli-bert`

本節將下載一個預訓練好的小版本的BERT，然後對其進行微調，以便在SNLI資料集上進行自然語言推斷。

```{.python .input}
from d2l import mxnet as d2l
import json
import multiprocessing
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import json
import multiprocessing
import torch
from torch import nn
import os
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import json
import multiprocessing
import paddle
from paddle import nn
import os
```

## [**載入預訓練的BERT**]

我們已經在 :numref:`sec_bert-dataset`和 :numref:`sec_bert-pretraining`WikiText-2資料集上預訓練BERT（請注意，原始的BERT模型是在更大的語料庫上預訓練的）。正如在 :numref:`sec_bert-pretraining`中所討論的，原始的BERT模型有數以億計的引數。在下面，我們提供了兩個版本的預訓練的BERT：『bert.base』與原始的BERT基礎模型一樣大，需要大量的計算資源才能進行微調，而『bert.small』是一個小版本，以便於示範。

```{.python .input}
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.zip',
                             '7b3820b35da691042e5d34c0971ac3edbd80d3f4')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.zip',
                              'a4e718a47137ccd1809c9107ab4f5edd317bae2c')
```

```{.python .input}
#@tab pytorch
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.torch.zip',
                             '225d66f04cae318b841a13d32af3acc165f253ac')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.torch.zip',
                              'c72329e68a732bef0452e4b96a1c341c8910f81f')
```

```{.python .input}
#@tab paddle
d2l.DATA_HUB['bert_small'] = ('https://paddlenlp.bj.bcebos.com/models/bert.small.paddle.zip', '9fcde07509c7e87ec61c640c1b277509c7e87ec6153d9041758e4')

d2l.DATA_HUB['bert_base'] = ('https://paddlenlp.bj.bcebos.com/models/bert.base.paddle.zip', '9fcde07509c7e87ec61c640c1b27509c7e87ec61753d9041758e4')
```

兩個預訓練好的BERT模型都包含一個定義詞表的『vocab.json』檔案和一個預訓練引數的『pretrained.params』檔案。我們實現了以下`load_pretrained_model`函式來[**載入預先訓練好的BERT引數**]。

```{.python .input}
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # 定義空詞表以載入預定義詞表
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir,
         'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens, ffn_num_hiddens, 
                         num_heads, num_layers, dropout, max_len)
    # 載入預訓練BERT引數
    bert.load_parameters(os.path.join(data_dir, 'pretrained.params'),
                         ctx=devices)
    return bert, vocab
```

```{.python .input}
#@tab pytorch
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # 定義空詞表以載入預定義詞表
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 
        'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens, norm_shape=[256],
                         ffn_num_input=256, ffn_num_hiddens=ffn_num_hiddens,
                         num_heads=4, num_layers=2, dropout=0.2,
                         max_len=max_len, key_size=256, query_size=256,
                         value_size=256, hid_in_features=256,
                         mlm_in_features=256, nsp_in_features=256)
    # 載入預訓練BERT引數
    bert.load_state_dict(torch.load(os.path.join(data_dir,
                                                 'pretrained.params')))
    return bert, vocab
```

```{.python .input}
#@tab paddle
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # 定義空詞表以載入預定義詞表
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir,
        'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens, norm_shape=[256],
                         ffn_num_input=256, ffn_num_hiddens=ffn_num_hiddens,
                         num_heads=4, num_layers=2, dropout=0.2,
                         max_len=max_len, key_size=256, query_size=256,
                         value_size=256, hid_in_features=256,
                         mlm_in_features=256, nsp_in_features=256)
    # 載入預訓練BERT引數
    bert.set_state_dict(paddle.load(os.path.join(data_dir,
                                                 'pretrained.pdparams')))

    return bert, vocab
```

為了便於在大多數機器上示範，我們將在本節中載入和微調經過預訓練BERT的小版本（『bert.small』）。在練習中，我們將展示如何微調大得多的『bert.base』以顯著提高測試精度。

```{.python .input}
#@tab mxnet, pytorch
devices = d2l.try_all_gpus()
bert, vocab = load_pretrained_model(
    'bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
    num_layers=2, dropout=0.1, max_len=512, devices=devices)
```

```{.python .input}
#@tab paddle
devices = d2l.try_all_gpus()
bert, vocab = load_pretrained_model(
    'bert_small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
    num_layers=2, dropout=0.1, max_len=512, devices=devices)
```

## [**微調BERT的資料集**]

對於SNLI資料集的下游任務自然語言推斷，我們定義了一個定製的資料集類`SNLIBERTDataset`。在每個樣本中，前提和假設形成一對文字序列，並被打包成一個BERT輸入序列，如 :numref:`fig_bert-two-seqs`所示。回想 :numref:`subsec_bert_input_rep`，片段索引用於區分BERT輸入序列中的前提和假設。利用預定義的BERT輸入序列的最大長度（`max_len`），持續移除輸入文字對中較長文字的最後一個標記，直到滿足`max_len`。為了加速生成用於微調BERT的SNLI資料集，我們使用4個工作程序並行生成訓練或測試樣本。

```{.python .input}
class SNLIBERTDataset(gluon.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[d2l.tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]
        
        self.labels = np.array(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # 使用4個程序
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (np.array(all_token_ids, dtype='int32'),
                np.array(all_segments, dtype='int32'), 
                np.array(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                             * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # 為BERT輸入中的'<CLS>'、'<SEP>'和'<SEP>'詞元保留位置
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input}
#@tab pytorch
class SNLIBERTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[d2l.tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]
        
        self.labels = torch.tensor(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # 使用4個程序
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long), 
                torch.tensor(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                             * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # 為BERT輸入中的'<CLS>'、'<SEP>'和'<SEP>'詞元保留位置
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input}
#@tab paddle
class SNLIBERTDataset(paddle.io.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[d2l.tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]

        self.labels = paddle.to_tensor(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        # pool = multiprocessing.Pool(1)  # 使用4個程序
        # out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        out = []
        for i in all_premise_hypothesis_tokens:
            tempOut = self._mp_worker(i)
            out.append(tempOut)
        
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (paddle.to_tensor(all_token_ids, dtype='int64'),
                paddle.to_tensor(all_segments, dtype='int64'),
                paddle.to_tensor(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                             * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # 為BERT輸入中的'<CLS>'、'<SEP>'和'<SEP>'詞元保留位置
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)
```

下載完SNLI資料集後，我們透過實例化`SNLIBERTDataset`類來[**生成訓練和測試樣本**]。這些樣本將在自然語言推斷的訓練和測試期間進行小批次讀取。

```{.python .input}
# 如果出現顯存不足錯誤，請減少『batch_size』。在原始的BERT模型中，max_len=512
batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
data_dir = d2l.download_extract('SNLI')
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                   num_workers=num_workers)
test_iter = gluon.data.DataLoader(test_set, batch_size,
                                  num_workers=num_workers)
```

```{.python .input}
#@tab pytorch
# 如果出現顯存不足錯誤，請減少『batch_size』。在原始的BERT模型中，max_len=512
batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
data_dir = d2l.download_extract('SNLI')
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                   num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                  num_workers=num_workers)
```

```{.python .input}
#@tab paddle
# 如果出現顯存不足錯誤，請減少『batch_size』。在原始的BERT模型中，max_len=512
batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
data_dir = d2l.download_extract('SNLI')
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
train_iter = paddle.io.DataLoader(train_set, batch_size=batch_size, shuffle=True, return_list=True)
test_iter = paddle.io.DataLoader(test_set, batch_size=batch_size, return_list=True)
```

## 微調BERT

如 :numref:`fig_bert-two-seqs`所示，用於自然語言推斷的微調BERT只需要一個額外的多層感知機，該多層感知機由兩個全連線層組成（請參見下面`BERTClassifier`類別中的`self.hidden`和`self.output`）。[**這個多層感知機將特殊的『&lt;cls&gt;』詞元**]的BERT表示進行了轉換，該詞元同時編碼前提和假設的訊息(**為自然語言推斷的三個輸出**)：蘊涵、矛盾和中性。

```{.python .input}
class BERTClassifier(nn.Block):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Dense(3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))
```

```{.python .input}
#@tab pytorch
class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Linear(256, 3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))
```

```{.python .input}
#@tab paddle
class BERTClassifier(nn.Layer):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Linear(256, 3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x.squeeze(1))
        return self.output(self.hidden(encoded_X[:, 0, :]))
```

在下文中，預訓練的BERT模型`bert`被送到用於下游應用的`BERTClassifier`實例`net`中。在BERT微調的常見實現中，只有額外的多層感知機（`net.output`）的輸出層的引數將從零開始學習。預訓練BERT編碼器（`net.encoder`）和額外的多層感知機的隱藏層（`net.hidden`）的所有引數都將進行微調。

```{.python .input}
net = BERTClassifier(bert)
net.output.initialize(ctx=devices)
```

```{.python .input}
#@tab pytorch, paddle
net = BERTClassifier(bert)
```

回想一下，在 :numref:`sec_bert`中，`MaskLM`類別和`NextSentencePred`類在其使用的多層感知機中都有一些引數。這些引數是預訓練BERT模型`bert`中引數的一部分，因此是`net`中的引數的一部分。然而，這些引數僅用於計算預訓練過程中的遮蔽語言模型損失和下一句預測損失。這兩個損失函式與微調下游應用無關，因此當BERT微調時，`MaskLM`和`NextSentencePred`中採用的多層感知機的引數不會更新（陳舊的，staled）。

為了允許具有陳舊梯度的引數，標誌`ignore_stale_grad=True`在`step`函式`d2l.train_batch_ch13`中被設定。我們透過該函式使用SNLI的訓練集（`train_iter`）和測試集（`test_iter`）對`net`模型進行訓練和評估。由於計算資源有限，[**訓練**]和測試精度可以進一步提高：我們把對它的討論留在練習中。

```{.python .input}
lr, num_epochs = 1e-4, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, 
    devices, d2l.split_batch_multi_inputs)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 1e-4, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction='none')
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, 
    devices)
```

```{.python .input}
#@tab paddle
lr, num_epochs = 1e-4, 5
trainer = paddle.optimizer.Adam(learning_rate=lr, parameters=net.parameters())
loss = nn.CrossEntropyLoss(reduction='none')
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
    devices)
```

## 小結

* 我們可以針對下游應用對預訓練的BERT模型進行微調，例如在SNLI資料集上進行自然語言推斷。
* 在微調過程中，BERT模型成為下游應用模型的一部分。僅與訓練前損失相關的引數在微調期間不會更新。

## 練習

1. 如果您的計算資源允許，請微調一個更大的預訓練BERT模型，該模型與原始的BERT基礎模型一樣大。修改`load_pretrained_model`函式中的引數設定：將『bert.small』替換為『bert.base』，將`num_hiddens=256`、`ffn_num_hiddens=512`、`num_heads=4`和`num_layers=2`的值分別增加到768、3072、12和12。透過增加微調迭代輪數（可能還會調優其他超引數），你可以獲得高於0.86的測試精度嗎？
1. 如何根據一對序列的長度比值截斷它們？將此對截斷方法與`SNLIBERTDataset`類別中使用的方法進行比較。它們的利弊是什麼？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/5715)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/5718)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11831)
:end_tab:
