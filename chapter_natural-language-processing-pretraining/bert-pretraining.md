# 預訓練BERT
:label:`sec_bert-pretraining`

利用 :numref:`sec_bert`中實現的BERT模型和 :numref:`sec_bert-dataset`中從WikiText-2資料集生成的預訓練樣本，我們將在本節中在WikiText-2資料集上對BERT進行預訓練。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx

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

首先，我們載入WikiText-2資料集作為小批次的預訓練樣本，用於遮蔽語言模型和下一句預測。批次大小是512，BERT輸入序列的最大長度是64。注意，在原始BERT模型中，最大長度是512。

```{.python .input}
#@tab mxnet, pytorch
batch_size, max_len = 512, 64
train_iter, vocab = d2l.load_data_wiki(batch_size, max_len)
```

```{.python .input}
#@tab paddle
def load_data_wiki(batch_size, max_len):
    """載入WikiText-2資料集

    Defined in :numref:`subsec_prepare_mlm_data`"""
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = d2l._read_wiki(data_dir)
    train_set = d2l._WikiTextDataset(paragraphs, max_len)
    train_iter = paddle.io.DataLoader(dataset=train_set, batch_size=batch_size, return_list=True,
                                        shuffle=True, num_workers=0)
    return train_iter, train_set.vocab

batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)
```

## 預訓練BERT

原始BERT :cite:`Devlin.Chang.Lee.ea.2018`有兩個不同模型尺寸的版本。基本模型（$\text{BERT}_{\text{BASE}}$）使用12層（Transformer編碼器區塊），768個隱藏單元（隱藏大小）和12個自注意頭。大模型（$\text{BERT}_{\text{LARGE}}$）使用24層，1024個隱藏單元和16個自注意頭。值得注意的是，前者有1.1億個引數，後者有3.4億個引數。為了便於示範，我們定義了一個小的BERT，使用了2層、128個隱藏單元和2個自注意頭。

```{.python .input}
net = d2l.BERTModel(len(vocab), num_hiddens=128, ffn_num_hiddens=256,
                    num_heads=2, num_layers=2, dropout=0.2)
devices = d2l.try_all_gpus()
net.initialize(init.Xavier(), ctx=devices)
loss = gluon.loss.SoftmaxCELoss()
```

```{.python .input}
#@tab pytorch, paddle
net = d2l.BERTModel(len(vocab), num_hiddens=128, norm_shape=[128],
                    ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
                    num_layers=2, dropout=0.2, key_size=128, query_size=128,
                    value_size=128, hid_in_features=128, mlm_in_features=128,
                    nsp_in_features=128)
devices = d2l.try_all_gpus()
loss = nn.CrossEntropyLoss()
```

在定義訓練程式碼實現之前，我們定義了一個輔助函式`_get_batch_loss_bert`。給定訓練樣本，該函式計算遮蔽語言模型和下一句子預測任務的損失。請注意，BERT預訓練的最終損失是遮蔽語言模型損失和下一句預測損失的和。

```{.python .input}
#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X_shards,
                         segments_X_shards, valid_lens_x_shards,
                         pred_positions_X_shards, mlm_weights_X_shards,
                         mlm_Y_shards, nsp_y_shards):
    mlm_ls, nsp_ls, ls = [], [], []
    for (tokens_X_shard, segments_X_shard, valid_lens_x_shard,
         pred_positions_X_shard, mlm_weights_X_shard, mlm_Y_shard,
         nsp_y_shard) in zip(
        tokens_X_shards, segments_X_shards, valid_lens_x_shards,
        pred_positions_X_shards, mlm_weights_X_shards, mlm_Y_shards,
        nsp_y_shards):
        # 前向傳播
        _, mlm_Y_hat, nsp_Y_hat = net(
            tokens_X_shard, segments_X_shard, valid_lens_x_shard.reshape(-1),
            pred_positions_X_shard)
        # 計算遮蔽語言模型損失
        mlm_l = loss(
            mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y_shard.reshape(-1),
            mlm_weights_X_shard.reshape((-1, 1)))
        mlm_l = mlm_l.sum() / (mlm_weights_X_shard.sum() + 1e-8)
        # 計算下一句子預測任務的損失
        nsp_l = loss(nsp_Y_hat, nsp_y_shard)
        nsp_l = nsp_l.mean()
        mlm_ls.append(mlm_l)
        nsp_ls.append(nsp_l)
        ls.append(mlm_l + nsp_l)
        npx.waitall()
    return mlm_ls, nsp_ls, ls
```

```{.python .input}
#@tab pytorch
#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # 前向傳播
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # 計算遮蔽語言模型損失
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # 計算下一句子預測任務的損失
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l
```

```{.python .input}
#@tab paddle
#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # 前向傳播
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape([-1]), 
                                  pred_positions_X)
    # 計算遮蔽語言模型損失
    mlm_l = loss(mlm_Y_hat.reshape([-1, vocab_size]), mlm_Y.reshape([-1])) *\
    mlm_weights_X.reshape([-1, 1])
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # 計算下一句子預測任務的損失
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l
```

透過呼叫上述兩個輔助函式，下面的`train_bert`函式定義了在WikiText-2（`train_iter`）資料集上預訓練BERT（`net`）的過程。訓練BERT可能需要很長時間。以下函式的輸入`num_steps`指定了訓練的迭代步數，而不是像`train_ch13`函式那樣指定訓練的輪數（參見 :numref:`sec_image_augmentation`）。

```{.python .input}
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': 0.01})
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # 遮蔽語言模型損失的和，下一句預測任務損失的和，句子對的數量，計數
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for batch in train_iter:
            (tokens_X_shards, segments_X_shards, valid_lens_x_shards,
             pred_positions_X_shards, mlm_weights_X_shards,
             mlm_Y_shards, nsp_y_shards) = [gluon.utils.split_and_load(
                elem, devices, even_split=False) for elem in batch]
            timer.start()
            with autograd.record():
                mlm_ls, nsp_ls, ls = _get_batch_loss_bert(
                    net, loss, vocab_size, tokens_X_shards, segments_X_shards,
                    valid_lens_x_shards, pred_positions_X_shards,
                    mlm_weights_X_shards, mlm_Y_shards, nsp_y_shards)
            for l in ls:
                l.backward()
            trainer.step(1)
            mlm_l_mean = sum([float(l) for l in mlm_ls]) / len(mlm_ls)
            nsp_l_mean = sum([float(l) for l in nsp_ls]) / len(nsp_ls)
            metric.add(mlm_l_mean, nsp_l_mean, batch[0].shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')
```

```{.python .input}
#@tab pytorch
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # 遮蔽語言模型損失的和，下一句預測任務損失的和，句子對的數量，計數
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')
```

```{.python .input}
#@tab paddle
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    trainer = paddle.optimizer.Adam(parameters=net.parameters(), learning_rate=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # 遮蔽語言模型損失的和，下一句預測任務損失的和，句子對的數量，計數
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            trainer.clear_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')
```

在預訓練過程中，我們可以繪製出遮蔽語言模型損失和下一句預測損失。

```{.python .input}
#@tab mxnet, pytorch
train_bert(train_iter, net, loss, len(vocab), devices, 50)
```

```{.python .input}
#@tab paddle
train_bert(train_iter, net, loss, len(vocab), devices[:1], 50)
```

## 用BERT表示文字

在預訓練BERT之後，我們可以用它來表示單個文字、文字對或其中的任何詞元。下面的函式回傳`tokens_a`和`tokens_b`中所有詞元的BERT（`net`）表示。

```{.python .input}
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = np.expand_dims(np.array(vocab[tokens], ctx=devices[0]),
                               axis=0)
    segments = np.expand_dims(np.array(segments, ctx=devices[0]), axis=0)
    valid_len = np.expand_dims(np.array(len(tokens), ctx=devices[0]), axis=0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```

```{.python .input}
#@tab pytorch
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```

```{.python .input}
#@tab paddle
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = paddle.to_tensor(vocab[tokens]).unsqueeze(0)
    segments = paddle.to_tensor(segments).unsqueeze(0)
    valid_len = paddle.to_tensor(len(tokens))
    
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```

考慮『a crane is flying』這句話。回想一下 :numref:`subsec_bert_input_rep`中討論的BERT的輸入表示。插入特殊標記『&lt;cls&gt;』（用於分類）和『&lt;sep&gt;』（用於分隔）後，BERT輸入序列的長度為6。因為零是『&lt;cls&gt;』詞元，`encoded_text[:, 0, :]`是整個輸入語句的BERT表示。為了評估一詞多義詞元『crane』，我們還列印出了該詞元的BERT表示的前三個元素。

```{.python .input}
#@tab all
tokens_a = ['a', 'crane', 'is', 'flying']
encoded_text = get_bert_encoding(net, tokens_a)
# 詞元：'<cls>','a','crane','is','flying','<sep>'
encoded_text_cls = encoded_text[:, 0, :]
encoded_text_crane = encoded_text[:, 2, :]
encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3]
```

現在考慮一個句子『a crane driver came』和『he just left』。類似地，`encoded_pair[:, 0, :]`是來自預訓練BERT的整個句子對的編碼結果。注意，多義詞元『crane』的前三個元素與上下文不同時的元素不同。這支援了BERT表示是上下文敏感的。

```{.python .input}
#@tab all
tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
encoded_pair = get_bert_encoding(net, tokens_a, tokens_b)
# 詞元：'<cls>','a','crane','driver','came','<sep>','he','just',
# 'left','<sep>'
encoded_pair_cls = encoded_pair[:, 0, :]
encoded_pair_crane = encoded_pair[:, 2, :]
encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3]
```

在 :numref:`chap_nlp_app`中，我們將為下游自然語言處理應用微調預訓練的BERT模型。

## 小結

* 原始的BERT有兩個版本，其中基本模型有1.1億個引數，大模型有3.4億個引數。
* 在預訓練BERT之後，我們可以用它來表示單個文字、文字對或其中的任何詞元。
* 在實驗中，同一個詞元在不同的上下文中具有不同的BERT表示。這支援BERT表示是上下文敏感的。

## 練習

1. 在實驗中，我們可以看到遮蔽語言模型損失明顯高於下一句預測損失。為什麼？
2. 將BERT輸入序列的最大長度設定為512（與原始BERT模型相同）。使用原始BERT模型的設定，如$\text{BERT}_{\text{LARGE}}$。執行此部分時是否遇到錯誤？為什麼？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/5742)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/5743)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11824)
:end_tab:
