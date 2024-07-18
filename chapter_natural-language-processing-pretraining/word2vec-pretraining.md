# 預訓練word2vec
:label:`sec_word2vec_pretraining`

我們繼續實現 :numref:`sec_word2vec`中定義的跳元語法模型。然後，我們將在PTB資料集上使用負取樣預訓練word2vec。首先，讓我們透過呼叫`d2l.load_data_ptb`函式來獲得該資料集的資料迭代器和詞表，該函式在 :numref:`sec_word2vec_data`中進行了描述。

```{.python .input}
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import torch
from torch import nn

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import math
import paddle
from paddle import nn

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

## 跳元模型

我們透過嵌入層和批次矩陣乘法實現了跳元模型。首先，讓我們回顧一下嵌入層是如何工作的。

### 嵌入層

如 :numref:`sec_seq2seq`中所述，嵌入層將詞元的索引對映到其特徵向量。該層的權重是一個矩陣，其行數等於字典大小（`input_dim`），列數等於每個標記的向量維數（`output_dim`）。在詞嵌入模型訓練之後，這個權重就是我們所需要的。

```{.python .input}
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
embed.weight
```

```{.python .input}
#@tab pytorch
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Parameter embedding_weight ({embed.weight.shape}, '
      f'dtype={embed.weight.dtype})')
```

```{.python .input}
#@tab paddle
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Parameter embedding_weight ({embed.weight.shape}, '
      f'dtype={embed.weight.dtype})')
```

嵌入層的輸入是詞元（詞）的索引。對於任何詞元索引$i$，其向量表示可以從嵌入層中的權重矩陣的第$i$行獲得。由於向量維度（`output_dim`）被設定為4，因此當小批次詞元索引的形狀為（2，3）時，嵌入層回傳具有形狀（2，3，4）的向量。

```{.python .input}
#@tab all
x = d2l.tensor([[1, 2, 3], [4, 5, 6]])
embed(x)
```

### 定義前向傳播

在前向傳播中，跳元語法模型的輸入包括形狀為（批次大小，1）的中心詞索引`center`和形狀為（批次大小，`max_len`）的上下文與噪聲詞索引`contexts_and_negatives`，其中`max_len`在 :numref:`subsec_word2vec-minibatch-loading`中定義。這兩個變數首先透過嵌入層從詞元索引轉換成向量，然後它們的批次矩陣相乘（在 :numref:`subsec_batch_dot`中描述）回傳形狀為（批次大小，1，`max_len`）的輸出。輸出中的每個元素是中心詞向量和上下文或噪聲詞向量的點積。

```{.python .input}
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = npx.batch_dot(v, u.swapaxes(1, 2))
    return pred
```

```{.python .input}
#@tab pytorch
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred
```

```{.python .input}
#@tab paddle
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = paddle.bmm(v, u.transpose(perm=[0, 2, 1]))
    return pred
```

讓我們為一些範例輸入列印此`skip_gram`函式的輸出形狀。

```{.python .input}
skip_gram(np.ones((2, 1)), np.ones((2, 4)), embed, embed).shape
```

```{.python .input}
#@tab pytorch
skip_gram(torch.ones((2, 1), dtype=torch.long),
          torch.ones((2, 4), dtype=torch.long), embed, embed).shape
```

```{.python .input}
#@tab paddle
skip_gram(paddle.ones((2, 1), dtype='int64'),
          paddle.ones((2, 4), dtype='int64'), embed, embed).shape
```

## 訓練

在訓練帶負取樣的跳元模型之前，我們先定義它的損失函式。

### 二元交叉熵損失

根據 :numref:`subsec_negative-sampling`中負取樣損失函式的定義，我們將使用二元交叉熵損失。

```{.python .input}
loss = gluon.loss.SigmoidBCELoss()
```

```{.python .input}
#@tab pytorch
class SigmoidBCELoss(nn.Module):
    # 帶掩碼的二元交叉熵損失
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)

loss = SigmoidBCELoss()
```

```{.python .input}
#@tab paddle
class SigmoidBCELoss(nn.Layer):
    # 帶掩碼的二元交叉熵損失
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            logit=inputs, label=target, weight=mask, reduction="none")
        return out.mean(axis=1)

loss = SigmoidBCELoss()
```

回想一下我們在 :numref:`subsec_word2vec-minibatch-loading`中對掩碼變數和標籤變數的描述。下面計算給定變數的二進位交叉熵損失。

```{.python .input}
#@tab mxnet, pytorch
pred = d2l.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = d2l.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = d2l.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1)
```

```{.python .input}
#@tab paddle
pred = d2l.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = d2l.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = d2l.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype='float32')
loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1)
```

下面顯示瞭如何使用二元交叉熵損失中的Sigmoid啟用函式（以較低效率的方式）計算上述結果。我們可以將這兩個輸出視為兩個規範化的損失，在非掩碼預測上進行平均。

```{.python .input}
#@tab all
def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))

print(f'{(sigmd(1.1) + sigmd(2.2) + sigmd(-3.3) + sigmd(4.4)) / 4:.4f}')
print(f'{(sigmd(-1.1) + sigmd(-2.2)) / 2:.4f}')
```

### 初始化模型引數

我們定義了兩個嵌入層，將詞表中的所有單詞分別作為中心詞和上下文詞使用。字向量維度`embed_size`被設定為100。

```{.python .input}
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=embed_size))
```

```{.python .input}
#@tab pytorch, paddle
embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size),
                    nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size))
```

### 定義訓練階段程式碼

訓練階段程式碼實現定義如下。由於填充的存在，損失函式的計算與以前的訓練函式略有不同。

```{.python .input}
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    net.initialize(ctx=device, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # 規範化的損失之和，規範化的損失數
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            center, context_negative, mask, label = [
                data.as_in_ctx(device) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                l = (loss(pred.reshape(label.shape), label, mask) *
                     mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(batch_size)
            metric.add(l.sum(), l.size)
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

```{.python .input}
#@tab pytorch
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # 規範化的損失之和，規範化的損失數
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

```{.python .input}
#@tab paddle
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.initializer.XavierUniform(m.weight)
    net.apply(init_weights)
    optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=net.parameters())
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # 規範化的損失之和，規範化的損失數
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.clear_grad()
            center, context_negative, mask, label = [
                paddle.to_tensor(data, place=device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape), paddle.to_tensor(label, dtype='float32'), 
                        paddle.to_tensor(mask, dtype='float32'))
                     / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

現在，我們可以使用負取樣來訓練跳元模型。

```{.python .input}
#@tab all
lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)
```

## 應用詞嵌入
:label:`subsec_apply-word-embed`

在訓練word2vec模型之後，我們可以使用訓練好模型中詞向量的餘弦相似度來從詞表中找到與輸入單詞語義最相似的單詞。

```{.python .input}
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[vocab[query_token]]
    # 計算餘弦相似性。增加1e-9以獲得數值穩定性
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + \
          1e-9)
    topk = npx.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # 刪除輸入詞
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

```{.python .input}
#@tab pytorch
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # 計算餘弦相似性。增加1e-9以獲得數值穩定性
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # 刪除輸入詞
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

```{.python .input}
#@tab paddle
def get_similar_tokens(query_token, k, embed):
    W = embed.weight
    x = W[vocab[query_token]]
    # 計算餘弦相似性。增加1e-9以獲得數值穩定性
    cos = paddle.mv(W, x) / paddle.sqrt(paddle.sum(W * W, axis=1) *
                                        paddle.sum(x * x) + 1e-9)
    topk = paddle.topk(cos, k=k+1)[1].numpy().astype('int32')
    for i in topk[1:]:  # 刪除輸入詞
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

## 小結

* 我們可以使用嵌入層和二元交叉熵損失來訓練帶負取樣的跳元模型。
* 詞嵌入的應用套件括基於詞向量的餘弦相似度為給定詞找到語義相似的詞。

## 練習

1. 使用訓練好的模型，找出其他輸入詞在語義上相似的詞。您能透過調優超引數來改進結果嗎？
1. 當訓練語料庫很大時，在更新模型引數時，我們經常對當前小批次的*中心詞*進行上下文詞和噪聲詞的取樣。換言之，同一中心詞在不同的訓練迭代輪數可以有不同的上下文詞或噪聲詞。這種方法的好處是什麼？嘗試實現這種訓練方法。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/5739)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/5740)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11817)
:end_tab:
