# 矩陣分解

矩陣分解（Matrix Factorization，MF）是推薦系統文獻中公認的一種演算法。最初版本的矩陣分解模型由Simon Funk發表在一篇非常知名的[博文](https://sifter.org/~simon/journal/20061211.html)中，他在這篇博文描述了分解矩陣的想法。隨後，由於2006年舉辦的Netflix競賽，該模型變得廣為人知。那時，流媒體和影片租賃公司Netflix為了增進其推薦系統的效能而舉辦了一項比賽。如果最佳團隊（例如Cinematch）能夠將Netflix的基線提高10%，那麼他們將贏得100萬美元的獎勵。這一比賽在推薦系統領域引起了廣泛的關注。隨後，BellKor's Pragmatic Chaos團隊（一個由BellKor、Pragmatic Theory和BigChaos混合組成的團隊）贏得了這一大獎。儘管他們的最終評分來自一個整合解決方案，矩陣分解演算法仍在其中起到了關鍵作用。Netflix Grand Prize的技術報告:cite:`Toscher.Jahrer.Bell.2009`詳細解釋了該方案所採用的模型。本節將深入研究矩陣分解模型的細節和實現過程。

## 矩陣分解模型

矩陣分解是一種協同過濾模型。具體來說，該模型將使用者-物品互動矩陣（例如評分矩陣）分解為兩個低秩矩陣的乘積，從而得到使用者和物品的低秩架構。

使用$\mathbf{R} \in \mathbb{R}^{m \times n}$表示具有$m$個使用者和$n$個物品的互動矩陣，矩陣$\mathbf{R}$的數值表示顯式評分。使用者-物品互動矩陣將被分解成使用者潛矩陣$\mathbf{P} \in \mathbb{R}^{m \times k}$和物品潛矩陣$\mathbf{Q} \in \mathbb{R}^{n \times k}$。其中，表示潛因子尺寸的$k \ll m, n$。使用$\mathbf{p}_u$表示矩陣$\mathbf{P}$的第$u$行，同時使用$\mathbf{q}_i$表示矩陣$\mathbf{Q}$的第$i$行。對於某一物品$i$，$\mathbf{q}_i$中的數值衡量了特徵（例如電影風格和語言等）的大小。對於某一使用者$u$，$\mathbf{p}_u$中的數值衡量他對物品相應特徵的感興趣程度。這些潛因子可能代表了之前提到的一些維度，但同時它們也可能是完全無法理解的。 使用者對物品的預測評分可以透過下式計算：

$$\hat{\mathbf{R}} = \mathbf{PQ}^\top$$

上式中的$\hat{\mathbf{R}}\in \mathbb{R}^{m \times n}$表示預測評分矩陣，它的形狀和真實評分矩陣$\mathbf{R}$是一致的。這種預測方式的主要問題是無法建模表示使用者和物品的偏置。例如，有一些使用者傾向於給出較高的評分，而有一些物品由於質量較差得到的評分普遍較低。這類偏置在實際應用中很常見。為了表示這種偏置，我們在此處引入了使用者偏置和物品偏置。具體來說，使用者$u$對物品$i$的評分由下式計算得到。

$$
\hat{\mathbf{R}}_{ui} = \mathbf{p}_u\mathbf{q}^\top_i + b_u + b_i
$$

然後，我們透過減少預測評分和實際評分的均方誤差來訓練矩陣分解模型。目標函式如下所示：

$$
\underset{\mathbf{P}, \mathbf{Q}, b}{\mathrm{argmin}} \sum_{(u, i) \in \mathcal{K}} \| \mathbf{R}_{ui} -
\hat{\mathbf{R}}_{ui} \|^2 + \lambda (\| \mathbf{P} \|^2_F + \| \mathbf{Q}
\|^2_F + b_u^2 + b_i^2 )
$$

上式中的$\lambda$表示正則化率。透過懲罰引數大小，正則化公式$\lambda (\| \mathbf{P} \|^2_F + \| \mathbf{Q}
\|^2_F + b_u^2 + b_i^2 )$被用來規避過擬合問題。已知$\mathbf{R}_{ui}$的$(u, i)$對儲存在集合$\mathcal{K}=\{(u, i) \mid \mathbf{R}_{ui} \text{ is known}\}$當中。模型引數可以透過最佳化演算法（例如隨機梯度下降法和Adam）學習得到。

矩陣分解模型的直觀示意圖如下所示：

![矩陣分解模型的圖示](../img/rec-mf.svg)

在本節的最後，我們將解釋矩陣分解模型的實現過程，並使用MovieLens資料集訓練模型。

```python
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
npx.set_np()
```

## 模型實現

首先，我們按照上述描述實現矩陣分解模型。使用者和物品的潛因子可以透過`nn.Embedding`構造。`input_dim`為使用者和物品的數量，而`output_dim`為潛因子$k$的維度。將`output_dim`設定為1後，我們也可以使用`nn.Embedding`構造使用者和物品的偏置量。在`forward`函式中，使用者和物品的id被用於索引嵌入向量。

```python
class MF(nn.Block):
    def __init__(self, num_factors, num_users, num_items, **kwargs):
        super(MF, self).__init__(**kwargs)
        self.P = nn.Embedding(input_dim=num_users, output_dim=num_factors)
        self.Q = nn.Embedding(input_dim=num_items, output_dim=num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_id, item_id):
        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        b_u = self.user_bias(user_id)
        b_i = self.item_bias(item_id)
        outputs = (P_u * Q_i).sum(axis=1) + np.squeeze(b_u) + np.squeeze(b_i)
        return outputs.flatten()
```

## 評估方法

接下來，我們使用均方根誤差（root-mean-square error，RMSE）作為度量。該度量方式常用於測量模型的預測評分和實際評分（真值）之間的差異:cite:`Gunawardana.Shani.2015`。RMSE定義如下：

$$
\mathrm{RMSE} = \sqrt{\frac{1}{|\mathcal{T}|}\sum_{(u, i) \in \mathcal{T}}(\mathbf{R}_{ui} -\hat{\mathbf{R}}_{ui})^2}
$$

其中，$\mathcal{T}$為包含了使用者-物品對的待評估集合，$|\mathcal{T}|$為集合的大小。我們可以使用`mx.metric`提供的RMSE函式。

```python
def evaluator(net, test_iter, ctx):
    rmse = mx.metric.RMSE()  # Get the RMSE
    rmse_list = []
    for idx, (users, items, ratings) in enumerate(test_iter):
        u = gluon.utils.split_and_load(users, ctx, even_split=False)
        i = gluon.utils.split_and_load(items, ctx, even_split=False)
        r_ui = gluon.utils.split_and_load(ratings, ctx, even_split=False)
        r_hat = [net(u, i) for u, i in zip(u, i)]
        rmse.update(labels=r_ui, preds=r_hat)
        rmse_list.append(rmse.get()[1])
    return float(np.mean(np.array(rmse_list)))
```

## 訓練和評估模型

在訓練函式中，我們採用了$L_2$損失作為權重衰減函式。該權重衰減機制和$L_2$正則化具有相同的效果。

```python
#@save
def train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        ctx_list=d2l.try_all_gpus(), evaluator=None,
                        **kwargs):
    timer = d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 2],
                            legend=['train loss', 'test RMSE'])
    for epoch in range(num_epochs):
        metric, l = d2l.Accumulator(3), 0.
        for i, values in enumerate(train_iter):
            timer.start()
            input_data = []
            values = values if isinstance(values, list) else [values]
            for v in values:
                input_data.append(gluon.utils.split_and_load(v, ctx_list))
            train_feat = input_data[0:-1] if len(values) > 1 else input_data
            train_label = input_data[-1]
            with autograd.record():
                preds = [net(*t) for t in zip(*train_feat)]
                ls = [loss(p, s) for p, s in zip(preds, train_label)]
            [l.backward() for l in ls]
            l += sum([l.asnumpy() for l in ls]).mean() / len(ctx_list)
            trainer.step(values[0].shape[0])
            metric.add(l, values[0].shape[0], values[0].size)
            timer.stop()
        if len(kwargs) > 0:  # It will be used in section AutoRec
            test_rmse = evaluator(net, test_iter, kwargs['inter_mat'],
                                  ctx_list)
        else:
            test_rmse = evaluator(net, test_iter, ctx_list)
        train_l = l / (i + 1)
        animator.add(epoch + 1, (train_l, test_rmse))
    print(f'train loss {metric[0] / metric[1]:.3f}, '
          f'test RMSE {test_rmse:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(ctx_list)}')
```

最後，把所有的東西全都結合起來然後開始訓練模型。此處，我們將潛因子的維度設定為30。

```python
ctx = d2l.try_all_gpus()
num_users, num_items, train_iter, test_iter = d2l.split_and_load_ml100k(
    test_ratio=0.1, batch_size=512)
net = MF(30, num_users, num_items)
net.initialize(ctx=ctx, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 20, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                    ctx, evaluator)
```

下面，我們使用訓練過的模型來預測ID為20的使用者對ID為30的物品的評分。

```python
scores = net(np.array([20], dtype='int', ctx=d2l.try_gpu()),
             np.array([30], dtype='int', ctx=d2l.try_gpu()))
scores
```

## 小結

* 矩陣分解模型在推薦系統中有著廣泛的應用。它可以用於預測使用者對物品的評分。
* 我們可以為推薦系統實現和訓練一個矩陣分解模型。

## 練習

* 修改潛因子的維度。潛因子的維度將怎樣影響模型的效能呢？
* 嘗試不同的最佳化器、學習率和權重衰減率。
* 檢查其他使用者對某一電影的評分。

[Discussions](https://discuss.d2l.ai/t/)

