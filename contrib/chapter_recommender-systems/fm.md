# 因子分解機

由Steffen Rendle發表於2010年的因子分解機（Factorization machines，FM）模型:cite:`Rendle.2010`是一種監督學習演算法，它可以用於分類、迴歸和排序等任務。它很快就引起了人們的注意，而後成為一個流行的、有影響的，可以用於預測和推薦的方法。具體而言，它是線性迴歸模型和矩陣分解模型的推廣。此外，它還讓人想起具有多項式核函式的支援向量機。和線性迴歸以及矩陣分解相比，因子分解機的優勢在於：一，它可以建模$\chi$路變數互動，其中$\chi$為多項式階數，通常為二；二、與因子分解機相關聯的快速最佳化演算法可以將計算時間從多項式複雜度降低為線性複雜度，如此一來，對於高維度稀疏輸入，它的計算效率會非常高。基於以上這些原因，因子分解機廣泛應用於現代廣告和產品推薦之中。技術細節和實現如下所示。

## 雙路因子分解機

使用$x \in \mathbb{R}^d$表示樣本的特徵向量，使用$y$表示樣本標籤。這裡的標籤$y$可以是實數值，也可以是二分類任務的類別標籤，例如點選/不點選。二階的因子分解機模型可以定義為：

$$
\hat{y}(x) = \mathbf{w}_0 + \sum_{i=1}^d \mathbf{w}_i x_i + \sum_{i=1}^d\sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j
$$

其中，$\mathbf{w}_0 \in \mathbb{R}$為全域偏置，$\mathbf{w} \in \mathbb{R}^d$為第i個變數的權重，$\mathbf{V} \in \mathbb{R}^{d\times k}$為特徵嵌入，$\mathbf{v}_i$為$\mathbf{V}$的第i行，$k$為隱向量的維度，$\langle\cdot, \cdot \rangle$為兩個向量的內積。$\langle \mathbf{v}_i, \mathbf{v}_j \rangle$對第i個特徵和第j個特徵的互動進行建模。有一些特徵互動很容易理解，因此它們可以由專家設計得到。但是，其他大多數特徵互動都隱藏在了資料之中，很難被識別出來。因此，自動化地建模特徵互動可以極大地減輕特徵工程的工作量。顯然，公式的前兩項對應了線性迴歸，而最後一項則對應了因子分解機。如果特徵$i$表示物品，而特徵$j$代表使用者，那麼第三項則恰好是使用者和物品嵌入向量的內積。需要注意的是，因子分解機可以推廣到更高的階數（階數大於2）。不過，數值穩定性可能會削弱模型的泛化效能。

## 高效的最佳化標準

直接對因子分解機進行最佳化的複雜度為$\mathcal{O}(kd^2)$，因為每一對互動作用都需要計算。為了解決效率低下的問題，我們可以重組第三項。重組後計算時間複雜度為$\mathcal{O}(kd)$)，計算成本大大降低。逐對互動項重組後的公式如下所示：

$$
\begin{aligned}
&\sum_{i=1}^d \sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j \\
 &= \frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d\langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j - \frac{1}{2}\sum_{i=1}^d \langle\mathbf{v}_i, \mathbf{v}_i\rangle x_i x_i \\
 &= \frac{1}{2} \big (\sum_{i=1}^d \sum_{j=1}^d \sum_{l=1}^k\mathbf{v}_{i, l} \mathbf{v}_{j, l} x_i x_j - \sum_{i=1}^d \sum_{l=1}^k \mathbf{v}_{i, l} \mathbf{v}_{i, l} x_i x_i \big)\\
 &=  \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i) (\sum_{j=1}^d \mathbf{v}_{j, l}x_j) - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2 \big ) \\
 &= \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i)^2 - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2)
 \end{aligned}
$$

重組後，模型的（最佳化）複雜度大大降低。此外，對於稀疏特徵，只有非零元素才需要計算，因此整體複雜度與非零特徵的數量呈線性關係。

為了學習因子分解機模型，我們在迴歸任務中使用MSE損失，在分類任務中使用交叉熵損失，在排序任務中使用貝葉斯個性化排序損失。標準最佳化器（如SGD和Adam等）均可用於引數最佳化過程。

```python
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os
import sys
npx.set_np()
```

## 模型實現

下面的程式碼實現了因子分解機。可以清楚地從中看到，因子分解機包含了一個線性迴歸模組和一個高效的特徵互動模組。由於點選率預測是一個分類任務，所以我們在最後的得分上應用了sigmoid函式。

```python
class FM(nn.Block):
    def __init__(self, field_dims, num_factors):
        super(FM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)
        
    def forward(self, x):
        square_of_sum = np.sum(self.embedding(x), axis=1) ** 2 # self.embedding(x).shape == (b, num_inputs, num_factors)
        sum_of_square = np.sum(self.embedding(x) ** 2, axis=1)
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True) # self.fc(x).shape == (b, num_inputs, 1)
        x = npx.sigmoid(x)
        return x
```

## 載入廣告資料集

我們使用上一節定義的資料裝飾器載入線上廣告資料集。

```python
batch_size = 2048
data_dir = d2l.download_extract('ctr')
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
```

## 訓練模型

之後，我們使用`Adam`最佳化器和`SigmoidBinaryCrossEntropyLoss`損失訓練模型。預設的學習率為0.01，而嵌入尺寸則設定為20。

```python
ctx = d2l.try_all_gpus()
net = FM(train_data.field_dims, num_factors=20)
net.initialize(init.Xavier(), ctx=ctx)
lr, num_epochs, optimizer = 0.02, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, ctx)
```

## 小結

* 因子分解機是一種通用框架，它可以應用在迴歸、分類和排序等一系列不同的任務上。
* 對於預測任務來說，特徵互動/交叉非常重要，而使用因子分解機可以高效地建模雙路特徵互動。

## 練習

* 你可以在Avazu、MovieLens和Criteo資料集上測試因子分級機嗎？
* 改變嵌入尺寸，觀察它對模型效能的影響。你能觀察到和矩陣分解模型相似的模式嗎？

[討論](https://discuss.d2l.ai/t/406)
