# AutoRec：使用自動編碼器預測評分

儘管矩陣分解模型在評分預測任務上取得了不錯的表現，但是它本質上仍然只是一個線性模型。因此，這類模型無法描述能夠預測使用者偏好的非線性複雜關係。在本節，我們介紹一個基於非線性神經網路的協同過濾模型AutoRec:cite:`Sedhain.Menon.Sanner.ea.2015`。AutoRec是一個基於顯式評分和自動編碼器架構，並將非線性變換整合到協同過濾（collaborative filtering，CF）中的模型。神經網路已經被證明能夠逼近任意連續函式，因此它能夠解決矩陣分解的不足，增強矩陣分解的表示能力。

一方面，AutoRec和自動編碼器擁有一樣的架構：輸入層、隱含層、重構層（輸出層）。自動編碼器是一種可以將輸入複製到輸出的神經網路，它能夠將輸入編碼成隱含層（通常維度更低）表示。AutoRec沒有顯式地將使用者和物品嵌入到低維空間。它使用互動矩陣的行或著列作為輸入，然後在輸出層重構互動矩陣。

另一方面，AutoRec和常規的自動編碼器也有所不同。AutoRec專注於學習重構層輸出，而不是隱含層表示。它使用一個只有部分資料的互動矩陣作為輸入，然後試圖重構一個完整的評分矩陣。同時，出於推薦的目的，重構過程在輸出層中將輸入層中缺失的條目補齊。

AutoRec有基於使用者的和基於物品的兩種變體。們在這裡只介紹基於物品的AutoRec，基於使用者的AutoRec可以據此匯出。

## 模型

$\mathbf{R}_{*i}$表示評分矩陣的第$i$列，其中未知評分在預設情況下設定為0。神經網路的定義如下所示：

$$
h(\mathbf{R}_{*i}) = f(\mathbf{W} \cdot g(\mathbf{V} \mathbf{R}_{*i} + \mu) + b)
$$

其中，$f(\cdot)$和$g(\cdot)$表示啟用函式，$\mathbf{W}$和$\mathbf{V}$表示權重矩陣，$\mu$和$b$表示偏置。使用$h( \cdot )$表示AutoRec的整個網路，因此$h(\mathbf{R}_{*i})$表示評分矩陣第$i$列的重構結果。

下面的目標函式旨在降低重構誤差。

$$
\underset{\mathbf{W},\mathbf{V},\mu, b}{\mathrm{argmin}} \sum_{i=1}^M{\parallel \mathbf{R}_{*i} - h(\mathbf{R}_{*i})\parallel_{\mathcal{O}}^2} +\lambda(\| \mathbf{W} \|_F^2 + \| \mathbf{V}\|_F^2)
$$

其中，$\| \cdot \|_{\mathcal{O}}$表示在訓練過程中只考慮已知評分。這也就是說，只有和已知輸入相關聯的權重矩陣才會在反向傳播的過程中得到更新。

```python
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
import sys
npx.set_np()
```

## 模型實現

一個典型的自動編碼器由編碼器和解碼器兩部分組成。編碼器將輸入對映為隱含層表示，解碼器則將隱含層表示對映到重構層。按照這一做法，我們使用全連線層建立編碼器和解碼器。在預設情況下，編碼器的啟用函式為`sigmoid`，而解碼器不使用啟用函式。為了減輕過擬合，在編碼器後新增了dropout層。透過掩碼遮蔽未定輸入值的梯度，如此一來，只有已確定的評分才能幫助到模型的學習。

```python
class AutoRec(nn.Block):
    def __init__(self, num_hidden, num_users, dropout=0.05):
        super(AutoRec, self).__init__()
        self.encoder = nn.Dense(num_hidden, activation='sigmoid',
                                use_bias=True)
        self.decoder = nn.Dense(num_users, use_bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        hidden = self.dropout(self.encoder(input))
        pred = self.decoder(hidden)
        if autograd.is_training():  # Mask the gradient during training
            return pred * np.sign(input)
        else:
            return pred
```

## 重新實現評估器

由於輸入和輸出均已改變，為了能繼續使用RMSE作為評估指標，我們需要重新實現評估函式。

```python
def evaluator(network, inter_matrix, test_data, ctx):
    scores = []
    for values in inter_matrix:
        feat = gluon.utils.split_and_load(values, ctx, even_split=False)
        scores.extend([network(i).asnumpy() for i in feat])
    recons = np.array([item for sublist in scores for item in sublist])
    # Calculate the test RMSE
    rmse = np.sqrt(np.sum(np.square(test_data - np.sign(test_data) * recons))
                   / np.sum(np.sign(test_data)))
    return float(rmse)
```

## 訓練和評估模型

現在，讓我們使用MovieLens資料集訓練和評估一下AutoRec模型。我們可以清楚地看到，測試集的RMSE低於矩陣分解模型，這表明神經網路在評分預測任務上的有效性。

```python
ctx = d2l.try_all_gpus()
# Load the MovieLens 100K dataset
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items)
_, _, _, train_inter_mat = d2l.load_data_ml100k(train_data, num_users,
                                                num_items)
_, _, _, test_inter_mat = d2l.load_data_ml100k(test_data, num_users,
                                               num_items)
train_iter = gluon.data.DataLoader(train_inter_mat, shuffle=True,
                                   last_batch="rollover", batch_size=256,
                                   num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(np.array(train_inter_mat), shuffle=False,
                                  last_batch="keep", batch_size=1024,
                                  num_workers=d2l.get_dataloader_workers())
# Model initialization, training, and evaluation
net = AutoRec(500, num_users)
net.initialize(ctx=ctx, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 25, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
d2l.train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        ctx, evaluator, inter_mat=test_inter_mat)
```

## 小結

* 我們可以使用自動編碼器建立矩陣分解演算法，同時還可以在其中整合非線性層和dropout正則化層。
* MovieLens-100K資料集上的實驗表明，自動編碼器的效能優於矩陣分解模型。

## 練習

* 修改自動編碼器的隱含層維度，觀察模型效能的變化。
* 嘗試新增更多的隱含層。這對提高模型的效能有幫助嗎？
* 可以找到更好的編碼器啟用函式和解碼器啟用函式嗎？

[討論](https://discuss.d2l.ai/t/401)
