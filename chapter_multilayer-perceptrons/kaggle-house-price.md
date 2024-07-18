# 實戰Kaggle比賽：預測房價
:label:`sec_kaggle_house`

之前幾節我們學習了一些訓練深度網路的基本工具和網路正則化的技術（如權重衰減、暫退法等）。
本節我們將透過Kaggle比賽，將所學知識付諸實踐。
Kaggle的房價預測比賽是一個很好的起點。
此資料集由Bart de Cock於2011年收集 :cite:`De-Cock.2011`，
涵蓋了2006-2010年期間亞利桑那州埃姆斯市的房價。
這個資料集是相當通用的，不會需要使用複雜模型架構。
它比哈里森和魯賓菲爾德的[波士頓房價](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names)
資料集要大得多，也有更多的特徵。

本節我們將詳細介紹資料預處理、模型設計和超引數選擇。
透過親身實踐，你將獲得一手經驗，這些經驗將有益資料科學家的職業成長。

## 下載和緩存資料集

在整本書中，我們將下載不同的資料集，並訓練和測試模型。
這裡我們(**實現幾個函式來方便下載資料**)。
首先，我們建立字典`DATA_HUB`，
它可以將資料集名稱的字串對映到資料集相關的二元組上，
這個二元組包含資料集的url和驗證檔案完整性的sha-1金鑰。
所有類似的資料集都託管在地址為`DATA_URL`的站點上。

```{.python .input}
#@tab all
import os
import requests
import zipfile
import tarfile
import hashlib

#@save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
```

下面的`download`函式用來下載資料集，
將資料集快取在本地目錄（預設情況下為`../data`）中，
並回傳下載檔案的名稱。
如果快取目錄中已經存在此資料集檔案，並且其sha-1與儲存在`DATA_HUB`中的相對應，
我們將使用快取的檔案，以避免重複的下載。

```{.python .input}
#@tab all
def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """下載一個DATA_HUB中的檔案，回傳本地檔案名"""
    assert name in DATA_HUB, f"{name} 不存在於 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中快取
    print(f'正在從{url}下載{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname
```

我們還需實現兩個實用函式：
一個將下載並解壓縮一個zip或tar檔案，
另一個是將本書中使用的所有資料集從`DATA_HUB`下載到快取目錄中。

```{.python .input}
#@tab all
def download_extract(name, folder=None):  #@save
    """下載並解壓zip/tar檔案"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar檔案可以被解壓縮'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():  #@save
    """下載DATA_HUB中的所有檔案"""
    for name in DATA_HUB:
        download(name)
```

## Kaggle

[Kaggle](https://www.kaggle.com)是一個當今流行舉辦機器學習比賽的平台，
每場比賽都以至少一個資料集為中心。
許多比賽有贊助方，他們為獲勝的解決方案提供獎金。
該平台幫助使用者透過論壇和共享程式碼進行互動，促進協作和競爭。
雖然排行榜的追逐往往令人失去理智：
有些研究人員短視地專注於預處理步驟，而不是考慮基礎性問題。
但一個客觀的平台有巨大的價值：該平台促進了競爭方法之間的直接定量比較，以及程式碼共享。
這便於每個人都可以學習哪些方法起作用，哪些沒有起作用。
如果我們想參加Kaggle比賽，首先需要註冊一個賬戶（見 :numref:`fig_kaggle`）。

![Kaggle網站](../img/kaggle.png)
:width:`400px`
:label:`fig_kaggle`

在房價預測比賽頁面（如 :numref:`fig_house_pricing` 所示）的"Data"頁籤下可以找到資料集。我們可以透過下面的網址送出預測，並檢視排名：

>https://www.kaggle.com/c/house-prices-advanced-regression-techniques

![房價預測比賽頁面](../img/house-pricing.png)
:width:`400px`
:label:`fig_house_pricing`

## 訪問和讀取資料集

注意，競賽資料分為訓練集和測試集。
每條記錄都包括房屋的屬性值和屬性，如街道型別、施工年份、屋頂型別、地下室狀況等。
這些特徵由各種資料型別組成。
例如，建築年份由整數表示，屋頂型別由離散類別表示，其他特徵由浮點數表示。
這就是現實讓事情變得複雜的地方：例如，一些資料完全丟失了，缺失值被簡單地標記為『NA』。
每套房子的價格只出現在訓練集中（畢竟這是一場比賽）。
我們將希望劃分訓練集以建立驗證集，但是在將預測結果上傳到Kaggle之後，
我們只能在官方測試集中評估我們的模型。
在 :numref:`fig_house_pricing` 中，"Data"頁籤有下載資料的連結。

開始之前，我們將[**使用`pandas`讀入並處理資料**]，
這是我們在 :numref:`sec_pandas`中引入的。
因此，在繼續操作之前，我們需要確保已安裝`pandas`。
幸運的是，如果我們正在用Jupyter閱讀該書，可以在不離開筆記本的情況下安裝`pandas`。

```{.python .input}
# 如果沒有安裝pandas，請取消下一行的註解
# !pip install pandas

%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, autograd, init, np, npx
from mxnet.gluon import nn
import pandas as pd
npx.set_np()
```

```{.python .input}
#@tab pytorch
# 如果沒有安裝pandas，請取消下一行的註解
# !pip install pandas

%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
import pandas as pd
import numpy as np
```

```{.python .input}
#@tab tensorflow
# 如果沒有安裝pandas，請取消下一行的註解
# !pip install pandas

%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import pandas as pd
import numpy as np
```

```{.python .input}
#@tab paddle
# 如果你沒有安裝pandas，請取消下一行的註解
# !pip install pandas

%matplotlib inline
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings(action='ignore')
import paddle
from paddle import nn
warnings.filterwarnings("ignore", category=DeprecationWarning)
from d2l import paddle as d2l
```

為方便起見，我們可以使用上面定義的指令碼下載並快取Kaggle房屋資料集。

```{.python .input}
#@tab all
DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
```

我們使用`pandas`分別載入包含訓練資料和測試資料的兩個CSV檔案。

```{.python .input}
#@tab all
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
```

訓練資料集包括1460個樣本，每個樣本80個特徵和1個標籤，
而測試資料集包含1459個樣本，每個樣本80個特徵。

```{.python .input}
#@tab all
print(train_data.shape)
print(test_data.shape)
```

讓我們看看[**前四個和最後兩個特徵，以及相應標籤**]（房價）。

```{.python .input}
#@tab all
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
```

我們可以看到，(**在每個樣本中，第一個特徵是ID，**)
這有助於模型識別每個訓練樣本。
雖然這很方便，但它不攜帶任何用於預測的訊息。
因此，在將資料提供給模型之前，(**我們將其從資料集中刪除**)。

```{.python .input}
#@tab all
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
```

## 資料預處理

如上所述，我們有各種各樣的資料型別。
在開始建模之前，我們需要對資料進行預處理。
首先，我們[**將所有缺失的值替換為相應特徵的平均值。**]然後，為了將所有特徵放在一個共同的尺度上，
我們(**透過將特徵重新縮放到零均值和單位方差來標準化資料**)：

$$x \leftarrow \frac{x - \mu}{\sigma},$$

其中$\mu$和$\sigma$分別表示均值和標準差。
現在，這些特徵具有零均值和單位方差，即 $E[\frac{x-\mu}{\sigma}] = \frac{\mu - \mu}{\sigma} = 0$和$E[(x-\mu)^2] = (\sigma^2 + \mu^2) - 2\mu^2+\mu^2 = \sigma^2$。
直觀地說，我們標準化資料有兩個原因：
首先，它方便最佳化。
其次，因為我們不知道哪些特徵是相關的，
所以我們不想讓懲罰分配給一個特徵的係數比分配給其他任何特徵的係數更大。

```{.python .input}
#@tab all
# 若無法獲得測試資料，則可根據訓練資料計算均值和標準差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在標準化資料之後，所有均值消失，因此我們可以將缺失值設定為0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
```

接下來，我們[**處理離散值。**]
這包括諸如『MSZoning』之類的特徵。
(**我們用獨熱編碼替換它們**)，
方法與前面將多類別標籤轉換為向量的方式相同
（請參見 :numref:`subsec_classification-problem`）。
例如，『MSZoning』包含值『RL』和『Rm』。
我們將建立兩個新的指示器特徵『MSZoning_RL』和『MSZoning_RM』，其值為0或1。
根據獨熱編碼，如果『MSZoning』的原始值為『RL』，
則：『MSZoning_RL』為1，『MSZoning_RM』為0。
`pandas`軟體套件會自動為我們實現這一點。

```{.python .input}
#@tab all
# 『Dummy_na=True』將『na』（缺失值）視為有效的特徵值，併為其建立指示符特徵
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape
```

可以看到此轉換會將特徵的總數量從79個增加到331個。
最後，透過`values`屬性，我們可以
[**從`pandas`格式中提取NumPy格式，並將其轉換為張量表示**]用於訓練。

```{.python .input}
#@tab all
n_train = train_data.shape[0]
train_features = d2l.tensor(all_features[:n_train].values, dtype=d2l.float32)
test_features = d2l.tensor(all_features[n_train:].values, dtype=d2l.float32)
train_labels = d2l.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=d2l.float32)
```

## [**訓練**]

首先，我們訓練一個帶有損失平方的線性模型。
顯然線性模型很難讓我們在競賽中獲勝，但線性模型提供了一種健全性檢查，
以檢視資料中是否存在有意義的訊息。
如果我們在這裡不能做得比隨機猜測更好，那麼我們很可能存在資料處理錯誤。
如果一切順利，線性模型將作為*基線*（baseline）模型，
讓我們直觀地知道最好的模型有超出簡單的模型多少。

```{.python .input}
loss = gluon.loss.L2Loss()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net
```

```{.python .input}
#@tab pytorch, paddle
loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.MeanSquaredError()

def get_net():
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Dense(
        1, kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    return net
```

房價就像股票價格一樣，我們關心的是相對數量，而不是絕對數量。
因此，[**我們更關心相對誤差$\frac{y - \hat{y}}{y}$，**]
而不是絕對誤差$y - \hat{y}$。
例如，如果我們在俄亥俄州農村地區估計一棟房子的價格時，
假設我們的預測偏差了10萬美元，
然而那裡一棟典型的房子的價值是12.5萬美元，
那麼模型可能做得很糟糕。
另一方面，如果我們在加州豪宅區的預測出現同樣的10萬美元的偏差，
（在那裡，房價中位數超過400萬美元）
這可能是一個不錯的預測。

(**解決這個問題的一種方法是用價格預測的對數來衡量差異**)。
事實上，這也是比賽中官方用來評價送出質量的誤差指標。
即將$\delta$ for $|\log y - \log \hat{y}| \leq \delta$
轉換為$e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^\delta$。
這使得預測價格的對數與真實標籤價格的對數之間出現以下均方根誤差：

$$\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}.$$

```{.python .input}
def log_rmse(net, features, labels):
    # 為了在取對數時進一步穩定該值，將小於1的值設定為1
    clipped_preds = np.clip(net(features), 1, float('inf'))
    return np.sqrt(2 * loss(np.log(clipped_preds), np.log(labels)).mean())
```

```{.python .input}
#@tab pytorch
def log_rmse(net, features, labels):
    # 為了在取對數時進一步穩定該值，將小於1的值設定為1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()
```

```{.python .input}
#@tab tensorflow
def log_rmse(y_true, y_pred):
    # 為了在取對數時進一步穩定該值，將小於1的值設定為1
    clipped_preds = tf.clip_by_value(y_pred, 1, float('inf'))
    return tf.sqrt(tf.reduce_mean(loss(
        tf.math.log(y_true), tf.math.log(clipped_preds))))
```

```{.python .input}
#@tab paddle
def log_rmse(net, features, labels):
    # 為了在取對數時進一步穩定該值，將小於1的值設定為1
    clipped_preds = paddle.clip(net(features), 1, float('inf'))
    rmse = paddle.sqrt(loss(paddle.log(clipped_preds),
                            paddle.log(labels)))
    return rmse.item()
```

與前面的部分不同，[**我們的訓練函式將藉助Adam最佳化器**]
（我們將在後面章節更詳細地描述它）。
Adam最佳化器的主要吸引力在於它對初始學習率不那麼敏感。

```{.python .input}
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 這裡使用的是Adam最佳化演算法
    trainer = gluon.Trainer(net.collect_params(), 'adam', {
        'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

```{.python .input}
#@tab pytorch
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 這裡使用的是Adam最佳化演算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

```{.python .input}
#@tab tensorflow
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 這裡使用的是Adam最佳化演算法
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    net.compile(loss=loss, optimizer=optimizer)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                y_hat = net(X)
                l = loss(y, y_hat)
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
        train_ls.append(log_rmse(train_labels, net(train_features)))
        if test_labels is not None:
            test_ls.append(log_rmse(test_labels, net(test_features)))
    return train_ls, test_ls
```

```{.python .input}
#@tab paddle
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 這裡使用的是Adam最佳化演算法
    optimizer = paddle.optimizer.Adam(learning_rate=learning_rate*1.0, 
                                      parameters=net.parameters(), 
                                      weight_decay=weight_decay*1.0)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
            optimizer.clear_grad()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

## $K$折交叉驗證

本書在討論模型選擇的部分（ :numref:`sec_model_selection`）
中介紹了[**K折交叉驗證**]，
它有助於模型選擇和超引數調整。
我們首先需要定義一個函式，在$K$折交叉驗證過程中回傳第$i$折的資料。
具體地說，它選擇第$i$個切片作為驗證資料，其餘部分作為訓練資料。
注意，這並不是處理資料的最有效方法，如果我們的資料集大得多，會有其他解決辦法。

```{.python .input}
#@tab all
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = d2l.concat([X_train, X_part], 0)
            y_train = d2l.concat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid
```

當我們在$K$折交叉驗證中訓練$K$次後，[**回傳訓練和驗證誤差的平均值**]。

```{.python .input}
#@tab all
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，訓練log rmse{float(train_ls[-1]):f}, '
              f'驗證log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k
```

## [**模型選擇**]

在本例中，我們選擇了一組未調優的超引數，並將其留給讀者來改進模型。
找到一組調優的超引數可能需要時間，這取決於一個人最佳化了多少變數。
有了足夠大的資料集和合理設定的超引數，$K$折交叉驗證往往對多次測試具有相當的穩定性。
然而，如果我們嘗試了不合理的超引數，我們可能會發現驗證效果不再代表真正的誤差。

```{.python .input}
#@tab all
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折驗證: 平均訓練log rmse: {float(train_l):f}, '
      f'平均驗證log rmse: {float(valid_l):f}')
```

請注意，有時一組超引數的訓練誤差可能非常低，但$K$折交叉驗證的誤差要高得多，
這表明模型過擬合了。
在整個訓練過程中，我們希望監控訓練誤差和驗證誤差這兩個數字。
較少的過擬合可能表明現有資料可以支撐一個更強大的模型，
較大的過擬合可能意味著我們可以透過正則化技術來獲益。

##  [**送出Kaggle預測**]

既然我們知道應該選擇什麼樣的超引數，
我們不妨使用所有資料對其進行訓練
（而不是僅使用交叉驗證中使用的$1-1/K$的資料）。
然後，我們透過這種方式獲得的模型可以應用於測試集。
將預測儲存在CSV檔案中可以簡化將結果上傳到Kaggle的過程。

```{.python .input}
#@tab all
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'訓練log rmse：{float(train_ls[-1]):f}')
    # 將網路應用於測試集。
    preds = d2l.numpy(net(test_features))
    # 將其重新格式化以匯出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
```

如果測試集上的預測與$K$倍交叉驗證過程中的預測相似，
那就是時候把它們上傳到Kaggle了。
下面的程式碼將生成一個名為`submission.csv`的檔案。

```{.python .input}
#@tab all
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
```

接下來，如 :numref:`fig_kaggle_submit2`中所示，
我們可以送出預測到Kaggle上，並檢視在測試集上的預測與實際房價（標籤）的比較情況。
步驟非常簡單。

* 登入Kaggle網站，訪問房價預測競賽頁面。
* 點選『Submit Predictions』或『Late Submission』按鈕（在撰寫本文時，該按鈕位於右側）。
* 點選頁面底部虛線框中的『Upload Submission File』按鈕，選擇要上傳的預測檔案。
* 點選頁面底部的『Make Submission』按鈕，即可檢視結果。

![向Kaggle送出資料](../img/kaggle-submit2.png)
:width:`400px`
:label:`fig_kaggle_submit2`

## 小結

* 真實資料通常混合了不同的資料型別，需要進行預處理。
* 常用的預處理方法：將實值資料重新縮放為零均值和單位方法；用均值替換缺失值。
* 將類別特徵轉化為指標特徵，可以使我們把這個特徵當作一個獨熱向量來對待。
* 我們可以使用$K$折交叉驗證來選擇模型並調整超引數。
* 對數對於相對誤差很有用。

## 練習

1. 把預測送出給Kaggle，它有多好？
1. 能透過直接最小化價格的對數來改進模型嗎？如果試圖預測價格的對數而不是價格，會發生什麼？
1. 用平均值替換缺失值總是好主意嗎？提示：能構造一個不隨機丟失值的情況嗎？
1. 透過$K$折交叉驗證調整超引數，從而提高Kaggle的得分。
1. 透過改進模型（例如，層、權重衰減和dropout）來提高分數。
1. 如果我們沒有像本節所做的那樣標準化連續的數值特徵，會發生什麼？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1823)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1824)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1825)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11775)
:end_tab:
