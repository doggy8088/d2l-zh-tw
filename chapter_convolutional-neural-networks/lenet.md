# 卷積神經網路（LeNet）
:label:`sec_lenet`

透過之前幾節，我們學習了建立一個完整卷積神經網路的所需元件。
回想一下，之前我們將softmax迴歸模型（ :numref:`sec_softmax_scratch`）和多層感知機模型（ :numref:`sec_mlp_scratch`）應用於Fashion-MNIST資料集中的服裝圖片。
為了能夠應用softmax迴歸和多層感知機，我們首先將每個大小為$28\times28$的影象展平為一個784維的固定長度的一維向量，然後用全連線層對其進行處理。
而現在，我們已經掌握了卷積層的處理方法，我們可以在影象中保留空間結構。
同時，用卷積層代替全連線層的另一個好處是：模型更簡潔、所需的引數更少。

本節將介紹LeNet，它是最早發布的卷積神經網路之一，因其在電腦視覺任務中的高效效能而受到廣泛關注。
這個模型是由AT&T貝爾實驗室的研究員Yann LeCun在1989年提出的（並以其命名），目的是識別影象 :cite:`LeCun.Bottou.Bengio.ea.1998`中的手寫數字。
當時，Yann LeCun發表了第一篇透過反向傳播成功訓練卷積神經網路的研究，這項工作代表了十多年來神經網路研究開發的成果。

當時，LeNet取得了與支援向量機（support vector machines）效能相媲美的成果，成為監督學習的主流方法。
LeNet被廣泛用於自動取款機（ATM）機中，幫助識別處理支票的數字。
時至今日，一些自動取款機仍在執行Yann LeCun和他的同事Leon Bottou在上世紀90年代寫的程式碼呢！

## LeNet

總體來看，(**LeNet（LeNet-5）由兩個部分組成：**)(~~卷積編碼器和全連線層密集區塊~~)

* 卷積編碼器：由兩個卷積層組成;
* 全連線層密集區塊：由三個全連線層組成。

該架構如 :numref:`img_lenet`所示。

![LeNet中的資料流。輸入是手寫數字，輸出為10種可能結果的機率。](../img/lenet.svg)
:label:`img_lenet`

每個卷積塊中的基本單元是一個卷積層、一個sigmoid啟用函式和平均彙聚層。請注意，雖然ReLU和最大彙聚層更有效，但它們在20世紀90年代還沒有出現。每個卷積層使用$5\times 5$卷積核和一個sigmoid啟用函式。這些層將輸入對映到多個二維特徵輸出，通常同時增加通道的數量。第一卷積層有6個輸出通道，而第二個卷積層有16個輸出通道。每個$2\times2$池操作（步幅2）透過空間下采樣將維數減少4倍。卷積的輸出形狀由批次大小、通道數、高度、寬度決定。

為了將卷積區塊的輸出傳遞給稠密區塊，我們必須在小批次中展平每個樣本。換言之，我們將這個四維輸入轉換成全連線層所期望的二維輸入。這裡的二維表示的第一個維度索引小批次中的樣本，第二個維度給出每個樣本的平面向量表示。LeNet的稠密區塊有三個全連線層，分別有120、84和10個輸出。因為我們在執行分類任務，所以輸出層的10維對應於最後輸出結果的數量。

透過下面的LeNet程式碼，可以看出用深度學習框架實現此類模型非常簡單。我們只需要實例化一個`Sequential`區塊並將需要的層連線在一起。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        # 預設情況下，『Dense』會自動將形狀為（批次大小，通道數，高度，寬度）的輸入，
        # 轉換為形狀為（批次大小，通道數*高度*寬度）的輸入
        nn.Dense(120, activation='sigmoid'),
        nn.Dense(84, activation='sigmoid'),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='sigmoid'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn, optimizer

net = nn.Sequential(
    nn.Conv2D(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2D(kernel_size=2, stride=2),
    nn.Conv2D(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2D(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

我們對原始模型做了一點小改動，去掉了最後一層的高斯啟用。除此之外，這個網路與最初的LeNet-5一致。

下面，我們將一個大小為$28 \times 28$的單通道（黑白）影象透過LeNet。透過在每一層列印輸出的形狀，我們可以[**檢查模型**]，以確保其操作與我們期望的 :numref:`img_lenet_vert`一致。

![LeNet 的簡化版。](../img/lenet-vert.svg)
:label:`img_lenet_vert`

```{.python .input}
X = np.random.uniform(size=(1, 1, 28, 28))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 28, 28, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
```

```{.python .input}
#@tab paddle
X = paddle.rand((1, 1, 28, 28), 'float32')
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
```

請注意，在整個卷積塊中，與上一層相比，每一層特徵的高度和寬度都減小了。
第一個卷積層使用2個畫素的填充，來補償$5 \times 5$卷積核導致的特徵減少。
相反，第二個卷積層沒有填充，因此高度和寬度都減少了4個畫素。
隨著層疊的上升，通道的數量從輸入時的1個，增加到第一個卷積層之後的6個，再到第二個卷積層之後的16個。
同時，每個彙聚層的高度和寬度都減半。最後，每個全連線層減少維數，最終輸出一個維數與結果分類數相對應的輸出。

## 模型訓練

現在我們已經實現了LeNet，讓我們看看[**LeNet在Fashion-MNIST資料集上的表現**]。

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
```

雖然卷積神經網路的引數較少，但與深度的多層感知機相比，它們的計算成本仍然很高，因為每個引數都參與更多的乘法。
透過使用GPU，可以用它加快訓練。

:begin_tab:`mxnet, pytorch`
為了進行評估，我們需要[**對**] :numref:`sec_softmax_scratch`中描述的(**`evaluate_accuracy`函式進行輕微的修改**)。
由於完整的資料集位於記憶體中，因此在模型使用GPU計算資料集之前，我們需要將其複製到顯存中。
:end_tab:

```{.python .input}
def evaluate_accuracy_gpu(net, data_iter, device=None):  #@save
    """使用GPU計算模型在資料集上的精度"""
    if not device:  # 查詢第一個引數所在的第一個裝置
        device = list(net.collect_params().values())[0].list_ctx()[0]
    metric = d2l.Accumulator(2)  # 正確預測的數量，總預測的數量
    for X, y in data_iter:
        X, y = X.as_in_ctx(device), y.as_in_ctx(device)
        metric.add(d2l.accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

```{.python .input}
#@tab pytorch
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU計算模型在資料集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 設定為評估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正確預測的數量，總預測的數量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微調所需的（之後將介紹）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

```{.python .input}
#@tab paddle
def evaluate_accuracy_gpu(net, data_iter, device=None):     #@save
    """使用GPU計算模型在資料集上的精度"""
    if isinstance(net, nn.Layer):
        net.eval()  # 設定為評估模式
        if not device:
            device = next(iter(net.parameters())).place
    paddle.set_device("gpu:{}".format(str(device)[-2]))
    # 正確預測的數量，總預測的數量
    metric = d2l.Accumulator(2)
    with paddle.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微調所需的
                X = [paddle.to_tensor(x, place=device) for x in X]
            else:
                X = paddle.to_tensor(X, place=device)
            y = paddle.to_tensor(y, place=device)
            metric.add(d2l.accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

[**為了使用GPU，我們還需要一點小改動**]。
與 :numref:`sec_softmax_scratch`中定義的`train_epoch_ch3`不同，在進行正向和反向傳播之前，我們需要將每一小批次資料移動到我們指定的裝置（例如GPU）上。

如下所示，訓練函式`train_ch6`也類似於 :numref:`sec_softmax_scratch`中定義的`train_ch3`。
由於我們將實現多層神經網路，因此我們將主要使用高階API。
以下訓練函式假定從高階API建立的模型作為輸入，並進行相應的最佳化。
我們使用在 :numref:`subsec_xavier`中介紹的Xavier隨機初始化模型引數。
與全連線層一樣，我們使用交叉熵損失函式和小批次隨機梯度下降。

```{.python .input}
#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU訓練模型(在第六章定義)"""
    net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(),
                            'sgd', {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # 訓練損失之和，訓練準確率之和，樣本數
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            # 下面是與『d2l.train_epoch_ch3』的主要不同
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
```

```{.python .input}
#@tab pytorch
#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU訓練模型(在第六章定義)"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 訓練損失之和，訓練準確率之和，樣本數
        metric = d2l.Accumulator(3)  
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
```

```{.python .input}
#@tab tensorflow
class TrainCallback(tf.keras.callbacks.Callback):  #@save
    """一個以視覺化的訓練進展的回呼"""
    def __init__(self, net, train_iter, test_iter, num_epochs, device_name):
        self.timer = d2l.Timer()
        self.animator = d2l.Animator(
            xlabel='epoch', xlim=[1, num_epochs], legend=[
                'train loss', 'train acc', 'test acc'])
        self.net = net
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.num_epochs = num_epochs
        self.device_name = device_name

    def on_epoch_begin(self, epoch, logs=None):
        self.timer.start()

    def on_epoch_end(self, epoch, logs):
        self.timer.stop()
        test_acc = self.net.evaluate(
            self.test_iter, verbose=0, return_dict=True)['accuracy']
        metrics = (logs['loss'], logs['accuracy'], test_acc)
        self.animator.add(epoch + 1, metrics)
        if epoch == self.num_epochs - 1:
            batch_size = next(iter(self.train_iter))[0].shape[0]
            num_examples = batch_size * tf.data.experimental.cardinality(
                self.train_iter).numpy()
            print(f'loss {metrics[0]:.3f}, train acc {metrics[1]:.3f}, '
                  f'test acc {metrics[2]:.3f}')
            print(f'{num_examples / self.timer.avg():.1f} examples/sec on '
                  f'{str(self.device_name)}')

#@save
def train_ch6(net_fn, train_iter, test_iter, num_epochs, lr, device):
    """用GPU訓練模型(在第六章定義)"""
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    net.fit(train_iter, epochs=num_epochs, verbose=0, callbacks=[callback])
    return net
```

```{.python .input}
#@tab paddle
#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU訓練模型(在第六章定義)"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2D:
            nn.initializer.XavierUniform(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = paddle.optimizer.SGD(learning_rate=lr, parameters=net.parameters())
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 訓練損失之和，訓練準確率之和，樣本數
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.clear_grad()
            X, y = paddle.to_tensor(X, place=device), paddle.to_tensor(y, place=device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with paddle.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
```

現在，我們[**訓練和評估LeNet-5模型**]。

```{.python .input}
#@tab all
lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## 小結

* 卷積神經網路（CNN）是一類使用卷積層的網路。
* 在卷積神經網路中，我們組合使用卷積層、非線性啟用函式和彙聚層。
* 為了構造高效能的卷積神經網路，我們通常對卷積層進行排列，逐漸降低其表示的空間解析度，同時增加通道數。
* 在傳統的卷積神經網路中，卷積區塊編碼得到的表徵在輸出之前需由一個或多個全連線層進行處理。
* LeNet是最早發布的卷積神經網路之一。

## 練習

1. 將平均彙聚層替換為最大彙聚層，會發生什麼？
1. 嘗試建立一個基於LeNet的更復雜的網路，以提高其準確性。
    1. 調整卷積視窗大小。
    1. 調整輸出通道的數量。
    1. 調整啟用函式（如ReLU）。
    1. 調整卷積層的數量。
    1. 調整全連線層的數量。
    1. 調整學習率和其他訓練細節（例如，初始化和輪數）。
1. 在MNIST資料集上嘗試以上改進的網路。
1. 顯示不同輸入（例如毛衣和外套）時，LeNet第一層和第二層的啟用值。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1861)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1860)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1859)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11787)
:end_tab:
