# 多GPU的簡潔實現
:label:`sec_multi_gpu_concise`

每個新模型的並行計算都從零開始實現是無趣的。此外，最佳化同步工具以獲得高效能也是有好處的。下面我們將展示如何使用深度學習框架的高階API來實現這一點。數學和演算法與 :numref:`sec_multi_gpu`中的相同。本節的程式碼至少需要兩個GPU來執行。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
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

## [**簡單網路**]

讓我們使用一個比 :numref:`sec_multi_gpu`的LeNet更有意義的網路，它依然能夠容易地和快速地訓練。我們選擇的是 :cite:`He.Zhang.Ren.ea.2016`中的ResNet-18。因為輸入的影象很小，所以稍微修改了一下。與 :numref:`sec_resnet`的區別在於，我們在開始時使用了更小的卷積核、步長和填充，而且刪除了最大彙聚層。

```{.python .input}
#@save
def resnet18(num_classes):
    """稍加修改的ResNet-18模型"""
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(d2l.Residual(
                    num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(d2l.Residual(num_channels))
        return blk

    net = nn.Sequential()
    # 該模型使用了更小的卷積核、步長和填充，而且刪除了最大彙聚層
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

```{.python .input}
#@tab pytorch
#@save
def resnet18(num_classes, in_channels=1):
    """稍加修改的ResNet-18模型"""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(in_channels, out_channels,
                                        use_1x1conv=True, strides=2))
            else:
                blk.append(d2l.Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # 該模型使用了更小的卷積核、步長和填充，而且刪除了最大彙聚層
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(
        64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net
```

```{.python .input}
#@tab paddle
#@save
def resnet18(num_classes, in_channels=1):
    """稍加修改的ResNet-18模型"""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(in_channels, out_channels,
                                        use_1x1conv=True, strides=2))
            else:
                blk.append(d2l.Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # 該模型使用了更小的卷積核、步長和填充，而且刪除了最大彙聚層
    net = nn.Sequential(
        nn.Conv2D(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2D(64),
        nn.ReLU())
    net.add_sublayer("resnet_block1", resnet_block(
        64, 64, 2, first_block=True))
    net.add_sublayer("resnet_block2", resnet_block(64, 128, 2))
    net.add_sublayer("resnet_block3", resnet_block(128, 256, 2))
    net.add_sublayer("resnet_block4", resnet_block(256, 512, 2))
    net.add_sublayer("global_avg_pool", nn.AdaptiveAvgPool2D((1, 1)))
    net.add_sublayer("fc", nn.Sequential(nn.Flatten(),
                                         nn.Linear(512, num_classes)))
    return net
```

## 網路初始化

:begin_tab:`mxnet`
`initialize`函式允許我們在所選裝置上初始化引數。請參閱 :numref:`sec_numerical_stability`複習初始化方法。這個函式在多個裝置上初始化網路時特別方便。下面在實踐中試一試它的運作方式。
:end_tab:

:begin_tab:`pytorch`
我們將在訓練迴路中初始化網路。請參見 :numref:`sec_numerical_stability`複習初始化方法。
:end_tab:

```{.python .input}
net = resnet18(10)
# 獲取GPU清單
devices = d2l.try_all_gpus()
# 初始化網路的所有引數
net.initialize(init=init.Normal(sigma=0.01), ctx=devices)
```

```{.python .input}
#@tab pytorch
net = resnet18(10)
# 獲取GPU清單
devices = d2l.try_all_gpus()
# 我們將在訓練程式碼實現中初始化網路
```

```{.python .input}
#@tab paddle
net = resnet18(10)
# 獲取GPU清單
devices = d2l.try_all_gpus()
# 我們將在訓練程式碼實現中初始化網路
```

:begin_tab:`mxnet`
使用 :numref:`sec_multi_gpu`中引入的`split_and_load`函式可以切分一個小批次資料，並將切分後的分塊資料複製到`devices`變數提供的裝置清單中。網路實例自動使用適當的GPU來計算前向傳播的值。我們將在下面生成$4$個觀測值，並在GPU上將它們拆分。
:end_tab:

```{.python .input}
x = np.random.uniform(size=(4, 1, 28, 28))
x_shards = gluon.utils.split_and_load(x, devices)
net(x_shards[0]), net(x_shards[1])
```

:begin_tab:`mxnet`
一旦資料透過網路，網路對應的引數就會在*有數據透過的裝置上初始化*。這意味著初始化是基於每個裝置進行的。由於我們選擇的是GPU0和GPU1，所以網路只在這兩個GPU上初始化，而不是在CPU上初始化。事實上，CPU上甚至沒有這些引數。我們可以透過列印引數和觀察可能出現的任何錯誤來驗證這一點。
:end_tab:

```{.python .input}
weight = net[0].params.get('weight')

try:
    weight.data()
except RuntimeError:
    print('not initialized on cpu')
weight.data(devices[0])[0], weight.data(devices[1])[0]
```

:begin_tab:`mxnet`
接下來，讓我們使用[**在多個裝置上並行工作**]的程式碼來替換前面的[**評估模型**]的程式碼。
這裡主要是 :numref:`sec_lenet`的`evaluate_accuracy_gpu`函式的替代，程式碼的主要區別在於在呼叫網路之前拆分了一個小批次，其他在本質上是一樣的。
:end_tab:

```{.python .input}
#@save
def evaluate_accuracy_gpus(net, data_iter, split_f=d2l.split_batch):
    """使用多個GPU計算資料集上模型的精度"""
    # 查詢裝置清單
    devices = list(net.collect_params().values())[0].list_ctx()
    # 正確預測的數量，預測的總數量
    metric = d2l.Accumulator(2)
    for features, labels in data_iter:
        X_shards, y_shards = split_f(features, labels, devices)
        # 並行執行
        pred_shards = [net(X_shard) for X_shard in X_shards]
        metric.add(sum(float(d2l.accuracy(pred_shard, y_shard)) for
                       pred_shard, y_shard in zip(
                           pred_shards, y_shards)), labels.size)
    return metric[0] / metric[1]
```

## [**訓練**]

如前所述，用於訓練的程式碼需要執行幾個基本功能才能實現高效並行：

* 需要在所有裝置上初始化網路引數；
* 在資料集上迭代時，要將小批次資料分配到所有裝置上；
* 跨裝置並行計算損失及其梯度；
* 聚合梯度，並相應地更新引數。

最後，並行地計算精確度和發布網路的最終效能。除了需要拆分和聚合資料外，訓練程式碼與前幾章的實現非常相似。

```{.python .input}
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    ctx = [d2l.try_gpu(i) for i in range(num_gpus)]
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        timer.start()
        for features, labels in train_iter:
            X_shards, y_shards = d2l.split_batch(features, labels, ctx)
            with autograd.record():
                ls = [loss(net(X_shard), y_shard) for X_shard, y_shard
                      in zip(X_shards, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        npx.waitall()
        timer.stop()
        animator.add(epoch + 1, (evaluate_accuracy_gpus(net, test_iter),))
    print(f'測試精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/輪，'
          f'在{str(ctx)}')
```

```{.python .input}
#@tab pytorch
def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)
    # 在多個GPU上設定模型
    net = nn.DataParallel(net, device_ids=devices)
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'測試精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/輪，'
          f'在{str(devices)}')
```

```{.python .input}
#@tab paddle
def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    
    init_normal = nn.initializer.Normal(mean=0.0, std=0.01)
    for i in net.sublayers():
        if type(i) in [nn.Linear, nn.Conv2D]:        
            init_normal(i.weight)

    # 在多個 GPU 上設定模型
    net = paddle.DataParallel(net)
    trainer = paddle.optimizer.SGD(parameters=net.parameters(), learning_rate=lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.clear_grad()
            X, y = paddle.to_tensor(X, place=devices[0]), paddle.to_tensor(y, place=devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'測試精度：{animator.Y[0][-1]:.2f}, {timer.avg():.1f}秒/輪，'
          f'在{str(devices)}')
```

接下來看看這在實踐中是如何運作的。我們先[**在單個GPU上訓練網路**]進行預熱。

```{.python .input}
train(num_gpus=1, batch_size=256, lr=0.1)
```

```{.python .input}
#@tab pytorch, paddle
train(net, num_gpus=1, batch_size=256, lr=0.1)
```

接下來我們[**使用2個GPU進行訓練**]。與 :numref:`sec_multi_gpu`中評估的LeNet相比，ResNet-18的模型要複雜得多。這就是顯示並行化優勢的地方，計算所需時間明顯大於同步引數需要的時間。因為並行化開銷的相關性較小，因此這種操作提高了模型的可伸縮性。

```{.python .input}
train(num_gpus=2, batch_size=512, lr=0.2)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=2, batch_size=512, lr=0.2)
```

## 小結

:begin_tab:`mxnet`
* Gluon透過提供一個上下文清單，為跨多個裝置的模型初始化提供原語。
* 神經網路可以在（可找到資料的）單GPU上進行自動評估。
* 每台裝置上的網路需要先初始化，然後再嘗試訪問該裝置上的引數，否則會遇到錯誤。
* 最佳化演算法在多個GPU上自動聚合。
:end_tab:

:begin_tab:`pytorch, paddle`
* 神經網路可以在（可找到資料的）單GPU上進行自動評估。
* 每台裝置上的網路需要先初始化，然後再嘗試訪問該裝置上的引數，否則會遇到錯誤。
* 最佳化演算法在多個GPU上自動聚合。
:end_tab:

## 練習

:begin_tab:`mxnet`
1. 本節使用ResNet-18，請嘗試不同的迭代週期數、批次大小和學習率，以及使用更多的GPU進行計算。如果使用$16$個GPU（例如，在AWS p2.16xlarge實例上）嘗試此操作，會發生什麼？
1. 有時候不同的裝置提供了不同的計算能力，我們可以同時使用GPU和CPU，那應該如何分配工作？為什麼？
1. 如果去掉`npx.waitall()`會怎樣？該如何修改訓練，以使並行操作最多有兩個步驟重疊？
:end_tab:

:begin_tab:`pytorch, paddle`
1. 本節使用ResNet-18，請嘗試不同的迭代週期數、批次大小和學習率，以及使用更多的GPU進行計算。如果使用$16$個GPU（例如，在AWS p2.16xlarge實例上）嘗試此操作，會發生什麼？
1. 有時候不同的裝置提供了不同的計算能力，我們可以同時使用GPU和CPU，那應該如何分配工作？為什麼？
:end_tab:

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2804)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2803)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11861)
:end_tab: