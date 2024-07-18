# 學習率排程器
:label:`sec_scheduler`

到目前為止，我們主要關注如何更新權重向量的最佳化演算法，而不是它們的更新速率。
然而，調整學習率通常與實際演算法同樣重要，有如下幾方面需要考慮：

* 首先，學習率的大小很重要。如果它太大，最佳化就會發散；如果它太小，訓練就會需要過長時間，或者我們最終只能得到次優的結果。我們之前看到問題的條件數很重要（有關詳細訊息，請參見 :numref:`sec_momentum`）。直觀地說，這是最不敏感與最敏感方向的變化量的比率。
* 其次，衰減速率同樣很重要。如果學習率持續過高，我們可能最終會在最小值附近彈跳，從而無法達到最優解。 :numref:`sec_minibatch_sgd`比較詳細地討論了這一點，在 :numref:`sec_sgd`中我們則分析了效能保證。簡而言之，我們希望速率衰減，但要比$\mathcal{O}(t^{-\frac{1}{2}})$慢，這樣能成為解決凸問題的不錯選擇。
* 另一個同樣重要的方面是初始化。這既涉及引數最初的設定方式（詳情請參閱 :numref:`sec_numerical_stability`），又關係到它們最初的演變方式。這被戲稱為*預熱*（warmup），即我們最初開始向著解決方案邁進的速度有多快。一開始的大步可能沒有好處，特別是因為最初的引數集是隨機的。最初的更新方向可能也是毫無意義的。
* 最後，還有許多最佳化變體可以執行週期性學習率調整。這超出了本章的範圍，我們建議讀者閱讀 :cite:`Izmailov.Podoprikhin.Garipov.ea.2018`來了解個中細節。例如，如何透過對整個路徑引數求平均值來獲得更好的解。

鑑於管理學習率需要很多細節，因此大多數深度學習框架都有自動應對這個問題的工具。
在本章中，我們將梳理不同的排程策略對準確性的影響，並展示如何透過*學習率排程器*（learning rate scheduler）來有效管理。

## 一個簡單的問題

我們從一個簡單的問題開始，這個問題可以輕鬆計算，但足以說明要義。
為此，我們選擇了一個稍微現代化的LeNet版本（啟用函式使用`relu`而不是`sigmoid`，彙聚層使用最大彙聚層而不是平均彙聚層），並應用於Fashion-MNIST資料集。
此外，我們混合網路以提高效能。
由於大多數程式碼都是標準的，我們只介紹基礎知識，而不做進一步的詳細討論。如果需要，請參閱 :numref:`chap_cnn`進行複習。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, lr_scheduler, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.HybridSequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10))
net.hybridize()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 程式碼幾乎與d2l.train_ch6定義在卷積神經網路一章LeNet一節中的相同
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device):
    net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss,train_acc,num_examples
        for i, (X, y) in enumerate(train_iter):
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.optim import lr_scheduler

def net_fn():
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))

    return model

loss = nn.CrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 程式碼幾乎與d2l.train_ch6定義在卷積神經網路一章LeNet一節中的相同
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
          scheduler=None):
    net.to(device)
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss,train_acc,num_examples
        for i, (X, y) in enumerate(train_iter):
            net.train()
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))

        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # UsingPyTorchIn-Builtscheduler
                scheduler.step()
            else:
                # Usingcustomdefinedscheduler
                for param_group in trainer.param_groups:
                    param_group['lr'] = scheduler(epoch)

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import math
from tensorflow.keras.callbacks import LearningRateScheduler

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='relu'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 程式碼幾乎與d2l.train_ch6定義在卷積神經網路一章LeNet一節中的相同
def train(net_fn, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu(), custom_callback = False):
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = d2l.TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    if custom_callback is False:
        net.fit(train_iter, epochs=num_epochs, verbose=0,
                callbacks=[callback])
    else:
         net.fit(train_iter, epochs=num_epochs, verbose=0,
                 callbacks=[callback, custom_callback])
    return net
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import math
import paddle
from paddle import nn
from paddle.optimizer import lr as lr_scheduler

def net_fn():
    model = nn.Sequential(
        nn.Conv2D(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2D(kernel_size=2, stride=2),
        nn.Conv2D(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2D(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))

    return model

loss = nn.CrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 程式碼幾乎與d2l.train_ch6定義在卷積神經網路一章LeNet一節中的相同
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
          scheduler=None):
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss,train_acc,num_examples
        for i, (X, y) in enumerate(train_iter):
            net.train()
            trainer.clear_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with paddle.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat,y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))

        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # UsingPaddleIn-Builtscheduler
                scheduler.step()
            else:
                # Usingcustomdefinedscheduler
                trainer.set_lr(scheduler(epoch))
    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, 'f'test acc {test_acc:.3f}')
```

讓我們來看看如果使用預設設定，呼叫此演算法會發生什麼。
例如設學習率為$0.3$並訓練$30$次迭代。
留意在超過了某點、測試準確度方面的進展停滯時，訓練準確度將如何繼續提高。
兩條曲線之間的間隙表示過擬合。

```{.python .input}
lr, num_epochs = 0.3, 30
net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.3, 30
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab tensorflow
lr, num_epochs = 0.3, 30
train(net, train_iter, test_iter, num_epochs, lr)
```

```{.python .input}
#@tab paddle
lr, num_epochs = 0.3, 30
net = net_fn()
trainer = paddle.optimizer.SGD(learning_rate=lr, parameters=net.parameters())
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

## 學習率排程器

我們可以在每個迭代輪數（甚至在每個小批次）之後向下調整學習率。
例如，以動態的方式來回應最佳化的進展情況。

```{.python .input}
trainer.set_learning_rate(0.1)
print(f'learning rate is now {trainer.learning_rate:.2f}')
```

```{.python .input}
#@tab pytorch
lr = 0.1
trainer.param_groups[0]["lr"] = lr
print(f'learning rate is now {trainer.param_groups[0]["lr"]:.2f}')
```

```{.python .input}
#@tab tensorflow
lr = 0.1
dummy_model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
dummy_model.compile(tf.keras.optimizers.SGD(learning_rate=lr), loss='mse')
print(f'learning rate is now ,', dummy_model.optimizer.lr.numpy())
```

```{.python .input}
#@tab paddle
lr = 0.1
trainer.set_lr(lr)
print(f'learning rate is now {trainer.get_lr():.2f}')
```

更通常而言，我們應該定義一個排程器。
當呼叫更新次數時，它將回傳學習率的適當值。
讓我們定義一個簡單的方法，將學習率設定為$\eta = \eta_0 (t + 1)^{-\frac{1}{2}}$。

```{.python .input}
#@tab all
class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)
```

讓我們在一系列值上繪製它的行為。

```{.python .input}
#@tab all
scheduler = SquareRootScheduler(lr=0.1)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

現在讓我們來看看這對在Fashion-MNIST資料集上的訓練有何影響。
我們只是提供排程器作為訓練演算法的額外引數。

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

```{.python .input}
#@tab paddle
net = net_fn()
trainer = paddle.optimizer.SGD(learning_rate=lr , parameters=net.parameters())
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

這比以前好一些：曲線比以前更加平滑，並且過擬合更小了。
遺憾的是，關於為什麼在理論上某些策略會導致較輕的過擬合，有一些觀點認為，較小的步長將導致引數更接近零，因此更簡單。
但是，這並不能完全解釋這種現象，因為我們並沒有真正地提前停止，而只是輕柔地降低了學習率。

## 策略

雖然我們不可能涵蓋所有型別的學習率排程器，但我們會嘗試在下面簡要概述常用的策略：多項式衰減和分段常數表。
此外，餘弦學習率排程在實踐中的一些問題上執行效果很好。
在某些問題上，最好在使用較高的學習率之前預熱最佳化器。

### 單因子排程器

多項式衰減的一種替代方案是乘法衰減，即$\eta_{t+1} \leftarrow \eta_t \cdot \alpha$其中$\alpha \in (0, 1)$。
為了防止學習率衰減到一個合理的下界之下，
更新方程經常修改為$\eta_{t+1} \leftarrow \mathop{\mathrm{max}}(\eta_{\mathrm{min}}, \eta_t \cdot \alpha)$。

```{.python .input}
#@tab all
class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
d2l.plot(d2l.arange(50), [scheduler(t) for t in range(50)])
```

接下來，我們將使用內建的排程器，但在這裡僅解釋它們的功能。

### 多因子排程器

訓練深度網路的常見策略之一是保持學習率為一組分段的常數，並且不時地按給定的引數對學習率做乘法衰減。
具體地說，給定一組降低學習率的時間點，例如$s = \{5, 10, 20\}$，
每當$t \in s$時，降低$\eta_{t+1} \leftarrow \eta_t \cdot \alpha$。
假設每步中的值減半，我們可以按如下方式實現這一點。

```{.python .input}
scheduler = lr_scheduler.MultiFactorScheduler(step=[15, 30], factor=0.5,
                                              base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)

def get_lr(trainer, scheduler):
    lr = scheduler.get_last_lr()[0]
    trainer.step()
    scheduler.step()
    return lr

d2l.plot(d2l.arange(num_epochs), [get_lr(trainer, scheduler)
                                  for t in range(num_epochs)])
```

```{.python .input}
#@tab tensorflow
class MultiFactorScheduler:
    def __init__(self, step, factor, base_lr):
        self.step = step
        self.factor = factor
        self.base_lr = base_lr

    def __call__(self, epoch):
        if epoch in self.step:
            self.base_lr = self.base_lr * self.factor
            return self.base_lr
        else:
            return self.base_lr

scheduler = MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab paddle
net = net_fn()
scheduler =paddle.optimizer.lr.MultiStepDecay(learning_rate=0.5, milestones=[15,30], gamma=0.5)
trainer = paddle.optimizer.SGD(learning_rate=scheduler, parameters=net.parameters())
def get_lr(trainer, scheduler):
    lr=trainer.state_dict()['LR_Scheduler']['last_lr']
    trainer.step()
    scheduler.step()
    return lr

d2l.plot(paddle.arange(num_epochs), [get_lr(trainer, scheduler)
                                  for t in range(num_epochs)])
```

這種分段恆定學習率排程背後的直覺是，讓最佳化持續進行，直到權重向量的分佈達到一個駐點。
此時，我們才將學習率降低，以獲得更高質量的代理來達到一個良好的區域性最小值。
下面的例子展示了如何使用這種方法產生更好的解決方案。

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

```{.python .input}
#@tab paddle
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

### 餘弦排程器

餘弦排程器是 :cite:`Loshchilov.Hutter.2016`提出的一種啟發式演算法。
它所依據的觀點是：我們可能不想在一開始就太大地降低學習率，而且可能希望最終能用非常小的學習率來『改進』解決方案。
這產生了一個類似於餘弦的排程，函式形式如下所示，學習率的值在$t \in [0, T]$之間。

$$\eta_t = \eta_T + \frac{\eta_0 - \eta_T}{2} \left(1 + \cos(\pi t/T)\right)$$

這裡$\eta_0$是初始學習率，$\eta_T$是當$T$時的目標學習率。
此外，對於$t > T$，我們只需將值固定到$\eta_T$而不再增加它。
在下面的範例中，我們設定了最大更新步數$T = 20$。

```{.python .input}
scheduler = lr_scheduler.CosineScheduler(max_update=20, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow, paddle
class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

在電腦視覺的背景下，這個排程方式可能產生改進的結果。
但請注意，如下所示，這種改進並不一定成立。

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

```{.python .input}
#@tab paddle
net = net_fn()
trainer = paddle.optimizer.SGD(learning_rate=0.3, parameters=net.parameters())
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

### 預熱

在某些情況下，初始化引數不足以得到良好的解。
這對某些高階網路設計來說尤其棘手，可能導致不穩定的最佳化結果。
對此，一方面，我們可以選擇一個足夠小的學習率，
從而防止一開始發散，然而這樣進展太緩慢。
另一方面，較高的學習率最初就會導致發散。

解決這種困境的一個相當簡單的解決方法是使用預熱期，在此期間學習率將增加至初始最大值，然後冷卻直到最佳化過程結束。
為了簡單起見，通常使用線性遞增。
這引出瞭如下表所示的時間表。

```{.python .input}
scheduler = lr_scheduler.CosineScheduler(20, warmup_steps=5, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(np.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow, paddle
scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

注意，觀察前5個迭代輪數的效能，網路最初收斂得更好。

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

```{.python .input}
#@tab paddle
net = net_fn()
trainer = paddle.optimizer.SGD(learning_rate=0.3, parameters=net.parameters())
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

預熱可以應用於任何排程器，而不僅僅是餘弦。
有關學習率排程的更多實驗和更詳細討論，請參閱 :cite:`Gotmare.Keskar.Xiong.ea.2018`。
其中，這篇論文的點睛之筆的發現：預熱階段限制了非常深的網路中引數的發散程度 。
這在直覺上是有道理的：在網路中那些一開始花費最多時間取得進展的部分，隨機初始化會產生巨大的發散。

## 小結

* 在訓練期間逐步降低學習率可以提高準確性，並且減少模型的過擬合。
* 在實驗中，每當進展趨於穩定時就降低學習率，這是很有效的。從本質上說，這可以確保我們有效地收斂到一個適當的解，也只有這樣才能透過降低學習率來減小引數的固有方差。
* 餘弦排程器在某些電腦視覺問題中很受歡迎。
* 最佳化之前的預熱期可以防止發散。
* 最佳化在深度學習中有多種用途。對於同樣的訓練誤差而言，選擇不同的最佳化演算法和學習率排程，除了最大限度地減少訓練時間，可以導致測試集上不同的泛化和過擬合量。

## 練習

1. 試驗給定固定學習率的最佳化行為。這種情況下可以獲得的最佳模型是什麼？
1. 如果改變學習率下降的指數，收斂性會如何改變？在實驗中方便起見，使用`PolyScheduler`。
1. 將餘弦排程器應用於大型計算機視覺問題，例如訓練ImageNet資料集。與其他排程器相比，它如何影響效能？
1. 預熱應該持續多長時間？
1. 可以試著把最佳化和取樣聯絡起來嗎？首先，在隨機梯度朗之萬動力學上使用 :cite:`Welling.Teh.2011`的結果。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/4333)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/4334)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/4335)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11856)
:end_tab:
