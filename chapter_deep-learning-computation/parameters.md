# 引數管理

在選擇了架構並設定了超引數後，我們就進入了訓練階段。
此時，我們的目標是找到使損失函式最小化的模型引數值。
經過訓練後，我們將需要使用這些引數來做出未來的預測。
此外，有時我們希望提取引數，以便在其他環境中複用它們，
將模型儲存下來，以便它可以在其他軟體中執行，
或者為了獲得科學的理解而進行檢查。

之前的介紹中，我們只依靠深度學習框架來完成訓練的工作，
而忽略了操作引數的具體細節。
本節，我們將介紹以下內容：

* 訪問引數，用於除錯、診斷和視覺化；
* 引數初始化；
* 在不同模型元件間共享引數。

(**我們首先看一下具有單隱藏層的多層感知機。**)

```{.python .input}
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # 使用預設初始化方法

X = np.random.uniform(size=(2, 4))
net(X)  # 正向傳播
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X)
```

```{.python .input}
#@tab paddle
import warnings
warnings.filterwarnings(action='ignore')
import paddle
from paddle import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = paddle.rand([2, 4])
net(X)
```

## [**引數訪問**]

我們從已有模型中訪問引數。
當透過`Sequential`類別定義模型時，
我們可以透過索引來訪問模型的任意層。
這就像模型是一個清單一樣，每層的引數都在其屬性中。
如下所示，我們可以檢查第二個全連線層的引數。

```{.python .input}
print(net[1].params)
```

```{.python .input}
#@tab pytorch, paddle
print(net[2].state_dict())
```

```{.python .input}
#@tab tensorflow
print(net.layers[2].weights)
```

輸出的結果告訴我們一些重要的事情：
首先，這個全連線層包含兩個引數，分別是該層的權重和偏置。
兩者都儲存為單精度浮點數（float32）。
注意，引數名稱允許唯一標識每個引數，即使在包含數百個層的網路中也是如此。

### [**目標引數**]

注意，每個引數都表示為引數類別的一個實例。
要對引數執行任何操作，首先我們需要訪問底層的數值。
有幾種方法可以做到這一點。有些比較簡單，而另一些則比較通用。
下面的程式碼從第二個全連線層（即第三個神經網路層）提取偏置，
提取後回傳的是一個引數類別實例，並進一步訪問該引數的值。

```{.python .input}
print(type(net[1].bias))
print(net[1].bias)
print(net[1].bias.data())
```

```{.python .input}
#@tab pytorch
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
```

```{.python .input}
#@tab tensorflow
print(type(net.layers[2].weights[1]))
print(net.layers[2].weights[1])
print(tf.convert_to_tensor(net.layers[2].weights[1]))
```

```{.python .input}
#@tab paddle
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.value)
```

:begin_tab:`mxnet,pytorch,paddle`
引數是複合的物件，包含值、梯度和額外訊息。
這就是我們需要顯式引數值的原因。
除了值之外，我們還可以訪問每個引數的梯度。
在上面這個網路中，由於我們還沒有呼叫反向傳播，所以引數的梯度處於初始狀態。
:end_tab:

```{.python .input}
net[1].weight.grad()
```

```{.python .input}
#@tab pytorch, paddle
net[2].weight.grad == None
```

### [**一次性訪問所有引數**]

當我們需要對所有引數執行操作時，逐個訪問它們可能會很麻煩。
當我們處理更復雜的區塊（例如，巢狀區塊）時，情況可能會變得特別複雜，
因為我們需要遞迴整個樹來提取每個子區塊的引數。
下面，我們將透過示範來比較訪問第一個全連線層的引數和訪問所有層。

```{.python .input}
print(net[0].collect_params())
print(net.collect_params())
```

```{.python .input}
#@tab pytorch, paddle
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
```

```{.python .input}
#@tab tensorflow
print(net.layers[1].weights)
print(net.get_weights())
```

這為我們提供了另一種訪問網路引數的方式，如下所示。

```{.python .input}
net.collect_params()['dense1_bias'].data()
```

```{.python .input}
#@tab pytorch
net.state_dict()['2.bias'].data
```

```{.python .input}
#@tab tensorflow
net.get_weights()[1]
```

```{.python .input}
#@tab paddle
net.state_dict()['2.bias']
```

### [**從巢狀區塊收集引數**]

讓我們看看，如果我們將多個區塊相互巢狀，引數命名約定是如何工作的。
我們首先定義一個生成區塊的函式（可以說是『區塊工廠』），然後將這些區塊組合到更大的塊中。

```{.python .input}
def block1():
    net = nn.Sequential()
    net.add(nn.Dense(32, activation='relu'))
    net.add(nn.Dense(16, activation='relu'))
    return net

def block2():
    net = nn.Sequential()
    for _ in range(4):
        # 在這裡巢狀
        net.add(block1())
    return net

rgnet = nn.Sequential()
rgnet.add(block2())
rgnet.add(nn.Dense(10))
rgnet.initialize()
rgnet(X)
```

```{.python .input}
#@tab pytorch
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在這裡巢狀
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
```

```{.python .input}
#@tab tensorflow
def block1(name):
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4, activation=tf.nn.relu)],
        name=name)

def block2():
    net = tf.keras.Sequential()
    for i in range(4):
        # 在這裡巢狀
        net.add(block1(name=f'block-{i}'))
    return net

rgnet = tf.keras.Sequential()
rgnet.add(block2())
rgnet.add(tf.keras.layers.Dense(1))
rgnet(X)
```

```{.python .input}
#@tab paddle
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), 
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在這裡巢狀
        net.add_sublayer(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
```

[**設計了網路後，我們看看它是如何工作的。**]

```{.python .input}
print(rgnet.collect_params)
print(rgnet.collect_params())
```

```{.python .input}
#@tab pytorch, paddle
print(rgnet)
```

```{.python .input}
#@tab tensorflow
print(rgnet.summary())
```

因為層是分層巢狀的，所以我們也可以像透過巢狀清單索引一樣訪問它們。
下面，我們訪問第一個主要的塊中、第二個子區塊的第一層的偏置項。

```{.python .input}
rgnet[0][1][0].bias.data()
```

```{.python .input}
#@tab pytorch
rgnet[0][1][0].bias.data
```

```{.python .input}
#@tab tensorflow
rgnet.layers[0].layers[1].layers[1].weights[1]
```

```{.python .input}
#@tab paddle
print(rgnet[0].state_dict()['block 0.0.bias'])
```

## 引數初始化

知道了如何訪問引數後，現在我們看看如何正確地初始化引數。
我們在 :numref:`sec_numerical_stability`中討論了良好初始化的必要性。
深度學習框架提供預設隨機初始化，
也允許我們建立自定義初始化方法，
滿足我們透過其他規則實現初始化權重。

:begin_tab:`mxnet`
預設情況下，MXNet透過初始化權重引數的方法是
從均勻分佈$U(-0.07, 0.07)$中隨機取樣權重，並將偏置引數設定為0。
MXNet的`init`模組提供了多種預置初始化方法。
:end_tab:

:begin_tab:`pytorch`
預設情況下，PyTorch會根據一個範圍均勻地初始化權重和偏置矩陣，
這個範圍是根據輸入和輸出維度計算出的。
PyTorch的`nn.init`模組提供了多種預置初始化方法。
:end_tab:

:begin_tab:`tensorflow`
預設情況下，Keras會根據一個範圍均勻地初始化權重矩陣，
這個範圍是根據輸入和輸出維度計算出的。
偏置引數設定為0。
TensorFlow在根模組和`keras.initializers`模組中提供了各種初始化方法。
:end_tab:

:begin_tab:`paddle`
預設情況下，PaddlePaddle會使用Xavier初始化權重矩陣，
偏置引數設定為0。
PaddlePaddle的`nn.initializer`模組提供了多種預置初始化方法。
:end_tab:

### [**內建初始化**]

讓我們首先呼叫內建的初始化器。
下面的程式碼將所有權重引數初始化為標準差為0.01的高斯隨機變數，
且將偏置引數設定為0。

```{.python .input}
# 這裡的force_reinit確保引數會被重新初始化，不論之前是否已經被初始化
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
#@tab pytorch
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1)])

net(X)
net.weights[0], net.weights[1]
```

```{.python .input}
#@tab paddle
def init_normal(m):
    if type(m) == nn.Linear:
        paddle.nn.initializer.Normal(mean=0.0, std=0.01)
        paddle.zeros(m.bias)    
net.apply(init_normal)
net[0].weight[0],net[0].state_dict()['bias']
```

我們還可以將所有引數初始化為給定的常數，比如初始化為1。

```{.python .input}
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
#@tab pytorch
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.Constant(1),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1),
])

net(X)
net.weights[0], net.weights[1]
```

```{.python .input}
#@tab paddle
def init_constant(m):
    if type(m) == nn.Linear:
        paddle.nn.initializer.Constant(value = 1)
        paddle.zeros(m.bias)
net.apply(init_constant)
net[0].weight[0],net[0].state_dict()['bias']
```

我們還可以[**對某些區塊應用不同的初始化方法**]。
例如，下面我們使用Xavier初始化方法初始化第一個神經網路層，
然後將第三個神經網路層初始化為常數值42。

```{.python .input}
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
print(net[0].weight.data()[0])
print(net[1].weight.data())
```

```{.python .input}
#@tab pytorch
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.Dense(
        1, kernel_initializer=tf.keras.initializers.Constant(1)),
])

net(X)
print(net.layers[1].weights[0])
print(net.layers[2].weights[0])
```

```{.python .input}
#@tab paddle
def xavier(m):
    if type(m) == nn.Linear:
        paddle.nn.initializer.XavierUniform(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        paddle.nn.initializer.Constant(42)
        
net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight[0])
print(net[2].weight)
```

### [**自定義初始化**]

有時，深度學習框架沒有提供我們需要的初始化方法。
在下面的例子中，我們使用以下的分佈為任意權重引數$w$定義初始化方法：

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \text{ 可能性 } \frac{1}{4} \\
            0    & \text{ 可能性 } \frac{1}{2} \\
        U(-10, -5) & \text{ 可能性 } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

:begin_tab:`mxnet`
在這裡，我們定義了`Initializer`類別的子類別。
通常，我們只需要實現`_init_weight`函式，
該函式接受張量引數（`data`）併為其分配所需的初始化值。
:end_tab:

:begin_tab:`pytorch`
同樣，我們實現了一個`my_init`函式來應用到`net`。
:end_tab:

:begin_tab:`tensorflow`
在這裡，我們定義了一個`Initializer`的子類別，
並實現了`__call__`函式。
該函式回傳給定形狀和數據型別的所需張量。
:end_tab:

:begin_tab:`paddle`
同樣，我們實現了一個`my_init`函式來應用到`net`。
:end_tab:

```{.python .input}
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[:2]
```

```{.python .input}
#@tab pytorch
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

```{.python .input}
#@tab tensorflow
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        data=tf.random.uniform(shape, -10, 10, dtype=dtype)
        factor=(tf.abs(data) >= 5)
        factor=tf.cast(factor, tf.float32)
        return data * factor

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=MyInit()),
    tf.keras.layers.Dense(1),
])

net(X)
print(net.layers[1].weights[0])
```

```{.python .input}
#@tab paddle
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) 
                        for name, param in m.named_parameters()][0])
        paddle.nn.initializer.XavierUniform(m.weight, -10, 10)
        h = paddle.abs(m.weight) >= 5
        h = paddle.to_tensor(h)
        m = paddle.to_tensor(m.weight)
        m *= h       

net.apply(my_init)
net[0].weight[:2]
```

注意，我們始終可以直接設定引數。

```{.python .input}
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
net[0].weight.data()[0]
```

```{.python .input}
#@tab pytorch
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

```{.python .input}
#@tab tensorflow
net.layers[1].weights[0][:].assign(net.layers[1].weights[0] + 1)
net.layers[1].weights[0][0, 0].assign(42)
net.layers[1].weights[0]
```

```{.python .input}
#@tab paddle
net[0].weight.set_value(net[0].weight.numpy() + 1)
val = net[0].weight.numpy()
val[0, 0] = 42
net[0].weight.set_value(val)
net[0].weight[0]
```

:begin_tab:`mxnet`
高階使用者請注意：如果要在`autograd`範圍內調整引數，
則需要使用`set_data`，以避免誤導自動微分機制。
:end_tab:

## [**引數繫結**]

有時我們希望在多個層間共享引數：
我們可以定義一個稠密層，然後使用它的引數來設定另一個層的引數。

```{.python .input}
net = nn.Sequential()
# 我們需要給共享層一個名稱，以便可以引用它的引數
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X)

# 檢查引數是否相同
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0, 0] = 100
# 確保它們實際上是同一個物件，而不只是有相同的值
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

```{.python .input}
#@tab pytorch
# 我們需要給共享層一個名稱，以便可以引用它的引數
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 檢查引數是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 確保它們實際上是同一個物件，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
```

```{.python .input}
#@tab tensorflow
# tf.keras的表現有點不同。它會自動刪除重複層
shared = tf.keras.layers.Dense(4, activation=tf.nn.relu)
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    shared,
    shared,
    tf.keras.layers.Dense(1),
])

net(X)
# 檢查引數是否不同
print(len(net.layers) == 3)
```

```{.python .input}
#@tab paddle
# 我們需要給共享層一個名稱，以便可以引用它的引數。
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 檢查引數是否相同
print(net[2].weight[0] == net[4].weight[0])
```

:begin_tab:`mxnet`
這個例子表明第二層和第三層的引數是繫結的。
它們不僅值相等，而且由相同的張量表示。
因此，如果我們改變其中一個引數，另一個引數也會改變。
這裡有一個問題：當引數繫結時，梯度會發生什麼情況？
答案是由於模型引數包含梯度，
因此在反向傳播期間第二個隱藏層和第三個隱藏層的梯度會加在一起。
:end_tab:

:begin_tab:`pytorch`
這個例子表明第三個和第五個神經網路層的引數是繫結的。
它們不僅值相等，而且由相同的張量表示。
因此，如果我們改變其中一個引數，另一個引數也會改變。
這裡有一個問題：當引數繫結時，梯度會發生什麼情況？
答案是由於模型引數包含梯度，因此在反向傳播期間第二個隱藏層
（即第三個神經網路層）和第三個隱藏層（即第五個神經網路層）的梯度會加在一起。
:end_tab:

## 小結

* 我們有幾種方法可以訪問、初始化和繫結模型引數。
* 我們可以使用自定義初始化方法。

## 練習

1. 使用 :numref:`sec_model_construction` 中定義的`FancyMLP`模型，訪問各個層的引數。
1. 檢視初始化模組文件以瞭解不同的初始化方法。
1. 建立包含共享引數層的多層感知機並對其進行訓練。在訓練過程中，觀察模型各層的引數和梯度。
1. 為什麼共享引數是個好主意？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1831)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1829)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1830)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11778)
:end_tab:
