# 層和區塊
:label:`sec_model_construction`

之前首次介紹神經網路時，我們關注的是具有單一輸出的線性模型。
在這裡，整個模型只有一個輸出。
注意，單個神經網路
（1）接受一些輸入；
（2）生成相應的標量輸出；
（3）具有一組相關 *引數*（parameters），更新這些引數可以最佳化某目標函式。

然後，當考慮具有多個輸出的網路時，
我們利用矢量化演算法來描述整層神經元。
像單個神經元一樣，層（1）接受一組輸入，
（2）生成相應的輸出，
（3）由一組可調整引數描述。
當我們使用softmax迴歸時，一個單層本身就是模型。
然而，即使我們隨後引入了多層感知機，我們仍然可以認為該模型保留了上面所說的基本架構。

對於多層感知機而言，整個模型及其組成層都是這種架構。
整個模型接受原始輸入（特徵），生成輸出（預測），
幷包含一些引數（所有組成層的引數集合）。
同樣，每個單獨的層接收輸入（由前一層提供），
生成輸出（到下一層的輸入），並且具有一組可調引數，
這些引數根據從下一層反向傳播的訊號進行更新。

事實證明，研究討論『比單個層大』但『比整個模型小』的元件更有價值。
例如，在電腦視覺中廣泛流行的ResNet-152架構就有數百層，
這些層是由*層組*（groups of layers）的重複模式組成。
這個ResNet架構贏得了2015年ImageNet和COCO電腦視覺比賽
的識別和檢測任務 :cite:`He.Zhang.Ren.ea.2016`。
目前ResNet架構仍然是許多視覺任務的首選架構。
在其他的領域，如自然語言處理和語音，
層組以各種重複模式排列的類似架構現在也是普遍存在。

為了實現這些複雜的網路，我們引入了神經網路*區塊*的概念。
*區塊*（block）可以描述單個層、由多個層組成的元件或整個模型本身。
使用區塊進行抽象的一個好處是可以將一些區塊組合成更大的元件，
這一過程通常是遞迴的，如 :numref:`fig_blocks`所示。
透過定義程式碼來按需生成任意複雜度的區塊，
我們可以透過簡潔的程式碼實現複雜的神經網路。

![多個層被組合成區塊，形成更大的模型](../img/blocks.svg)
:label:`fig_blocks`

從程式設計的角度來看，區塊由*類*（class）表示。
它的任何子類別都必須定義一個將其輸入轉換為輸出的前向傳播函式，
並且必須儲存任何必需的引數。
注意，有些區塊不需要任何引數。
最後，為了計算梯度，區塊必須具有反向傳播函式。
在定義我們自己的區塊時，由於自動微分（在 :numref:`sec_autograd` 中引入）
提供了一些後端實現，我們只需要考慮前向傳播函式和必需的引數。

在構造自定義區塊之前，(**我們先回顧一下多層感知機**)
（ :numref:`sec_mlp_concise` ）的程式碼。
下面的程式碼生成一個網路，其中包含一個具有256個單元和ReLU啟用函式的全連線隱藏層，
然後是一個具有10個隱藏單元且不帶啟用函式的全連線輸出層。

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
net(X)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])

X = tf.random.uniform((2, 20))
net(X)
```

```{.python .input}
#@tab paddle
import warnings
warnings.filterwarnings(action='ignore')
import paddle
from paddle import nn
from paddle.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = paddle.rand([2, 20])
net(X)
```

:begin_tab:`mxnet`
在這個例子中，我們透過實例化`nn.Sequential`來建立我們的模型，
回傳的物件賦給`net`變數。
接下來，我們反覆呼叫`net`變數的`add`函式，按照想要執行的順序新增層。
簡而言之，`nn.Sequential`定義了一種特殊型別的`Block`，
即在Gluon中表示區塊的類，它維護`Block`的有序清單。
`add`函式方便將每個連續的`Block`新增到清單中。
請注意，每層都是`Dense`類別的一個實例，`Dense`類本身就是`Block`的子類別。
到目前為止，我們一直在透過`net(X)`呼叫我們的模型來獲得模型的輸出。
這實際上是`net.forward(X)`的簡寫，
這是透過`Block`類別的`__call__`函式實現的一個Python技巧。
前向傳播（`forward`）函式非常簡單：它將清單中的每個`Block`連線在一起，
將每個`Block`的輸出作為輸入傳遞給下一層。

:end_tab:

:begin_tab:`pytorch`
在這個例子中，我們透過實例化`nn.Sequential`來建立我們的模型，
層的執行順序是作為引數傳遞的。
簡而言之，(**`nn.Sequential`定義了一種特殊的`Module`**)，
即在PyTorch中表示一個區塊的類，
它維護了一個由`Module`組成的有序清單。
注意，兩個全連線層都是`Linear`類別的實例，
`Linear`類本身就是`Module`的子類別。
另外，到目前為止，我們一直在透過`net(X)`呼叫我們的模型來獲得模型的輸出。
這實際上是`net.__call__(X)`的簡寫。
這個前向傳播函式非常簡單：
它將清單中的每個區塊連線在一起，將每個區塊的輸出作為下一個區塊的輸入。

:end_tab:

:begin_tab:`tensorflow`
在這個例子中，我們透過實例化`keras.models.Sequential`來建立我們的模型，
層的執行順序是作為引數傳遞的。
簡而言之，`Sequential`定義了一種特殊的`keras.Model`，
即在Keras中表示一個區塊的類。
它維護了一個由`Model`組成的有序清單，
注意兩個全連線層都是`Model`類別的實例，
這個類本身就是`Model`的子類別。
前向傳播（`call`）函式也非常簡單：
它將清單中的每個區塊連線在一起，將每個區塊的輸出作為下一個區塊的輸入。
注意，到目前為止，我們一直在透過`net(X)`呼叫我們的模型來獲得模型的輸出。
這實際上是`net.call(X)`的簡寫，
這是透過Block類別的`__call__`函式實現的一個Python技巧。
:end_tab:

:begin_tab:`paddle`
在這個例子中，我們透過實例化`nn.Sequential`來建立我們的模型，
層的執行順序是作為引數傳遞的。
簡而言之，(**`nn.Sequential`定義了一種特殊的`Layer`**)，
即在PaddlePaddle中表示一個區塊的類，
它維護了一個由`Layer`組成的有序清單。
注意，兩個全連線層都是`Linear`類別的實例，
`Linear`類本身就是`Layer`的子類別。
另外，到目前為止，我們一直在透過`net(X)`呼叫我們的模型來獲得模型的輸出。
這實際上是`net.__call__(X)`的簡寫。
這個前向傳播函式非常簡單：
它將清單中的每個區塊連線在一起，將每個區塊的輸出作為下一個區塊的輸入。
:end_tab:

## [**自定義區塊**]

要想直觀地瞭解區塊是如何工作的，最簡單的方法就是自己實現一個。
在實現我們自定義區塊之前，我們簡要總結一下每個區塊必須提供的基本功能。

:begin_tab:`mxnet, tensorflow`
1. 將輸入資料作為其前向傳播函式的引數。
1. 透過前向傳播函式來生成輸出。請注意，輸出的形狀可能與輸入的形狀不同。例如，我們上面模型中的第一個全連線的層接收任意維的輸入，但是回傳一個維度256的輸出。
1. 計算其輸出關於輸入的梯度，可透過其反向傳播函式進行訪問。通常這是自動發生的。
1. 儲存和訪問前向傳播計算所需的引數。
1. 根據需要初始化模型引數。
:end_tab:

:begin_tab:`pytorch, paddle`
1. 將輸入資料作為其前向傳播函式的引數。
1. 透過前向傳播函式來生成輸出。請注意，輸出的形狀可能與輸入的形狀不同。例如，我們上面模型中的第一個全連線的層接收一個20維的輸入，但是回傳一個維度為256的輸出。
1. 計算其輸出關於輸入的梯度，可透過其反向傳播函式進行訪問。通常這是自動發生的。
1. 儲存和訪問前向傳播計算所需的引數。
1. 根據需要初始化模型引數。
:end_tab:


在下面的程式碼片段中，我們從零開始編寫一個區塊。
它包含一個多層感知機，其具有256個隱藏單元的隱藏層和一個10維輸出層。
注意，下面的`MLP`類繼承了表示區塊的類。
我們的實現只需要提供我們自己的建構式函式（Python中的`__init__`函式）和前向傳播函式。

```{.python .input}
class MLP(nn.Block):
    # 用模型引數宣告層。這裡，我們宣告兩個全連線的層
    def __init__(self, **kwargs):
        # 呼叫MLP的父類別Block的建構式函式來執行必要的初始化。
        # 這樣，在類別實例化時也可以指定其他函式引數，例如模型引數params（稍後將介紹）
        super().__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # 隱藏層
        self.out = nn.Dense(10)  # 輸出層

    # 定義模型的前向傳播，即如何根據輸入X回傳所需的模型輸出
    def forward(self, X):
        return self.out(self.hidden(X))
```

```{.python .input}
#@tab pytorch
class MLP(nn.Module):
    # 用模型引數宣告層。這裡，我們宣告兩個全連線的層
    def __init__(self):
        # 呼叫MLP的父類別Module的建構式函式來執行必要的初始化。
        # 這樣，在類別實例化時也可以指定其他函式引數，例如模型引數params（稍後將介紹）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隱藏層
        self.out = nn.Linear(256, 10)  # 輸出層

    # 定義模型的前向傳播，即如何根據輸入X回傳所需的模型輸出
    def forward(self, X):
        # 注意，這裡我們使用ReLU的函式版本，其在nn.functional模組中定義。
        return self.out(F.relu(self.hidden(X)))
```

```{.python .input}
#@tab tensorflow
class MLP(tf.keras.Model):
    # 用模型引數宣告層。這裡，我們宣告兩個全連線的層
    def __init__(self):
        # 呼叫MLP的父類別Model的建構式函式來執行必要的初始化。
        # 這樣，在類別實例化時也可以指定其他函式引數，例如模型引數params（稍後將介紹）
        super().__init__()
        # Hiddenlayer
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)  # Outputlayer

    # 定義模型的前向傳播，即如何根據輸入X回傳所需的模型輸出
    def call(self, X):
        return self.out(self.hidden((X)))
```

```{.python .input}
#@tab paddle
class MLP(nn.Layer):
    # 用模型引數宣告層。這裡，我們宣告兩個全連線的層
    def __init__(self):
        # 呼叫`MLP`的父類別Layer的建構式函式來執行必要的初始化。
        # 這樣，在類別實例化時也可以指定其他函式引數，例如模型引數`params`（稍後將介紹）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隱藏層
        self.out = nn.Linear(256, 10)  # 輸出層

    # 定義模型的正向傳播，即如何根據輸入`X`回傳所需的模型輸出
    def forward(self, X):
        # 注意，這裡我們使用ReLU的函式版本，其在nn.functional模組中定義。
        return self.out(F.relu(self.hidden(X)))
```

我們首先看一下前向傳播函式，它以`X`作為輸入，
計算帶有啟用函式的隱藏表示，並輸出其未規範化的輸出值。
在這個`MLP`實現中，兩個層都是實例變數。
要了解這為什麼是合理的，可以想象實例化兩個多層感知機（`net1`和`net2`），
並根據不同的資料對它們進行訓練。
當然，我們希望它們學到兩種不同的模型。

接著我們[**實例化多層感知機的層，然後在每次呼叫前向傳播函式時調用這些層**]。
注意一些關鍵細節：
首先，我們定製的`__init__`函式透過`super().__init__()`
呼叫父類別的`__init__`函式，
省去了重複編寫模版程式碼的痛苦。
然後，我們實例化兩個全連線層，
分別為`self.hidden`和`self.out`。
注意，除非我們實現一個新的運算符，
否則我們不必擔心反向傳播函式或引數初始化，
系統將自動生成這些。

我們來試一下這個函式：

```{.python .input}
net = MLP()
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch, paddle
net = MLP()
net(X)
```

```{.python .input}
#@tab tensorflow
net = MLP()
net(X)
```

區塊的一個主要優點是它的多功能性。
我們可以子類別化區塊以建立層（如全連線層的類）、
整個模型（如上面的`MLP`類）或具有中等複雜度的各種元件。
我們在接下來的章節中充分利用了這種多功能性，
比如在處理卷積神經網路時。

## [**順序區塊**]

現在我們可以更仔細地看看`Sequential`類是如何工作的，
回想一下`Sequential`的設計是為了把其他模組串起來。
為了建立我們自己的簡化的`MySequential`，
我們只需要定義兩個關鍵函式：

1. 一種將區塊逐個追加到清單中的函式；
1. 一種前向傳播函式，用於將輸入按追加區塊的順序傳遞給區塊組成的『鏈條』。

下面的`MySequential`類提供了與預設`Sequential`類相同的功能。

```{.python .input}
class MySequential(nn.Block):
    def add(self, block):
    # 這裡，block是Block子類別的一個實例，我們假設它有一個唯一的名稱。我們把它
    # 儲存在'Block'類別的成員變數_children中。block的型別是OrderedDict。
    # 當MySequential實例呼叫initialize函式時，系統會自動初始化_children
    # 的所有成員
        self._children[block.name] = block

    def forward(self, X):
        # OrderedDict保證了按照成員新增的順序遍歷它們
        for block in self._children.values():
            X = block(X)
        return X
```

```{.python .input}
#@tab pytorch
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 這裡，module是Module子類別的一個實例。我們把它儲存在'Module'類別的成員
            # 變數_modules中。_module的型別是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保證了按照成員新增的順序遍歷它們
        for block in self._modules.values():
            X = block(X)
        return X
```

```{.python .input}
#@tab tensorflow
class MySequential(tf.keras.Model):
    def __init__(self, *args):
        super().__init__()
        self.modules = []
        for block in args:
            # 這裡，block是tf.keras.layers.Layer子類別的一個實例
            self.modules.append(block)

    def call(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

```{.python .input}
#@tab paddle
class MySequential(nn.Layer):
    def __init__(self, *layers):
        super(MySequential, self).__init__()
        # 如果傳入的是一個tuple
        if len(layers) > 0 and isinstance(layers[0], tuple): 
            for name, layer in layers:
                # add_sublayer方法會將layer新增到self._sub_layers(一個tuple)
                self.add_sublayer(name, layer)  
        else:
            for idx, layer in enumerate(layers):
                self.add_sublayer(str(idx), layer)

    def forward(self, X):
        # OrderedDict保證了按照成員新增的順序遍歷它們
        for layer in self._sub_layers.values():
            X = layer(X)
        return X
```

:begin_tab:`mxnet`
`add`函式向有序字典`_children`新增一個區塊。
讀者可能會好奇為什麼每個Gluon中的`Block`都有一個`_children`屬性？
以及為什麼我們使用它而不是自己定義一個Python清單？
簡而言之，`_children`的主要優點是：
在區塊的引數初始化過程中，
Gluon知道在`_children`字典中查詢需要初始化引數的子區塊。
:end_tab:

:begin_tab:`pytorch`
`__init__`函式將每個模組逐個新增到有序字典`_modules`中。
讀者可能會好奇為什麼每個`Module`都有一個`_modules`屬性？
以及為什麼我們使用它而不是自己定義一個Python清單？
簡而言之，`_modules`的主要優點是：
在模組的引數初始化過程中，
系統知道在`_modules`字典中查詢需要初始化引數的子區塊。
:end_tab:

:begin_tab:`paddle`
`__init__`函式將每個模組逐個新增到有序字典`_sub_layers`中。
你可能會好奇為什麼每個`Layer`都有一個`_sub_layers`屬性？
以及為什麼我們使用它而不是自己定義一個Python清單？
簡而言之，`_sub_layers`的主要優點是：
在模組的引數初始化過程中，
系統知道在`_sub_layers`字典中查詢需要初始化引數的子區塊。
:end_tab:

當`MySequential`的前向傳播函式被呼叫時，
每個新增的區塊都按照它們被新增的順序執行。
現在可以使用我們的`MySequential`類重新實現多層感知機。

```{.python .input}
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch, paddle
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
```

```{.python .input}
#@tab tensorflow
net = MySequential(
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10))
net(X)
```

請注意，`MySequential`的用法與之前為`Sequential`類編寫的程式碼相同
（如 :numref:`sec_mlp_concise` 中所述）。

## [**在前向傳播函式中執行程式碼**]

`Sequential`類使模型構造變得簡單，
允許我們組合新的架構，而不必定義自己的類。
然而，並不是所有的架構都是簡單的順序架構。
當需要更強的靈活性時，我們需要定義自己的區塊。
例如，我們可能希望在前向傳播函式中執行Python的控制流。
此外，我們可能希望執行任意的數學運算，
而不是簡單地依賴預定義的神經網路層。

到目前為止，
我們網路中的所有操作都對網路的啟用值及網路的引數起作用。
然而，有時我們可能希望合併既不是上一層的結果也不是可更新引數的項，
我們稱之為*常數引數*（constant parameter）。
例如，我們需要一個計算函式
$f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$的層，
其中$\mathbf{x}$是輸入，
$\mathbf{w}$是引數，
$c$是某個在最佳化過程中沒有更新的指定常數。
因此我們實現了一個`FixedHiddenMLP`類，如下所示：

```{.python .input}
class FixedHiddenMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 使用get_constant函式建立的隨機權重引數在訓練期間不會更新（即為常數引數）
        self.rand_weight = self.params.get_constant(
            'rand_weight', np.random.uniform(size=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, X):
        X = self.dense(X)
        # 使用建立的常數引數以及relu和dot函式
        X = npx.relu(np.dot(X, self.rand_weight.data()) + 1)
        # 複用全連線層。這相當於兩個全連線層共享引數
        X = self.dense(X)
        # 控制流
        while np.abs(X).sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
#@tab pytorch
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不計算梯度的隨機權重引數。因此其在訓練期間保持不變
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用建立的常數引數以及relu和mm函式
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 複用全連線層。這相當於兩個全連線層共享引數
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
#@tab tensorflow
class FixedHiddenMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        # 使用tf.constant函式建立的隨機權重引數在訓練期間不會更新（即為常數引數）
        self.rand_weight = tf.constant(tf.random.uniform((20, 20)))
        self.dense = tf.keras.layers.Dense(20, activation=tf.nn.relu)

    def call(self, inputs):
        X = self.flatten(inputs)
        # 使用建立的常數引數以及relu和matmul函式
        X = tf.nn.relu(tf.matmul(X, self.rand_weight) + 1)
        # 複用全連線層。這相當於兩個全連線層共享引數。
        X = self.dense(X)
        # 控制流
        while tf.reduce_sum(tf.math.abs(X)) > 1:
            X /= 2
        return tf.reduce_sum(X)
```

```{.python .input}
#@tab paddle
class FixedHiddenMLP(nn.Layer):
    def __init__(self):
        super().__init__()
        # 不計算梯度的隨機權重引數。因此其在訓練期間保持不變。
        self.rand_weight = paddle.rand([20, 20])
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用建立的常數引數以及relu和mm函式。
        X = F.relu(paddle.tensor.mm(X, self.rand_weight) + 1)
        # 複用全連線層。這相當於兩個全連線層共享引數。
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

在這個`FixedHiddenMLP`模型中，我們實現了一個隱藏層，
其權重（`self.rand_weight`）在實例化時被隨機初始化，之後為常數。
這個權重不是一個模型引數，因此它永遠不會被反向傳播更新。
然後，神經網路將這個固定層的輸出透過一個全連線層。

注意，在回傳輸出之前，模型做了一些不尋常的事情：
它執行了一個while迴圈，在$L_1$範數大於$1$的條件下，
將輸出向量除以$2$，直到它滿足條件為止。
最後，模型回傳了`X`中所有項的和。
注意，此操作可能不會常用於在任何實際任務中，
我們只展示如何將任意程式碼整合到神經網路計算的流程中。

```{.python .input}
net = FixedHiddenMLP()
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch, tensorflow, paddle
net = FixedHiddenMLP()
net(X)
```

我們可以[**混合搭配各種組合區塊的方法**]。
在下面的例子中，我們以一些想到的方法巢狀區塊。

```{.python .input}
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, X):
        return self.dense(self.net(X))

chimera = nn.Sequential()
chimera.add(NestMLP(), nn.Dense(20), FixedHiddenMLP())
chimera.initialize()
chimera(X)
```

```{.python .input}
#@tab pytorch
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
```

```{.python .input}
#@tab tensorflow
class NestMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        self.net.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        self.dense = tf.keras.layers.Dense(16, activation=tf.nn.relu)

    def call(self, inputs):
        return self.dense(self.net(inputs))

chimera = tf.keras.Sequential()
chimera.add(NestMLP())
chimera.add(tf.keras.layers.Dense(20))
chimera.add(FixedHiddenMLP())
chimera(X)
```

```{.python .input}
#@tab paddle
class NestMLP(nn.Layer):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
```

## 效率

:begin_tab:`mxnet`
讀者可能會開始擔心操作效率的問題。
畢竟，我們在一個高效能的深度學習庫中進行了大量的字典查詢、
程式碼執行和許多其他的Python程式碼。
Python的問題[全域直譯器鎖](https://wiki.python.org/moin/GlobalInterpreterLock)
是眾所周知的。
在深度學習環境中，我們擔心速度極快的GPU可能要等到CPU執行Python程式碼後才能執行另一個作業。

提高Python速度的最好方法是完全避免使用Python。
Gluon這樣做的一個方法是允許*混合式程式設計*（hybridization），這將在後面描述。
Python直譯器在第一次呼叫區塊時執行它。
Gluon執行時記錄正在發生的事情，以及下一次它將對Python呼叫加速。
在某些情況下，這可以大大加快執行速度，
但當控制流（如上所述）在不同的網路通路上引導不同的分支時，需要格外小心。
我們建議感興趣的讀者在讀完本章後，閱讀混合式程式設計部分（ :numref:`sec_hybridize` ）來了解編譯。
:end_tab:

:begin_tab:`pytorch`
讀者可能會開始擔心操作效率的問題。
畢竟，我們在一個高效能的深度學習庫中進行了大量的字典查詢、
程式碼執行和許多其他的Python程式碼。
Python的問題[全域直譯器鎖](https://wiki.python.org/moin/GlobalInterpreterLock)
是眾所周知的。
在深度學習環境中，我們擔心速度極快的GPU可能要等到CPU執行Python程式碼後才能執行另一個作業。
:end_tab:

:begin_tab:`tensorflow`
讀者可能會開始擔心操作效率的問題。
畢竟，我們在一個高效能的深度學習庫中進行了大量的字典查詢、
程式碼執行和許多其他的Python程式碼。
Python的問題[全域直譯器鎖](https://wiki.python.org/moin/GlobalInterpreterLock)
是眾所周知的。
在深度學習環境中，我們擔心速度極快的GPU可能要等到CPU執行Python程式碼後才能執行另一個作業。
:end_tab:

:begin_tab:`paddle`
你可能會開始擔心操作效率的問題。
畢竟，我們在一個高效能的深度學習庫中進行了大量的字典查詢、
程式碼執行和許多其他的Python程式碼。
Python的問題[全域直譯器鎖](https://wiki.python.org/moin/GlobalInterpreterLock)
是眾所周知的。
在深度學習環境中，我們擔心速度極快的GPU可能要等到CPU執行Python程式碼後才能執行另一個作業。
:end_tab:


## 小結

* 一個區塊可以由許多層組成；一個區塊可以由許多區塊組成。
* 區塊可以包含程式碼。
* 區塊負責大量的內部處理，包括引數初始化和反向傳播。
* 層和區塊的順序連線由`Sequential`區塊處理。

## 練習

1. 如果將`MySequential`中儲存區塊的方式更改為Python清單，會出現什麼樣的問題？
1. 實現一個區塊，它以兩個區塊為引數，例如`net1`和`net2`，並回傳前向傳播中兩個網路的串聯輸出。這也被稱為平行區塊。
1. 假設我們想要連線同一網路的多個實例。實現一個函式，該函式生成同一個區塊的多個實例，並在此基礎上建立更大的網路。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1828)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1827)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1826)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11777)
:end_tab:
