# 線性迴歸的簡潔實現
:label:`sec_linear_concise`

在過去的幾年裡，出於對深度學習強烈的興趣，
許多公司、學者和業餘愛好者開發了各種成熟的開源框架。
這些框架可以自動化基於梯度的學習演算法中重複性的工作。
在 :numref:`sec_linear_scratch`中，我們只運用了：
（1）透過張量來進行資料儲存和線性代數；
（2）透過自動微分來計算梯度。
實際上，由於資料迭代器、損失函式、最佳化器和神經網路層很常用，
現代深度學習函式庫也為我們實現了這些元件。

本節將介紹如何(**透過使用深度學習框架來簡潔地實現**)
 :numref:`sec_linear_scratch`中的(**線性迴歸模型**)。

## 生成資料集

與 :numref:`sec_linear_scratch`中類似，我們首先[**生成資料集**]。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch.utils import data
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import paddle
```

```{.python .input}
#@tab all
true_w = d2l.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```

## 讀取資料集

我們可以[**呼叫框架中現有的API來讀取資料**]。
我們將`features`和`labels`作為API的引數傳遞，並透過資料迭代器指定`batch_size`。
此外，布林值`is_train`表示是否希望資料迭代器物件在每個迭代週期內打亂數據。

```{.python .input}
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """構造一個Gluon資料迭代器"""
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)
```

```{.python .input}
#@tab pytorch
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """構造一個PyTorch資料迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
```

```{.python .input}
#@tab tensorflow
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """構造一個TensorFlow資料迭代器"""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset
```

```{.python .input}
#@tab paddle
#@save
def load_array(data_arrays, batch_size, is_train=True):
    """構造一個Paddle資料迭代器"""
    dataset = paddle.io.TensorDataset(data_arrays)
    return paddle.io.DataLoader(dataset, batch_size=batch_size,
                                shuffle=is_train,
                                return_list=True)
```

```{.python .input}
#@tab all
batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

使用`data_iter`的方式與我們在 :numref:`sec_linear_scratch`中使用`data_iter`函式的方式相同。為了驗證是否正常工作，讓我們讀取並列印第一個小批次樣本。
與 :numref:`sec_linear_scratch`不同，這裡我們使用`iter`構造Python迭代器，並使用`next`從迭代器中獲取第一項。

```{.python .input}
#@tab all
next(iter(data_iter))
```

## 定義模型

當我們在 :numref:`sec_linear_scratch`中實現線性迴歸時，
我們明確定義了模型引數變數，並編寫了計算的程式碼，這樣透過基本的線性代數運算得到輸出。
但是，如果模型變得更加複雜，且當我們幾乎每天都需要實現模型時，自然會想簡化這個過程。
這種情況類似於為自己的部落格從零開始編寫網頁。
做一兩次是有益的，但如果每個新部落格就需要工程師花一個月的時間重新開始編寫網頁，那並不高效。

對於標準深度學習模型，我們可以[**使用框架的預定義好的層**]。這使我們只需關注使用哪些層來構造模型，而不必關注層的實現細節。
我們首先定義一個模型變數`net`，它是一個`Sequential`類別的實例。
`Sequential`類將多個層串聯在一起。
當給定輸入資料時，`Sequential`實例將資料傳入到第一層，
然後將第一層的輸出作為第二層的輸入，以此類推。
在下面的例子中，我們的模型只包含一個層，因此實際上不需要`Sequential`。
但是由於以後幾乎所有的模型都是多層的，在這裡使用`Sequential`會讓你熟悉『標準的流水線』。

回顧 :numref:`fig_single_neuron`中的單層網路架構，
這一單層被稱為*全連線層*（fully-connected layer），
因為它的每一個輸入都透過矩陣-向量乘法得到它的每個輸出。

:begin_tab:`mxnet`
在Gluon中，全連線層在`Dense`類別中定義。
由於我們只想得到一個標量輸出，所以我們將該數字設定為1。

值得注意的是，為了方便使用，Gluon並不要求我們為每個層指定輸入的形狀。
所以在這裡，我們不需要告訴Gluon有多少輸入進入這一層。
當我們第一次嘗試透過我們的模型傳遞資料時，例如，當後面執行`net(X)`時，
Gluon會自動推斷每個層輸入的形狀。
本節稍後將詳細介紹這種工作機制。
:end_tab:

:begin_tab:`pytorch`
在PyTorch中，全連線層在`Linear`類別中定義。
值得注意的是，我們將兩個引數傳遞到`nn.Linear`中。
第一個指定輸入特徵形狀，即2，第二個指定輸出特徵形狀，輸出特徵形狀為單個標量，因此為1。
:end_tab:

:begin_tab:`tensorflow`
在Keras中，全連線層在`Dense`類別中定義。
由於我們只想得到一個標量輸出，所以我們將該數字設定為1。


值得注意的是，為了方便使用，Keras不要求我們為每個層指定輸入形狀。
所以在這裡，我們不需要告訴Keras有多少輸入進入這一層。
當我們第一次嘗試透過我們的模型傳遞資料時，例如，當後面執行`net(X)`時，
Keras會自動推斷每個層輸入的形狀。
本節稍後將詳細介紹這種工作機制。
:end_tab:

:begin_tab:`paddle`
在PaddlePaddle中，全連線層在`Linear`類別中定義。
值得注意的是，我們將兩個引數傳遞到`nn.Linear`中。
第一個指定輸入特徵形狀，即2，第二個指定輸出特徵形狀，輸出特徵形狀為單個標量，因此為1。
:end_tab:

```{.python .input}
# nn是神經網路的縮寫
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(1))
```

```{.python .input}
#@tab pytorch
# nn是神經網路的縮寫
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
```

```{.python .input}
#@tab tensorflow
# keras是TensorFlow的高階API
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1))
```

```{.python .input}
#@tab paddle
# nn是神經網路的縮寫
from paddle import nn
net = nn.Sequential(nn.Linear(2, 1))
```

## (**初始化模型引數**)

在使用`net`之前，我們需要初始化模型引數。
如線上性迴歸模型中的權重和偏置。
深度學習框架通常有預定義的方法來初始化引數。
在這裡，我們指定每個權重引數應該從均值為0、標準差為0.01的常態分佈中隨機取樣，
偏置引數將初始化為零。

:begin_tab:`mxnet`
我們從MXNet匯入`initializer`模組，這個模組提供了各種模型引數初始化方法。
Gluon將`init`作為訪問`initializer`套件的快捷方式。
我們可以透過呼叫`init.Normal(sigma=0.01)`來指定初始化權重的方法。
預設情況下，偏置引數初始化為零。
:end_tab:

:begin_tab:`pytorch`
正如我們在構造`nn.Linear`時指定輸入和輸出尺寸一樣，
現在我們能直接訪問引數以設定它們的初始值。
我們透過`net[0]`選擇網路中的第一個圖層，
然後使用`weight.data`和`bias.data`方法訪問引數。
我們還可以使用替換方法`normal_`和`fill_`來重寫引數值。
:end_tab:

:begin_tab:`tensorflow`
TensorFlow中的`initializers`模組提供了多種模型引數初始化方法。
在Keras中最簡單的指定初始化方法是在建立層時指定`kernel_initializer`。
在這裡，我們重新建立了`net`。
:end_tab:

:begin_tab:`paddle`
正如我們在構造`nn.Linear`時指定輸入和輸出尺寸一樣，
現在我們能直接訪問引數以設定它們的初始值。 
我們透過`net[0]`選擇網路中的第一個圖層，
然後使用`weight`和`bias`方法訪問引數。
我們可以透過呼叫`nn.initializer.Normal(0, 0.01)`來指定初始化權重的方法。
預設情況下，偏置引數初始化為零。
:end_tab:

```{.python .input}
from mxnet import init
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```

```{.python .input}
#@tab tensorflow
initializer = tf.initializers.RandomNormal(stddev=0.01)
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))
```

```{.python .input}
#@tab paddle
weight_attr = paddle.ParamAttr(initializer=
                               paddle.nn.initializer.Normal(0, 0.01))
bias_attr = paddle.ParamAttr(initializer=None)
net = nn.Sequential(nn.Linear(2, 1, weight_attr=weight_attr,
                              bias_attr=bias_attr))
```

:begin_tab:`mxnet`
上面的程式碼可能看起來很簡單，但是這裡有一個應該注意到的細節：
我們正在為網路初始化引數，而Gluon還不知道輸入將有多少維!
網路的輸入可能有2維，也可能有2000維。
Gluon讓我們避免了這個問題，在後端執行時，初始化實際上是*推遲*（deferred）執行的，
只有在我們第一次嘗試透過網路傳遞資料時才會進行真正的初始化。
請注意，因為引數還沒有初始化，所以我們不能訪問或操作它們。
:end_tab:

:begin_tab:`pytorch`

:end_tab:

:begin_tab:`tensorflow`
上面的程式碼可能看起來很簡單，但是這裡有一個應該注意到的細節：
我們正在為網路初始化引數，而Keras還不知道輸入將有多少維!
網路的輸入可能有2維，也可能有2000維。
Keras讓我們避免了這個問題，在後端執行時，初始化實際上是*推遲*（deferred）執行的。
只有在我們第一次嘗試透過網路傳遞資料時才會進行真正的初始化。
請注意，因為引數還沒有初始化，所以我們不能訪問或操作它們。
:end_tab:

## 定義損失函式

:begin_tab:`mxnet`
在Gluon中，`loss`模組定義了各種損失函式。
在這個例子中，我們將使用Gluon中的均方誤差（`L2Loss`）。
:end_tab:

:begin_tab:`pytorch`
[**計算均方誤差使用的是`MSELoss`類，也稱為平方$L_2$範數**]。
預設情況下，它回傳所有樣本損失的平均值。
:end_tab:

:begin_tab:`tensorflow`
計算均方誤差使用的是`MeanSquaredError`類，也稱為平方$L_2$範數。
預設情況下，它回傳所有樣本損失的平均值。
:end_tab:

:begin_tab:`paddle`
[**計算均方誤差使用的是`MSELoss`類，也稱為平方$L_2$範數**]。
預設情況下，它回傳所有樣本損失的平均值。
:end_tab:

```{.python .input}
loss = gluon.loss.L2Loss()
```

```{.python .input}
#@tab pytorch
loss = nn.MSELoss()
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.MeanSquaredError()
```

```{.python .input}
#@tab paddle
loss = nn.MSELoss()
```

## 定義最佳化演算法

:begin_tab:`mxnet`
小批次隨機梯度下降演算法是一種最佳化神經網路的標準工具，
Gluon透過`Trainer`類支援該演算法的許多變種。
當我們實例化`Trainer`時，我們要指定最佳化的引數
（可透過`net.collect_params()`從我們的模型`net`中獲得）、
我們希望使用的最佳化演算法（`sgd`）以及最佳化演算法所需的超引數字典。
小批次隨機梯度下降只需要設定`learning_rate`值，這裡設定為0.03。
:end_tab:

:begin_tab:`pytorch`
小批次隨機梯度下降演算法是一種最佳化神經網路的標準工具，
PyTorch在`optim`模組中實現了該演算法的許多變種。
當我們(**實例化一個`SGD`實例**)時，我們要指定最佳化的引數
（可透過`net.parameters()`從我們的模型中獲得）以及最佳化演算法所需的超引數字典。
小批次隨機梯度下降只需要設定`lr`值，這裡設定為0.03。
:end_tab:

:begin_tab:`tensorflow`
小批次隨機梯度下降演算法是一種最佳化神經網路的標準工具，
Keras在`optimizers`模組中實現了該演算法的許多變種。
小批次隨機梯度下降只需要設定`learning_rate`值，這裡設定為0.03。
:end_tab:

:begin_tab:`paddle`
小批次隨機梯度下降演算法是一種最佳化神經網路的標準工具，
PaddlePaddle在`optimizer`模組中實現了該演算法的許多變種。
小批次隨機梯度下降只需要設定`learning_rate`值，這裡設定為0.03。
:end_tab:

```{.python .input}
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=0.03)
```

```{.python .input}
#@tab paddle
trainer =  paddle.optimizer.SGD(learning_rate=0.03,
                                parameters=net.parameters())
```

## 訓練

透過深度學習框架的高階API來實現我們的模型只需要相對較少的程式碼。
我們不必單獨分配引數、不必定義我們的損失函式，也不必手動實現小批次隨機梯度下降。
當我們需要更復雜的模型時，高階API的優勢將大大增加。
當我們有了所有的基本元件，[**訓練過程程式碼與我們從零開始實現時所做的非常相似**]。

回顧一下：在每個迭代週期裡，我們將完整遍歷一次資料集（`train_data`），
不停地從中獲取一個小批次的輸入和相應的標籤。
對於每一個小批次，我們會進行以下步驟:

* 透過呼叫`net(X)`生成預測並計算損失`l`（前向傳播）。
* 透過進行反向傳播來計算梯度。
* 透過呼叫最佳化器來更新模型引數。

為了更好的衡量訓練效果，我們計算每個迭代週期後的損失，並列印它來監控訓練過程。

```{.python .input}
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l.mean().asnumpy():f}')
```

```{.python .input}
#@tab pytorch
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

```{.python .input}
#@tab tensorflow
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with tf.GradientTape() as tape:
            l = loss(net(X, training=True), y)
        grads = tape.gradient(l, net.trainable_variables)
        trainer.apply_gradients(zip(grads, net.trainable_variables))
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

```{.python .input}
#@tab paddle
num_epochs = 3
for epoch in range(num_epochs):
    for i,(X, y) in enumerate (data_iter()):
        l = loss(net(X) ,y)
        trainer.clear_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1},'f'loss {l}')
```

下面我們[**比較生成資料集的真實引數和透過有限資料訓練獲得的模型引數**]。
要訪問引數，我們首先從`net`訪問所需的層，然後讀取該層的權重和偏置。
正如在從零開始實現中一樣，我們估計得到的引數與生成資料的真實引數非常接近。

```{.python .input}
w = net[0].weight.data()
print(f'w的估計誤差： {true_w - d2l.reshape(w, true_w.shape)}')
b = net[0].bias.data()
print(f'b的估計誤差： {true_b - b}')
```

```{.python .input}
#@tab pytorch
w = net[0].weight.data
print('w的估計誤差：', true_w - d2l.reshape(w, true_w.shape))
b = net[0].bias.data
print('b的估計誤差：', true_b - b)
```

```{.python .input}
#@tab tensorflow
w = net.get_weights()[0]
print('w的估計誤差：', true_w - d2l.reshape(w, true_w.shape))
b = net.get_weights()[1]
print('b的估計誤差：', true_b - b)
```

```{.python .input}
#@tab paddle
w = net[0].weight
print('w的估計誤差：', true_w - w.reshape(true_w.shape))
b = net[0].bias
print('b的估計誤差：', true_b - b)
```

## 小結

:begin_tab:`mxnet`
* 我們可以使用Gluon更簡潔地實現模型。
* 在Gluon中，`data`模組提供了資料處理工具，`nn`模組定義了大量的神經網路層，`loss`模組定義了許多常見的損失函式。
* MXNet的`initializer`模組提供了各種模型引數初始化方法。
* 維度和儲存可以自動推斷，但注意不要在初始化引數之前嘗試訪問引數。
:end_tab:

:begin_tab:`pytorch`
* 我們可以使用PyTorch的高階API更簡潔地實現模型。
* 在PyTorch中，`data`模組提供了資料處理工具，`nn`模組定義了大量的神經網路層和常見損失函式。
* 我們可以透過`_`結尾的方法將引數替換，從而初始化引數。
:end_tab:

:begin_tab:`tensorflow`
* 我們可以使用TensorFlow的高階API更簡潔地實現模型。
* 在TensorFlow中，`data`模組提供了資料處理工具，`keras`模組定義了大量神經網路層和常見損耗函式。
* TensorFlow的`initializers`模組提供了多種模型引數初始化方法。
* 維度和儲存可以自動推斷，但注意不要在初始化引數之前嘗試訪問引數。
:end_tab:

## 練習

1. 如果將小批次的總損失替換為小批次損失的平均值，需要如何更改學習率？
1. 檢視深度學習框架文件，它們提供了哪些損失函式和初始化方法？用Huber損失代替原損失，即
    $$l(y,y') = \begin{cases}|y-y'| -\frac{\sigma}{2} & \text{ if } |y-y'| > \sigma \\ \frac{1}{2 \sigma} (y-y')^2 & \text{ 其它情況}\end{cases}$$
1. 如何訪問線性迴歸的梯度？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1782)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1781)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1780)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11690)
:end_tab:
