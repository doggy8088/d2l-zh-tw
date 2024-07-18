# 讀寫檔案

到目前為止，我們討論瞭如何處理資料，
以及如何建立、訓練和測試深度學習模型。
然而，有時我們希望儲存訓練的模型，
以備將來在各種環境中使用（比如在部署中進行預測）。
此外，當執行一個耗時較長的訓練過程時，
最佳的做法是定期儲存中間結果，
以確保在服務器電源被不小心斷掉時，我們不會損失幾天的計算結果。
因此，現在是時候學習如何載入和儲存權重向量和整個模型了。

## (**載入和儲存張量**)

對於單個張量，我們可以直接呼叫`load`和`save`函式分別讀寫它們。
這兩個函式都要求我們提供一個名稱，`save`要求將要儲存的變數作為輸入。

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

x = np.arange(4)
npx.save('x-file', x)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
import numpy as np

x = tf.range(4)
np.save('x-file.npy', x)
```

```{.python .input}
#@tab paddle
import warnings
warnings.filterwarnings(action='ignore')
import paddle
from paddle import nn 
from paddle.nn import functional as F

x = paddle.arange(4)  
paddle.save(x, 'x-file')
```

我們現在可以將儲存在檔案中的資料讀回記憶體。

```{.python .input}
x2 = npx.load('x-file')
x2
```

```{.python .input}
#@tab pytorch
x2 = torch.load('x-file')
x2
```

```{.python .input}
#@tab tensorflow
x2 = np.load('x-file.npy', allow_pickle=True)
x2
```

```{.python .input}
#@tab paddle
x2 = paddle.load('x-file')
x2
```

我們可以[**儲存一個張量清單，然後把它們讀回記憶體。**]

```{.python .input}
y = np.zeros(4)
npx.save('x-files', [x, y])
x2, y2 = npx.load('x-files')
(x2, y2)
```

```{.python .input}
#@tab pytorch
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

```{.python .input}
#@tab tensorflow
y = tf.zeros(4)
np.save('xy-files.npy', [x, y])
x2, y2 = np.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

```{.python .input}
#@tab paddle
y = paddle.zeros([4])
paddle.save([x,y], 'x-file')
x2, y2 = paddle.load('x-file')
(x2, y2)
```

我們甚至可以(**寫入或讀取從字串對映到張量的字典**)。
當我們要讀取或寫入模型中的所有權重時，這很方便。

```{.python .input}
mydict = {'x': x, 'y': y}
npx.save('mydict', mydict)
mydict2 = npx.load('mydict')
mydict2
```

```{.python .input}
#@tab pytorch
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

```{.python .input}
#@tab tensorflow
mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
mydict2
```

```{.python .input}
#@tab paddle
mydict = {'x': x, 'y': y}
paddle.save(mydict, 'mydict')
mydict2 = paddle.load('mydict')
mydict2
```

## [**載入和儲存模型引數**]

儲存單個權重向量（或其他張量）確實有用，
但是如果我們想儲存整個模型，並在以後載入它們，
單獨儲存每個向量則會變得很麻煩。
畢竟，我們可能有數百個引數散佈在各處。
因此，深度學習框架提供了內建函式來儲存和載入整個網路。
需要注意的一個重要細節是，這將儲存模型的引數而不是儲存整個模型。
例如，如果我們有一個3層多層感知機，我們需要單獨指定架構。
因為模型本身可以包含任意程式碼，所以模型本身難以序列化。
因此，為了恢復模型，我們需要用程式碼生成架構，
然後從磁碟載入引數。
讓我們從熟悉的多層感知機開始嘗試一下。

```{.python .input}
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = np.random.uniform(size=(2, 20))
Y = net(X)
```

```{.python .input}
#@tab pytorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

```{.python .input}
#@tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)
```

```{.python .input}
#@tab paddle
class MLP(nn.Layer):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = paddle.randn(shape=[2, 20])
Y = net(X)
```

接下來，我們[**將模型的引數儲存在一個叫做『mlp.params』的檔案中。**]

```{.python .input}
net.save_parameters('mlp.params')
```

```{.python .input}
#@tab pytorch
torch.save(net.state_dict(), 'mlp.params')
```

```{.python .input}
#@tab tensorflow
net.save_weights('mlp.params')
```

```{.python .input}
#@tab paddle
paddle.save(net.state_dict(), 'mlp.pdparams')
```

為了恢復模型，我們[**實例化了原始多層感知機模型的一個備份。**]
這裡我們不需要隨機初始化模型引數，而是(**直接讀取檔案中儲存的引數。**)

```{.python .input}
clone = MLP()
clone.load_parameters('mlp.params')
```

```{.python .input}
#@tab pytorch
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

```{.python .input}
#@tab tensorflow
clone = MLP()
clone.load_weights('mlp.params')
```

```{.python .input}
#@tab paddle
clone = MLP()
clone.set_state_dict(paddle.load('mlp.pdparams'))
clone.eval()
```

由於兩個實例具有相同的模型引數，在輸入相同的`X`時，
兩個實例的計算結果應該相同。
讓我們來驗證一下。

```{.python .input}
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input}
#@tab pytorch, paddle
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input}
#@tab tensorflow
Y_clone = clone(X)
Y_clone == Y
```

## 小結

* `save`和`load`函式可用於張量物件的檔案讀寫。
* 我們可以透過引數字典儲存和載入網路的全部引數。
* 儲存架構必須在程式碼中完成，而不是在引數中完成。

## 練習

1. 即使不需要將經過訓練的模型部署到不同的裝置上，儲存模型引數還有什麼實際的好處？
1. 假設我們只想複用網路的一部分，以將其合併到不同的網路架構中。比如想在一個新的網路中使用之前網路的前兩層，該怎麼做？
1. 如何同時儲存網路架構和引數？需要對架構加上什麼限制？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1840)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1839)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1838)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11781)
:end_tab:
