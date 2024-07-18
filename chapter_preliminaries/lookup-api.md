# 查閱文件

:begin_tab:`mxnet`
由於篇幅限制，本書不可能介紹每一個MXNet函式和類。
API文件、其他教程和範例提供了本書之外的大量文件。
本節提供了一些檢視MXNet API的指導。
:end_tab:

:begin_tab:`pytorch`
由於篇幅限制，本書不可能介紹每一個PyTorch函式和類。
API文件、其他教程和範例提供了本書之外的大量文件。
本節提供了一些檢視PyTorch API的指導。
:end_tab:

:begin_tab:`tensorflow`
由於篇幅限制，本書不可能介紹每一個TensorFlow函式和類。
API文件、其他教程和範例提供了本書之外的大量文件。
本節提供了一些查TensorFlow API的指導。
:end_tab:

## 查詢模組中的所有函式和類

為了知道模組中可以呼叫哪些函式和類，可以呼叫`dir`函式。
例如，我們可以(**查詢隨機數生成模組中的所有屬性：**)

```{.python .input}
from mxnet import np
print(dir(np.random))
```

```{.python .input}
#@tab pytorch
import torch
print(dir(torch.distributions))
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
print(dir(tf.random))
```

```{.python .input}
#@tab paddle
import warnings
warnings.filterwarnings(action='ignore')
import paddle
print(dir(paddle.distribution))
```

通常可以忽略以『`__`』（雙下劃線）開始和結束的函式，它們是Python中的特殊物件，
或以單個『`_`』（單下劃線）開始的函式，它們通常是內部函式。
根據剩餘的函式名或屬性名，我們可能會猜測這個模組提供了各種生成隨機數的方法，
包括從均勻分佈（`uniform`）、常態分佈（`normal`）和多項分佈（`multinomial`）中取樣。

## 查詢特定函式和類別的用法

有關如何使用給定函式或類別的更具體說明，可以呼叫`help`函式。
例如，我們來[**檢視張量`ones`函式的用法。**]

```{.python .input}
help(np.ones)
```

```{.python .input}
#@tab pytorch
help(torch.ones)
```

```{.python .input}
#@tab tensorflow
help(tf.ones)
```

```{.python .input}
#@tab paddle
help(paddle.ones)
```

從文件中，我們可以看到`ones`函式建立一個具有指定形狀的新張量，並將所有元素值設定為1。
下面來[**執行一個快速測試**]來確認這一解釋：

```{.python .input}
np.ones(4)
```

```{.python .input}
#@tab pytorch
torch.ones(4)
```

```{.python .input}
#@tab tensorflow
tf.ones(4)
```

```{.python .input}
#@tab paddle
paddle.ones([4], dtype='float32')
```

在Jupyter記事本中，我們可以使用`?`指令在另一個瀏覽器視窗中顯示文件。
例如，`list?`指令將建立與`help(list)`指令幾乎相同的內容，並在新的瀏覽器視窗中顯示它。
此外，如果我們使用兩個問號，如`list??`，將顯示實現該函式的Python程式碼。

## 小結

* 官方文件提供了本書之外的大量描述和範例。
* 可以透過呼叫`dir`和`help`函式或在Jupyter記事本中使用`?`和`??`檢視API的用法文件。

## 練習

1. 在深度學習框架中查詢任何函式或類別的文件。請嘗試在這個框架的官方網站上找到文件。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1764)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1765)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1763)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11686)
:end_tab:
