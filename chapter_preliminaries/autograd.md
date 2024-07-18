# 自動微分
:label:`sec_autograd`

正如 :numref:`sec_calculus`中所說，求導是幾乎所有深度學習最佳化演算法的關鍵步驟。
雖然求導的計算很簡單，只需要一些基本的微積分。
但對於複雜的模型，手工進行更新是一件很痛苦的事情（而且經常容易出錯）。

深度學習框架透過自動計算導數，即*自動微分*（automatic differentiation）來加快求導。
實際中，根據設計好的模型，系統會建立一個*計算圖*（computational graph），
來跟蹤計算是哪些資料透過哪些操作組合起來產生輸出。
自動微分使系統能夠隨後反向傳播梯度。
這裡，*反向傳播*（backpropagate）意味著跟蹤整個計算圖，填充關於每個引數的偏導數。

## 一個簡單的例子

作為一個示範例子，(**假設我們想對函式$y=2\mathbf{x}^{\top}\mathbf{x}$關於列向量$\mathbf{x}$求導**)。
首先，我們建立變數`x`併為其分配一個初始值。

```{.python .input}
from mxnet import autograd, np, npx
npx.set_np()

x = np.arange(4.0)
x
```

```{.python .input}
#@tab pytorch
import torch

x = torch.arange(4.0)
x
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

x = tf.range(4, dtype=tf.float32)
x
```

```{.python .input}
#@tab paddle
import warnings
warnings.filterwarnings(action='ignore')
import paddle

x = paddle.arange(4, dtype='float32')
x
```

[**在我們計算$y$關於$\mathbf{x}$的梯度之前，需要一個地方來儲存梯度。**]
重要的是，我們不會在每次對一個引數求導時都分配新的記憶體。
因為我們經常會成千上萬次地更新相同的引數，每次都分配新的記憶體可能很快就會將記憶體耗盡。
注意，一個標量函式關於向量$\mathbf{x}$的梯度是向量，並且與$\mathbf{x}$具有相同的形狀。

```{.python .input}
# 透過呼叫attach_grad來為一個張量的梯度分配記憶體
x.attach_grad()
# 在計算關於x的梯度後，將能夠透過'grad'屬性訪問它，它的值被初始化為0
x.grad
```

```{.python .input}
#@tab pytorch
x.requires_grad_(True)  # 等價於x=torch.arange(4.0,requires_grad=True)
x.grad  # 預設值是None
```

```{.python .input}
#@tab tensorflow
x = tf.Variable(x)
```

```{.python .input}
#@tab paddle
x = paddle.to_tensor(x, stop_gradient=False)
x.grad  # 預設值是None
```

(**現在計算$y$。**)

```{.python .input}
# 把程式碼放到autograd.record內，以建立計算圖
with autograd.record():
    y = 2 * np.dot(x, x)
y
```

```{.python .input}
#@tab pytorch
y = 2 * torch.dot(x, x)
y
```

```{.python .input}
#@tab tensorflow
# 把所有計算記錄在磁帶上
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```

```{.python .input}
#@tab paddle
y = 2 * paddle.dot(x, x)
y
```

`x`是一個長度為4的向量，計算`x`和`x`的點積，得到了我們賦值給`y`的標量輸出。
接下來，[**透過呼叫反向傳播函式來自動計算`y`關於`x`每個分量的梯度**]，並列印這些梯度。

```{.python .input}
y.backward()
x.grad
```

```{.python .input}
#@tab pytorch
y.backward()
x.grad
```

```{.python .input}
#@tab tensorflow
x_grad = t.gradient(y, x)
x_grad
```

```{.python .input}
#@tab paddle
y.backward()
x.grad
```

函式$y=2\mathbf{x}^{\top}\mathbf{x}$關於$\mathbf{x}$的梯度應為$4\mathbf{x}$。
讓我們快速驗證這個梯度是否計算正確。

```{.python .input}
x.grad == 4 * x
```

```{.python .input}
#@tab pytorch
x.grad == 4 * x
```

```{.python .input}
#@tab tensorflow
x_grad == 4 * x
```

```{.python .input}
#@tab paddle
x.grad == 4 * x
```

[**現在計算`x`的另一個函式。**]

```{.python .input}
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # 被新計算的梯度覆蓋
```

```{.python .input}
#@tab pytorch
# 在預設情況下，PyTorch會累積梯度，我們需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
x.grad
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # 被新計算的梯度覆蓋
```

```{.python .input}
#@tab paddle
# 在預設情況下，PaddlePaddle會累積梯度，我們需要清除之前的值
x.clear_gradient()
y = paddle.sum(x)
y.backward()
x.grad
```

## 非標量變數的反向傳播

當`y`不是標量時，向量`y`關於向量`x`的導數的最自然解釋是一個矩陣。
對於高階和高維的`y`和`x`，求導的結果可以是一個高階張量。

然而，雖然這些更奇特的物件確實出現在高階機器學習中（包括[**深度學習中**]），
但當呼叫向量的反向計算時，我們通常會試圖計算一批訓練樣本中每個組成部分的損失函式的導數。
這裡(**，我們的目的不是計算微分矩陣，而是單獨計算批次中每個樣本的偏導數之和。**)

```{.python .input}
# 當對向量值變數y（關於x的函式）呼叫backward時，將透過對y中的元素求和來建立
# 一個新的標量變數。然後計算這個標量變數相對於x的梯度
with autograd.record():
    y = x * x  # y是一個向量
y.backward()
x.grad  # 等價於y=sum(x*x)
```

```{.python .input}
#@tab pytorch
# 對非標量呼叫backward需要傳入一個gradient引數，該引數指定微分函式關於self的梯度。
# 本例只想求偏導數的和，所以傳遞一個1的梯度是合適的
x.grad.zero_()
y = x * x
# 等價於y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # 等價於y=tf.reduce_sum(x*x)
```

```{.python .input}
#@tab paddle
x.clear_gradient()
y = x * x
paddle.sum(y).backward() 
x.grad
```

## 分離計算

有時，我們希望[**將某些計算移動到記錄的計算圖之外**]。
例如，假設`y`是作為`x`的函式計算的，而`z`則是作為`y`和`x`的函式計算的。
想象一下，我們想計算`z`關於`x`的梯度，但由於某種原因，希望將`y`視為一個常數，
並且只考慮到`x`在`y`被計算後發揮的作用。

這裡可以分離`y`來回傳一個新變數`u`，該變數與`y`具有相同的值，
但丟棄計算圖中如何計算`y`的任何訊息。
換句話說，梯度不會向後流經`u`到`x`。
因此，下面的反向傳播函式計算`z=u*x`關於`x`的偏導數，同時將`u`作為常數處理，
而不是`z=x*x*x`關於`x`的偏導數。

```{.python .input}
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
```

```{.python .input}
#@tab pytorch
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

```{.python .input}
#@tab tensorflow
# 設定persistent=True來執行t.gradient多次
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
```

```{.python .input}
#@tab paddle
x.clear_gradient()
y = x * x
u = y.detach()
z = u * x

paddle.sum(z).backward()
x.grad == u
```

由於記錄了`y`的計算結果，我們可以隨後在`y`上呼叫反向傳播，
得到`y=x*x`關於的`x`的導數，即`2*x`。

```{.python .input}
y.backward()
x.grad == 2 * x
```

```{.python .input}
#@tab pytorch
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

```{.python .input}
#@tab tensorflow
t.gradient(y, x) == 2 * x
```

```{.python .input}
#@tab paddle
x.clear_gradient()
paddle.sum(y).backward()
x.grad == 2 * x
```

## Python控制流的梯度計算

使用自動微分的一個好處是：
[**即使建立函式的計算圖需要透過Python控制流（例如，條件、迴圈或任意函式呼叫），我們仍然可以計算得到的變數的梯度**]。
在下面的程式碼中，`while`迴圈的迭代次數和`if`語句的結果都取決於輸入`a`的值。

```{.python .input}
def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
#@tab pytorch
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
#@tab tensorflow
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
#@tab paddle
def f(a):
    b = a * 2
    while paddle.norm(b) < 1000:
        b = b * 2
    if paddle.sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

讓我們計算梯度。

```{.python .input}
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

```{.python .input}
#@tab pytorch
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

```{.python .input}
#@tab tensorflow
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
```

```{.python .input}
#@tab paddle
a = paddle.to_tensor(paddle.randn(shape=[1]), stop_gradient=False)
d = f(a)
d.backward()
```

我們現在可以分析上面定義的`f`函式。
請注意，它在其輸入`a`中是分段線性的。
換言之，對於任何`a`，存在某個常數標量`k`，使得`f(a)=k*a`，其中`k`的值取決於輸入`a`，因此可以用`d/a`驗證梯度是否正確。

```{.python .input}
a.grad == d / a
```

```{.python .input}
#@tab pytorch
a.grad == d / a
```

```{.python .input}
#@tab tensorflow
d_grad == d / a
```

```{.python .input}
#@tab paddle
a.grad == d / a
```

## 小結

* 深度學習框架可以自動計算導數：我們首先將梯度附加到想要對其計算偏導數的變數上，然後記錄目標值的計算，執行它的反向傳播函式，並訪問得到的梯度。

## 練習

1. 為什麼計算二階導數比一階導數的開銷要更大？
1. 在執行反向傳播函式之後，立即再次執行它，看看會發生什麼。
1. 在控制流的例子中，我們計算`d`關於`a`的導數，如果將變數`a`更改為隨機向量或矩陣，會發生什麼？
1. 重新設計一個求控制流梯度的例子，執行並分析結果。
1. 使$f(x)=\sin(x)$，繪製$f(x)$和$\frac{df(x)}{dx}$的影象，其中後者不使用$f'(x)=\cos(x)$。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1758)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1759)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1757)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11684)
:end_tab:
