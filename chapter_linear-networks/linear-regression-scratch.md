# 線性迴歸的從零開始實現
:label:`sec_linear_scratch`

在瞭解線性迴歸的關鍵思想之後，我們可以開始透過程式碼來動手實現線性迴歸了。
在這一節中，(**我們將從零開始實現整個方法，
包括資料流水線、模型、損失函式和小批次隨機梯度下降最佳化器**)。
雖然現代的深度學習框架幾乎可以自動化地進行所有這些工作，但從零開始實現可以確保我們真正知道自己在做什麼。
同時，瞭解更細緻的工作原理將方便我們自定義模型、自定義層或自定義損失函式。
在這一節中，我們將只使用張量和自動求導。
在之後的章節中，我們會充分利用深度學習框架的優勢，介紹更簡潔的實現方式。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import random
import paddle
```

## 生成資料集

為了簡單起見，我們將[**根據帶有噪聲的線性模型構造一個人造資料集。**]
我們的任務是使用這個有限樣本的資料集來恢復這個模型的引數。
我們將使用低維資料，這樣可以很容易地將其視覺化。
在下面的程式碼中，我們生成一個包含1000個樣本的資料集，
每個樣本包含從標準常態分佈中取樣的2個特徵。
我們的合成數據集是一個矩陣$\mathbf{X}\in \mathbb{R}^{1000 \times 2}$。

(**我們使用線性模型引數$\mathbf{w} = [2, -3.4]^\top$、$b = 4.2$
和噪聲項$\epsilon$生成資料集及其標籤：

$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \mathbf\epsilon.$$
**)

$\epsilon$可以視為模型預測和標籤時的潛在觀測誤差。
在這裡我們認為標準假設成立，即$\epsilon$服從均值為0的常態分佈。
為了簡化問題，我們將標準差設為0.01。
下面的程式碼生成合成數據集。

```{.python .input}
#@tab mxnet, pytorch, paddle
def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪聲"""
    X = d2l.normal(0, 1, (num_examples, len(w)))
    y = d2l.matmul(X, w) + b
    y += d2l.normal(0, 0.01, y.shape)
    return X, d2l.reshape(y, (-1, 1))
```

```{.python .input}
#@tab tensorflow
def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪聲"""
    X = d2l.zeros((num_examples, w.shape[0]))
    X += tf.random.normal(shape=X.shape)
    y = d2l.matmul(X, tf.reshape(w, (-1, 1))) + b
    y += tf.random.normal(shape=y.shape, stddev=0.01)
    y = d2l.reshape(y, (-1, 1))
    return X, y
```

```{.python .input}
#@tab all
true_w = d2l.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

注意，[**`features`中的每一行都包含一個二維資料樣本，
`labels`中的每一行都包含一維標籤值（一個標量）**]。

```{.python .input}
#@tab all
print('features:', features[0],'\nlabel:', labels[0])
```

透過生成第二個特徵`features[:, 1]`和`labels`的散點圖，
可以直觀觀察到兩者之間的線性關係。

```{.python .input}
#@tab all
d2l.set_figsize()
d2l.plt.scatter(d2l.numpy(features[:, 1]), d2l.numpy(labels), 1);
```

## 讀取資料集

回想一下，訓練模型時要對資料集進行遍歷，每次抽取一小批次樣本，並使用它們來更新我們的模型。
由於這個過程是訓練機器學習演算法的基礎，所以有必要定義一個函式，
該函式能打亂數據集中的樣本並以小批次方式獲取資料。

在下面的程式碼中，我們[**定義一個`data_iter`函式，
該函式接收批次大小、特徵矩陣和標籤向量作為輸入，生成大小為`batch_size`的小批次**]。
每個小批次包含一組特徵和標籤。

```{.python .input}
#@tab mxnet, pytorch, paddle
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 這些樣本是隨機讀取的，沒有特定的順序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = d2l.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
```

```{.python .input}
#@tab tensorflow
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 這些樣本是隨機讀取的，沒有特定的順序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = tf.constant(indices[i: min(i + batch_size, num_examples)])
        yield tf.gather(features, j), tf.gather(labels, j)
```

通常，我們利用GPU並行運算的優勢，處理合理大小的『小批次』。
每個樣本都可以並行地進行模型計算，且每個樣本損失函式的梯度也可以被並行計算。
GPU可以在處理幾百個樣本時，所花費的時間不比處理一個樣本時多太多。

我們直觀感受一下小批次運算：讀取第一個小批次資料樣本並列印。
每個批次的特徵維度顯示批次大小和輸入特徵數。
同樣的，批次的標籤形狀與`batch_size`相等。

```{.python .input}
#@tab all
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
```

當我們執行迭代時，我們會連續地獲得不同的小批次，直至遍歷完整個數據集。
上面實現的迭代對教學來說很好，但它的執行效率很低，可能會在實際問題上陷入麻煩。
例如，它要求我們將所有資料載入到記憶體中，並執行大量的隨機記憶體訪問。
在深度學習框架中實現的內建迭代器效率要高得多，
它可以處理儲存在檔案中的資料和數據流提供的資料。

## 初始化模型引數

[**在我們開始用小批次隨機梯度下降最佳化我們的模型引數之前**]，
(**我們需要先有一些引數**)。
在下面的程式碼中，我們透過從均值為0、標準差為0.01的常態分佈中取樣隨機數來初始化權重，
並將偏置初始化為0。

```{.python .input}
w = np.random.normal(0, 0.01, (2, 1))
b = np.zeros(1)
w.attach_grad()
b.attach_grad()
```

```{.python .input}
#@tab pytorch
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

```{.python .input}
#@tab tensorflow
w = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=0.01),
                trainable=True)
b = tf.Variable(tf.zeros(1), trainable=True)
```

```{.python .input}
#@tab paddle
w = d2l.normal(0, 0.01, shape=(2,1))
b = d2l.zeros(shape=[1])
# w和b為建立的模型引數，stop_gradient預設為True，即梯度不更新，因此需要指定為False已更新梯度
w.stop_gradient = False
b.stop_gradient = False
```

在初始化引數之後，我們的任務是更新這些引數，直到這些引數足夠擬合我們的資料。
每次更新都需要計算損失函式關於模型引數的梯度。
有了這個梯度，我們就可以向減小損失的方向更新每個引數。
因為手動計算梯度很枯燥而且容易出錯，所以沒有人會手動計算梯度。
我們使用 :numref:`sec_autograd`中引入的自動微分來計算梯度。

## 定義模型

接下來，我們必須[**定義模型，將模型的輸入和引數同模型的輸出關聯起來。**]
回想一下，要計算線性模型的輸出，
我們只需計算輸入特徵$\mathbf{X}$和模型權重$\mathbf{w}$的矩陣-向量乘法後加上偏置$b$。
注意，上面的$\mathbf{Xw}$是一個向量，而$b$是一個標量。
回想一下 :numref:`subsec_broadcasting`中描述的廣播機制：
當我們用一個向量加一個標量時，標量會被加到向量的每個分量上。

```{.python .input}
#@tab all
def linreg(X, w, b):  #@save
    """線性迴歸模型"""
    return d2l.matmul(X, w) + b
```

## [**定義損失函式**]

因為需要計算損失函式的梯度，所以我們應該先定義損失函式。
這裡我們使用 :numref:`sec_linear_regression`中描述的平方損失函式。
在實現中，我們需要將真實值`y`的形狀轉換為和預測值`y_hat`的形狀相同。

```{.python .input}
#@tab all
def squared_loss(y_hat, y):  #@save
    """均方損失"""
    return (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
```

## (**定義最佳化演算法**)

正如我們在 :numref:`sec_linear_regression`中討論的，線性迴歸有解析解。
儘管線性迴歸有解析解，但本書中的其他模型卻沒有。
這裡我們介紹小批次隨機梯度下降。

在每一步中，使用從資料集中隨機抽取的一個小批次，然後根據引數計算損失的梯度。
接下來，朝著減少損失的方向更新我們的引數。
下面的函式實現小批次隨機梯度下降更新。
該函式接受模型引數集合、學習速率和批次大小作為輸入。每
一步更新的大小由學習速率`lr`決定。
因為我們計算的損失是一個批次樣本的總和，所以我們用批次大小（`batch_size`）
來規範化步長，這樣步長大小就不會取決於我們對批次大小的選擇。

```{.python .input}
def sgd(params, lr, batch_size):  #@save
    """小批次隨機梯度下降"""
    for param in params:
        param[:] = param - lr * param.grad / batch_size
```

```{.python .input}
#@tab pytorch
def sgd(params, lr, batch_size):  #@save
    """小批次隨機梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, lr, batch_size):  #@save
    """小批次隨機梯度下降"""
    for param, grad in zip(params, grads):
        param.assign_sub(lr*grad/batch_size)
```

```{.python .input}
#@tab paddle
#@save
def sgd(params, lr, batch_size):
    """小批次隨機梯度下降"""
    with paddle.no_grad():
        for i, param in enumerate(params):
            param -= lr * params[i].grad / batch_size
            params[i].set_value(param)
            params[i].clear_gradient()
```

## 訓練

現在我們已經準備好了模型訓練所有需要的要素，可以實現主要的[**訓練過程**]部分了。
理解這段程式碼至關重要，因為從事深度學習後，
相同的訓練過程幾乎一遍又一遍地出現。
在每次迭代中，我們讀取一小批次訓練樣本，並透過我們的模型來獲得一組預測。
計算完損失後，我們開始反向傳播，儲存每個引數的梯度。
最後，我們呼叫最佳化演算法`sgd`來更新模型引數。

概括一下，我們將執行以下迴圈：

* 初始化引數
* 重複以下訓練，直到完成
    * 計算梯度$\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
    * 更新引數$(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$

在每個*迭代週期*（epoch）中，我們使用`data_iter`函式遍歷整個資料集，
並將訓練資料集中所有樣本都使用一次（假設樣本數能夠被批次大小整除）。
這裡的迭代週期個數`num_epochs`和學習率`lr`都是超引數，分別設為3和0.03。
設定超引數很棘手，需要透過反覆試驗進行調整。
我們現在忽略這些細節，以後會在 :numref:`chap_optimization`中詳細介紹。

```{.python .input}
#@tab all
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
```

```{.python .input}
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)  # X和y的小批次損失
        # 計算l關於[w,b]的梯度
        l.backward()
        sgd([w, b], lr, batch_size)  # 使用引數的梯度更新引數
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

```{.python .input}
#@tab pytorch
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批次損失
        # 因為l形狀是(batch_size,1)，而不是一個標量。l中的所有元素被加到一起，
        # 並以此計算關於[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用引數的梯度更新引數
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

```{.python .input}
#@tab tensorflow
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with tf.GradientTape() as g:
            l = loss(net(X, w, b), y)  # X和y的小批次損失
        # 計算l關於[w,b]的梯度
        dw, db = g.gradient(l, [w, b])
        # 使用引數的梯度更新引數
        sgd([w, b], [dw, db], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')
```

```{.python .input}
#@tab paddle
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批次損失
        # 因為l形狀是(batch_size,1)，而不是一個標量。l中的所有元素被加到一起，
        # 並以此計算關於[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用引數的梯度更新引數
    with paddle.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

因為我們使用的是自己合成的資料集，所以我們知道真正的引數是什麼。
因此，我們可以透過[**比較真實引數和透過訓練學到的引數來評估訓練的成功程度**]。
事實上，真實引數和透過訓練學到的引數確實非常接近。

```{.python .input}
#@tab all
print(f'w的估計誤差: {true_w - d2l.reshape(w, true_w.shape)}')
print(f'b的估計誤差: {true_b - b}')
```

注意，我們不應該想當然地認為我們能夠完美地求解引數。
在機器學習中，我們通常不太關心恢復真正的引數，而更關心如何高度準確預測引數。
幸運的是，即使是在複雜的最佳化問題上，隨機梯度下降通常也能找到非常好的解。
其中一個原因是，在深度網路中存在許多引數組合能夠實現高度精確的預測。

## 小結

* 我們學習了深度網路是如何實現和最佳化的。在這一過程中只使用張量和自動微分，不需要定義層或複雜的最佳化器。
* 這一節只觸及到了表面知識。在下面的部分中，我們將基於剛剛介紹的概念描述其他模型，並學習如何更簡潔地實現其他模型。

## 練習

1. 如果我們將權重初始化為零，會發生什麼。演算法仍然有效嗎？
1. 假設試圖為電壓和電流的關係建立一個模型。自動微分可以用來學習模型的引數嗎?
1. 能基於[普朗克定律](https://en.wikipedia.org/wiki/Planck%27s_law)使用光譜能量密度來確定物體的溫度嗎？
1. 計算二階導數時可能會遇到什麼問題？這些問題可以如何解決？
1. 為什麼在`squared_loss`函式中需要使用`reshape`函式？
1. 嘗試使用不同的學習率，觀察損失函式值下降的快慢。
1. 如果樣本個數不能被批次大小整除，`data_iter`函式的行為會有什麼變化？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1779)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1778)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1777)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11689)
:end_tab:
