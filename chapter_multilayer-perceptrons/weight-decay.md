# 權重衰減
:label:`sec_weight_decay`

前一節我們描述了過擬合的問題，本節我們將介紹一些正則化模型的技術。
我們總是可以透過去收集更多的訓練資料來緩解過擬合。
但這可能成本很高，耗時頗多，或者完全超出我們的控制，因而在短期內不可能做到。
假設我們已經擁有儘可能多的高質量資料，我們便可以將重點放在正則化技術上。

回想一下，在多項式迴歸的例子（ :numref:`sec_model_selection`）中，
我們可以透過調整擬合多項式的階數來限制模型的容量。
實際上，限制特徵的數量是緩解過擬合的一種常用技術。
然而，簡單地丟棄特徵對這項工作來說可能過於生硬。
我們繼續思考多項式迴歸的例子，考慮高維輸入可能發生的情況。
多項式對多變量資料的自然擴充套件稱為*單項式*（monomials），
也可以說是變數冪的乘積。
單項式的階數是冪的和。
例如，$x_1^2 x_2$和$x_3 x_5^2$都是3次單項式。

注意，隨著階數$d$的增長，帶有階數$d$的項數迅速增加。 
給定$k$個變數，階數為$d$的項的個數為
${k - 1 + d} \choose {k - 1}$，即$C^{k-1}_{k-1+d} = \frac{(k-1+d)!}{(d)!(k-1)!}$。
因此即使是階數上的微小變化，比如從$2$到$3$，也會顯著增加我們模型的複雜性。
僅僅透過簡單的限制特徵數量（在多項式迴歸中體現為限制階數），可能仍然使模型在過簡單和過複雜中徘徊，
我們需要一個更細粒度的工具來調整函式的複雜性，使其達到一個合適的平衡位置。
## 範數與權重衰減

在 :numref:`subsec_lin-algebra-norms`中，
我們已經描述了$L_2$範數和$L_1$範數，
它們是更為一般的$L_p$範數的特殊情況。
(~~權重衰減是最廣泛使用的正則化的技術之一~~)
在訓練引數化機器學習模型時，
*權重衰減*（weight decay）是最廣泛使用的正則化的技術之一，
它通常也被稱為$L_2$*正則化*。
這項技術透過函式與零的距離來衡量函式的複雜度，
因為在所有函式$f$中，函式$f = 0$（所有輸入都得到值$0$）
在某種意義上是最簡單的。
但是我們應該如何精確地測量一個函式和零之間的距離呢？
沒有一個正確的答案。
事實上，函式分析和巴拿赫空間理論的研究，都在致力於回答這個問題。

一種簡單的方法是透過線性函式
$f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$
中的權重向量的某個範數來度量其複雜性，
例如$\| \mathbf{w} \|^2$。
要保證權重向量比較小，
最常用方法是將其範數作為懲罰項加到最小化損失的問題中。
將原來的訓練目標*最小化訓練標籤上的預測損失*，
調整為*最小化預測損失和懲罰項之和*。
現在，如果我們的權重向量增長的太大，
我們的學習演算法可能會更集中於最小化權重範數$\| \mathbf{w} \|^2$。
這正是我們想要的。
讓我們回顧一下 :numref:`sec_linear_regression`中的線性迴歸例子。
我們的損失由下式給出：

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

回想一下，$\mathbf{x}^{(i)}$是樣本$i$的特徵，
$y^{(i)}$是樣本$i$的標籤，
$(\mathbf{w}, b)$是權重和偏置引數。
為了懲罰權重向量的大小，
我們必須以某種方式在損失函式中新增$\| \mathbf{w} \|^2$，
但是模型應該如何平衡這個新的額外懲罰的損失？
實際上，我們透過*正則化常數*$\lambda$來描述這種權衡，
這是一個非負超引數，我們使用驗證資料擬合：

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2,$$

對於$\lambda = 0$，我們恢復了原來的損失函式。
對於$\lambda > 0$，我們限制$\| \mathbf{w} \|$的大小。
這裡我們仍然除以$2$：當我們取一個二次函式的導數時，
$2$和$1/2$會抵消，以確保更新表示式看起來既漂亮又簡單。
為什麼在這裡我們使用平方範數而不是標準範數（即歐幾里得距離）？
我們這樣做是為了便於計算。
透過平方$L_2$範數，我們去掉平方根，留下權重向量每個分量的平方和。
這使得懲罰的導數很容易計算：導數的和等於和的導數。

此外，為什麼我們首先使用$L_2$範數，而不是$L_1$範數。
事實上，這個選擇在整個統計領域中都是有效的和受歡迎的。
$L_2$正則化線性模型構成經典的*嶺迴歸*（ridge regression）演算法，
$L_1$正則化線性迴歸是統計學中類似的基本模型，
通常被稱為*套索迴歸*（lasso regression）。
使用$L_2$範數的一個原因是它對權重向量的大分量施加了巨大的懲罰。
這使得我們的學習演算法偏向於在大量特徵上均勻分佈權重的模型。
在實踐中，這可能使它們對單個變數中的觀測誤差更為穩定。
相比之下，$L_1$懲罰會導致模型將權重集中在一小部分特徵上，
而將其他權重清除為零。
這稱為*特徵選擇*（feature selection），這可能是其他場景下需要的。

使用與 :eqref:`eq_linreg_batch_update`中的相同符號，
$L_2$正則化迴歸的小批次隨機梯度下降更新如下式：

$$
\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}
$$

根據之前章節所講的，我們根據估計值與觀測值之間的差異來更新$\mathbf{w}$。
然而，我們同時也在試圖將$\mathbf{w}$的大小縮小到零。
這就是為什麼這種方法有時被稱為*權重衰減*。
我們僅考慮懲罰項，最佳化演算法在訓練的每一步*衰減*權重。
與特徵選擇相比，權重衰減為我們提供了一種連續的機制來調整函式的複雜度。
較小的$\lambda$值對應較少約束的$\mathbf{w}$，
而較大的$\lambda$值對$\mathbf{w}$的約束更大。

是否對相應的偏置$b^2$進行懲罰在不同的實踐中會有所不同，
在神經網路的不同層中也會有所不同。
通常，網路輸出層的偏置項不會被正則化。

## 高維線性迴歸

我們透過一個簡單的例子來示範權重衰減。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
```

首先，我們[**像以前一樣生成一些資料**]，生成公式如下：

(**$$y = 0.05 + \sum_{i = 1}^d 0.01 x_i + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.01^2).$$**)

我們選擇標籤是關於輸入的線性函式。
標籤同時被均值為0，標準差為0.01高斯噪聲破壞。
為了使過擬合的效果更加明顯，我們可以將問題的維數增加到$d = 200$，
並使用一個只包含20個樣本的小訓練集。

```{.python .input}
#@tab all
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)
```

## 從零開始實現

下面我們將從頭開始實現權重衰減，只需將$L_2$的平方懲罰新增到原始目標函式中。

### [**初始化模型引數**]

首先，我們將定義一個函式來隨機初始化模型引數。

```{.python .input}
def init_params():
    w = np.random.normal(scale=1, size=(num_inputs, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    return [w, b]
```

```{.python .input}
#@tab pytorch
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]
```

```{.python .input}
#@tab tensorflow
def init_params():
    w = tf.Variable(tf.random.normal(mean=1, shape=(num_inputs, 1)))
    b = tf.Variable(tf.zeros(shape=(1, )))
    return [w, b]
```

```{.python .input}
#@tab paddle
def init_params():
    w = paddle.normal(0, 1, shape=(num_inputs, 1))
    w.stop_gradient = False
    b = paddle.zeros(shape=[1])
    b.stop_gradient = False
    return [w, b]
```

### (**定義$L_2$範數懲罰**)

實現這一懲罰最方便的方法是對所有項求平方後並將它們求和。

```{.python .input}
def l2_penalty(w):
    return (w**2).sum() / 2
```

```{.python .input}
#@tab pytorch
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2
```

```{.python .input}
#@tab tensorflow
def l2_penalty(w):
    return tf.reduce_sum(tf.pow(w, 2)) / 2
```

```{.python .input}
#@tab paddle
def l2_penalty(w):
    return paddle.sum(w.pow(2)) / 2
```

### [**定義訓練程式碼實現**]

下面的程式碼將模型擬合訓練資料集，並在測試資料集上進行評估。
從 :numref:`chap_linear`以來，線性網路和平方損失沒有變化，
所以我們透過`d2l.linreg`和`d2l.squared_loss`匯入它們。
唯一的變化是損失現在包括了懲罰項。

```{.python .input}
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                # 增加了L2範數懲罰項，
                # 廣播機制使l2_penalty(w)成為一個長度為batch_size的向量
                l = loss(net(X), y) + lambd * l2_penalty(w)
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2範數是：', np.linalg.norm(w))
```

```{.python .input}
#@tab pytorch
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2範數懲罰項，
            # 廣播機制使l2_penalty(w)成為一個長度為batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2範數是：', torch.norm(w).item())
```

```{.python .input}
#@tab tensorflow
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                # 增加了L2範數懲罰項，
                # 廣播機制使l2_penalty(w)成為一個長度為batch_size的向量
                l = loss(net(X), y) + lambd * l2_penalty(w)
            grads = tape.gradient(l, [w, b])
            d2l.sgd([w, b], grads, lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2範數是：', tf.norm(w).numpy())
```

```{.python .input}
#@tab paddle
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter():
            # 增加了L2範數懲罰項,
            # 廣播機制使l2_penalty(w)成為一個長度為`batch_size`的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2範數是：', paddle.norm(w).item())
```

### [**忽略正則化直接訓練**]

我們現在用`lambd = 0`停用權重衰減後執行這個程式碼。
注意，這裡訓練誤差有了減少，但測試誤差沒有減少，
這意味著出現了嚴重的過擬合。

```{.python .input}
#@tab all
train(lambd=0)
```

### [**使用權重衰減**]

下面，我們使用權重衰減來執行程式碼。
注意，在這裡訓練誤差增大，但測試誤差減小。
這正是我們期望從正則化中得到的效果。

```{.python .input}
#@tab all
train(lambd=3)
```

## [**簡潔實現**]

由於權重衰減在神經網路最佳化中很常用，
深度學習框架為了便於我們使用權重衰減，
將權重衰減整合到最佳化演算法中，以便與任何損失函式結合使用。
此外，這種整合還有計算上的好處，
允許在不增加任何額外的計算開銷的情況下向演算法中新增權重衰減。
由於更新的權重衰減部分僅依賴於每個引數的當前值，
因此最佳化器必須至少接觸每個引數一次。

:begin_tab:`mxnet`
在下面的程式碼中，我們在實例化`Trainer`時直接透過`wd`指定weight decay超引數。
預設情況下，Gluon同時衰減權重和偏置。
注意，更新模型引數時，超引數`wd`將乘以`wd_mult`。
因此，如果我們將`wd_mult`設定為零，則偏置引數$b$將不會被衰減。
:end_tab:

:begin_tab:`pytorch`
在下面的程式碼中，我們在實例化最佳化器時直接透過`weight_decay`指定weight decay超引數。
預設情況下，PyTorch同時衰減權重和偏移。
這裡我們只為權重設定了`weight_decay`，所以偏置引數$b$不會衰減。
:end_tab:

:begin_tab:`tensorflow`
在下面的程式碼中，我們使用權重衰減超引數`wd`建立一個$L_2$正則化器，
並透過`kernel_regularizer`引數將其應用於網路層。
:end_tab:

```{.python .input}
def train_concise(wd):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=1))
    loss = gluon.loss.L2Loss()
    num_epochs, lr = 100, 0.003
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'wd': wd})
    # 偏置引數沒有衰減。偏置名稱通常以『bias』結尾
    net.collect_params('.*bias').setattr('wd_mult', 0)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2範數：', np.linalg.norm(net[0].weight.data()))
```

```{.python .input}
#@tab pytorch
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置引數沒有衰減
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (d2l.evaluate_loss(net, train_iter, loss),
                          d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2範數：', net[0].weight.norm().item())
```

```{.python .input}
#@tab tensorflow
def train_concise(wd):
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Dense(
        1, kernel_regularizer=tf.keras.regularizers.l2(wd)))
    net.build(input_shape=(1, num_inputs))
    w, b = net.trainable_variables
    loss = tf.keras.losses.MeanSquaredError()
    num_epochs, lr = 100, 0.003
    trainer = tf.keras.optimizers.SGD(learning_rate=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                # tf.keras需要為自定義訓練程式碼手動新增損失。
                l = loss(net(X), y) + net.losses
            grads = tape.gradient(l, net.trainable_variables)
            trainer.apply_gradients(zip(grads, net.trainable_variables))
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2範數：', tf.norm(net.get_weights()[0]).numpy())
```

```{.python .input}
#@tab paddle
def train_concise(wd):
    weight_attr = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Normal(mean=0.0, std=1.0))
    bias_attr = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Normal(mean=0.0, std=1.0))
    net = nn.Sequential(nn.Linear(num_inputs, 1, weight_attr=weight_attr, bias_attr=bias_attr))
    loss = nn.MSELoss()
    num_epochs, lr = 100, 0.003
    # 偏置引數沒有衰減。
    trainer = paddle.optimizer.SGD(parameters=net[0].parameters(), learning_rate=lr, weight_decay=wd*1.0)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y)
            l.backward()
            trainer.step()
            trainer.clear_grad()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2範數：', net[0].weight.norm().item())
```

[**這些圖看起來和我們從零開始實現權重衰減時的圖相同**]。
然而，它們執行得更快，更容易實現。
對於更復雜的問題，這一好處將變得更加明顯。

```{.python .input}
#@tab all
train_concise(0)
```

```{.python .input}
#@tab all
train_concise(3)
```

到目前為止，我們只接觸到一個簡單線性函式的概念。
此外，由什麼構成一個簡單的非線性函式可能是一個更復雜的問題。
例如，[再生核希爾伯特空間（RKHS）](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space)
允許在非線性環境中應用為線性函式引入的工具。
不幸的是，基於RKHS的演算法往往難以應用到大型、高維的資料。
在這本書中，我們將預設使用簡單的啟發式方法，即在深層網路的所有層上應用權重衰減。

## 小結

* 正則化是處理過擬合的常用方法：在訓練集的損失函式中加入懲罰項，以降低學習到的模型的複雜度。
* 保持模型簡單的一個特別的選擇是使用$L_2$懲罰的權重衰減。這會導致學習演算法更新步驟中的權重衰減。
* 權重衰減功能在深度學習框架的最佳化器中提供。
* 在同一訓練程式碼實現中，不同的引數集可以有不同的更新行為。

## 練習

1. 在本節的估計問題中使用$\lambda$的值進行實驗。繪製訓練和測試精度關於$\lambda$的函式。觀察到了什麼？
1. 使用驗證集來找到最佳值$\lambda$。它真的是最優值嗎？這有關系嗎？
1. 如果我們使用$\sum_i |w_i|$作為我們選擇的懲罰（$L_1$正則化），那麼更新方程會是什麼樣子？
1. 我們知道$\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$。能找到類似的矩陣方程嗎（見 :numref:`subsec_lin-algebra-norms` 中的Frobenius範數）？
1. 回顧訓練誤差和泛化誤差之間的關係。除了權重衰減、增加訓練資料、使用適當複雜度的模型之外，還能想出其他什麼方法來處理過擬合？
1. 在貝葉斯統計中，我們使用先驗和似然的乘積，透過公式$P(w \mid x) \propto P(x \mid w) P(w)$得到後驗。如何得到帶正則化的$P(w)$？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1810)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1808)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1809)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11773)
:end_tab:
