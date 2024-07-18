# 注意力彙聚：Nadaraya-Watson 核迴歸
:label:`sec_nadaraya-watson`

上節介紹了框架下的注意力機制的主要成分 :numref:`fig_qkv`：
查詢（自主提示）和鍵（非自主提示）之間的互動形成了注意力彙聚；
注意力彙聚有選擇地聚合了值（感官輸入）以生成最終的輸出。
本節將介紹注意力彙聚的更多細節，
以便從宏觀上了解注意力機制在實踐中的運作方式。
具體來說，1964年提出的Nadaraya-Watson核迴歸模型
是一個簡單但完整的例子，可以用於示範具有注意力機制的機器學習。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
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
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
tf.random.set_seed(seed=1322)
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
```

## [**生成資料集**]

簡單起見，考慮下面這個迴歸問題：
給定的成對的『輸入－輸出』資料集
$\{(x_1, y_1), \ldots, (x_n, y_n)\}$，
如何學習$f$來預測任意新輸入$x$的輸出$\hat{y} = f(x)$？

根據下面的非線性函式生成一個人工資料集，
其中加入的噪聲項為$\epsilon$：

$$y_i = 2\sin(x_i) + x_i^{0.8} + \epsilon,$$

其中$\epsilon$服從均值為$0$和標準差為$0.5$的常態分佈。
在這裡生成了$50$個訓練樣本和$50$個測試樣本。
為了更好地視覺化之後的注意力模式，需要將訓練樣本進行排序。

```{.python .input}
n_train = 50  # 訓練樣本數
x_train = np.sort(d2l.rand(n_train) * 5)   # 排序後的訓練樣本
```

```{.python .input}
#@tab pytorch
n_train = 50  # 訓練樣本數
x_train, _ = torch.sort(d2l.rand(n_train) * 5)   # 排序後的訓練樣本
```

```{.python .input}
#@tab tensorflow
n_train = 50
x_train = tf.sort(tf.random.uniform(shape=(n_train,), maxval=5))
```

```{.python .input}
#@tab paddle
n_train = 50  # 訓練樣本數
x_train = paddle.sort(paddle.rand([n_train]) * 5)   # 排序後的訓練樣本
```

```{.python .input}
def f(x):
    return 2 * d2l.sin(x) + x**0.8

y_train = f(x_train) + d2l.normal(0.0, 0.5, (n_train,))  # 訓練樣本的輸出
x_test = d2l.arange(0, 5, 0.1)  # 測試樣本
y_truth = f(x_test)  # 測試樣本的真實輸出
n_test = len(x_test)  # 測試樣本數
n_test
```

```{.python .input}
#@tab pytorch
def f(x):
    return 2 * d2l.sin(x) + x**0.8

y_train = f(x_train) + d2l.normal(0.0, 0.5, (n_train,))  # 訓練樣本的輸出
x_test = d2l.arange(0, 5, 0.1)  # 測試樣本
y_truth = f(x_test)  # 測試樣本的真實輸出
n_test = len(x_test)  # 測試樣本數
n_test
```

```{.python .input}
#@tab tensorflow
def f(x):
    return 2 * d2l.sin(x) + x**0.8

y_train = f(x_train) + d2l.normal((n_train,), 0.0, 0.5)  # 訓練樣本的輸出
x_test = d2l.arange(0, 5, 0.1)  # 測試樣本
y_truth = f(x_test)  # 測試樣本的真實輸出
n_test = len(x_test)  # 測試樣本數
n_test
```

```{.python .input}
#@tab paddle
def f(x):
    return 2 * paddle.sin(x) + x**0.8

y_train = f(x_train) + paddle.normal(0.0, 0.5, (n_train,))  # 訓練樣本的輸出
x_test = d2l.arange(0, 5, 0.1, dtype='float32')   # 測試樣本
y_truth = f(x_test)  # 測試樣本的真實輸出
n_test = len(x_test)  # 測試樣本數
n_test
```

下面的函式將繪製所有的訓練樣本（樣本由圓圈表示），
不帶噪聲項的真實資料生成函式$f$（標記為『Truth』），
以及學習得到的預測函式（標記為『Pred』）。

```{.python .input}
#@tab all
def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5);
```

## 平均彙聚

先使用最簡單的估計器來解決迴歸問題。
基於平均彙聚來計算所有訓練樣本輸出值的平均值：

$$f(x) = \frac{1}{n}\sum_{i=1}^n y_i,$$
:eqlabel:`eq_avg-pooling`

如下圖所示，這個估計器確實不夠聰明。
真實函式$f$（『Truth』）和預測函式（『Pred』）相差很大。

```{.python .input}
y_hat = y_train.mean().repeat(n_test)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab tensorflow
y_hat = tf.repeat(tf.reduce_mean(y_train), repeats=n_test)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab paddle
y_hat = paddle.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)
```

## [**非引數注意力彙聚**]

顯然，平均彙聚忽略了輸入$x_i$。
於是Nadaraya :cite:`Nadaraya.1964`和
Watson :cite:`Watson.1964`提出了一個更好的想法，
根據輸入的位置對輸出$y_i$進行加權：

$$f(x) = \sum_{i=1}^n \frac{K(x - x_i)}{\sum_{j=1}^n K(x - x_j)} y_i,$$
:eqlabel:`eq_nadaraya-watson`

其中$K$是*核*（kernel）。
公式 :eqref:`eq_nadaraya-watson`所描述的估計器被稱為
*Nadaraya-Watson核迴歸*（Nadaraya-Watson kernel regression）。
這裡不會深入討論核函式的細節，
但受此啟發，
我們可以從 :numref:`fig_qkv`中的注意力機制框架的角度
重寫 :eqref:`eq_nadaraya-watson`，
成為一個更加通用的*注意力彙聚*（attention pooling）公式：

$$f(x) = \sum_{i=1}^n \alpha(x, x_i) y_i,$$
:eqlabel:`eq_attn-pooling`

其中$x$是查詢，$(x_i, y_i)$是鍵值對。
比較 :eqref:`eq_attn-pooling`和 :eqref:`eq_avg-pooling`，
注意力彙聚是$y_i$的加權平均。
將查詢$x$和鍵$x_i$之間的關係建模為
*注意力權重*（attention weight）$\alpha(x, x_i)$，
如 :eqref:`eq_attn-pooling`所示，
這個權重將被分配給每一個對應值$y_i$。
對於任何查詢，模型在所有鍵值對注意力權重都是一個有效的機率分佈：
它們是非負的，並且總和為1。

為了更好地理解注意力彙聚，
下面考慮一個*高斯核*（Gaussian kernel），其定義為：

$$K(u) = \frac{1}{\sqrt{2\pi}} \exp(-\frac{u^2}{2}).$$

將高斯核代入 :eqref:`eq_attn-pooling`和
 :eqref:`eq_nadaraya-watson`可以得到：

$$\begin{aligned} f(x) &=\sum_{i=1}^n \alpha(x, x_i) y_i\\ &= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}(x - x_i)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}(x - x_j)^2\right)} y_i \\&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}(x - x_i)^2\right) y_i. \end{aligned}$$
:eqlabel:`eq_nadaraya-watson-gaussian`

在 :eqref:`eq_nadaraya-watson-gaussian`中，
如果一個鍵$x_i$越是接近給定的查詢$x$，
那麼分配給這個鍵對應值$y_i$的注意力權重就會越大，
也就『獲得了更多的注意力』。

值得注意的是，Nadaraya-Watson核迴歸是一個非引數模型。
因此， :eqref:`eq_nadaraya-watson-gaussian`是
*非引數的注意力彙聚*（nonparametric attention pooling）模型。
接下來，我們將基於這個非引數的注意力彙聚模型來繪製預測結果。
從繪製的結果會發現新的模型預測線是平滑的，並且比平均彙聚的預測更接近真實。

```{.python .input}
# X_repeat的形狀:(n_test,n_train),
# 每一行都包含著相同的測試輸入（例如：同樣的查詢）
X_repeat = d2l.reshape(x_test.repeat(n_train), (-1, n_train))
# x_train包含著鍵。attention_weights的形狀：(n_test,n_train),
# 每一行都包含著要在給定的每個查詢的值（y_train）之間分配的注意力權重
attention_weights = npx.softmax(-(X_repeat - x_train)**2 / 2)
# y_hat的每個元素都是值的加權平均值，其中的權重是注意力權重
y_hat = d2l.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
# X_repeat的形狀:(n_test,n_train),
# 每一行都包含著相同的測試輸入（例如：同樣的查詢）
X_repeat = d2l.reshape(x_test.repeat_interleave(n_train), (-1, n_train))
# x_train包含著鍵。attention_weights的形狀：(n_test,n_train),
# 每一行都包含著要在給定的每個查詢的值（y_train）之間分配的注意力權重
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# y_hat的每個元素都是值的加權平均值，其中的權重是注意力權重
y_hat = d2l.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab tensorflow
# X_repeat的形狀:(n_test,n_train),
# 每一行都包含著相同的測試輸入（例如：同樣的查詢）
X_repeat = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_train, axis=0)
# x_train包含著鍵。attention_weights的形狀：(n_test,n_train),
# 每一行都包含著要在給定的每個查詢的值（y_train）之間分配的注意力權重
attention_weights = tf.nn.softmax(-(X_repeat - tf.expand_dims(x_train, axis=1))**2/2, axis=1)
# y_hat的每個元素都是值的加權平均值，其中的權重是注意力權重
y_hat = tf.matmul(attention_weights, tf.expand_dims(y_train, axis=1))
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab paddle
# X_repeat的形狀:(n_test,n_train),
# 每一行都包含著相同的測試輸入（例如：同樣的查詢）
X_repeat = d2l.reshape(x_test.repeat_interleave(n_train), (-1, n_train))
# x_train包含著鍵。attention_weights的形狀：(n_test,n_train),
# 每一行都包含著要在給定的每個查詢的值（y_train）之間分配的注意力權重
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, axis=1)
# y_hat的每個元素都是值的加權平均值，其中的權重是注意力權重
y_hat = d2l.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
```

現在來觀察注意力的權重。
這裡測試資料的輸入相當於查詢，而訓練資料的輸入相當於鍵。
因為兩個輸入都是經過排序的，因此由觀察可知『查詢-鍵』對越接近，
注意力彙聚的[**注意力權重**]就越高。

```{.python .input}
d2l.show_heatmaps(np.expand_dims(np.expand_dims(attention_weights, 0), 0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab pytorch
d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab tensorflow
d2l.show_heatmaps(tf.expand_dims(
                      tf.expand_dims(attention_weights, axis=0), axis=0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab paddle
d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

## [**帶引數注意力彙聚**]

非引數的Nadaraya-Watson核迴歸具有*一致性*（consistency）的優點：
如果有足夠的資料，此模型會收斂到最優結果。
儘管如此，我們還是可以輕鬆地將可學習的引數整合到注意力彙聚中。

例如，與 :eqref:`eq_nadaraya-watson-gaussian`略有不同，
在下面的查詢$x$和鍵$x_i$之間的距離乘以可學習引數$w$：

$$\begin{aligned}f(x) &= \sum_{i=1}^n \alpha(x, x_i) y_i \\&= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}((x - x_i)w)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}((x - x_j)w)^2\right)} y_i \\&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}((x - x_i)w)^2\right) y_i.\end{aligned}$$
:eqlabel:`eq_nadaraya-watson-gaussian-para`

本節的餘下部分將透過訓練這個模型
 :eqref:`eq_nadaraya-watson-gaussian-para`來學習注意力彙聚的引數。

### 批次矩陣乘法

:label:`subsec_batch_dot`

為了更有效地計算小批次資料的注意力，
我們可以利用深度學習開發框架中提供的批次矩陣乘法。

假設第一個小批次資料包含$n$個矩陣$\mathbf{X}_1,\ldots, \mathbf{X}_n$，
形狀為$a\times b$，
第二個小批次包含$n$個矩陣$\mathbf{Y}_1, \ldots, \mathbf{Y}_n$，
形狀為$b\times c$。
它們的批次矩陣乘法得到$n$個矩陣
$\mathbf{X}_1\mathbf{Y}_1, \ldots, \mathbf{X}_n\mathbf{Y}_n$，
形狀為$a\times c$。
因此，[**假定兩個張量的形狀分別是$(n,a,b)$和$(n,b,c)$，
它們的批次矩陣乘法輸出的形狀為$(n,a,c)$**]。

```{.python .input}
X = d2l.ones((2, 1, 4))
Y = d2l.ones((2, 4, 6))
npx.batch_dot(X, Y).shape
```

```{.python .input}
#@tab pytorch
X = d2l.ones((2, 1, 4))
Y = d2l.ones((2, 4, 6))
torch.bmm(X, Y).shape
```

```{.python .input}
#@tab tensorflow
X = tf.ones((2, 1, 4))
Y = tf.ones((2, 4, 6))
tf.matmul(X, Y).shape
```

```{.python .input}
#@tab paddle
X = paddle.ones((2, 1, 4))
Y = paddle.ones((2, 4, 6))
paddle.bmm(X, Y).shape
```

在注意力機制的背景中，我們可以[**使用小批次矩陣乘法來計算小批次資料中的加權平均值**]。

```{.python .input}
weights = d2l.ones((2, 10)) * 0.1
values = d2l.reshape(d2l.arange(20), (2, 10))
npx.batch_dot(np.expand_dims(weights, 1), np.expand_dims(values, -1))
```

```{.python .input}
#@tab pytorch
weights = d2l.ones((2, 10)) * 0.1
values = d2l.reshape(d2l.arange(20.0), (2, 10))
torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))
```

```{.python .input}
#@tab tensorflow
weights = tf.ones((2, 10)) * 0.1
values = tf.reshape(tf.range(20.0), shape = (2, 10))
tf.matmul(tf.expand_dims(weights, axis=1), tf.expand_dims(values, axis=-1)).numpy()
```

```{.python .input}
#@tab paddle
weights = paddle.ones((2, 10)) * 0.1
values = paddle.arange(20, dtype='float32').reshape((2, 10))
paddle.bmm(weights.unsqueeze(1), values.unsqueeze(-1))
```

### 定義模型

基於 :eqref:`eq_nadaraya-watson-gaussian-para`中的
[**帶引數的注意力彙聚**]，使用小批次矩陣乘法，
定義Nadaraya-Watson核迴歸的帶引數版本為：

```{.python .input}
class NWKernelRegression(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = self.params.get('w', shape=(1,))

    def forward(self, queries, keys, values):
        # queries和attention_weights的形狀為(查詢數，『鍵－值』對數)
        queries = d2l.reshape(
            queries.repeat(keys.shape[1]), (-1, keys.shape[1]))
        self.attention_weights = npx.softmax(
            -((queries - keys) * self.w.data())**2 / 2)
        # values的形狀為(查詢數，『鍵－值』對數)
        return npx.batch_dot(np.expand_dims(self.attention_weights, 1),
                             np.expand_dims(values, -1)).reshape(-1)
```

```{.python .input}
#@tab pytorch
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # queries和attention_weights的形狀為(查詢個數，『鍵－值』對個數)
        queries = d2l.reshape(
            queries.repeat_interleave(keys.shape[1]), (-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # values的形狀為(查詢個數，『鍵－值』對個數)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)
```

```{.python .input}
#@tab tensorflow
class NWKernelRegression(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(initial_value=tf.random.uniform(shape=(1,)))
        
    def call(self, queries, keys, values, **kwargs):
        # 對於訓練，『查詢』是x_train。『鍵』是每個點的訓練資料的距離。『值』為'y_train'。
        # queries和attention_weights的形狀為(查詢個數，『鍵－值』對個數)
        queries = tf.repeat(tf.expand_dims(queries, axis=1), repeats=keys.shape[1], axis=1)
        self.attention_weights = tf.nn.softmax(-((queries - keys) * self.w)**2 /2, axis =1)
        # values的形狀為(查詢個數，『鍵－值』對個數)
        return tf.squeeze(tf.matmul(tf.expand_dims(self.attention_weights, axis=1), tf.expand_dims(values, axis=-1)))
```

```{.python .input}
#@tab paddle
class NWKernelRegression(nn.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = paddle.create_parameter((1,), dtype='float32')

    def forward(self, queries, keys, values):
        # queries和attention_weights的形狀為(查詢個數，『鍵－值』對個數)
        queries = queries.reshape((queries.shape[0], 1)) \
        .tile([keys.shape[1]]) \
        .reshape((-1, keys.shape[1]))
        self.attention_weight = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, axis=1)
        # values的形狀為(查詢個數，『鍵－值』對個數)
        return paddle.bmm(self.attention_weight.unsqueeze(1),
                          values.unsqueeze(-1)).reshape((-1, ))
```

### 訓練

接下來，[**將訓練資料集變換為鍵和值**]用於訓練注意力模型。
在帶引數的注意力彙聚模型中，
任何一個訓練樣本的輸入都會和除自己以外的所有訓練樣本的『鍵－值』對進行計算，
從而得到其對應的預測輸出。

```{.python .input}
# X_tile的形狀:(n_train，n_train)，每一行都包含著相同的訓練輸入
X_tile = np.tile(x_train, (n_train, 1))
# Y_tile的形狀:(n_train，n_train)，每一行都包含著相同的訓練輸出
Y_tile = np.tile(y_train, (n_train, 1))
# keys的形狀:('n_train'，'n_train'-1)
keys = d2l.reshape(X_tile[(1 - d2l.eye(n_train)).astype('bool')],
                   (n_train, -1))
# values的形狀:('n_train'，'n_train'-1)
values = d2l.reshape(Y_tile[(1 - d2l.eye(n_train)).astype('bool')],
                     (n_train, -1))
```

```{.python .input}
#@tab pytorch
# X_tile的形狀:(n_train，n_train)，每一行都包含著相同的訓練輸入
X_tile = x_train.repeat((n_train, 1))
# Y_tile的形狀:(n_train，n_train)，每一行都包含著相同的訓練輸出
Y_tile = y_train.repeat((n_train, 1))
# keys的形狀:('n_train'，'n_train'-1)
keys = d2l.reshape(X_tile[(1 - d2l.eye(n_train)).type(torch.bool)],
                   (n_train, -1))
# values的形狀:('n_train'，'n_train'-1)
values = d2l.reshape(Y_tile[(1 - d2l.eye(n_train)).type(torch.bool)],
                     (n_train, -1))
```

```{.python .input}
#@tab tensorflow
# X_tile的形狀:(n_train，n_train)，每一行都包含著相同的訓練輸入
X_tile = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_train, axis=0)
# Y_tile的形狀:(n_train，n_train)，每一行都包含著相同的訓練輸出
Y_tile = tf.repeat(tf.expand_dims(y_train, axis=0), repeats=n_train, axis=0)
# keys的形狀:('n_train'，'n_train'-1)
keys = tf.reshape(X_tile[tf.cast(1 - tf.eye(n_train), dtype=tf.bool)], shape=(n_train, -1))
# values的形狀:('n_train'，'n_train'-1)
values = tf.reshape(Y_tile[tf.cast(1 - tf.eye(n_train), dtype=tf.bool)], shape=(n_train, -1))
```

```{.python .input}
#@tab paddle
# X_tile的形狀:(n_train，n_train)，每一行都包含著相同的訓練輸入
X_tile = x_train.tile([n_train, 1])
# Y_tile的形狀:(n_train，n_train)，每一行都包含著相同的訓練輸出
Y_tile = y_train.tile([n_train, 1])
# keys的形狀:('n_train'，'n_train'-1)
keys = X_tile[(1 - paddle.eye(n_train)).astype(paddle.bool)].reshape((n_train, -1))
# values的形狀:('n_train'，'n_train'-1)
values = Y_tile[(1 - paddle.eye(n_train)).astype(paddle.bool)].reshape((n_train, -1))
```

[**訓練帶引數的注意力彙聚模型**]時，使用平方損失函式和隨機梯度下降。

```{.python .input}
net = NWKernelRegression()
net.initialize()
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    with autograd.record():
        l = loss(net(x_train, keys, values), y_train)
    l.backward()
    trainer.step(1)
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
```

```{.python .input}
#@tab pytorch
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
```

```{.python .input}
#@tab tensorflow
net = NWKernelRegression()
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])


for epoch in range(5):
    with tf.GradientTape() as t:
        loss = loss_object(y_train, net(x_train, keys, values)) * len(y_train)
    grads = t.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    print(f'epoch {epoch + 1}, loss {float(loss):.6f}')
    animator.add(epoch + 1, float(loss))
```

```{.python .input}
#@tab paddle
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = paddle.optimizer.SGD(learning_rate=0.5, parameters=net.parameters())
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.clear_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
```

如下所示，訓練完帶引數的注意力彙聚模型後可以發現：
在嘗試擬合帶噪聲的訓練資料時，
[**預測結果繪製**]的線不如之前非引數模型的平滑。

```{.python .input}
# keys的形狀:(n_test，n_train)，每一行包含著相同的訓練輸入（例如，相同的鍵）
keys = np.tile(x_train, (n_test, 1))
# value的形狀:(n_test，n_train)
values = np.tile(y_train, (n_test, 1))
y_hat = net(x_test, keys, values)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
# keys的形狀:(n_test，n_train)，每一行包含著相同的訓練輸入（例如，相同的鍵）
keys = x_train.repeat((n_test, 1))
# value的形狀:(n_test，n_train)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab tensorflow
# keys的形狀:(n_test，n_train)，每一行包含著相同的訓練輸入（例如，相同的鍵）
keys = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_test, axis=0)
# value的形狀:(n_test，n_train)
values = tf.repeat(tf.expand_dims(y_train, axis=0), repeats=n_test, axis=0)
y_hat = net(x_test, keys, values)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab paddle
# keys的形狀:(n_test，n_train)，每一行包含著相同的訓練輸入（例如，相同的鍵）
keys = x_train.tile([n_test, 1])
# value的形狀:(n_test，n_train)
values = y_train.tile([n_test, 1])
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)
```

為什麼新的模型更不平滑了呢？
下面看一下輸出結果的繪製圖：
與非引數的注意力彙聚模型相比，
帶引數的模型加入可學習的引數後，
[**曲線在注意力權重較大的區域變得更不平滑**]。

```{.python .input}
d2l.show_heatmaps(np.expand_dims(
                      np.expand_dims(net.attention_weights, 0), 0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab pytorch
d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab tensorflow
d2l.show_heatmaps(tf.expand_dims(
                      tf.expand_dims(net.attention_weights, axis=0), axis=0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab paddle
d2l.show_heatmaps(net.attention_weight.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorter testing, inputs')
```

## 小結

* Nadaraya-Watson核迴歸是具有注意力機制的機器學習範例。
* Nadaraya-Watson核迴歸的注意力彙聚是對訓練資料中輸出的加權平均。從注意力的角度來看，分配給每個值的注意力權重取決於將值所對應的鍵和查詢作為輸入的函式。
* 注意力彙聚可以分為非引數型和帶引數型。

## 練習

1. 增加訓練資料的樣本數量，能否得到更好的非引數的Nadaraya-Watson核迴歸模型？
1. 在帶引數的注意力彙聚的實驗中學習得到的引數$w$的價值是什麼？為什麼在視覺化注意力權重時，它會使加權區域更加尖銳？
1. 如何將超引數新增到非引數的Nadaraya-Watson核迴歸中以實現更好地預測結果？
1. 為本節的核迴歸設計一個新的帶引數的注意力彙聚模型。訓練這個新模型並可視化其注意力權重。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/5759)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/5760)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11840)
:end_tab: