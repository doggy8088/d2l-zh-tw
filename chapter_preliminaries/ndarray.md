# 資料操作
:label:`sec_ndarray`

為了能夠完成各種資料操作，我們需要某種方法來儲存和操作數據。
通常，我們需要做兩件重要的事：（1）獲取資料；（2）將資料讀入電腦後對其進行處理。
如果沒有某種方法來儲存資料，那麼獲取資料是沒有意義的。

首先，我們介紹$n$維陣列，也稱為*張量*（tensor）。
使用過Python中NumPy計算套件的讀者會對本部分很熟悉。
無論使用哪個深度學習框架，它的*張量類*（在MXNet中為`ndarray`，
在PyTorch和TensorFlow中為`Tensor`）都與Numpy的`ndarray`類似。
但深度學習框架又比Numpy的`ndarray`多一些重要功能：
首先，GPU很好地支持加速計算，而NumPy僅支援CPU計算；
其次，張量類支援自動微分。
這些功能使得張量類更適合深度學習。
如果沒有特殊說明，本書中所說的張量均指的是張量類別的實例。

## 入門

本節的目標是幫助讀者瞭解並執行一些在閱讀本書的過程中會用到的基本數值計算工具。
如果你很難理解一些數學概念或函式庫函式，請不要擔心。
後面的章節將透過一些實際的例子來回顧這些內容。
如果你已經具有相關經驗，想要深入學習數學內容，可以跳過本節。

:begin_tab:`mxnet`
首先，我們從MXNet匯入`np`（`numpy`）模組和`npx`（`numpy_extension`）模組。
`np`模組包含NumPy支援的函式；
而`npx`模組包含一組擴充套件函式，用來在類似NumPy的環境中實現深度學習開發。
當使用張量時，幾乎總是會呼叫`set_np`函式，這是為了相容MXNet的其他張量處理元件。
:end_tab:

:begin_tab:`pytorch`
(**首先，我們匯入`torch`。請注意，雖然它被稱為PyTorch，但是程式碼中使用`torch`而不是`pytorch`。**)
:end_tab:

:begin_tab:`tensorflow`
首先，我們匯入`tensorflow`。
由於`tensorflow`名稱有點長，我們經常在匯入它後使用短別名`tf`。
:end_tab:

```{.python .input}
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
```

```{.python .input}
#@tab paddle
import warnings
warnings.filterwarnings(action='ignore')
import paddle
```

[**張量表示一個由數值組成的陣列，這個陣列可能有多個維度**]。
具有一個軸的張量對應數學上的*向量*（vector）；
具有兩個軸的張量對應數學上的*矩陣*（matrix）；
具有兩個軸以上的張量沒有特殊的數學名稱。

:begin_tab:`mxnet`
首先，我們可以使用 `arange` 建立一個行向量 `x`。這個行向量包含以0開始的前12個整數，它們預設建立為浮點數。張量中的每個值都稱為張量的 *元素*（element）。例如，張量 `x` 中有 12 個元素。除非額外指定，新的張量將儲存在記憶體中，並採用基於CPU的計算。
:end_tab:

:begin_tab:`pytorch`
首先，我們可以使用 `arange` 建立一個行向量 `x`。這個行向量包含以0開始的前12個整數，它們預設建立為整數。也可指定建立型別為浮點數。張量中的每個值都稱為張量的 *元素*（element）。例如，張量 `x` 中有 12 個元素。除非額外指定，新的張量將儲存在記憶體中，並採用基於CPU的計算。
:end_tab:

:begin_tab:`tensorflow`
首先，我們可以使用 `range` 建立一個行向量 `x`。這個行向量包含以0開始的前12個整數，它們預設建立為整數。也可指定建立型別為浮點數。張量中的每個值都稱為張量的 *元素*（element）。例如，張量 `x` 中有 12 個元素。除非額外指定，新的張量將儲存在記憶體中，並採用基於CPU的計算。
:end_tab:

```{.python .input}
x = np.arange(12)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(12)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(12)
x
```

```{.python .input}
#@tab paddle
x = paddle.arange(12)
x
```

[**可以透過張量的`shape`屬性來訪問張量（沿每個軸的長度）的*形狀***]
(~~和張量中元素的總數~~)。

```{.python .input}
#@tab all
x.shape
```

如果只想知道張量中元素的總數，即形狀的所有元素乘積，可以檢查它的大小（size）。
因為這裡在處理的是一個向量，所以它的`shape`與它的`size`相同。

```{.python .input}
x.size
```

```{.python .input}
#@tab pytorch
x.numel()
```

```{.python .input}
#@tab tensorflow
tf.size(x)
```

```{.python .input}
#@tab paddle
x.numel()
```

[**要想改變一個張量的形狀而不改變元素數量和元素值，可以呼叫`reshape`函式。**]
例如，可以把張量`x`從形狀為（12,）的行向量轉換為形狀為（3,4）的矩陣。
這個新的張量包含與轉換前相同的值，但是它被看成一個3行4列的矩陣。
要重點說明一下，雖然張量的形狀發生了改變，但其元素值並沒有變。
注意，透過改變張量的形狀，張量的大小不會改變。

```{.python .input}
#@tab mxnet, pytorch
X = x.reshape(3, 4)
X
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(x, (3, 4))
X
```

```{.python .input}
#@tab paddle
X = paddle.reshape(x, (3, 4))
X
```

我們不需要透過手動指定每個維度來改變形狀。
也就是說，如果我們的目標形狀是（高度,寬度），
那麼在知道寬度後，高度會被自動計算得出，不必我們自己做除法。
在上面的例子中，為了獲得一個3行的矩陣，我們手動指定了它有3行和4列。
幸運的是，我們可以透過`-1`來呼叫此自動計算出維度的功能。
即我們可以用`x.reshape(-1,4)`或`x.reshape(3,-1)`來取代`x.reshape(3,4)`。

有時，我們希望[**使用全0、全1、其他常數，或者從特定分佈中隨機取樣的數字**]來初始化矩陣。
我們可以建立一個形狀為（2,3,4）的張量，其中所有元素都設定為0。程式碼如下：

```{.python .input}
np.zeros((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.zeros((2, 3, 4))
```

```{.python .input}
#@tab tensorflow
tf.zeros((2, 3, 4))
```

```{.python .input}
#@tab paddle
paddle.zeros((2, 3, 4))
```

同樣，我們可以建立一個形狀為`(2,3,4)`的張量，其中所有元素都設定為1。程式碼如下：

```{.python .input}
np.ones((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.ones((2, 3, 4))
```

```{.python .input}
#@tab tensorflow
tf.ones((2, 3, 4))
```

```{.python .input}
#@tab paddle
paddle.ones((2, 3, 4))
```

有時我們想透過從某個特定的機率分佈中隨機取樣來得到張量中每個元素的值。
例如，當我們構造陣列來作為神經網路中的引數時，我們通常會隨機初始化引數的值。
以下程式碼建立一個形狀為（3,4）的張量。
其中的每個元素都從均值為0、標準差為1的標準高斯分佈（常態分佈）中隨機取樣。

```{.python .input}
np.random.normal(0, 1, size=(3, 4))
```

```{.python .input}
#@tab pytorch
torch.randn(3, 4)
```

```{.python .input}
#@tab tensorflow
tf.random.normal(shape=[3, 4])
```

```{.python .input}
#@tab paddle
paddle.randn((3, 4),'float32')
```

我們還可以[**透過提供包含數值的Python清單（或巢狀清單），來為所需張量中的每個元素賦予確定值**]。
在這裡，最外層的清單對應於軸0，內層的清單對應於軸1。

```{.python .input}
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab pytorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab tensorflow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab paddle
paddle.to_tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## 運算符

我們的興趣不僅限於讀取資料和寫入資料。
我們想在這些資料上執行數學運算，其中最簡單且最有用的操作是*按元素*（elementwise）運算。
它們將標準標量運算符應用於陣列的每個元素。
對於將兩個陣列作為輸入的函式，按元素運算將二元運算符應用於兩個陣列中的每對位置對應的元素。
我們可以基於任何從標量到標量的函式來建立按元素函式。

在數學表示法中，我們將透過符號$f: \mathbb{R} \rightarrow \mathbb{R}$
來表示*一元*標量運算符（只接收一個輸入）。
這意味著該函式從任何實數（$\mathbb{R}$）對映到另一個實數。
同樣，我們透過符號$f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$
表示*二元*標量運算符，這意味著該函式接收兩個輸入，併產生一個輸出。
給定同一形狀的任意兩個向量$\mathbf{u}$和$\mathbf{v}$和二元運算符$f$，
我們可以得到向量$\mathbf{c} = F(\mathbf{u},\mathbf{v})$。
具體計算方法是$c_i \gets f(u_i, v_i)$，
其中$c_i$、$u_i$和$v_i$分別是向量$\mathbf{c}$、$\mathbf{u}$和$\mathbf{v}$中的元素。
在這裡，我們透過將標量函式升級為按元素向量運算來生成向量值
$F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$。

對於任意具有相同形狀的張量，
[**常見的標準算術運算符（`+`、`-`、`*`、`/`和`**`）都可以被升級為按元素運算**]。
我們可以在同一形狀的任意兩個張量上呼叫按元素操作。
在下面的例子中，我們使用逗號來表示一個具有5個元素的元組，其中每個元素都是按元素操作的結果。

```{.python .input}
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # **運算符是求冪運算
```

```{.python .input}
#@tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # **運算符是求冪運算
```

```{.python .input}
#@tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # **運算符是求冪運算
```

```{.python .input}
#@tab paddle
x = paddle.to_tensor([1.0, 2, 4, 8])
y = paddle.to_tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x**y  # **運算符是求冪運算
```

(**『按元素』方式可以應用更多的計算**)，包括像求冪這樣的一元運算符。

```{.python .input}
np.exp(x)
```

```{.python .input}
#@tab pytorch
torch.exp(x)
```

```{.python .input}
#@tab tensorflow
tf.exp(x)
```

```{.python .input}
#@tab paddle
paddle.exp(x)
```

除了按元素計算外，我們還可以執行線性代數運算，包括向量點積和矩陣乘法。
我們將在 :numref:`sec_linear-algebra`中解釋線性代數的重點內容。

[**我們也可以把多個張量*連結*（concatenate）在一起**]，
把它們端對端地疊起來形成一個更大的張量。
我們只需要提供張量清單，並給出沿哪個軸連結。
下面的例子分別示範了當我們沿行（軸-0，形狀的第一個元素）
和按列（軸-1，形狀的第二個元素）連結兩個矩陣時，會發生什麼情況。
我們可以看到，第一個輸出張量的軸-0長度（$6$）是兩個輸入張量軸-0長度的總和（$3 + 3$）；
第二個輸出張量的軸-1長度（$8$）是兩個輸入張量軸-1長度的總和（$4 + 4$）。

```{.python .input}
X = np.arange(12).reshape(3, 4)
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([X, Y], axis=0), np.concatenate([X, Y], axis=1)
```

```{.python .input}
#@tab pytorch
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1)
```

```{.python .input}
#@tab paddle
X = paddle.arange(12, dtype='float32').reshape((3, 4))
Y = paddle.to_tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
paddle.concat((X, Y), axis=0), paddle.concat((X, Y), axis=1)
```

有時，我們想[**透過*邏輯運算符*建立二元張量**]。
以`X == Y`為例：
對於每個位置，如果`X`和`Y`在該位置相等，則新張量中相應項的值為1。
這意味著邏輯語句`X == Y`在該位置處為真，否則該位置為0。

```{.python .input}
#@tab all
X == Y
```

[**對張量中的所有元素進行求和，會產生一個單元素張量。**]

```{.python .input}
#@tab mxnet, pytorch, paddle
X.sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(X)
```

## 廣播機制
:label:`subsec_broadcasting`

在上面的部分中，我們看到瞭如何在相同形狀的兩個張量上執行按元素操作。
在某些情況下，[**即使形狀不同，我們仍然可以透過呼叫
*廣播機制*（broadcasting mechanism）來執行按元素操作**]。
這種機制的工作方式如下：

1. 透過適當複製元素來擴充套件一個或兩個陣列，以便在轉換之後，兩個張量具有相同的形狀；
2. 對生成的陣列執行按元素操作。

在大多數情況下，我們將沿著陣列中長度為1的軸進行廣播，如下例子：

```{.python .input}
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
#@tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
#@tab tensorflow
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
a, b
```

```{.python .input}
#@tab paddle
a = paddle.reshape(paddle.arange(3), (3, 1))
b = paddle.reshape(paddle.arange(2), (1, 2))
a, b
```

由於`a`和`b`分別是$3\times1$和$1\times2$矩陣，如果讓它們相加，它們的形狀不對應。
我們將兩個矩陣*廣播*為一個更大的$3\times2$矩陣，如下所示：矩陣`a`將複製列，
矩陣`b`將複製行，然後再按元素相加。

```{.python .input}
#@tab all
a + b
```

## 索引和切片

就像在任何其他Python陣列中一樣，張量中的元素可以透過索引訪問。
與任何Python陣列一樣：第一個元素的索引是0，最後一個元素索引是-1；
可以指定範圍以包含第一個元素和最後一個之前的元素。

如下所示，我們[**可以用`[-1]`選擇最後一個元素，可以用`[1:3]`選擇第二個和第三個元素**]：

```{.python .input}
#@tab all
X[-1], X[1:3]
```

:begin_tab:`mxnet, pytorch`
[**除讀取外，我們還可以透過指定索引來將元素寫入矩陣。**]
:end_tab:

:begin_tab:`tensorflow`
TensorFlow中的`Tensors`是不可變的，也不能被賦值。
TensorFlow中的`Variables`是支援賦值的可變容器。
請記住，TensorFlow中的梯度不會透過`Variable`反向傳播。

除了為整個`Variable`分配一個值之外，我們還可以透過索引來寫入`Variable`的元素。
:end_tab:

```{.python .input}
#@tab mxnet, pytorch, paddle
X[1, 2] = 9
X
```

```{.python .input}
#@tab tensorflow
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
X_var
```

如果我們想[**為多個元素賦值相同的值，我們只需要索引所有元素，然後為它們賦值。**]
例如，`[0:2, :]`訪問第1行和第2行，其中『:』代表沿軸1（列）的所有元素。
雖然我們討論的是矩陣的索引，但這也適用於向量和超過2個維度的張量。

```{.python .input}
#@tab mxnet, pytorch, paddle
X[0:2, :] = 12
X
```

```{.python .input}
#@tab tensorflow
X_var = tf.Variable(X)
X_var[0:2, :].assign(tf.ones(X_var[0:2,:].shape, dtype = tf.float32) * 12)
X_var
```

## 節省記憶體

[**執行一些操作可能會導致為新結果分配記憶體**]。
例如，如果我們用`Y = X + Y`，我們將取消引用`Y`指向的張量，而是指向新分配的記憶體處的張量。

在下面的例子中，我們用Python的`id()`函式示範了這一點，
它給我們提供了記憶體中引用物件的確切地址。
執行`Y = Y + X`後，我們會發現`id(Y)`指向另一個位置。
這是因為Python首先計算`Y + X`，為結果分配新的記憶體，然後使`Y`指向記憶體中的這個新位置。

```{.python .input}
#@tab all
before = id(Y)
Y = Y + X
id(Y) == before
```

這可能是不可取的，原因有兩個：

1. 首先，我們不想總是不必要地分配記憶體。在機器學習中，我們可能有數百兆的引數，並且在一秒內多次更新所有引數。通常情況下，我們希望原地執行這些更新；
2. 如果我們不原地更新，其他引用仍然會指向舊的記憶體位置，這樣我們的某些程式碼可能會無意中引用舊的引數。

:begin_tab:`mxnet, pytorch`
幸運的是，(**執行原地操作**)非常簡單。
我們可以使用切片表示法將操作的結果分配給先前分配的陣列，例如`Y[:] = <expression>`。
為了說明這一點，我們首先建立一個新的矩陣`Z`，其形狀與另一個`Y`相同，
使用`zeros_like`來分配一個全$0$的區塊。
:end_tab:

:begin_tab:`tensorflow`
`Variables`是TensorFlow中的可變容器，它們提供了一種儲存模型引數的方法。
我們可以透過`assign`將一個操作的結果分配給一個`Variable`。
為了說明這一點，我們建立了一個與另一個張量`Y`相同的形狀的`Z`，
使用`zeros_like`來分配一個全$0$的區塊。
:end_tab:

```{.python .input}
Z = np.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
#@tab pytorch
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
#@tab tensorflow
Z = tf.Variable(tf.zeros_like(Y))
print('id(Z):', id(Z))
Z.assign(X + Y)
print('id(Z):', id(Z))
```

```{.python .input}
#@tab paddle
Z = paddle.zeros_like(Y)
print('id(Z):', id(Z))
Z = X + Y
print('id(Z):', id(Z))
```

:begin_tab:`mxnet, pytorch`
[**如果在後續計算中沒有重複使用`X`，
我們也可以使用`X[:] = X + Y`或`X += Y`來減少操作的記憶體開銷。**]
:end_tab:

:begin_tab:`tensorflow`
即使你將狀態持久儲存在`Variable`中，
你也可能希望避免為不是模型引數的張量過度分配記憶體，從而進一步減少記憶體使用量。

由於TensorFlow的`Tensors`是不可變的，而且梯度不會透過`Variable`流動，
因此TensorFlow沒有提供一種明確的方式來原地執行單個操作。

但是，TensorFlow提供了`tf.function`修飾符，
將計算封裝在TensorFlow圖中，該圖在執行前經過編譯和最佳化。
這允許TensorFlow刪除未使用的值，並複用先前分配的且不再需要的值。
這樣可以最大限度地減少TensorFlow計算的記憶體開銷。
:end_tab:

```{.python .input}
#@tab mxnet, pytorch, paddle
before = id(X)
X += Y
id(X) == before
```

```{.python .input}
#@tab tensorflow
@tf.function
def computation(X, Y):
    Z = tf.zeros_like(Y)  # 這個未使用的值將被刪除
    A = X + Y  # 當不再需要時，分配將被複用
    B = A + Y
    C = B + Y
    return C + Y

computation(X, Y)
```

## 轉換為其他Python物件
:begin_tab:`mxnet, tensorflow`
將深度學習框架定義的張量[**轉換為NumPy張量（`ndarray`）**]很容易，反之也同樣容易。
轉換後的結果不共享記憶體。
這個小的不便實際上是非常重要的：當在CPU或GPU上執行操作的時候，
如果Python的NumPy套件也希望使用相同的記憶體區塊執行其他操作，人們不希望停下計算來等它。
:end_tab:

:begin_tab:`pytorch`
將深度學習框架定義的張量[**轉換為NumPy張量（`ndarray`）**]很容易，反之也同樣容易。
torch張量和numpy陣列將共享它們的底層記憶體，就地操作更改一個張量也會同時更改另一個張量。
:end_tab:

```{.python .input}
A = X.asnumpy()
B = np.array(A)
type(A), type(B)
```

```{.python .input}
#@tab pytorch
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)
```

```{.python .input}
#@tab tensorflow
A = X.numpy()
B = tf.constant(A)
type(A), type(B)
```

```{.python .input}
#@tab paddle
A = X.numpy()
B = paddle.to_tensor(A)
type(A), type(B)
```

要(**將大小為1的張量轉換為Python標量**)，我們可以呼叫`item`函式或Python的內建函式。

```{.python .input}
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab paddle
a = paddle.to_tensor([3.5])
a, a.item(), float(a), int(a)
```

## 小結

* 深度學習儲存和操作數據的主要介面是張量（$n$維陣列）。它提供了各種功能，包括基本數學運算、廣播、索引、切片、記憶體節省和轉換其他Python物件。

## 練習

1. 執行本節中的程式碼。將本節中的條件語句`X == Y`更改為`X < Y`或`X > Y`，然後看看你可以得到什麼樣的張量。
1. 用其他形狀（例如三維張量）替換廣播機制中按元素操作的兩個張量。結果是否與預期相同？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1745)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1747)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1746)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11680)
:end_tab:
