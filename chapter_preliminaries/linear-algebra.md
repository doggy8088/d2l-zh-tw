# 線性代數
:label:`sec_linear-algebra`

在介紹完如何儲存和操作數據後，接下來將簡要地回顧一下部分基本線性代數內容。
這些內容有助於讀者瞭解和實現本書中介紹的大多數模型。
本節將介紹線性代數中的基本數學物件、算術和運算，並用數學符號和相應的程式碼實現來表示它們。

## 標量


如果你曾經在餐廳支付餐費，那麼應該已經知道一些基本的線性代數，比如在數字間相加或相乘。
例如，北京的溫度為$52^{\circ}F$（華氏度，除攝氏度外的另一種溫度計量單位）。
嚴格來說，僅包含一個數值被稱為*標量*（scalar）。
如果要將此華氏度值轉換為更常用的攝氏度，
則可以計算表示式$c=\frac{5}{9}(f-32)$，並將$f$賦為$52$。
在此等式中，每一項（$5$、$9$和$32$）都是標量值。
符號$c$和$f$稱為*變數*（variable），它們表示未知的標量值。

本書採用了數學表示法，其中標量變數由普通小寫字母表示（例如，$x$、$y$和$z$）。
本書用$\mathbb{R}$表示所有（連續）*實數*標量的空間，之後將嚴格定義*空間*（space）是什麼，
但現在只要記住表示式$x\in\mathbb{R}$是表示$x$是一個實值標量的正式形式。
符號$\in$稱為『屬於』，它表示『是集合中的成員』。
例如$x, y \in \{0,1\}$可以用來表明$x$和$y$是值只能為$0$或$1$的數字。

(**標量由只有一個元素的張量表示**)。
下面的程式碼將實例化兩個標量，並執行一些熟悉的算術運算，即加法、乘法、除法和指數。

```{.python .input}
from mxnet import np, npx
npx.set_np()

x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

```{.python .input}
#@tab pytorch
import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

x = tf.constant(3.0)
y = tf.constant(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
#@tab paddle
import warnings
warnings.filterwarnings(action='ignore')
import paddle

x = paddle.to_tensor([3.0])
y = paddle.to_tensor([2.0])

x + y, x * y, x / y, x**y
```

## 向量

[**向量可以被視為標量值組成的清單**]。
這些標量值被稱為向量的*元素*（element）或*分量*（component）。
當向量表示資料集中的樣本時，它們的值具有一定的現實意義。
例如，如果我們正在訓練一個模型來預測貸款違約風險，可能會將每個申請人與一個向量相關聯，
其分量與其收入、工作年限、過往違約次數和其他因素相對應。
如果我們正在研究醫院患者可能面臨的心臟病發作風險，可能會用一個向量來表示每個患者，
其分量為最近的生命體徵、膽固醇水平、每天運動時間等。
在數學表示法中，向量通常記為粗體、小寫的符號
（例如，$\mathbf{x}$、$\mathbf{y}$和$\mathbf{z})$）。

人們透過一維張量表示向量。一般來說，張量可以具有任意長度，取決於機器的記憶體限制。

```{.python .input}
x = np.arange(4)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(4)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(4)
x
```

```{.python .input}
#@tab paddle
x = paddle.arange(4)
x
```

我們可以使用下標來引用向量的任一元素，例如可以透過$x_i$來引用第$i$個元素。
注意，元素$x_i$是一個標量，所以我們在引用它時不會加粗。
大量文獻認為列向量是向量的預設方向，在本書中也是如此。
在數學中，向量$\mathbf{x}$可以寫為：

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\x_{2}  \\ \vdots  \\x_{n}\end{bmatrix},$$
:eqlabel:`eq_vec_def`

其中$x_1,\ldots,x_n$是向量的元素。在程式碼中，我們(**透過張量的索引來訪問任一元素**)。

```{.python .input}
x[3]
```

```{.python .input}
#@tab pytorch
x[3]
```

```{.python .input}
#@tab tensorflow
x[3]
```

```{.python .input}
#@tab paddle
x[3]
```

### 長度、維度和形狀

向量只是一個數字陣列，就像每個陣列都有一個長度一樣，每個向量也是如此。
在數學表示法中，如果我們想說一個向量$\mathbf{x}$由$n$個實值標量組成，
可以將其表示為$\mathbf{x}\in\mathbb{R}^n$。
向量的長度通常稱為向量的*維度*（dimension）。

與普通的Python陣列一樣，我們可以透過呼叫Python的內建`len()`函式來[**訪問張量的長度**]。

```{.python .input}
len(x)
```

```{.python .input}
#@tab pytorch
len(x)
```

```{.python .input}
#@tab tensorflow
len(x)
```

```{.python .input}
#@tab paddle
len(x)
```

當用張量表示一個向量（只有一個軸）時，我們也可以透過`.shape`屬性訪問向量的長度。
形狀（shape）是一個元素組，列出了張量沿每個軸的長度（維數）。
對於(**只有一個軸的張量，形狀只有一個元素。**)

```{.python .input}
x.shape
```

```{.python .input}
#@tab pytorch
x.shape
```

```{.python .input}
#@tab tensorflow
x.shape
```

```{.python .input}
#@tab paddle
x.shape
```

請注意，*維度*（dimension）這個詞在不同上下文時往往會有不同的含義，這經常會使人感到困惑。
為了清楚起見，我們在此明確一下：
*向量*或*軸*的維度被用來表示*向量*或*軸*的長度，即向量或軸的元素數量。
然而，張量的維度用來表示張量具有的軸數。
在這個意義上，張量的某個軸的維數就是這個軸的長度。

## 矩陣

正如向量將標量從零階推廣到一階，矩陣將向量從一階推廣到二階。
矩陣，我們通常用粗體、大寫字母來表示
（例如，$\mathbf{X}$、$\mathbf{Y}$和$\mathbf{Z}$），
在程式碼中表示為具有兩個軸的張量。

數學表示法使用$\mathbf{A} \in \mathbb{R}^{m \times n}$
來表示矩陣$\mathbf{A}$，其由$m$行和$n$列的實值標量組成。
我們可以將任意矩陣$\mathbf{A} \in \mathbb{R}^{m \times n}$視為一個表格，
其中每個元素$a_{ij}$屬於第$i$行第$j$列：

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`

對於任意$\mathbf{A} \in \mathbb{R}^{m \times n}$，
$\mathbf{A}$的形狀是（$m$,$n$）或$m \times n$。
當矩陣具有相同數量的行和列時，其形狀將變為正方形；
因此，它被稱為*方陣*（square matrix）。

當呼叫函式來實例化張量時，
我們可以[**透過指定兩個分量$m$和$n$來建立一個形狀為$m \times n$的矩陣**]。

```{.python .input}
A = np.arange(20).reshape(5, 4)
A
```

```{.python .input}
#@tab pytorch
A = torch.arange(20).reshape(5, 4)
A
```

```{.python .input}
#@tab tensorflow
A = tf.reshape(tf.range(20), (5, 4))
A
```

```{.python .input}
#@tab paddle
A = paddle.reshape(paddle.arange(20), (5, 4))
A
```

我們可以透過行索引（$i$）和列索引（$j$）來訪問矩陣中的標量元素$a_{ij}$，
例如$[\mathbf{A}]_{ij}$。
如果沒有給出矩陣$\mathbf{A}$的標量元素，如在 :eqref:`eq_matrix_def`那樣，
我們可以簡單地使用矩陣$\mathbf{A}$的小寫字母索引下標$a_{ij}$
來引用$[\mathbf{A}]_{ij}$。
為了表示起來簡單，只有在必要時才會將逗號插入到單獨的索引中，
例如$a_{2,3j}$和$[\mathbf{A}]_{2i-1,3}$。

當我們交換矩陣的行和列時，結果稱為矩陣的*轉置*（transpose）。
通常用$\mathbf{a}^\top$來表示矩陣的轉置，如果$\mathbf{B}=\mathbf{A}^\top$，
則對於任意$i$和$j$，都有$b_{ij}=a_{ji}$。
因此，在 :eqref:`eq_matrix_def`中的轉置是一個形狀為$n \times m$的矩陣：

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

現在在程式碼中訪問(**矩陣的轉置**)。

```{.python .input}
A.T
```

```{.python .input}
#@tab pytorch
A.T
```

```{.python .input}
#@tab tensorflow
tf.transpose(A)
```

```{.python .input}
#@tab paddle
paddle.transpose(A, perm=[1, 0])
```

作為方陣的一種特殊型別，[***對稱矩陣*（symmetric matrix）$\mathbf{A}$等於其轉置：$\mathbf{A} = \mathbf{A}^\top$**]。
這裡定義一個對稱矩陣$\mathbf{B}$：

```{.python .input}
B = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```{.python .input}
#@tab pytorch
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```{.python .input}
#@tab tensorflow
B = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```{.python .input}
#@tab paddle
B = paddle.to_tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

現在我們將`B`與它的轉置進行比較。

```{.python .input}
B == B.T
```

```{.python .input}
#@tab pytorch
B == B.T
```

```{.python .input}
#@tab tensorflow
B == tf.transpose(B)
```

```{.python .input}
#@tab paddle
B == paddle.transpose(B, perm=[1, 0])
```

矩陣是有用的資料結構：它們允許我們組織具有不同模式的資料。
例如，我們矩陣中的行可能對應於不同的房屋（資料樣本），而列可能對應於不同的屬性。
曾經使用過電子表格軟體或已閱讀過 :numref:`sec_pandas`的人，應該對此很熟悉。
因此，儘管單個向量的預設方向是列向量，但在表示表格資料集的矩陣中，
將每個資料樣本作為矩陣中的行向量更為常見。
後面的章節將講到這點，這種約定將支援常見的深度學習實踐。
例如，沿著張量的最外軸，我們可以訪問或遍歷小批次的資料樣本。


## 張量

[**就像向量是標量的推廣，矩陣是向量的推廣一樣，我們可以建立具有更多軸的資料結構**]。
張量（本小節中的『張量』指代數物件）是描述具有任意數量軸的$n$維陣列的通用方法。
例如，向量是一階張量，矩陣是二階張量。
張量用特殊字型的大寫字母表示（例如，$\mathsf{X}$、$\mathsf{Y}$和$\mathsf{Z}$），
它們的索引機制（例如$x_{ijk}$和$[\mathsf{X}]_{1,2i-1,3}$）與矩陣類似。

當我們開始處理影象時，張量將變得更加重要，影象以$n$維陣列形式出現，
其中3個軸對應於高度、寬度，以及一個*通道*（channel）軸，
用於表示顏色通道（紅色、綠色和藍色）。
現在先將高階張量暫放一邊，而是專注學習其基礎知識。

```{.python .input}
X = np.arange(24).reshape(2, 3, 4)
X
```

```{.python .input}
#@tab pytorch
X = torch.arange(24).reshape(2, 3, 4)
X
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(24), (2, 3, 4))
X
```

```{.python .input}
#@tab paddle
X = paddle.reshape(paddle.arange(24), (2, 3, 4))
X
```

## 張量演算法的基本性質

標量、向量、矩陣和任意數量軸的張量（本小節中的『張量』指代數物件）有一些實用的屬性。
例如，從按元素操作的定義中可以注意到，任何按元素的一元運算都不會改變其操作數的形狀。
同樣，[**給定具有相同形狀的任意兩個張量，任何按元素二元運算的結果都將是相同形狀的張量**]。
例如，將兩個相同形狀的矩陣相加，會在這兩個矩陣上執行元素加法。

```{.python .input}
A = np.arange(20).reshape(5, 4)
B = A.copy()  # 透過分配新記憶體，將A的一個副本分配給B
A, A + B
```

```{.python .input}
#@tab pytorch
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 透過分配新記憶體，將A的一個副本分配給B
A, A + B
```

```{.python .input}
#@tab tensorflow
A = tf.reshape(tf.range(20, dtype=tf.float32), (5, 4))
B = A  # 不能透過分配新記憶體將A克隆到B
A, A + B
```

```{.python .input}
#@tab paddle
A = paddle.reshape(paddle.arange(20, dtype=paddle.float32), (5, 4))
B = A.clone()  # 透過分配新記憶體，將A的一個副本分配給B
A, A + B
```

具體而言，[**兩個矩陣的按元素乘法稱為*Hadamard積*（Hadamard product）（數學符號$\odot$）**]。
對於矩陣$\mathbf{B} \in \mathbb{R}^{m \times n}$，
其中第$i$行和第$j$列的元素是$b_{ij}$。
矩陣$\mathbf{A}$（在 :eqref:`eq_matrix_def`中定義）和$\mathbf{B}$的Hadamard積為：
$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

```{.python .input}
A * B
```

```{.python .input}
#@tab pytorch
A * B
```

```{.python .input}
#@tab tensorflow
A * B
```

```{.python .input}
#@tab paddle
A * B
```

將張量乘以或加上一個標量不會改變張量的形狀，其中張量的每個元素都將與標量相加或相乘。

```{.python .input}
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
#@tab pytorch
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
#@tab tensorflow
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
a + X, (a * X).shape
```

```{.python .input}
#@tab paddle
a = 2
X = paddle.reshape(paddle.arange(24), (2, 3, 4))
a + X, (a * X).shape
```

## 降維

:label:`subseq_lin-alg-reduction`

我們可以對任意張量進行的一個有用的操作是[**計算其元素的和**]。
數學表示法使用$\sum$符號表示求和。
為了表示長度為$d$的向量中元素的總和，可以記為$\sum_{i=1}^dx_i$。
在程式碼中可以呼叫計算求和的函式：

```{.python .input}
x = np.arange(4)
x, x.sum()
```

```{.python .input}
#@tab pytorch
x = torch.arange(4, dtype=torch.float32)
x, x.sum()
```

```{.python .input}
#@tab tensorflow
x = tf.range(4, dtype=tf.float32)
x, tf.reduce_sum(x)
```

```{.python .input}
#@tab paddle
x = paddle.arange(4, dtype=paddle.float32)
x, x.sum()
```

我們可以(**表示任意形狀張量的元素和**)。
例如，矩陣$\mathbf{A}$中元素的和可以記為$\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$。

```{.python .input}
A.shape, A.sum()
```

```{.python .input}
#@tab pytorch
A.shape, A.sum()
```

```{.python .input}
#@tab tensorflow
A.shape, tf.reduce_sum(A)
```

```{.python .input}
#@tab paddle
A.shape, A.sum()
```

預設情況下，呼叫求和函式會沿所有的軸降低張量的維度，使它變為一個標量。
我們還可以[**指定張量沿哪一個軸來透過求和降低維度**]。
以矩陣為例，為了透過求和所有行的元素來降維（軸0），可以在呼叫函式時指定`axis=0`。
由於輸入矩陣沿0軸降維以生成輸出向量，因此輸入軸0的維數在輸出形狀中消失。

```{.python .input}
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

```{.python .input}
#@tab pytorch
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

```{.python .input}
#@tab tensorflow
A_sum_axis0 = tf.reduce_sum(A, axis=0)
A_sum_axis0, A_sum_axis0.shape
```

```{.python .input}
#@tab paddle
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

指定`axis=1`將透過彙總所有列的元素降維（軸1）。因此，輸入軸1的維數在輸出形狀中消失。

```{.python .input}
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

```{.python .input}
#@tab pytorch
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

```{.python .input}
#@tab tensorflow
A_sum_axis1 = tf.reduce_sum(A, axis=1)
A_sum_axis1, A_sum_axis1.shape
```

```{.python .input}
#@tab paddle
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

沿著行和列對矩陣求和，等價於對矩陣的所有元素進行求和。

```{.python .input}
A.sum(axis=[0, 1])  # 結果和A.sum()相同
```

```{.python .input}
#@tab pytorch
A.sum(axis=[0, 1])  # 結果和A.sum()相同
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(A, axis=[0, 1])  # 結果和tf.reduce_sum(A)相同
```

```{.python .input}
#@tab paddle
A.sum(axis=[0, 1])
```

[**一個與求和相關的量是*平均值*（mean或average）**]。
我們透過將總和除以元素總數來計算平均值。
在程式碼中，我們可以呼叫函式來計算任意形狀張量的平均值。

```{.python .input}
A.mean(), A.sum() / A.size
```

```{.python .input}
#@tab pytorch
A.mean(), A.sum() / A.numel()
```

```{.python .input}
#@tab tensorflow
tf.reduce_mean(A), tf.reduce_sum(A) / tf.size(A).numpy()
```

```{.python .input}
#@tab paddle
A.mean(), A.sum() / A.numel()
```

同樣，計算平均值的函式也可以沿指定軸降低張量的維度。

```{.python .input}
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
#@tab pytorch
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
#@tab tensorflow
tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0]
```

```{.python .input}
#@tab paddle
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

### 非降維求和

:label:`subseq_lin-alg-non-reduction`

但是，有時在呼叫函式來[**計算總和或均值時保持軸數不變**]會很有用。

```{.python .input}
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```

```{.python .input}
#@tab pytorch
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```

```{.python .input}
#@tab tensorflow
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
sum_A
```

```{.python .input}
#@tab paddle
sum_A = paddle.sum(A, axis=1, keepdim=True)
sum_A
```

例如，由於`sum_A`在對每行進行求和後仍保持兩個軸，我們可以(**透過廣播將`A`除以`sum_A`**)。

```{.python .input}
A / sum_A
```

```{.python .input}
#@tab pytorch
A / sum_A
```

```{.python .input}
#@tab tensorflow
A / sum_A
```

```{.python .input}
#@tab paddle
A / sum_A
```

如果我們想沿[**某個軸計算`A`元素的累積總和**]，
比如`axis=0`（按行計算），可以呼叫`cumsum`函式。
此函式不會沿任何軸降低輸入張量的維度。

```{.python .input}
A.cumsum(axis=0)
```

```{.python .input}
#@tab pytorch
A.cumsum(axis=0)
```

```{.python .input}
#@tab tensorflow
tf.cumsum(A, axis=0)
```

```{.python .input}
#@tab paddle
A.cumsum(axis=0)
```

## 點積（Dot Product）

我們已經學習了按元素操作、求和及平均值。
另一個最基本的操作之一是點積。
給定兩個向量$\mathbf{x},\mathbf{y}\in\mathbb{R}^d$，
它們的*點積*（dot product）$\mathbf{x}^\top\mathbf{y}$
（或$\langle\mathbf{x},\mathbf{y}\rangle$）
是相同位置的按元素乘積的和：$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$。

[~~點積是相同位置的按元素乘積的和~~]

```{.python .input}
y = np.ones(4)
x, y, np.dot(x, y)
```

```{.python .input}
#@tab pytorch
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```{.python .input}
#@tab tensorflow
y = tf.ones(4, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)
```

```{.python .input}
#@tab paddle
y = paddle.ones(shape=[4], dtype='float32')
x, y, paddle.dot(x, y)
```

注意，(**我們可以透過執行按元素乘法，然後進行求和來表示兩個向量的點積**)：

```{.python .input}
np.sum(x * y)
```

```{.python .input}
#@tab pytorch
torch.sum(x * y)
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(x * y)
```

```{.python .input}
#@tab paddle
paddle.sum(x * y)
```

點積在很多場合都很有用。
例如，給定一組由向量$\mathbf{x} \in \mathbb{R}^d$表示的值，
和一組由$\mathbf{w} \in \mathbb{R}^d$表示的權重。
$\mathbf{x}$中的值根據權重$\mathbf{w}$的加權和，
可以表示為點積$\mathbf{x}^\top \mathbf{w}$。
當權重為非負數且和為1（即$\left(\sum_{i=1}^{d}{w_i}=1\right)$）時，
點積表示*加權平均*（weighted average）。
將兩個向量規範化得到單位長度後，點積表示它們夾角的餘弦。
本節後面的內容將正式介紹*長度*（length）的概念。

## 矩陣-向量積

現在我們知道如何計算點積，可以開始理解*矩陣-向量積*（matrix-vector product）。
回顧分別在 :eqref:`eq_matrix_def`和 :eqref:`eq_vec_def`中定義的矩陣$\mathbf{A} \in \mathbb{R}^{m \times n}$和向量$\mathbf{x} \in \mathbb{R}^n$。
讓我們將矩陣$\mathbf{A}$用它的行向量表示：

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

其中每個$\mathbf{a}^\top_{i} \in \mathbb{R}^n$都是行向量，表示矩陣的第$i$行。
[**矩陣向量積$\mathbf{A}\mathbf{x}$是一個長度為$m$的列向量，
其第$i$個元素是點積$\mathbf{a}^\top_i \mathbf{x}$**]：

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$

我們可以把一個矩陣$\mathbf{A} \in \mathbb{R}^{m \times n}$乘法看作一個從$\mathbb{R}^{n}$到$\mathbb{R}^{m}$向量的轉換。
這些轉換是非常有用的，例如可以用方陣的乘法來表示旋轉。
後續章節將講到，我們也可以使用矩陣-向量積來描述在給定前一層的值時，
求解神經網路每一層所需的複雜計算。

:begin_tab:`mxnet`
在程式碼中使用張量表示矩陣-向量積，我們使用與點積相同的`dot`函式。
當我們為矩陣`A`和向量`x`呼叫`np.dot(A,x)`時，會執行矩陣-向量積。
注意，`A`的列維數（沿軸1的長度）必須與`x`的維數（其長度）相同。
:end_tab:

:begin_tab:`pytorch`
在程式碼中使用張量表示矩陣-向量積，我們使用`mv`函式。
當我們為矩陣`A`和向量`x`呼叫`torch.mv(A, x)`時，會執行矩陣-向量積。
注意，`A`的列維數（沿軸1的長度）必須與`x`的維數（其長度）相同。
:end_tab:

:begin_tab:`tensorflow`
在程式碼中使用張量表示矩陣-向量積，我們使用與點積相同的`matvec`函式。
當我們為矩陣`A`和向量`x`呼叫`tf.linalg.matvec(A, x)`時，會執行矩陣-向量積。
注意，`A`的列維數（沿軸1的長度）必須與`x`的維數（其長度）相同。
:end_tab:

```{.python .input}
A.shape, x.shape, np.dot(A, x)
```

```{.python .input}
#@tab pytorch
A.shape, x.shape, torch.mv(A, x)
```

```{.python .input}
#@tab tensorflow
A.shape, x.shape, tf.linalg.matvec(A, x)
```

```{.python .input}
#@tab paddle
A.shape, x.shape, paddle.mv(A, x)
```

## 矩陣-矩陣乘法

在掌握點積和矩陣-向量積的知識後，
那麼**矩陣-矩陣乘法**（matrix-matrix multiplication）應該很簡單。

假設有兩個矩陣$\mathbf{A} \in \mathbb{R}^{n \times k}$和$\mathbf{B} \in \mathbb{R}^{k \times m}$：

$$\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.$$

用行向量$\mathbf{a}^\top_{i} \in \mathbb{R}^k$表示矩陣$\mathbf{A}$的第$i$行，並讓列向量$\mathbf{b}_{j} \in \mathbb{R}^k$作為矩陣$\mathbf{B}$的第$j$列。要生成矩陣積$\mathbf{C} = \mathbf{A}\mathbf{B}$，最簡單的方法是考慮$\mathbf{A}$的行向量和$\mathbf{B}$的列向量:

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$
當我們簡單地將每個元素$c_{ij}$計算為點積$\mathbf{a}^\top_i \mathbf{b}_j$:

$$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$

[**我們可以將矩陣-矩陣乘法$\mathbf{AB}$看作簡單地執行$m$次矩陣-向量積，並將結果拼接在一起，形成一個$n \times m$矩陣**]。
在下面的程式碼中，我們在`A`和`B`上執行矩陣乘法。
這裡的`A`是一個5行4列的矩陣，`B`是一個4行3列的矩陣。
兩者相乘後，我們得到了一個5行3列的矩陣。

```{.python .input}
B = np.ones(shape=(4, 3))
np.dot(A, B)
```

```{.python .input}
#@tab pytorch
B = torch.ones(4, 3)
torch.mm(A, B)
```

```{.python .input}
#@tab tensorflow
B = tf.ones((4, 3), tf.float32)
tf.matmul(A, B)
```

```{.python .input}
#@tab paddle
B = paddle.ones(shape=[4, 3], dtype='float32')
paddle.mm(A, B)
```

矩陣-矩陣乘法可以簡單地稱為**矩陣乘法**，不應與"Hadamard積"混淆。

## 範數
:label:`subsec_lin-algebra-norms`

線性代數中最有用的一些運算符是*範數*（norm）。
非正式地說，向量的*範數*是表示一個向量有多大。
這裡考慮的*大小*（size）概念不涉及維度，而是分量的大小。

線上性代數中，向量範數是將向量對映到標量的函式$f$。
給定任意向量$\mathbf{x}$，向量範數要滿足一些屬性。
第一個性質是：如果我們按常數因子$\alpha$縮放向量的所有元素，
其範數也會按相同常數因子的*絕對值*縮放：

$$f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x}).$$

第二個性質是熟悉的三角不等式:

$$f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y}).$$

第三個性質簡單地說範數必須是非負的:

$$f(\mathbf{x}) \geq 0.$$

這是有道理的。因為在大多數情況下，任何東西的最小的*大小*是0。
最後一個性質要求範數最小為0，若且唯若向量全由0組成。

$$\forall i, [\mathbf{x}]_i = 0 \Leftrightarrow f(\mathbf{x})=0.$$

範數聽起來很像距離的度量。
歐幾里得距離和畢達哥拉斯定理中的非負性概念和三角不等式可能會給出一些啟發。
事實上，歐幾里得距離是一個$L_2$範數：
假設$n$維向量$\mathbf{x}$中的元素是$x_1,\ldots,x_n$，其[**$L_2$*範數*是向量元素平方和的平方根：**]

(**$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2},$$**)

其中，在$L_2$範數中常常省略下標$2$，也就是說$\|\mathbf{x}\|$等同於$\|\mathbf{x}\|_2$。
在程式碼中，我們可以按如下方式計算向量的$L_2$範數。

```{.python .input}
u = np.array([3, -4])
np.linalg.norm(u)
```

```{.python .input}
#@tab pytorch
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

```{.python .input}
#@tab tensorflow
u = tf.constant([3.0, -4.0])
tf.norm(u)
```

```{.python .input}
#@tab paddle
u = paddle.to_tensor([3.0, -4.0])
paddle.norm(u)
```

深度學習中更經常地使用$L_2$範數的平方，也會經常遇到[**$L_1$範數，它表示為向量元素的絕對值之和：**]

(**$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$**)

與$L_2$範數相比，$L_1$範數受異常值的影響較小。
為了計算$L_1$範數，我們將絕對值函式和按元素求和組合起來。

```{.python .input}
np.abs(u).sum()
```

```{.python .input}
#@tab pytorch
torch.abs(u).sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(tf.abs(u))
```

```{.python .input}
#@tab paddle
paddle.abs(u).sum()
```

$L_2$範數和$L_1$範數都是更一般的$L_p$範數的特例：

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

類似於向量的$L_2$範數，[**矩陣**]$\mathbf{X} \in \mathbb{R}^{m \times n}$(**的*Frobenius範數*（Frobenius norm）是矩陣元素平方和的平方根：**)

(**$$\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$**)

Frobenius範數滿足向量範數的所有性質，它就像是矩陣形向量的$L_2$範數。
呼叫以下函式將計算矩陣的Frobenius範數。

```{.python .input}
np.linalg.norm(np.ones((4, 9)))
```

```{.python .input}
#@tab pytorch
torch.norm(torch.ones((4, 9)))
```

```{.python .input}
#@tab tensorflow
tf.norm(tf.ones((4, 9)))
```

```{.python .input}
#@tab paddle
paddle.norm(paddle.ones(shape=[4, 9], dtype='float32'))
```

### 範數和目標

:label:`subsec_norms_and_objectives`

在深度學習中，我們經常試圖解決最佳化問題：
*最大化*分配給觀測資料的機率;
*最小化*預測和真實觀測之間的距離。
用向量表示物品（如單詞、產品或新聞文章），以便最小化相似專案之間的距離，最大化不同專案之間的距離。
目標，或許是深度學習演算法最重要的組成部分（除了資料），通常被表達為範數。

## 關於線性代數的更多訊息

僅用一節，我們就教會了閱讀本書所需的、用以理解現代深度學習的線性代數。
線性代數還有很多，其中很多數學對於機器學習非常有用。
例如，矩陣可以分解為因子，這些分解可以顯示真實世界資料集中的低維結構。
機器學習的整個子領域都側重於使用矩陣分解及其向高階張量的泛化，來發現資料集中的結構並解決預測問題。
當開始動手嘗試並在真實資料集上應用了有效的機器學習模型，你會更傾向於學習更多數學。
因此，這一節到此結束，本書將在後面介紹更多數學知識。

如果渴望瞭解有關線性代數的更多訊息，可以參考[線性代數運算的線上附錄](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/geometry-linear-algebraic-ops.html)或其他優秀資源 :cite:`Strang.1993,Kolter.2008,Petersen.Pedersen.ea.2008`。

## 小結

* 標量、向量、矩陣和張量是線性代數中的基本數學物件。
* 向量泛化自標量，矩陣泛化自向量。
* 標量、向量、矩陣和張量分別具有零、一、二和任意數量的軸。
* 一個張量可以透過`sum`和`mean`沿指定的軸降低維度。
* 兩個矩陣的按元素乘法被稱為他們的Hadamard積。它與矩陣乘法不同。
* 在深度學習中，我們經常使用範數，如$L_1$範數、$L_2$範數和Frobenius範數。
* 我們可以對標量、向量、矩陣和張量執行各種操作。

## 練習

1. 證明一個矩陣$\mathbf{A}$的轉置的轉置是$\mathbf{A}$，即$(\mathbf{A}^\top)^\top = \mathbf{A}$。
1. 給出兩個矩陣$\mathbf{A}$和$\mathbf{B}$，證明『它們轉置的和』等於『它們和的轉置』，即$\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$。
1. 給定任意方陣$\mathbf{A}$，$\mathbf{A} + \mathbf{A}^\top$總是對稱的嗎?為什麼?
1. 本節中定義了形狀$(2,3,4)$的張量`X`。`len(X)`的輸出結果是什麼？
1. 對於任意形狀的張量`X`,`len(X)`是否總是對應於`X`特定軸的長度?這個軸是什麼?
1. 執行`A/A.sum(axis=1)`，看看會發生什麼。請分析一下原因？
1. 考慮一個具有形狀$(2,3,4)$的張量，在軸0、1、2上的求和輸出是什麼形狀?
1. 為`linalg.norm`函式提供3個或更多軸的張量，並觀察其輸出。對於任意形狀的張量這個函式計算得到什麼?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1752)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1751)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1753)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11682)
:end_tab:
