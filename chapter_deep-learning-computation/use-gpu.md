# GPU
:label:`sec_use_gpu`

在 :numref:`tab_intro_decade`中，
我們回顧了過去20年計算能力的快速增長。
簡而言之，自2000年以來，GPU效能每十年增長1000倍。

本節，我們將討論如何利用這種計算效能進行研究。
首先是如何使用單個GPU，然後是如何使用多個GPU和多個服務器（具有多個GPU）。

我們先看看如何使用單個NVIDIA GPU進行計算。
首先，確保至少安裝了一個NVIDIA GPU。
然後，下載[NVIDIA驅動和CUDA](https://developer.nvidia.com/cuda-downloads)
並按照提示設定適當的路徑。
當這些準備工作完成，就可以使用`nvidia-smi`指令來(**檢視顯卡訊息。**)

```{.python .input}
#@tab all
!nvidia-smi
```

:begin_tab:`mxnet`
讀者可能已經注意到MXNet張量看起來與NumPy的`ndarray`幾乎相同。
但有一些關鍵區別，其中之一是MXNet支援不同的硬體裝置。

在MXNet中，每個陣列都有一個環境（context）。
預設情況下，所有變數和相關的計算都分配給CPU。
有時環境可能是GPU。
當我們跨多個服務器部署作業時，事情會變得更加棘手。
透過智慧地將陣列分配給環境，
我們可以最大限度地減少在裝置之間傳輸資料的時間。
例如，當在帶有GPU的伺服器上訓練神經網路時，
我們通常希望模型的引數在GPU上。

接下來，我們需要確認是否安裝了MXNet的GPU版本。
如果已經安裝了MXNet的CPU版本，我們需要先解除安裝它。
例如，使用`pip uninstall mxnet`指令，
然後根據CUDA版本安裝相應的MXNet的GPU版本。
例如，假設已經安裝了CUDA10.0，可以透過`pip install mxnet-cu100`安裝支援CUDA10.0的MXNet版本。
:end_tab:

:begin_tab:`pytorch`
在PyTorch中，每個陣列都有一個裝置（device），
我們通常將其稱為環境（context）。
預設情況下，所有變數和相關的計算都分配給CPU。
有時環境可能是GPU。
當我們跨多個服務器部署作業時，事情會變得更加棘手。
透過智慧地將陣列分配給環境，
我們可以最大限度地減少在裝置之間傳輸資料的時間。
例如，當在帶有GPU的伺服器上訓練神經網路時，
我們通常希望模型的引數在GPU上。
:end_tab:

:begin_tab:`paddle`
在PaddlePaddle中，每個張量都有一個裝置（device），
我們通常將其稱為上下文（context）。
預設情況下，所有變數和相關的計算都分配給CPU。
有時上下文可能是GPU。
當我們跨多個服務器部署作業時，事情會變得更加棘手。
透過智慧地將陣列分配給上下文，
我們可以最大限度地減少在裝置之間傳輸資料的時間。
例如，當在帶有GPU的伺服器上訓練神經網路時，
我們通常希望模型的引數在GPU上。

接下來，我們需要確認安裝了PaddlePaddle的GPU版本。
如果已經安裝了PaddlePaddle的CPU版本，我們需要先解除安裝它。
然後根據你的CUDA版本安裝相應的PaddlePaddle的GPU版本。
例如，假設你安裝了CUDA10.1，你可以透過`conda install paddlepaddle-gpu==2.2.2 cudatoolkit=10.1 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/`安裝支援CUDA10.1的PaddlePaddle版本。
:end_tab:

要執行此部分中的程式，至少需要兩個GPU。
注意，對大多數桌面電腦來說，這可能是奢侈的，但在雲中很容易獲得。
例如可以使用AWS EC2的多GPU實例。
本書的其他章節大都不需要多個GPU，
而本節只是為了展示資料如何在不同的裝置之間傳遞。

## [**計算裝置**]

我們可以指定用於儲存和計算的裝置，如CPU和GPU。
預設情況下，張量是在記憶體中建立的，然後使用CPU計算它。

:begin_tab:`mxnet`
在MXNet中，CPU和GPU可以用`cpu()`和`gpu()`表示。
需要注意的是，`cpu()`（或括號中的任意整數）表示所有物理CPU和記憶體，
這意味著MXNet的計算將嘗試使用所有CPU核心。
然而，`gpu()`只代表一個卡和相應的顯存。
如果有多個GPU，我們使用`gpu(i)`表示第$i$區塊GPU（$i$從0開始）。
另外，`gpu(0)`和`gpu()`是等價的。
:end_tab:

:begin_tab:`pytorch`
在PyTorch中，CPU和GPU可以用`torch.device('cpu')`
和`torch.device('cuda')`表示。
應該注意的是，`cpu`裝置意味著所有物理CPU和記憶體，
這意味著PyTorch的計算將嘗試使用所有CPU核心。
然而，`gpu`裝置只代表一個卡和相應的顯存。
如果有多個GPU，我們使用`torch.device(f'cuda:{i}')`
來表示第$i$區塊GPU（$i$從0開始）。
另外，`cuda:0`和`cuda`是等價的。
:end_tab:

:begin_tab:`paddle`
在飛槳中，CPU和GPU可以用`paddle.device.set_device('cpu')` 
和`paddle.device.set_device('gpu')`表示。 
應該注意的是，`cpu`裝置意味著所有物理CPU和記憶體,
這意味著飛槳的計算將嘗試使用所有CPU核心。 
然而，`gpu`裝置只代表一個卡和相應的顯存。 
如果有多個GPU，我們使用`paddle.device.get_device()`
其中輸出的數字是表示的是卡號（比如`gpu:3`，表示的是卡3，注意GPU的卡號是從0開始的）。 
另外，`gpu:0`和`gpu`是等價的。
:end_tab:

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

npx.cpu(), npx.gpu(), npx.gpu(1)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1')
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

tf.device('/CPU:0'), tf.device('/GPU:0'), tf.device('/GPU:1')
```

```{.python .input}
#@tab paddle
import paddle
from paddle import nn

paddle.device.set_device("cpu"), paddle.CUDAPlace(0), paddle.CUDAPlace(1)
```

我們可以(**查詢可用gpu的數量。**)

```{.python .input}
npx.num_gpus()
```

```{.python .input}
#@tab pytorch
torch.cuda.device_count()
```

```{.python .input}
#@tab tensorflow
len(tf.config.experimental.list_physical_devices('GPU'))
```

```{.python .input}
#@tab paddle
paddle.device.cuda.device_count()
```

現在我們定義了兩個方便的函式，
[**這兩個函式允許我們在不存在所需所有GPU的情況下執行程式碼。**]

```{.python .input}
def try_gpu(i=0):  #@save
    """如果存在，則回傳gpu(i)，否則回傳cpu()"""
    return npx.gpu(i) if npx.num_gpus() >= i + 1 else npx.cpu()

def try_all_gpus():  #@save
    """回傳所有可用的GPU，如果沒有GPU，則回傳[cpu()]"""
    devices = [npx.gpu(i) for i in range(npx.num_gpus())]
    return devices if devices else [npx.cpu()]

try_gpu(), try_gpu(10), try_all_gpus()
```

```{.python .input}
#@tab pytorch
def try_gpu(i=0):  #@save
    """如果存在，則回傳gpu(i)，否則回傳cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """回傳所有可用的GPU，如果沒有GPU，則回傳[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

try_gpu(), try_gpu(10), try_all_gpus()
```

```{.python .input}
#@tab tensorflow
def try_gpu(i=0):  #@save
    """如果存在，則回傳gpu(i)，否則回傳cpu()"""
    if len(tf.config.experimental.list_physical_devices('GPU')) >= i + 1:
        return tf.device(f'/GPU:{i}')
    return tf.device('/CPU:0')

def try_all_gpus():  #@save
    """回傳所有可用的GPU，如果沒有GPU，則回傳[cpu(),]"""
    num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    devices = [tf.device(f'/GPU:{i}') for i in range(num_gpus)]
    return devices if devices else [tf.device('/CPU:0')]

try_gpu(), try_gpu(10), try_all_gpus()
```

```{.python .input}
#@tab paddle
#@save
def try_gpu(i=0):  
    """如果存在，則回傳gpu(i)，否則回傳cpu()。"""
    if paddle.device.cuda.device_count() >= i + 1:
        return paddle.CUDAPlace(i)
    return paddle.CPUPlace()

#@save
def try_all_gpus():  
    """回傳所有可用的GPU，如果沒有GPU，則回傳[cpu(),]。"""
    devices = [paddle.CUDAPlace(i)
               for i in range(paddle.device.cuda.device_count())]
    return devices if devices else paddle.CPUPlace()

try_gpu(),try_gpu(10),try_all_gpus()
```

## 張量與GPU

我們可以[**查詢張量所在的裝置。**]
預設情況下，張量是在CPU上建立的。

```{.python .input}
x = np.array([1, 2, 3])
x.ctx
```

```{.python .input}
#@tab pytorch
x = torch.tensor([1, 2, 3])
x.device
```

```{.python .input}
#@tab tensorflow
x = tf.constant([1, 2, 3])
x.device
```

```{.python .input}
#@tab paddle
x = paddle.to_tensor([1, 2, 3])
x.place
```

需要注意的是，無論何時我們要對多個項進行操作，
它們都必須在同一個裝置上。
例如，如果我們對兩個張量求和，
我們需要確保兩個張量都位於同一個裝置上，
否則框架將不知道在哪裡儲存結果，甚至不知道在哪裡執行計算。

### [**儲存在GPU上**]

有幾種方法可以在GPU上儲存張量。
例如，我們可以在建立張量時指定儲存裝置。接
下來，我們在第一個`gpu`上建立張量變量`X`。
在GPU上建立的張量只消耗這個GPU的顯存。
我們可以使用`nvidia-smi`指令檢視顯存使用情況。
一般來說，我們需要確保不建立超過GPU顯存限制的資料。

```{.python .input}
X = np.ones((2, 3), ctx=try_gpu())
X
```

```{.python .input}
#@tab pytorch
X = torch.ones(2, 3, device=try_gpu())
X
```

```{.python .input}
#@tab tensorflow
with try_gpu():
    X = tf.ones((2, 3))
X
```

```{.python .input}
#@tab paddle
X = paddle.to_tensor(paddle.ones(shape=[2, 3]), place=try_gpu())
X
```

假設我們至少有兩個GPU，下面的程式碼將在(**第二個GPU上建立一個隨機張量。**)

```{.python .input}
Y = np.random.uniform(size=(2, 3), ctx=try_gpu(1))
Y
```

```{.python .input}
#@tab pytorch
Y = torch.rand(2, 3, device=try_gpu(1))
Y
```

```{.python .input}
#@tab tensorflow
with try_gpu(1):
    Y = tf.random.uniform((2, 3))
Y
```

```{.python .input}
#@tab paddle
Y = paddle.to_tensor(paddle.rand([2, 3]), place=try_gpu(1))
Y
```

### 複製

如果我們[**要計算`X + Y`，我們需要決定在哪裡執行這個操作**]。
例如，如 :numref:`fig_copyto`所示，
我們可以將`X`傳輸到第二個GPU並在那裡執行操作。
*不要*簡單地`X`加上`Y`，因為這會導致異常，
執行時引擎不知道該怎麼做：它在同一裝置上找不到資料會導致失敗。
由於`Y`位於第二個GPU上，所以我們需要將`X`移到那裡，
然後才能執行相加運算。

![複製資料以在同一裝置上執行操作](../img/copyto.svg)
:label:`fig_copyto`

```{.python .input}
Z = X.copyto(try_gpu(1))
print(X)
print(Z)
```

```{.python .input}
#@tab pytorch, paddle
Z = X.cuda(1)
print(X)
print(Z)
```

```{.python .input}
#@tab tensorflow
with try_gpu(1):
    Z = X
print(X)
print(Z)
```

[**現在資料在同一個GPU上（`Z`和`Y`都在），我們可以將它們相加。**]

```{.python .input}
#@tab all
Y + Z
```

:begin_tab:`mxnet`
假設變數`Z`已經存在於第二個GPU上。
如果現在我們還是呼叫`Z.copyto(gpu(1))`會發生什麼？
即使該變數已經存在於目標裝置（第二個GPU）上，
它仍將被複製並儲存在新分配的顯存中。
有時，我們只想在變數存在於不同裝置中時進行複製。
在這種情況下，我們可以呼叫`as_in_ctx`。
如果變數已經存在於指定的裝置中，則這不會進行任何操作。
除非我們特別想建立一個複製，否則選擇`as_in_ctx`方法。
:end_tab:

:begin_tab:`pytorch`
假設變數`Z`已經存在於第二個GPU上。
如果我們還是呼叫`Z.cuda(1)`會發生什麼？
它將回傳`Z`，而不會複製並分配新記憶體。
:end_tab:

:begin_tab:`tensorflow`
假設變數`Z`已經存在於第二個GPU上。
如果我們仍然在同一個裝置作用域下調用`Z2 = Z`會發生什麼？
它將回傳`Z`，而不會複製並分配新記憶體。
:end_tab:

```{.python .input}
Z.as_in_ctx(try_gpu(1)) is Z
```

```{.python .input}
#@tab pytorch, paddle
Z.cuda(1) is Z
```

```{.python .input}
#@tab tensorflow
with try_gpu(1):
    Z2 = Z
Z2 is Z
```

### 旁註

人們使用GPU來進行機器學習，因為單個GPU相對執行速度快。
但是在裝置（CPU、GPU和其他機器）之間傳輸資料比計算慢得多。
這也使得並行化變得更加困難，因為我們必須等待資料被髮送（或者接收），
然後才能繼續進行更多的操作。
這就是為什麼複製操作要格外小心。
根據經驗，多個小操作比一個大操作糟糕得多。
此外，一次執行幾個操作比程式碼中散佈的許多單個操作要好得多。
如果一個裝置必須等待另一個裝置才能執行其他操作，
那麼這樣的操作可能會阻塞。
這有點像排隊訂購咖啡，而不像透過電話預先訂購：
當客人到店的時候，咖啡已經準備好了。

最後，當我們列印張量或將張量轉換為NumPy格式時，
如果資料不在記憶體中，框架會首先將其複製到記憶體中，
這會導致額外的傳輸開銷。
更糟糕的是，它現在受制於全域直譯器鎖，使得一切都得等待Python完成。

## [**神經網路與GPU**]

類似地，神經網路模型可以指定裝置。
下面的程式碼將模型引數放在GPU上。

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=try_gpu())
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
```

```{.python .input}
#@tab tensorflow
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)])
```

```{.python .input}
#@tab paddle
net = nn.Sequential(nn.Linear(3, 1))
net=net.to(try_gpu())
```

在接下來的幾章中，
我們將看到更多關於如何在GPU上執行模型的例子，
因為它們將變得更加計算密集。

當輸入為GPU上的張量時，模型將在同一GPU上計算結果。

```{.python .input}
#@tab all
net(X)
```

讓我們(**確認模型引數儲存在同一個GPU上。**)

```{.python .input}
net[0].weight.data().ctx
```

```{.python .input}
#@tab pytorch
net[0].weight.data.device
```

```{.python .input}
#@tab tensorflow
net.layers[0].weights[0].device, net.layers[0].weights[1].device
```

```{.python .input}
#@tab paddle
net[0].weight.place
```

總之，只要所有的資料和引數都在同一個裝置上，
我們就可以有效地學習模型。
在下面的章節中，我們將看到幾個這樣的例子。

## 小結

* 我們可以指定用於儲存和計算的裝置，例如CPU或GPU。預設情況下，資料在主記憶體中建立，然後使用CPU進行計算。
* 深度學習框架要求計算的所有輸入資料都在同一裝置上，無論是CPU還是GPU。
* 不經意地移動資料可能會顯著降低效能。一個典型的錯誤如下：計算GPU上每個小批次的損失，並在指令行中將其報告給使用者（或將其記錄在NumPy `ndarray`中）時，將觸發全域直譯器鎖，從而使所有GPU阻塞。最好是為GPU內部的日誌分配記憶體，並且只移動較大的日誌。

## 練習

1. 嘗試一個計算量更大的任務，比如大矩陣的乘法，看看CPU和GPU之間的速度差異。再試一個計算量很小的任務呢？
1. 我們應該如何在GPU上讀寫模型引數？
1. 測量計算1000個$100 \times 100$矩陣的矩陣乘法所需的時間，並記錄輸出矩陣的Frobenius範數，一次記錄一個結果，而不是在GPU上儲存日誌並僅傳輸最終結果。
1. 測量同時在兩個GPU上執行兩個矩陣乘法與在一個GPU上按順序執行兩個矩陣乘法所需的時間。提示：應該看到近乎線性的縮放。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1843)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1841)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1842)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11782)
:end_tab:
