# 多GPU訓練
:label:`sec_multi_gpu`

到目前為止，我們討論瞭如何在CPU和GPU上高效地訓練模型，同時在 :numref:`sec_auto_para`中展示了深度學習框架如何在CPU和GPU之間自動地並行化計算和通訊，還在 :numref:`sec_use_gpu`中展示了如何使用`nvidia-smi`指令列出電腦上所有可用的GPU。
但是我們沒有討論如何真正實現深度學習訓練的並行化。
是否一種方法，以某種方式分割資料到多個裝置上，並使其能夠正常工作呢？
本節將詳細介紹如何從零開始並行地訓練網路，
這裡需要運用小批次隨機梯度下降演算法（詳見 :numref:`sec_minibatch_sgd`）。
後面我還講介紹如何使用高階API並行訓練網路（請參閱 :numref:`sec_multi_gpu_concise`）。

## 問題拆分

我們從一個簡單的電腦視覺問題和一個稍稍過時的網路開始。
這個網路有多個卷積層和彙聚層，最後可能有幾個全連線的層，看起來非常類似於LeNet :cite:`LeCun.Bottou.Bengio.ea.1998`或AlexNet :cite:`Krizhevsky.Sutskever.Hinton.2012`。
假設我們有多個GPU（如果是桌面伺服器則有$2$個，AWS g4dn.12xlarge上有$4$個，p3.16xlarge上有$8$個，p2.16xlarge上有$16$個）。
我們希望以一種方式對訓練進行拆分，為實現良好的加速比，還能同時受益於簡單且可重複的設計選擇。
畢竟，多個GPU同時增加了記憶體和計算能力。
簡而言之，對於需要分類的小批次訓練資料，我們有以下選擇。

第一種方法，在多個GPU之間拆分網路。
也就是說，每個GPU將流入特定層的資料作為輸入，跨多個後續層對資料進行處理，然後將資料傳送到下一個GPU。
與單個GPU所能處理的資料相比，我們可以用更大的網路處理資料。
此外，每個GPU佔用的*顯存*（memory footprint）可以得到很好的控制，雖然它只是整個網路顯存的一小部分。

然而，GPU的介面之間需要的密集同步可能是很難辦的，特別是層之間計算的工作負載不能正確對應的時候，
還有層之間的介面需要大量的數據傳輸的時候（例如：啟用值和梯度，資料量可能會超出GPU匯流排的頻寬）。
此外，計算密集型操作的順序對拆分來說也是非常重要的，這方面的最好研究可參見 :cite:`Mirhoseini.Pham.Le.ea.2017`，其本質仍然是一個困難的問題，目前還不清楚研究是否能在特定問題上實現良好的線性縮放。
綜上所述，除非存框架或作業系統本身支援將多個GPU連線在一起，否則不建議這種方法。

第二種方法，拆分層內的工作。
例如，將問題分散到$4$個GPU，每個GPU生成$16$個通道的資料，而不是在單個GPU上計算$64$個通道。
對於全連線的層，同樣可以拆分輸出單元的數量。
 :numref:`fig_alexnet_original`描述了這種設計，其策略用於處理顯存非常小（當時為2GB）的GPU。
當通道或單元的數量不太小時，使計算效能有良好的提升。
此外，由於可用的顯存呈線性擴充套件，多個GPU能夠處理不斷變大的網路。

![由於GPU顯存有限，原有AlexNet設計中的模型並行](../img/alexnet-original.svg)
:label:`fig_alexnet_original`

然而，我們需要大量的同步或*屏障操作*（barrier operation），因為每一層都依賴於所有其他層的結果。
此外，需要傳輸的資料量也可能比跨GPU拆分層時還要大。
因此，基於頻寬的成本和複雜性，我們同樣不推薦這種方法。

最後一種方法，跨多個GPU對資料進行拆分。
這種方式下，所有GPU儘管有不同的觀測結果，但是執行著相同型別的工作。
在完成每個小批次資料的訓練之後，梯度在GPU上聚合。
這種方法最簡單，並可以應用於任何情況，同步只需要在每個小批次資料處理之後進行。
也就是說，當其他梯度引數仍在計算時，完成計算的梯度引數就可以開始交換。
而且，GPU的數量越多，小批次包含的資料量就越大，從而就能提高訓練效率。
但是，新增更多的GPU並不能讓我們訓練更大的模型。

![在多個GPU上並行化。從左到右：原始問題、網路並行、分層並行、資料並行](../img/splitting.svg)
:label:`fig_splitting`

 :numref:`fig_splitting`中比較了多個GPU上不同的並行方式。
總體而言，只要GPU的顯存足夠大，資料並行是最方便的。
有關分散式訓練分割槽的詳細描述，請參見 :cite:`Li.Andersen.Park.ea.2014`。
在深度學習的早期，GPU的顯存曾經是一個棘手的問題，然而如今除了非常特殊的情況，這個問題已經解決。
下面我們將重點討論資料並行性。

## 資料並行性

假設一台機器有$k$個GPU。
給定需要訓練的模型，雖然每個GPU上的引數值都是相同且同步的，但是每個GPU都將獨立地維護一組完整的模型引數。
例如， :numref:`fig_data_parallel`示範了在$k=2$時基於資料並行方法訓練模型。

![利用兩個GPU上的資料，並行計算小批次隨機梯度下降](../img/data-parallel.svg)
:label:`fig_data_parallel`

一般來說，$k$個GPU並行訓練過程如下：

* 在任何一次訓練迭代中，給定的隨機的小批次樣本都將被分成$k$個部分，並均勻地分配到GPU上；
* 每個GPU根據分配給它的小批次子集，計算模型引數的損失和梯度；
* 將$k$個GPU中的區域性梯度聚合，以獲得當前小批次的隨機梯度；
* 聚合梯度被重新分發到每個GPU中；
* 每個GPU使用這個小批次隨機梯度，來更新它所維護的完整的模型引數集。


在實踐中請注意，當在$k$個GPU上訓練時，需要擴大小批次的大小為$k$的倍數，這樣每個GPU都有相同的工作量，就像只在單個GPU上訓練一樣。
因此，在16-GPU伺服器上可以顯著地增加小批次資料量的大小，同時可能還需要相應地提高學習率。
還請注意， :numref:`sec_batch_norm`中的批次規範化也需要調整，例如，為每個GPU保留單獨的批次規範化引數。

下面我們將使用一個簡單網路來示範多GPU訓練。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
from paddle.nn import functional as F
```

## [**簡單網路**]

我們使用 :numref:`sec_lenet`中介紹的（稍加修改的）LeNet，
從零開始定義它，從而詳細說明引數交換和同步。

```{.python .input}
# 初始化模型引數
scale = 0.01
W1 = np.random.normal(scale=scale, size=(20, 1, 3, 3))
b1 = np.zeros(20)
W2 = np.random.normal(scale=scale, size=(50, 20, 5, 5))
b2 = np.zeros(50)
W3 = np.random.normal(scale=scale, size=(800, 128))
b3 = np.zeros(128)
W4 = np.random.normal(scale=scale, size=(128, 10))
b4 = np.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# 定義模型
def lenet(X, params):
    h1_conv = npx.convolution(data=X, weight=params[0], bias=params[1],
                              kernel=(3, 3), num_filter=20)
    h1_activation = npx.relu(h1_conv)
    h1 = npx.pooling(data=h1_activation, pool_type='avg', kernel=(2, 2),
                     stride=(2, 2))
    h2_conv = npx.convolution(data=h1, weight=params[2], bias=params[3],
                              kernel=(5, 5), num_filter=50)
    h2_activation = npx.relu(h2_conv)
    h2 = npx.pooling(data=h2_activation, pool_type='avg', kernel=(2, 2),
                     stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = np.dot(h2, params[4]) + params[5]
    h3 = npx.relu(h3_linear)
    y_hat = np.dot(h3, params[6]) + params[7]
    return y_hat

# 交叉熵損失函式
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
# 初始化模型引數
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# 定義模型
def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

# 交叉熵損失函式
loss = nn.CrossEntropyLoss(reduction='none')
```

```{.python .input}
#@tab paddle
# 初始化模型引數
scale = 0.01
W1 = paddle.randn(shape=[20, 1, 3, 3]) * scale
b1 = paddle.zeros(shape=[20])
W2 = paddle.randn(shape=[50, 20, 5, 5]) * scale
b2 = paddle.zeros(shape=[50])
W3 = paddle.randn(shape=[800, 128]) * scale
b3 = paddle.zeros(shape=[128])
W4 = paddle.randn(shape=[128, 10]) * scale
b4 = paddle.zeros(shape=[10])
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# 定義模型
def lenet(X, params):
    h1_conv = F.conv2d(x=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(x=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(x=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(x=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape([h2.shape[0], -1])
    h3_linear = paddle.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = paddle.mm(h3, params[6]) + params[7]
    return y_hat

# 交叉熵損失函式
loss = nn.CrossEntropyLoss(reduction='none')
```

## 資料同步

對於高效的多GPU訓練，我們需要兩個基本操作。
首先，我們需要[**向多個裝置分發引數**]並附加梯度（`get_params`）。
如果沒有引數，就不可能在GPU上評估網路。
第二，需要跨多個裝置對引數求和，也就是說，需要一個`allreduce`函式。

```{.python .input}
def get_params(params, device):
    new_params = [p.copyto(device) for p in params]
    for p in new_params:
        p.attach_grad()
    return new_params
```

```{.python .input}
#@tab pytorch
def get_params(params, device):
    new_params = [p.to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params
```

```{.python .input}
#@tab paddle
def get_params(params, device):
    new_params = [paddle.to_tensor(p, place=device) for p in params]
    for p in new_params:
        p.stop_gradient = False
    return new_params
```

透過將模型引數複製到一個GPU。

```{.python .input}
#@tab all
new_params = get_params(params, d2l.try_gpu(0))
print('b1 權重:', new_params[1])
print('b1 梯度:', new_params[1].grad)
```

由於還沒有進行任何計算，因此權重引數的梯度仍然為零。
假設現在有一個向量分佈在多個GPU上，下面的[**`allreduce`函式將所有向量相加，並將結果廣播給所有GPU**]。
請注意，我們需要將資料複製到累積結果的裝置，才能使函式正常工作。

```{.python .input}
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].copyto(data[0].ctx)
    for i in range(1, len(data)):
        data[0].copyto(data[i])
```

```{.python .input}
#@tab pytorch
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)
```

```{.python .input}
#@tab paddle
def allreduce(data):
    paddle.set_device("gpu:0")
    for i in range(1, len(data)):
        data[0] += paddle.to_tensor(data[i], place=data[0].place)
    for i in range(1, len(data)):
        data[i] = paddle.to_tensor(data[0], place=data[i].place) 
```

透過在不同裝置上建立具有不同值的向量並聚合它們。

```{.python .input}
data = [np.ones((1, 2), ctx=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('allreduce之前：\n', data[0], '\n', data[1])
allreduce(data)
print('allreduce之後：\n', data[0], '\n', data[1])
```

```{.python .input}
#@tab pytorch
data = [torch.ones((1, 2), device=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('allreduce之前：\n', data[0], '\n', data[1])
allreduce(data)
print('allreduce之後：\n', data[0], '\n', data[1])
```

```{.python .input}
#@tab paddle
num_gpus = 2
devices = [d2l.try_gpu(i) for i in range(num_gpus)]

data = [paddle.to_tensor(paddle.ones(shape=[1, 2]) * (i + 1), place=devices[i]) for i in range(2)]
print('allreduce之前：\n', data[0], '\n', data[1])
allreduce(data)
print('allreduce之後：\n', data[0], '\n', data[1])
```

## 資料分發

我們需要一個簡單的工具函式，[**將一個小批次資料均勻地分佈在多個GPU上**]。
例如，有兩個GPU時，我們希望每個GPU可以複製一半的資料。
因為深度學習框架的內建函式編寫程式碼更方便、更簡潔，所以在$4 \times 5$矩陣上使用它進行嘗試。

```{.python .input}
data = np.arange(20).reshape(4, 5)
devices = [npx.gpu(0), npx.gpu(1)]
split = gluon.utils.split_and_load(data, devices)
print('輸入：', data)
print('裝置：', devices)
print('輸出：', split)
```

```{.python .input}
#@tab pytorch
data = torch.arange(20).reshape(4, 5)
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
split = nn.parallel.scatter(data, devices)
print('input :', data)
print('load into', devices)
print('output:', split)
```

```{.python .input}
#@tab paddle
def paddlescatter(XY, devices): 
    xy = XY.shape[0]//len(devices) # 根據GPU數目計算分塊大小
    return [paddle.to_tensor(XY[i*xy:(i+1)*xy], place=device) for i,device in enumerate(devices)]

# 資料分發
data = paddle.arange(20).reshape([4, 5])
split = paddlescatter(data, devices)

print('input :', data)
print('load into', devices)
print('output:', split)
```

為了方便以後複用，我們定義了可以同時拆分資料和標籤的`split_batch`函式。

```{.python .input}
#@save
def split_batch(X, y, devices):
    """將X和y拆分到多個裝置上"""
    assert X.shape[0] == y.shape[0]
    return (gluon.utils.split_and_load(X, devices),
            gluon.utils.split_and_load(y, devices))
```

```{.python .input}
#@tab pytorch
#@save
def split_batch(X, y, devices):
    """將X和y拆分到多個裝置上"""
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))
```

```{.python .input}
#@tab paddle
#@save
def split_batch(X, y, devices):
    """將X和y拆分到多個裝置上"""
    assert X.shape[0] == y.shape[0]
    return (paddlescatter(X, devices),
            paddlescatter(y, devices))
```

## 訓練

現在我們可以[**在一個小批次上實現多GPU訓練**]。
在多個GPU之間同步資料將使用剛才討論的輔助函式`allreduce`和`split_and_load`。
我們不需要編寫任何特定的程式碼來實現並行性。
因為計算圖在小批次內的裝置之間沒有任何依賴關係，因此它是『自動地』並行執行。

```{.python .input}
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    with autograd.record():  # 在每個GPU上分別計算損失
        ls = [loss(lenet(X_shard, device_W), y_shard)
              for X_shard, y_shard, device_W in zip(
                  X_shards, y_shards, device_params)]
    for l in ls:  # 反向傳播在每個GPU上分別執行
        l.backward()
    # 將每個GPU的所有梯度相加，並將其廣播到所有GPU
    for i in range(len(device_params[0])):
        allreduce([device_params[c][i].grad for c in range(len(devices))])
    # 在每個GPU上分別更新模型引數
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0])  # 在這裡，我們使用全尺寸的小批次
```

```{.python .input}
#@tab pytorch
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    # 在每個GPU上分別計算損失
    ls = [loss(lenet(X_shard, device_W), y_shard).sum()
          for X_shard, y_shard, device_W in zip(
              X_shards, y_shards, device_params)]
    for l in ls:  # 反向傳播在每個GPU上分別執行
        l.backward()
    # 將每個GPU的所有梯度相加，並將其廣播到所有GPU
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce(
                [device_params[c][i].grad for c in range(len(devices))])
    # 在每個GPU上分別更新模型引數
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0]) # 在這裡，我們使用全尺寸的小批次
```

```{.python .input}
#@tab paddle
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    # 在每個GPU上分別計算損失
    for i, (X_shard, y_shard, device_W) in enumerate(zip(
              X_shards, y_shards, device_params)) :
        # 設定全域變數，以便在指定的GPU執行計算
        paddle.set_device(f"gpu:{i}") 
        y_shard = paddle.squeeze(y_shard)
        l = loss(lenet(X_shard, device_W), y_shard).sum()
        # 反向傳播在每個GPU上分別執行
        l.backward()
    # 將每個GPU的所有梯度相加，並將其廣播到所有GPU
    with paddle.no_grad():
        for i in range(len(device_params[0])):
            allreduce(
                [device_params[c][i].grad for c in range(len(devices))])
    # 在每個GPU上分別更新模型引數
    for i in range(len(device_params)):
        paddle.set_device(f"gpu:{i}")
        param = device_params[i]
        d2l.sgd(param, lr, X.shape[0]) # 在這裡，我們使用全尺寸的小批次
```

現在，我們可以[**定義訓練函式**]。
與前幾章中略有不同：訓練函式需要分配GPU並將所有模型引數複製到所有裝置。
顯然，每個小批次都是使用`train_batch`函式來處理多個GPU。
我們只在一個GPU上計算模型的精確度，而讓其他GPU保持空閒，儘管這是相對低效的，但是使用方便且程式碼簡潔。

```{.python .input}
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # 將模型引數複製到num_gpus個GPU
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # 為單個小批次執行多GPU訓練
            train_batch(X, y, device_params, devices, lr)
            npx.waitall()
        timer.stop()
        # 在GPU0上評估模型
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'測試精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/輪，'
          f'在{str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # 將模型引數複製到num_gpus個GPU
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # 為單個小批次執行多GPU訓練
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize()
        timer.stop()
        # 在GPU0上評估模型
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'測試精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/輪，'
          f'在{str(devices)}')
```

```{.python .input}
#@tab paddle
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # 將模型引數複製到num_gpus個GPU
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10 
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # 為單個小批次執行多GPU訓練
            train_batch(X, y, device_params, devices, lr)
            paddle.device.cuda.synchronize()
        timer.stop()
        # 在GPU0上評估模型
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'測試精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/輪，'
        f'在{str(devices)}')
```

讓我們看看[**在單個GPU上執行**]效果得有多好。
首先使用的批次大小是$256$，學習率是$0.2$。

```{.python .input}
#@tab all
train(num_gpus=1, batch_size=256, lr=0.2)
```

保持批次大小和學習率不變，並[**增加為2個GPU**]，我們可以看到測試精度與之前的實驗基本相同。
不同的GPU個數在演算法尋優方面是相同的。
不幸的是，這裡沒有任何有意義的加速：模型實在太小了；而且資料集也太小了。在這個資料集中，我們實現的多GPU訓練的簡單方法受到了巨大的Python開銷的影響。
在未來，我們將遇到更復雜的模型和更復雜的並行化方法。
儘管如此，讓我們看看Fashion-MNIST資料集上會發生什麼。

```{.python .input}
#@tab mxnet, pytorch
train(num_gpus=2, batch_size=256, lr=0.2)
```

## 小結

* 有多種方法可以在多個GPU上拆分深度網路的訓練。拆分可以在層之間、跨層或跨資料上實現。前兩者需要對數據傳輸過程進行嚴格編排，而最後一種則是最簡單的策略。
* 資料並行訓練本身是不復雜的，它透過增加有效的小批次資料量的大小提高了訓練效率。
* 在資料並行中，資料需要跨多個GPU拆分，其中每個GPU執行自己的前向傳播和反向傳播，隨後所有的梯度被聚合為一，之後聚合結果向所有的GPU廣播。
* 小批次資料量更大時，學習率也需要稍微提高一些。

## 練習

1. 在$k$個GPU上進行訓練時，將批次大小從$b$更改為$k \cdot b$，即按GPU的數量進行擴充套件。
1. 比較不同學習率時模型的精確度，隨著GPU數量的增加學習率應該如何擴充套件？
1. 實現一個更高效的`allreduce`函式用於在不同的GPU上聚合不同的引數？為什麼這樣的效率更高？
1. 實現模型在多GPU下測試精度的計算。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2801)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2800)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11860)
:end_tab: