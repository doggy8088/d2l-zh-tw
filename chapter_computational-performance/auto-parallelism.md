# 自動並行
:label:`sec_auto_para`

深度學習框架（例如，MxNet、飛槳和PyTorch）會在後端自動建立計算圖。利用計算圖，系統可以瞭解所有依賴關係，並且可以選擇性地並行執行多個不相互依賴的任務以提高速度。例如， :numref:`sec_async`中的 :numref:`fig_asyncgraph`獨立初始化兩個變數。因此，系統可以選擇並行執行它們。

通常情況下單個運算子將使用所有CPU或單個GPU上的所有計算資源。例如，即使在一台機器上有多個CPU處理器，`dot`運算子也將使用所有CPU上的所有核心（和執行緒）。這樣的行為同樣適用於單個GPU。因此，並行化對單裝置電腦來說並不是很有用，而並行化對於多個裝置就很重要了。雖然並行化通常應用在多個GPU之間，但增加本地CPU以後還將提高少許效能。例如， :cite:`Hadjis.Zhang.Mitliagkas.ea.2016`則把結合GPU和CPU的訓練應用到電腦視覺模型中。藉助自動並行化框架的便利性，我們可以依靠幾行Python程式碼實現相同的目標。對自動並行計算的討論主要集中在使用CPU和GPU的並行計算上，以及計算和通訊的並行化內容。

請注意，本節中的實驗至少需要兩個GPU來執行。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
import numpy as np
```

## 基於GPU的並行計算

從定義一個具有參考性的用於測試的工作負載開始：下面的`run`函式將執行$10$次*矩陣－矩陣*乘法時需要使用的資料分配到兩個變數（`x_gpu1`和`x_gpu2`）中，這兩個變數分別位於選擇的不同裝置上。

```{.python .input}
devices = d2l.try_all_gpus()
def run(x):
    return [x.dot(x) for _ in range(50)]

x_gpu1 = np.random.uniform(size=(4000, 4000), ctx=devices[0])
x_gpu2 = np.random.uniform(size=(4000, 4000), ctx=devices[1])
```

```{.python .input}
#@tab pytorch
devices = d2l.try_all_gpus()
def run(x):
    return [x.mm(x) for _ in range(50)]

x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])
x_gpu2 = torch.rand(size=(4000, 4000), device=devices[1])
```

```{.python .input}
#@tab paddle
devices = d2l.try_all_gpus()
def run(x, index=0):
    paddle.set_device(f"gpu:{index}")
    return [x.matmul(x) for _ in range(50)]
    
data = np.random.rand(4000, 4000)
x_gpu1 = paddle.to_tensor(data, place=devices[0])
x_gpu2 = paddle.to_tensor(data, place=devices[1])
```

:begin_tab:`mxnet`
現在使用函式來處理資料。透過在測量之前需要預熱裝置（對裝置執行一次傳遞）來確保快取的作用不影響最終的結果。
:end_tab:

:begin_tab:`pytorch`
現在使用函式來處理資料。透過在測量之前需要預熱裝置（對裝置執行一次傳遞）來確保快取的作用不影響最終的結果。`torch.cuda.synchronize()`函式將會等待一個CUDA裝置上的所有流中的所有核心的計算完成。函式接受一個`device`引數，代表是哪個裝置需要同步。如果device引數是`None`（預設值），它將使用`current_device()`找出的當前裝置。
:end_tab:

:begin_tab:`paddle`
現在我們使用函式來資料。我們透過在測量之前預熱裝置（對裝置執行一次傳遞）來確保快取的作用不影響最終的結果。`paddle.device.cuda.synchronize()`函式將會等待一個CUDA裝置上的所有流中的所有核心的計算完成。函式接受一個`device`引數，代表是哪個裝置需要同步。如果device引數是`None`（預設值），它將使用`current_device()`找出的當前裝置。
:end_tab:

```{.python .input}
run(x_gpu1)  # 預熱裝置
run(x_gpu2)
npx.waitall()  

with d2l.Benchmark('GPU1 時間'):
    run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('GPU2 時間'):
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
run(x_gpu1)
run(x_gpu2)  # 預熱裝置
torch.cuda.synchronize(devices[0])
torch.cuda.synchronize(devices[1])

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    torch.cuda.synchronize(devices[0])

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    torch.cuda.synchronize(devices[1])
```

```{.python .input}
#@tab paddle
run(x_gpu1, 0)
run(x_gpu2, 1)  # 預熱裝置
paddle.device.cuda.synchronize(devices[0])
paddle.device.cuda.synchronize(devices[1])

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1, 0)
    paddle.device.cuda.synchronize(devices[0])

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2, 1)
    paddle.device.cuda.synchronize(devices[1])
```

:begin_tab:`mxnet`
如果刪除兩個任務之間的`waitall`語句，系統就可以在兩個裝置上自動實現並行計算。
:end_tab:

:begin_tab:`pytorch`
如果刪除兩個任務之間的`synchronize`語句，系統就可以在兩個裝置上自動實現並行計算。
:end_tab:

:begin_tab:`paddle`
如果我們刪除兩個任務之間的`synchronize`語句，系統就可以在兩個裝置上自動實現並行計算。
:end_tab:

```{.python .input}
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    torch.cuda.synchronize()
```

```{.python .input}
#@tab paddle
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1, 0)
    run(x_gpu2, 1)
    paddle.device.cuda.synchronize()
```

在上述情況下，總執行時間小於兩個部分執行時間的總和，因為深度學習框架自動排程兩個GPU裝置上的計算，而不需要使用者編寫複雜的程式碼。

## 並行計算與通訊

在許多情況下，我們需要在不同的裝置之間移動資料，比如在CPU和GPU之間，或者在不同的GPU之間。例如，當執行分散式最佳化時，就需要移動資料來聚合多個加速卡上的梯度。讓我們透過在GPU上計算，然後將結果複製回CPU來模擬這個過程。

```{.python .input}
def copy_to_cpu(x):
    return [y.copyto(npx.cpu()) for y in x]

with d2l.Benchmark('在GPU1上執行'):
    y = run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('複製到CPU'):
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
def copy_to_cpu(x, non_blocking=False):
    return [y.to('cpu', non_blocking=non_blocking) for y in x]

with d2l.Benchmark('在GPU1上執行'):
    y = run(x_gpu1)
    torch.cuda.synchronize()

with d2l.Benchmark('複製到CPU'):
    y_cpu = copy_to_cpu(y)
    torch.cuda.synchronize()
```

```{.python .input}
#@tab paddle
def copy_to_cpu(x):
    return [paddle.to_tensor(y, place=paddle.CPUPlace()) for y in x]

with d2l.Benchmark('在GPU1上執行'):
    y = run(x_gpu1, 0)
    paddle.device.cuda.synchronize()
    
with d2l.Benchmark('複製到CPU'):
    y_cpu = copy_to_cpu(y)
    paddle.device.cuda.synchronize()
```

:begin_tab:`mxnet`
這種方式效率不高。注意到當清單中的其餘部分還在計算時，我們可能就已經開始將`y`的部分複製到CPU了。例如，當計算一個小批次的梯度時，某些引數的梯度將比其他引數的梯度更早可用。因此，在GPU仍在執行時就開始使用PCI-Express匯流排頻寬來移動資料是有利的。刪除這兩個部分之間的`waitall`以模擬這個場景。
:end_tab:

:begin_tab:`pytorch`
這種方式效率不高。注意到當清單中的其餘部分還在計算時，我們可能就已經開始將`y`的部分複製到CPU了。例如，當計算一個小批次的（反傳）梯度時。某些引數的梯度將比其他引數的梯度更早可用。因此，在GPU仍在執行時就開始使用PCI-Express匯流排頻寬來移動資料是有利的。在PyTorch中，`to()`和`copy_()`等函式都允許顯式的`non_blocking`引數，這允許在不需要同步時調用方可以繞過同步。設定`non_blocking=True`以模擬這個場景。
:end_tab:

:begin_tab:`paddle`
這種方式效率不高。注意到當清單中的其餘部分還在計算時，我們可能就已經開始將`y`的部分複製到CPU了。例如，當我們計算一個小批次的（反傳）梯度時。某些引數的梯度將比其他引數的梯度更早可用。因此，在GPU仍在執行時就開始使用PCI-Express匯流排頻寬來移動資料對我們是有利的。
:end_tab:

```{.python .input}
with d2l.Benchmark('在GPU1上執行並複製到CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('在GPU1上執行並複製到CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y, True)
    torch.cuda.synchronize()
```

```{.python .input}
#@tab paddle
with d2l.Benchmark('在GPU1上執行並複製到CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y)
    paddle.device.cuda.synchronize()
```

兩個操作所需的總時間少於它們各部分操作所需時間的總和。請注意，與並行計算的區別是通訊操作使用的資源：CPU和GPU之間的匯流排。事實上，我們可以在兩個裝置上同時進行計算和通訊。如上所述，計算和通訊之間存在的依賴關係是必須先計算`y[i]`，然後才能將其複製到CPU。幸運的是，系統可以在計算`y[i]`的同時複製`y[i-1]`，以減少總的執行時間。

最後，本節給出了一個簡單的兩層多層感知機在CPU和兩個GPU上訓練時的計算圖及其依賴關係的例子，如 :numref:`fig_twogpu`所示。手動排程由此產生的並行程序將是相當痛苦的。這就是基於圖的計算後端進行最佳化的優勢所在。

![在一個CPU和兩個GPU上的兩層的多層感知機的計算圖及其依賴關係](../img/twogpu.svg)
:label:`fig_twogpu`

## 小結

* 現代系統擁有多種裝置，如多個GPU和多個CPU，還可以並行地、非同步地使用它們。
* 現代系統還擁有各種通訊資源，如PCI Express、儲存（通常是固態硬碟或網路儲存）和網路頻寬，為了達到最高效率可以並行使用它們。
* 後端可以透過自動化地並行計算和通訊來提高效能。

## 練習

1. 在本節定義的`run`函式中執行了八個操作，並且操作之間沒有依賴關係。設計一個實驗，看看深度學習框架是否會自動地並行地執行它們。
1. 當單個運算子的工作量足夠小，即使在單個CPU或GPU上，並行化也會有所幫助。設計一個實驗來驗證這一點。
1. 設計一個實驗，在CPU和GPU這兩種裝置上使用並行計算和通訊。
1. 使用諸如NVIDIA的[Nsight](https://developer.nvidia.com/nsight-compute-2019_5)之類的除錯器來驗證程式碼是否有效。
1. 設計並實驗具有更加複雜的資料依賴關係的計算任務，以檢視是否可以在提高效能的同時獲得正確的結果。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2795)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2794)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11859)
:end_tab: