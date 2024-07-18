# 非同步計算
:label:`sec_async`

今天的電腦是高度並行的系統，由多個CPU核、多個GPU、多個處理單元組成。通常每個CPU核有多個執行緒，每個裝置通常有多個GPU，每個GPU有多個處理單元。總之，我們可以同時處理許多不同的事情，並且通常是在不同的裝置上。不幸的是，Python並不善於編寫並行和非同步程式碼，至少在沒有額外幫助的情況下不是好選擇。歸根結底，Python是單線程的，將來也是不太可能改變的。因此在諸多的深度學習框架中，MXNet和TensorFlow之類則採用了一種*非同步程式設計*（asynchronous programming）模型來提高效能，而PyTorch則使用了Python自己的排程器來實現不同的效能權衡。對PyTorch來說GPU操作在預設情況下是非同步的。當呼叫一個使用GPU的函式時，操作會排隊到特定的裝置上，但不一定要等到以後才執行。這允許我們並行執行更多的計算，包括在CPU或其他GPU上的操作。

因此，瞭解非同步程式設計是如何工作的，透過主動地減少計算需求和相互依賴，有助於我們開發更高效的程式。這能夠減少記憶體開銷並提高處理器利用率。

```{.python .input}
from d2l import mxnet as d2l
import numpy, os, subprocess
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy, os, subprocess
import torch
from torch import nn
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import numpy, os, subprocess
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
d2l.try_gpu()
```

## 透過後端非同步處理

:begin_tab:`mxnet`
作為熱身，考慮一個簡單問題：生成一個隨機矩陣並將其相乘。讓我們在NumPy和`mxnet.np`中都這樣做，看看有什麼不同。
:end_tab:

:begin_tab:`pytorch`
作為熱身，考慮一個簡單問題：生成一個隨機矩陣並將其相乘。讓我們在NumPy和PyTorch張量中都這樣做，看看它們的區別。請注意，PyTorch的`tensor`是在GPU上定義的。
:end_tab:

:begin_tab:`paddle`
作為熱身，考慮一個簡單問題：我們要生成一個隨機矩陣並將其相乘。讓我們在NumPy和飛槳張量中都這樣做，看看它們的區別。請注意，飛槳的`tensor`是在GPU上定義的。
:end_tab:

```{.python .input}
with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('mxnet.np'):
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
```

```{.python .input}
#@tab pytorch
# GPU計算熱身
device = d2l.try_gpu()
a = torch.randn(size=(1000, 1000), device=device)
b = torch.mm(a, a)

with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('torch'):
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
```

```{.python .input}
#@tab paddle
# GPU計算熱身
a = paddle.randn(shape=(1000, 1000))
b = paddle.mm(a, a)

with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('paddle'):
    for _ in range(10):
        a = paddle.randn(shape=(1000, 1000))
        b = paddle.mm(a, a)
```

:begin_tab:`mxnet`
透過MXNet的基準輸出比較快了幾個數量級。由於兩者都在同一處理器上執行，因此一定有其他原因。強制MXNet在回傳之前完成所有後端計算，這種強制說明了之前發生的情況：計算是由後端執行，而前端將控制權回傳給了Python。
:end_tab:

:begin_tab:`pytorch`
透過PyTorch的基準輸出比較快了幾個數量級。NumPy點積是在CPU上執行的，而PyTorch矩陣乘法是在GPU上執行的，後者的速度要快得多。但巨大的時間差距表明一定還有其他原因。預設情況下，GPU操作在PyTorch中是非同步的。強制PyTorch在回傳之前完成所有計算，這種強制說明了之前發生的情況：計算是由後端執行，而前端將控制權回傳給了Python。
:end_tab:

:begin_tab:`paddle`
透過飛槳的基準輸出比較快了幾個數量級。NumPy點積是在CPU上執行的，而飛槳矩陣乘法是在GPU上執行的，後者的速度要快得多。但巨大的時間差距表明一定還有其他原因。預設情況下，GPU操作在飛槳中是非同步的。強制飛槳在回傳之前完成所有計算，這種強制說明了之前發生的情況：計算是由後端執行，而前端將控制權回傳給了Python。
:end_tab:

```{.python .input}
with d2l.Benchmark():
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark():
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
    torch.cuda.synchronize(device)
```

```{.python .input}
#@tab paddle
with d2l.Benchmark():
    for _ in range(10):
        a = paddle.randn(shape=(1000, 1000))
        b = paddle.mm(a, a)
    paddle.device.cuda.synchronize()
```

:begin_tab:`mxnet`
廣義上說，MXNet有一個用於與使用者直接互動的前端（例如透過Python），還有一個由系統用來執行計算的後端。如 :numref:`fig_frontends`所示，使用者可以用各種前端語言編寫MXNet程式，如Python、R、Scala和C++。不管使用的前端程式語言是什麼，MXNet程式的執行主要發生在C++實現的後端。由前端語言發出的操作被傳遞到後端執行。後端管理自己的執行緒，這些執行緒不斷收集和執行排隊的任務。請注意，要使其工作，後端必須能夠跟蹤計算圖中各個步驟之間的依賴關係。因此，不可能並行化相互依賴的操作。
:end_tab:

:begin_tab:`pytorch`
廣義上說，PyTorch有一個用於與使用者直接互動的前端（例如透過Python），還有一個由系統用來執行計算的後端。如 :numref:`fig_frontends`所示，使用者可以用各種前端語言編寫PyTorch程式，如Python和C++。不管使用的前端程式語言是什麼，PyTorch程式的執行主要發生在C++實現的後端。由前端語言發出的操作被傳遞到後端執行。後端管理自己的執行緒，這些執行緒不斷收集和執行排隊的任務。請注意，要使其工作，後端必須能夠跟蹤計算圖中各個步驟之間的依賴關係。因此，不可能並行化相互依賴的操作。
:end_tab:

:begin_tab:`paddle`
廣義上說，飛槳有一個用於與使用者直接互動的前端（例如透過Python），還有一個由系統用來執行計算的後端。如 :numref:`fig_frontends`所示，使用者可以用各種前端語言編寫Python程式，如Python和C++。不管使用的前端程式語言是什麼，飛槳程式的執行主要發生在C++實現的後端。由前端語言發出的操作被傳遞到後端執行。後端管理自己的執行緒，這些執行緒不斷收集和執行排隊的任務。請注意，要使其工作，後端必須能夠跟蹤計算圖中各個步驟之間的依賴關係。因此，不可能並行化相互依賴的操作。
:end_tab:

![程式語言前端和深度學習框架後端](../img/frontends.png)
:width:`300px`
:label:`fig_frontends`

接下來看看另一個簡單例子，以便更好地理解依賴關係圖。

```{.python .input}
x = np.ones((1, 2))
y = np.ones((1, 2))
z = x * y + 2
z
```

```{.python .input}
#@tab pytorch
x = torch.ones((1, 2), device=device)
y = torch.ones((1, 2), device=device)
z = x * y + 2
z
```

```{.python .input}
#@tab paddle
x = paddle.ones((1, 2))
y = paddle.ones((1, 2))
z = x * y + 2
z
```

![後端跟蹤計算圖中各個步驟之間的依賴關係](../img/asyncgraph.svg)
:label:`fig_asyncgraph`

上面的程式碼片段在 :numref:`fig_asyncgraph`中進行了說明。每當Python前端執行緒執行前三條語句中的一條語句時，它只是將任務回傳到後端佇列。當最後一個語句的結果需要被列印出來時，Python前端執行緒將等待C++後端執行緒完成變數`z`的結果計算。這種設計的一個好處是Python前端執行緒不需要執行實際的計算。因此，不管Python的效能如何，對程式的整體性能幾乎沒有影響。 :numref:`fig_threading`示範了前端和後端如何互動。

![前端和後端的互動](../img/threading.svg)
:label:`fig_threading`

## 障礙器與阻塞器

:begin_tab:`mxnet`
有許多操作用於強制Python等待完成：

* 最明顯的是，`npx.waitall()`不考慮計算指令的發出時間，等待直到所有計算完成。除非絕對必要，否則在實踐中使用此運算符不是個好主意，因為它可能會導致較差的效能；
* 如果只想等待一個特定的變數可用，我們可以呼叫`z.wait_to_read()`。在這種情況下，MXNet阻止程式回傳Python，直到計算出變數`z`為止。`z`之後的其他計算才可能很好地繼續。

接下來看看這在實踐中是如何運作的。
:end_tab:

```{.python .input}
with d2l.Benchmark('waitall'):
    b = np.dot(a, a)
    npx.waitall()

with d2l.Benchmark('wait_to_read'):
    b = np.dot(a, a)
    b.wait_to_read()
```

:begin_tab:`mxnet`
兩個操作的完成時間大致相同。除了顯式地阻塞操作之外，建議注意*隱式*的阻塞器。列印變數就是一個阻塞器，因為其要求變數可用。最後，透過`z.asnumpy()`轉換為NumPy型別的變數和透過`z.item()`轉換為標量也是阻塞器。因為NumPy中沒有非同步的概念，因此它需要像`print`函式（等待變數可用）一樣訪問這些值。

頻繁地將少量資料從MXNet的作用域複製到NumPy，可能會破壞原本高效程式碼的效能，因為每一個這樣的操作都需要使用計算圖來求得所有的中間結果，從而獲得相關項，然後才能做其他事情。
:end_tab:

```{.python .input}
with d2l.Benchmark('numpy conversion'):
    b = np.dot(a, a)
    b.asnumpy()

with d2l.Benchmark('scalar conversion'):
    b = np.dot(a, a)
    b.sum().item()
```

## 改進計算

:begin_tab:`mxnet`
在重度多執行緒的系統中（即使普通筆記型電腦也有4個或更多執行緒，然而在多插槽伺服器上這個數字可能超過256），排程操作的開銷可能會變得非常大。這也是極度希望計算和排程是非同步和並行的原因。為了說明這樣做的好處，讓我們看看按順序（同步執行）或非同步執行多次將變數遞增$1$會發生什麼情況。這裡透過在每個加法之間插入`wait_to_read`障礙器來模擬同步執行。
:end_tab:

```{.python .input}
with d2l.Benchmark('synchronous'):
    for _ in range(10000):
        y = x + 1
        y.wait_to_read()

with d2l.Benchmark('asynchronous'):
    for _ in range(10000):
        y = x + 1
    npx.waitall()
```

Python前端執行緒和C++後端執行緒之間的簡化互動可以概括如下：

1. 前端指令後端將計算任務`y = x + 1`插入佇列；
1. 然後後端從佇列接收計算任務並執行；
1. 然後後端將計算結果回傳到前端。

假設這三個階段的持續時間分別為$t_1, t_2, t_3$。如果不使用非同步程式設計，執行10000次計算所需的總時間約為$10000 (t_1+ t_2 + t_3)$。如果使用非同步程式設計，因為前端不必等待後端為每個迴圈回傳計算結果，執行$10000$次計算所花費的總時間可以減少到$t_1 + 10000 t_2 + t_3$（假設$10000 t_2 > 9999t_1$）。


## 小結

* 深度學習框架可以將Python前端的控制與後端的執行解耦，使得指令可以快速地非同步插入後端、並行執行。
* 非同步產生了一個相當靈活的前端，但請注意：過度填充任務佇列可能會導致記憶體消耗過多。建議對每個小批次進行同步，以保持前端和後端大致同步。
* 晶片供應商提供了複雜的效能分析工具，以獲得對深度學習效率更精確的洞察。

:begin_tab:`mxnet`
* 將MXNet管理的記憶體轉換到Python將迫使後端等待特定變數就緒，`print`、`asnumpy`和`item`等函式也具有這個效果。請注意，錯誤地使用同步會破壞程式效能。
:end_tab:

## 練習

:begin_tab:`mxnet`
1. 上面提到使用非同步計算可以將執行$10000$次計算所需的總時間減少到$t_1 + 10000 t_2 + t_3$。為什麼要假設這裡是$10000 t_2 > 9999 t_1$？
1. 測量`waitall`和`wait_to_read`之間的差值。提示：執行多條指令並同步以獲得中間結果。
:end_tab:

:begin_tab:`pytorch`
1. 在CPU上，對本節中相同的矩陣乘法操作進行基準測試，仍然可以透過後端觀察非同步嗎？
:end_tab:

:begin_tab:`paddle`
1. 在CPU上，對本節中相同的矩陣乘法操作進行基準測試。你仍然可以透過後端觀察非同步嗎？
:end_tab:

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2792)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2791)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11858)
:end_tab: