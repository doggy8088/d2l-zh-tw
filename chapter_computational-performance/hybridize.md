# 編譯器和解釋器
:label:`sec_hybridize`

目前為止，本書主要關注的是*指令式程式設計*（imperative programming）。
指令式程式設計使用諸如`print`、『`+`』和`if`之類的語句來更改程式的狀態。
考慮下面這段簡單的指令式程式：

```{.python .input}
#@tab all
def add(a, b):
    return a + b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g

print(fancy_func(1, 2, 3, 4))
```

Python是一種*解釋型語言*（interpreted language）。因此，當對上面的`fancy_func`函式求值時，它按順序執行函式體的操作。也就是說，它將透過對`e = add(a, b)`求值，並將結果儲存為變數`e`，從而更改程式的狀態。接下來的兩個語句`f = add(c, d)`和`g = add(e, f)`也將執行類似地操作，即執行加法計算並將結果儲存為變數。 :numref:`fig_compute_graph`說明了資料流。

![指令式程式設計中的資料流](../img/computegraph.svg)
:label:`fig_compute_graph`

儘管指令式程式設計很方便，但可能效率不高。一方面原因，Python會單獨執行這三個函式的呼叫，而沒有考慮`add`函式在`fancy_func`中被重複呼叫。如果在一個GPU（甚至多個GPU）上執行這些指令，那麼Python直譯器產生的開銷可能會非常大。此外，它需要儲存`e`和`f`的變數值，直到`fancy_func`中的所有語句都執行完畢。這是因為程式不知道在執行語句`e = add(a, b)`和`f = add(c, d)`之後，其他部分是否會使用變數`e`和`f`。

## 符號式程式設計

考慮另一種選擇*符號式程式設計*（symbolic programming），即程式碼通常只在完全定義了過程之後才執行計算。這個策略被多個深度學習框架使用，包括Theano和TensorFlow（後者已經獲得了指令式程式設計的擴充套件）。一般包括以下步驟：

1. 定義計算流程；
1. 將流程編譯成可執行的程式；
1. 給定輸入，呼叫編譯好的程式執行。

這將允許進行大量的最佳化。首先，在大多數情況下，我們可以跳過Python直譯器。從而消除因為多個更快的GPU與單個CPU上的單個Python執行緒搭配使用時產生的效能瓶頸。其次，編譯器可以將上述程式碼最佳化和重寫為`print((1 + 2) + (3 + 4))`甚至`print(10)`。因為編譯器在將其轉換為機器指令之前可以看到完整的程式碼，所以這種最佳化是可以實現的。例如，只要某個變數不再需要，編譯器就可以釋放記憶體（或者從不分配記憶體），或者將程式碼轉換為一個完全等價的片段。下面，我們將透過模擬指令式程式設計來進一步瞭解符號式程式設計的概念。

```{.python .input}
#@tab all
def add_():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_():
    return add_() + fancy_func_() + 'print(fancy_func(1, 2, 3, 4))'

prog = evoke_()
print(prog)
y = compile(prog, '', 'exec')
exec(y)
```

指令式（解釋型）程式設計和符號式程式設計的區別如下：

* 指令式程式設計更容易使用。在Python中，指令式程式設計的大部分程式碼都是簡單易懂的。指令式程式設計也更容易除錯，這是因為無論是獲取和列印所有的中間變數值，或者使用Python的內建除錯工具都更加簡單；
* 符號式程式設計執行效率更高，更易於移植。符號式程式設計更容易在編譯期間最佳化程式碼，同時還能夠將程式移植到與Python無關的格式中，從而允許程式在非Python環境中執行，避免了任何潛在的與Python直譯器相關的效能問題。

## 混合式程式設計

歷史上，大部分深度學習框架都在指令式程式設計與符號式程式設計之間進行選擇。例如，Theano、TensorFlow（靈感來自前者）、Keras和CNTK採用了符號式程式設計。相反地，Chainer和PyTorch採取了指令式程式設計。在後來的版本更新中，TensorFlow2.0和Keras增加了指令式程式設計。

:begin_tab:`mxnet`
開發人員在設計Gluon時思考了這個問題，有沒有可能將這兩種程式設計模式的優點結合起來。於是得到了一個混合式程式設計模型，既允許使用者使用純指令式程式設計進行開發和除錯，還能夠將大多數程式轉換為符號式程式，以便在需要產品級計算效能和部署時使用。

這意味著我們在實際開發中使用的是`HybridBlock`類或`HybridSequential`類在建立模型。預設情況下，它們都與指令式程式設計中使用`Block`類或`Sequential`類別的方式相同。其中，`HybridSequential`類是`HybridBlock`的子類別（就如`Sequential`是`Block`的子類別一樣）。當`hybridize`函式被呼叫時，Gluon將模型編譯成符號式程式設計中使用的形式。這將允許在不犧牲模型實現方式的情況下最佳化計算密集型元件。下面，我們透過將重點放在`Sequential`和`Block`上來詳細描述其優點。
:end_tab:

:begin_tab:`pytorch`
如上所述，PyTorch是基於指令式程式設計並且使用動態計算圖。為了能夠利用符號式程式設計的可移植性和效率，開發人員思考能否將這兩種程式設計模型的優點結合起來，於是就產生了torchscript。torchscript允許使用者使用純指令式程式設計進行開發和除錯，同時能夠將大多數程式轉換為符號式程式，以便在需要產品級計算效能和部署時使用。
:end_tab:

:begin_tab:`tensorflow`
指令式程式設計現在是TensorFlow2的預設選擇，對那些剛接觸該語言的人來說是一個很好的改變。不過，符號式程式設計技術和計算圖仍然存在於TensorFlow中，並且可以透過易於使用的裝飾器`tf.function`進行訪問。這為TensorFlow帶來了指令式程式設計正規化，允許使用者定義更加直觀的函式，然後使用被TensorFlow團隊稱為[autograph](https://www.tensorflow.org/api_docs/python/tf/autograph)的屬性將它們封裝，再自動編譯成計算圖。
:end_tab:

:begin_tab:`paddle`
如上所述，飛槳是基於指令式程式設計並且使用動態計算圖。為了能夠利用符號式程式設計的可移植性和效率，開發人員思考能否將這兩種程式設計模型的優點結合起來，於是就產生了飛槳2.0版本。飛槳2.0及以上版本允許使用者使用純指令式程式設計進行開發和除錯，同時能夠將大多數程式轉換為符號式程式，以便在需要產品級計算效能和部署時使用。
:end_tab:

## `Sequential`的混合式程式設計

要了解混合式程式設計的工作原理，最簡單的方法是考慮具有多層的深層網路。按照慣例，Python直譯器需要執行所有層的程式碼來生成一條指令，然後將該指令轉發到CPU或GPU。對於單個的（快速的）計算裝置，這不會導致任何重大問題。另一方面，如果我們使用先進的8-GPU伺服器，比如AWS P3dn.24xlarge實例，Python將很難讓所有的GPU都保持忙碌。在這裡，瓶頸是單線程的Python直譯器。讓我們看看如何透過將`Sequential`替換為`HybridSequential`來解決程式碼中這個瓶頸。首先，我們定義一個簡單的多層感知機。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

# 生產網路的工廠模式
def get_net():
    net = nn.HybridSequential()  
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(2))
    net.initialize()
    return net

x = np.random.normal(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

# 生產網路的工廠模式
def get_net():
    net = nn.Sequential(nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2))
    return net

x = torch.randn(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from tensorflow.keras.layers import Dense

# 生產網路的工廠模式
def get_net():
    net = tf.keras.Sequential()
    net.add(Dense(256, input_shape = (512,), activation = "relu"))
    net.add(Dense(128, activation = "relu"))
    net.add(Dense(2, activation = "linear"))
    return net

x = tf.random.normal([1,512])
net = get_net()
net(x)
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
from paddle.jit import to_static
from paddle.static import InputSpec

# 生產網路的工廠模式
def get_net():
    blocks = [
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    ]
    net = nn.Sequential(*blocks)
    return net

x = paddle.randn((1, 512))
net = get_net()
net(x)
```

:begin_tab:`mxnet`
透過呼叫`hybridize`函式，我們就有能力編譯和最佳化多層感知機中的計算，而模型的計算結果保持不變。
:end_tab:

:begin_tab:`pytorch`
透過使用`torch.jit.script`函式來轉換模型，我們就有能力編譯和最佳化多層感知機中的計算，而模型的計算結果保持不變。
:end_tab:

:begin_tab:`tensorflow`
一開始，TensorFlow中建立的所有函式都是作為計算圖建立的，因此預設情況下是JIT編譯的。但是，隨著TensorFlow2.X和EargeTensor的發布，計算圖就不再是預設行為。我們可以使用tf.function重新啟用這個功能。tf.function更常被用作函式裝飾器，如下所示，它也可以直接將其作為普通的Python函式呼叫。模型的計算結果保持不變。
:end_tab:

:begin_tab:`paddle`
透過使用`paddle.jit.to_static`函式來轉換模型，我們就有能力編譯和最佳化多層感知機中的計算，而模型的計算結果保持不變。
:end_tab:

```{.python .input}
net.hybridize()
net(x)
```

```{.python .input}
#@tab pytorch
net = torch.jit.script(net)
net(x)
```

```{.python .input}
#@tab tensorflow
net = tf.function(net)
net(x)
```

```{.python .input}
#@tab paddle
net = paddle.jit.to_static(net)
net(x)
```

:begin_tab:`mxnet`
我們只需將一個區塊指定為`HybridSequential`，然後編寫與之前相同的程式碼，再呼叫`hybridize`，當完成這些任務後，網路就將得到最佳化（我們將在下面對效能進行基準測試）。不幸的是，這種魔法並不適用於每一層。也就是說，如果某個層是從`Block`類而不是從`HybridBlock`類繼承的，那麼它將不會得到最佳化。
:end_tab:

:begin_tab:`pytorch`
我們編寫與之前相同的程式碼，再使用`torch.jit.script`簡單地轉換模型，當完成這些任務後，網路就將得到最佳化（我們將在下面對效能進行基準測試）。
:end_tab:

:begin_tab:`tensorflow`
我們編寫與之前相同的程式碼，再使用`tf.function`簡單地轉換模型，當完成這些任務後，網路將以TensorFlow的MLIR中間表示形式建立為一個計算圖，並在編譯器級別進行大量最佳化以滿足快速執行的需要（我們將在下面對效能進行基準測試）。透過將`jit_compile = True`標誌新增到`tf.function()`的函式呼叫中可以顯式地啟用TensorFlow中的XLA（線性代數加速）功能。在某些情況下，XLA可以進一步最佳化JIT的編譯程式碼。如果沒有這種顯式定義，圖形模式將會被啟用，但是XLA可以使某些大規模的線性代數的運算速度更快（與我們在深度學習程式中看到的操作類似），特別是在GPU環境中。
:end_tab:

:begin_tab:`paddle`
我們編寫與之前相同的程式碼，再使用`paddle.jit.to_static`簡單地轉換模型，當完成這些任務後，網路就將得到最佳化（我們將在下面對效能進行基準測試）。
:end_tab:

### 透過混合式程式設計加速

為了證明透過編譯獲得了效能改進，我們比較了混合程式設計前後執行`net(x)`所需的時間。讓我們先定義一個度量時間的類，它在本章中在衡量（和改進）模型效能時將非常有用。

```{.python .input}
#@tab all
#@save
class Benchmark:
    """用於測量執行時間"""
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = d2l.Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')
```

:begin_tab:`mxnet`
現在我們可以呼叫網路兩次，一次使用混合式，一次不使用混合式。
:end_tab:

:begin_tab:`pytorch`
現在我們可以呼叫網路兩次，一次使用torchscript，一次不使用torchscript。
:end_tab:

:begin_tab:`tensorflow`
現在我們可以呼叫網路三次，一次使用eager模式，一次是使用圖模式，一次使用JIT編譯的XLA。
:end_tab:

:begin_tab:`paddle`
現在我們可以呼叫網路兩次，一次使用動態圖指令式程式設計，一次使用靜態圖符號式程式設計。
:end_tab:

```{.python .input}
net = get_net()
with Benchmark('無混合式'):
    for i in range(1000): net(x)
    npx.waitall()

net.hybridize()
with Benchmark('混合式'):
    for i in range(1000): net(x)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
net = get_net()
with Benchmark('無torchscript'):
    for i in range(1000): net(x)

net = torch.jit.script(net)
with Benchmark('有torchscript'):
    for i in range(1000): net(x)
```

```{.python .input}
#@tab tensorflow
net = get_net()
with Benchmark('Eager模式'):
    for i in range(1000): net(x)

net = tf.function(net)
with Benchmark('Graph模式'):
    for i in range(1000): net(x)
```

```{.python .input}
#@tab paddle
net = get_net()
with Benchmark('飛槳動態圖指令式程式設計'):
    for i in range(1000): net(x)

# InputSpec用於描述模型輸入的簽名信息，包括shape、dtype和name
x_spec = InputSpec(shape=[-1, 512], name='x') 
net = paddle.jit.to_static(get_net(),input_spec=[x_spec])
with Benchmark('飛槳靜態圖符號式程式設計'):
    for i in range(1000): net(x)
```

:begin_tab:`mxnet`
如以上結果所示，在`HybridSequential`的實例呼叫`hybridize`函式後，透過使用符號式程式設計提高了計算效能。
:end_tab:

:begin_tab:`pytorch`
如以上結果所示，在`nn.Sequential`的實例被函式`torch.jit.script`指令碼化後，透過使用符號式程式設計提高了計算效能。
:end_tab:

:begin_tab:`tensorflow`
如以上結果所示，在`tf.keras.Sequential`的實例被函式`tf.function`指令碼化後，透過使用TensorFlow中的圖模式執行方式實現的符號式程式設計提高了計算效能。
:end_tab:

:begin_tab:`paddle`
如以上結果所示，在`nn.Sequential`的實例被函式`paddle.jit.to_static`指令碼化後，透過使用符號式程式設計提高了計算效能。
:end_tab:

### 序列化

:begin_tab:`mxnet`
編譯模型的好處之一是我們可以將模型及其引數序列化（儲存）到磁碟。這允許這些訓練好的模型部署到其他裝置上，並且還能方便地使用其他前端程式語言。同時，通常編譯模型的程式碼執行速度也比指令式程式設計更快。讓我們看看`export`的實際功能。
:end_tab:

:begin_tab:`pytorch`
編譯模型的好處之一是我們可以將模型及其引數序列化（儲存）到磁碟。這允許這些訓練好的模型部署到其他裝置上，並且還能方便地使用其他前端程式語言。同時，通常編譯模型的程式碼執行速度也比指令式程式設計更快。讓我們看看`save`的實際功能。
:end_tab:

:begin_tab:`tensorflow`
編譯模型的好處之一是我們可以將模型及其引數序列化（儲存）到磁碟。這允許這些訓練好的模型部署到其他裝置上，並且還能方便地使用其他前端程式語言。同時，通常編譯模型的程式碼執行速度也比指令式程式設計更快。在TensorFlow中保存模型的底層API是`tf.saved_model`，讓我們來看看`saved_model`的執行情況。
:end_tab:

:begin_tab:`paddle`
編譯模型的好處之一是我們可以將模型及其引數序列化（儲存）到磁碟。這允許這些訓練好的模型部署到其他裝置上，並且還能方便地使用其他前端程式語言。同時，通常編譯模型的程式碼執行速度也比指令式程式設計更快。讓我們看看`paddle.jit.save`的實際功能。
:end_tab:

```{.python .input}
net.export('my_mlp')
!ls -lh my_mlp*
```

```{.python .input}
#@tab pytorch
net.save('my_mlp')
!ls -lh my_mlp*
```

```{.python .input}
#@tab tensorflow
net = get_net()
tf.saved_model.save(net, 'my_mlp')
!ls -lh my_mlp*
```

```{.python .input}
#@tab paddle
paddle.jit.save(net, './my_mlp')
!ls -lh my_mlp*
```

:begin_tab:`mxnet`
模型被分解成兩個檔案，一個是大的二進位引數檔案，一個是執行模型計算所需要的程式的JSON描述檔案。這些檔案可以被其他前端語言讀取，例如C++、R、Scala和Perl，只要這些語言能夠被Python或者MXNet支援。讓我們看看模型描述中的前幾行。
:end_tab:

```{.python .input}
!head my_mlp-symbol.json
```

:begin_tab:`mxnet`
之前，我們示範了在呼叫`hybridize`函式之後，模型能夠獲得優異的計算效能和可移植性。注意，混合式可能會影響模型的靈活性，特別是在控制流方面。

此外，與`Block`實例需要使用`forward`函式不同的是`HybridBlock`實例需要使用`hybrid_forward`函式。
:end_tab:

```{.python .input}
class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(4)
        self.output = nn.Dense(2)

    def hybrid_forward(self, F, x):
        print('module F: ', F)
        print('value  x: ', x)
        x = F.npx.relu(self.hidden(x))
        print('result  : ', x)
        return self.output(x)
```

:begin_tab:`mxnet`
上述程式碼實現了一個具有$4$個隱藏單元和$2$個輸出的簡單網路。`hybrid_forward`函式增加了一個必需的引數`F`，因為是否採用混合模式將影響程式碼使用稍微不同的函式庫（`ndarray`或`symbol`）進行處理。這兩個類執行了非常相似的函式，於是MXNet將自動確定這個引數。為了理解發生了什麼，我們將列印引數作為了函式呼叫的一部分。
:end_tab:

```{.python .input}
net = HybridNet()
net.initialize()
x = np.random.normal(size=(1, 3))
net(x)
```

:begin_tab:`mxnet`
重複的前向傳播將導致相同的輸出（細節已被省略）。現在看看呼叫`hybridize`函式會發生什麼。
:end_tab:

```{.python .input}
net.hybridize()
net(x)
```

:begin_tab:`mxnet`
程式使用`symbol`模組替換了`ndarray`模組來表示`F`。而且，即使輸入是`ndarray`型別，流過網路的資料現在也轉換為`symbol`型別，這種轉換正是編譯過程的一部分。再次的函式呼叫產生了令人驚訝的結果：
:end_tab:

```{.python .input}
net(x)
```

:begin_tab:`mxnet`
這與我們在前面看到的情況大不相同。`hybrid_forward`中定義的所有列印語句都被忽略了。實際上，在`net(x)`被混合執行時就不再使用Python直譯器。這意味著任何Python程式碼（例如`print`語句）都會被忽略，以利於更精簡的執行和更好的效能。MXNet透過直接呼叫C++後端替代Python直譯器。另外請注意，`symbol`模組不能支援某些函式（例如`asnumpy`），因此`a += b`和`a[:] = a + b`等操作必須重寫為`a = a + b`。儘管如此，當速度很重要時，模型的編譯也是值得的。速度的優勢可以從很小的百分比到兩倍以上，主要取決於模型的複雜性、CPU的速度以及GPU的速度和數量。
:end_tab:

## 小結

* 指令式程式設計使得新模型的設計變得容易，因為可以依據控制流編寫程式碼，並擁有相對成熟的Python軟體生態。
* 符號式程式設計要求我們先定義並且編譯程序，然後再執行程式，其好處是提高了計算效能。

:begin_tab:`mxnet`
* MXNet能夠根據使用者需要，結合這兩種方法（指令式程式設計和符號式程式設計）的優點。
* 由`HybridSequential`和`HybridBlock`類構造的模型能夠透過呼叫`hybridize`函式將指令式程式轉換為符號式程式。
:end_tab:

## 練習

:begin_tab:`mxnet`
1. 在本節的`HybridNet`類別的`hybrid_forward`函式的第一行中新增`x.asnumpy()`，執行程式碼並觀察遇到的錯誤。為什麼會這樣？
1. 如果我們在`hybrid_forward`函式中新增控制流，即Python語句`if`和`for`，會發生什麼？
1. 回顧前幾章中感興趣的模型，能透過重新實現它們來提高它們的計算效能嗎？
:end_tab:

:begin_tab:`pytorch,tensorflow`
1. 回顧前幾章中感興趣的模型，能提高它們的計算效能嗎？
:end_tab:

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2789)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2788)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/2787)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11857)
:end_tab: