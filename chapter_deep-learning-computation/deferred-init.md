# 延後初始化
:label:`sec_deferred_init`

到目前為止，我們忽略了建立網路時需要做的以下這些事情：

* 我們定義了網路架構，但沒有指定輸入維度。
* 我們新增層時沒有指定前一層的輸出維度。
* 我們在初始化引數時，甚至沒有足夠的訊息來確定模型應該包含多少引數。

有些讀者可能會對我們的程式碼能執行感到驚訝。
畢竟，深度學習框架無法判斷網路的輸入維度是什麼。
這裡的訣竅是框架的*延後初始化*（defers initialization），
即直到資料第一次透過模型傳遞時，框架才會動態地推斷出每個層的大小。

在以後，當使用卷積神經網路時，
由於輸入維度（即影象的解析度）將影響每個後續層的維數，
有了該技術將更加方便。
現在我們在編寫程式碼時無須知道維度是什麼就可以設定引數，
這種能力可以大大簡化定義和修改模型的任務。
接下來，我們將更深入地研究初始化機制。

## 實例化網路

首先，讓我們實例化一個多層感知機。

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    return net

net = get_net()
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
```

此時，因為輸入維數是未知的，所以網路不可能知道輸入層權重的維數。
因此，框架尚未初始化任何引數，我們透過嘗試訪問以下引數進行確認。

```{.python .input}
print(net.collect_params)
print(net.collect_params())
```

```{.python .input}
#@tab tensorflow
[net.layers[i].get_weights() for i in range(len(net.layers))]
```

:begin_tab:`mxnet`
注意，當引數物件存在時，每個層的輸入維度為-1。
MXNet使用特殊值-1表示引數維度仍然未知。
此時，嘗試訪問`net[0].weight.data()`將觸發執行時錯誤，
提示必須先初始化網路，然後才能訪問引數。
現在讓我們看看當我們試圖透過`initialize`函式初始化引數時會發生什麼。
:end_tab:

:begin_tab:`tensorflow`
請注意，每個層物件都存在，但權重為空。
使用`net.get_weights()`將丟擲一個錯誤，因為權重尚未初始化。
:end_tab:

```{.python .input}
net.initialize()
net.collect_params()
```

:begin_tab:`mxnet`
如我們所見，一切都沒有改變。
當輸入維度未知時，呼叫`initialize`不會真正初始化引數。
而是會在MXNet內部宣告希望初始化引數，並且可以選擇初始化分佈。
:end_tab:

接下來讓我們將資料透過網路，最終使框架初始化引數。

```{.python .input}
X = np.random.uniform(size=(2, 20))
net(X)

net.collect_params()
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((2, 20))
net(X)
[w.shape for w in net.get_weights()]
```

一旦我們知道輸入維數是20，框架可以透過代入值20來識別第一層權重矩陣的形狀。
識別出第一層的形狀後，框架處理第二層，依此類推，直到所有形狀都已知為止。
注意，在這種情況下，只有第一層需要延遲初始化，但是框架仍是按順序初始化的。
等到知道了所有的引數形狀，框架就可以初始化引數。

## 小結

* 延後初始化使框架能夠自動推斷引數形狀，使修改模型架構變得容易，避免了一些常見的錯誤。
* 我們可以透過模型傳遞資料，使框架最終初始化引數。

## 練習

1. 如果指定了第一層的輸入尺寸，但沒有指定後續層的尺寸，會發生什麼？是否立即進行初始化？
1. 如果指定了不對應的維度會發生什麼？
1. 如果輸入具有不同的維度，需要做什麼？提示：檢視引數繫結的相關內容。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/5770)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/5770)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1833)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11779)
:end_tab:
