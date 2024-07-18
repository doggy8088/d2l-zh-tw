# 資料預處理
:label:`sec_pandas`

為了能用深度學習來解決現實世界的問題，我們經常從預處理原始資料開始，
而不是從那些準備好的張量格式資料開始。
在Python中常用的資料分析工具中，我們通常使用`pandas`軟體套件。
像龐大的Python生態系統中的許多其他擴充套件套件一樣，`pandas`可以與張量相容。
本節我們將簡要介紹使用`pandas`預處理原始資料，並將原始資料轉換為張量格式的步驟。
後面的章節將介紹更多的資料預處理技術。

## 讀取資料集

舉一個例子，我們首先(**建立一個人工資料集，並存儲在CSV（逗號分隔值）檔案**)
`../data/house_tiny.csv`中。
以其他格式儲存的資料也可以透過類似的方式進行處理。
下面我們將資料集按行寫入CSV檔案中。

```{.python .input}
#@tab all
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一個資料樣本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
```

要[**從建立的CSV檔案中載入原始資料集**]，我們匯入`pandas`套件並呼叫`read_csv`函式。該資料集有四行三列。其中每行描述了房間數量（『NumRooms』）、巷子類別型（『Alley』）和房屋價格（『Price』）。

```{.python .input}
#@tab all
# 如果沒有安裝pandas，只需取消對以下行的註解來安裝pandas
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## 處理缺失值

注意，『NaN』項代表缺失值。
[**為了處理缺失的資料，典型的方法包括*插值法*和*刪除法*，**]
其中插值法用一個替代值彌補缺失值，而刪除法則直接忽略缺失值。
在(**這裡，我們將考慮插值法**)。

透過位置索引`iloc`，我們將`data`分成`inputs`和`outputs`，
其中前者為`data`的前兩列，而後者為`data`的最後一列。
對於`inputs`中缺少的數值，我們用同一列的均值替換『NaN』項。

```{.python .input}
#@tab all
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

[**對於`inputs`中的類別值或離散值，我們將『NaN』視為一個類別。**]
由於『巷子類別型』（『Alley』）列只接受兩種型別的類別值『Pave』和『NaN』，
`pandas`可以自動將此列轉換為兩列『Alley_Pave』和『Alley_nan』。
巷子類別型為『Pave』的行會將『Alley_Pave』的值設定為1，『Alley_nan』的值設定為0。
缺少巷子類別型的行會將『Alley_Pave』和『Alley_nan』分別設定為0和1。

```{.python .input}
#@tab all
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

## 轉換為張量格式

[**現在`inputs`和`outputs`中的所有條目都是數值型別，它們可以轉換為張量格式。**]
當資料採用張量格式後，可以透過在 :numref:`sec_ndarray`中引入的那些張量函式來進一步操作。

```{.python .input}
from mxnet import np

X, y = np.array(inputs.to_numpy(dtype=float)), np.array(outputs.to_numpy(dtype=float))
X, y
```

```{.python .input}
#@tab pytorch
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
X, y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

X = tf.constant(inputs.to_numpy(dtype=float))
y = tf.constant(outputs.to_numpy(dtype=float))
X, y
```

```{.python .input}
#@tab paddle
import warnings
warnings.filterwarnings(action='ignore')
import paddle

X, y = paddle.to_tensor(inputs.values), paddle.to_tensor(outputs.values)
X, y
```

## 小結

* `pandas`軟體套件是Python中常用的資料分析工具中，`pandas`可以與張量相容。
* 用`pandas`處理缺失的資料時，我們可根據情況選擇用插值法和刪除法。

## 練習

建立包含更多行和列的原始資料集。

1. 刪除缺失值最多的列。
2. 將預處理後的資料集轉換為張量格式。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1749)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1750)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1748)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11681)
:end_tab:
