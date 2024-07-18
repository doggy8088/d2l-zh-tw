# 使用Jupyter Notebook
:label:`sec_jupyter`

本節介紹如何使用Jupyter Notebook編輯和執行本書各章中的程式碼。確保你已按照 :ref:`chap_installation`中的說明安裝了Jupyter並下載了程式碼。如果你想了解更多關於Jupyter的訊息，請參閱其[文件](https://jupyter.readthedocs.io/en/latest/)中的優秀教程。 

## 在本地編輯和執行程式碼

假設本書程式碼的本地路徑為`xx/yy/d2l-en/`。使用shell將目錄更改為此路徑（`cd xx/yy/d2l-en`）並執行指令`jupyter notebook`。如果瀏覽器未自動開啟，請開啟http://localhost:8888。此時你將看到Jupyter的介面以及包含本書程式碼的所有資料夾，如 :numref:`fig_jupyter00`所示

![包含本書程式碼的資料夾](../img/jupyter00.png)
:width:`600px`
:label:`fig_jupyter00`

你可以透過單擊網頁上顯示的資料夾來訪問notebook檔案。它們通常有後綴『.ipynb』。為了簡潔起見，我們建立了一個臨時的『test.ipynb』檔案。單擊後顯示的內容如 :numref:`fig_jupyter01`所示。此notebook包括一個標記單元格和一個程式碼單元格。標記單元格中的內容包括『This Is a Title』和『This is text.』。程式碼單元包含兩行Python程式碼。 

![『test.ipynb』檔案中的markdown和程式碼區塊](../img/jupyter01.png)
:width:`600px`
:label:`fig_jupyter01`

雙擊標記單元格以進入編輯模式。在單元格末尾新增一個新的文字字串『Hello world.』，如 :numref:`fig_jupyter02`所示。 

![編輯markdown單元格](../img/jupyter02.png)
:width:`600px`
:label:`fig_jupyter02`

如 :numref:`fig_jupyter03`所示，單擊選單欄中的『Cell』 $\rightarrow$ 『Run Cells』以執行編輯後的單元格。 

![執行單元格](../img/jupyter03.png)
:width:`600px`
:label:`fig_jupyter03`

執行後，markdown單元格如 :numref:`fig_jupyter04`所示。 

![編輯後的markdown單元格](../img/jupyter04.png)
:width:`600px`
:label:`fig_jupyter04`

接下來，單擊程式碼單元。將最後一行程式碼後的元素乘以2，如 :numref:`fig_jupyter05`所示。 

![編輯程式碼單元格](../img/jupyter05.png)
:width:`600px`
:label:`fig_jupyter05`

你還可以使用快捷鍵（預設情況下為Ctrl+Enter）執行單元格，並從 :numref:`fig_jupyter06`獲取輸出結果。 

![執行程式碼單元格以獲得輸出](../img/jupyter06.png)
:width:`600px`
:label:`fig_jupyter06`

當一個notebook包含更多單元格時，我們可以單擊選單欄中的『Kernel』$\rightarrow$『Restart & Run All』來執行整個notebook中的所有單元格。透過單擊選單欄中的『Help』$\rightarrow$『Edit Keyboard Shortcuts』，可以根據你的首選項編輯快捷鍵。 

## 高階選項

除了本地編輯，還有兩件事非常重要：以markdown格式編輯notebook和遠端執行Jupyter。當我們想要在更快的伺服器上執行程式碼時，後者很重要。前者很重要，因為Jupyter原生的ipynb格式儲存了大量輔助資料，這些資料實際上並不特定於notebook中的內容，主要與程式碼的執行方式和執行位置有關。這讓git感到困惑，並且使得合併貢獻非常困難。幸運的是，還有另一種選擇——在markdown中進行本地編輯。 

### Jupyter中的Markdown檔案

如果你希望對本書的內容有所貢獻，則需要在GitHub上修改源檔案（md檔案，而不是ipynb檔案）。使用notedown外掛，我們可以直接在Jupyter中修改md格式的notebook。 

首先，安裝notedown外掛，執行Jupyter Notebook並載入外掛：

```
pip install d2l-notedown  # 你可能需要解除安裝原始notedown
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```

要在執行Jupyter Notebook時預設開啟notedown外掛，請執行以下操作：首先，生成一個Jupyter Notebook設定檔案（如果已經生成了，可以跳過此步驟）。

```
jupyter notebook --generate-config
```

然後，在Jupyter Notebook設定檔案的末尾新增以下行（對於Linux/macOS，通常位於`~/.jupyter/jupyter_notebook_config.py`）：

```
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```

在這之後，你只需要執行`jupyter notebook`指令就可以預設開啟notedown外掛。 

### 在遠端伺服器上執行Jupyter Notebook

有時，你可能希望在遠端伺服器上執行Jupyter Notebook，並透過本地電腦上的瀏覽器訪問它。如果本地電腦上安裝了Linux或MacOS（Windows也可以透過PuTTY等第三方軟體支援此功能），則可以使用連接埠轉發：

```
ssh myserver -L 8888:localhost:8888
```

以上是遠端伺服器`myserver`的地址。然後我們可以使用http://localhost:8888 訪問執行Jupyter Notebook的遠端伺服器`myserver`。下一節將詳細介紹如何在AWS實例上執行Jupyter Notebook。 

### 執行時間

我們可以使用`ExecuteTime`外掛來計算Jupyter Notebook中每個程式碼單元的執行時間。使用以下指令安裝外掛：

```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable execute_time/ExecuteTime
```

## 小結

* 使用Jupyter Notebook工具，我們可以編輯、執行和為本書做貢獻。
* 使用連接埠轉發在遠端伺服器上執行Jupyter Notebook。

## 練習

1. 在本地電腦上使用Jupyter Notebook編輯並執行本書中的程式碼。
1. 使用Jupyter Notebook透過連接埠轉發來遠端編輯和執行本書中的程式碼。
1. 對於兩個方矩陣，測量$\mathbf{A}^\top \mathbf{B}$與$\mathbf{A} \mathbf{B}$在$\mathbb{R}^{1024 \times 1024}$中的執行時間。哪一個更快？

[Discussions](https://discuss.d2l.ai/t/5731)
