# 安裝
:label:`chap_installation`

我們需要設定一個環境來執行 Python、Jupyter Notebook、相關函式庫以及執行本書所需的程式碼，以快速入門並獲得動手學習經驗。

## 安裝 Miniconda

最簡單的方法就是安裝依賴Python 3.x的[Miniconda](https://conda.io/en/latest/miniconda.html)。
如果已安裝conda，則可以跳過以下步驟。訪問Miniconda網站，根據Python3.x版本確定適合的版本。

如果我們使用macOS，假設Python版本是3.9（我們的測試版本），將下載名稱包含字串『MacOSX』的bash指令碼，並執行以下操作：

```bash
# 以Intel處理器為例，檔案名可能會更改
sh Miniconda3-py39_4.12.0-MacOSX-x86_64.sh -b
```


如果我們使用Linux，假設Python版本是3.9（我們的測試版本），將下載名稱包含字串『Linux』的bash指令碼，並執行以下操作：

```bash
# 檔案名可能會更改
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh -b
```


接下來，初始化終端Shell，以便我們可以直接執行`conda`。

```bash
~/miniconda3/bin/conda init
```


現在關閉並重新開啟當前的shell。並使用下面的指令建立一個新的環境：

```bash
conda create --name d2l python=3.9 -y
```


現在啟用 `d2l` 環境：

```bash
conda activate d2l
```


## 安裝深度學習框架和`d2l`軟體套件

在安裝深度學習框架之前，請先檢查電腦上是否有可用的GPU。
例如可以檢視電腦是否裝有NVIDIA GPU並已安裝[CUDA](https://developer.nvidia.com/cuda-downloads)。
如果機器沒有任何GPU，沒有必要擔心，因為CPU在前幾章完全夠用。
但是，如果想流暢地學習全部章節，請提早獲取GPU並且安裝深度學習框架的GPU版本。


:begin_tab:`mxnet`

安裝MXNet的GPU版本，首先需要知道已安裝的CUDA版本。
（可以透過執行`nvcc --version`或`cat /usr/local/cuda/version.txt`來檢驗。）
假設已安裝CUDA 10.1版本，請執行以下指令：

```bash
# 對於Linux和macOS使用者
pip install mxnet-cu101==1.7.0

# 對於Windows使用者
pip install mxnet-cu101==1.7.0 -f https://dist.mxnet.io/python
```


可以根據CUDA版本更改如上`mxnet-cu101`的最後一位數字，
例如：CUDA 10.0是`cu100`， CUDA 9.0是`cu90`。


如果機器沒有NVIDIA GPU或CUDA，可以按如下方式MXNet的CPU版本：

```bash
pip install mxnet==1.7.0.post1
```


:end_tab:


:begin_tab:`pytorch`

我們可以按如下方式安裝PyTorch的CPU或GPU版本：

```bash
pip install torch==1.12.0
pip install torchvision==0.13.0
```


:end_tab:

:begin_tab:`tensorflow`
我們可以按如下方式安裝TensorFlow的CPU或GPU版本：

```bash
pip install tensorflow==2.8.0
pip install tensorflow-probability==0.16.0
```


:end_tab:

:begin_tab:`paddle`
安裝PaddlePaddle的GPU版本，首先需要知道已安裝的CUDA版本。
（可以透過執行`nvcc --version`或`cat /usr/local/cuda/version.txt`來檢驗。）
假設已安裝CUDA 11.2版本，請執行以下指令：

```bash
python -m pip install paddlepaddle-gpu==2.3.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```


如果機器沒有NVIDIA GPU或CUDA，可以按如下方式PaddlePaddle的CPU版本：

```bash
python -m pip install paddlepaddle==2.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```


:end_tab:

我們的下一步是安裝`d2l`套件，以方便調取本書中經常使用的函式和類：

```bash
pip install d2l==0.17.6
```


## 下載 D2L Notebook

接下來，需要下載這本書的程式碼。
可以點選本書HTML頁面頂部的『Jupyter 記事本』選項下載後解壓程式碼，或者可以按照如下方式進行下載：


:begin_tab:`mxnet`

```bash
mkdir d2l-zh && cd d2l-zh
curl https://zh-v2.d2l.ai/d2l-zh-2.0.0.zip -o d2l-zh.zip
unzip d2l-zh.zip && rm d2l-zh.zip
cd mxnet
```


注意：如果沒有安裝`unzip`，則可以透過執行`sudo apt install unzip`進行安裝。

:end_tab:


:begin_tab:`pytorch`

```bash
mkdir d2l-zh && cd d2l-zh
curl https://zh-v2.d2l.ai/d2l-zh-2.0.0.zip -o d2l-zh.zip
unzip d2l-zh.zip && rm d2l-zh.zip
cd pytorch
```


注意：如果沒有安裝`unzip`，則可以透過執行`sudo apt install unzip`進行安裝。

:end_tab:


:begin_tab:`tensorflow`

```bash
mkdir d2l-zh && cd d2l-zh
curl https://zh-v2.d2l.ai/d2l-zh-2.0.0.zip -o d2l-zh.zip
unzip d2l-zh.zip && rm d2l-zh.zip
cd tensorflow
```


注意：如果沒有安裝`unzip`，則可以透過執行`sudo apt install unzip`進行安裝。

:end_tab:


:begin_tab:`paddle`

```bash
mkdir d2l-zh && cd d2l-zh
curl https://zh-v2.d2l.ai/d2l-zh-2.0.0.zip -o d2l-zh.zip
unzip d2l-zh.zip && rm d2l-zh.zip
cd paddle
```


注意：如果沒有安裝`unzip`，則可以透過執行`sudo apt install unzip`進行安裝。

:end_tab:


安裝完成後我們可以透過執行以下指令開啟Jupyter筆記本（在Window系統的指令行視窗中執行以下指令前，需先將當前路徑定位到剛下載的本書程式碼解壓後的目錄）：

```bash
jupyter notebook
```


現在可以在Web瀏覽器中開啟<http://localhost:8888>（通常會自動開啟）。
由此，我們可以執行這本書中每個部分的程式碼。
在執行書籍程式碼、更新深度學習框架或`d2l`軟體套件之前，請始終執行`conda activate d2l`以啟用執行時環境。
要退出環境，請執行`conda deactivate`。



:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2082)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2083)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/2084)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11679)
:end_tab:
