# 使用Amazon SageMaker
:label:`sec_sagemaker`

深度學習程式可能需要很多計算資源，這很容易超出你的本地電腦所能提供的範圍。雲端運算服務允許你使用功能更強大的電腦更輕鬆地執行本書的GPU密集型程式碼。本節將介紹如何使用Amazon SageMaker執行本書的程式碼。

## 註冊

首先，我們需要在註冊一個帳戶https://aws.amazon.com/。 為了增加安全性，鼓勵使用雙因素身份驗證。設定詳細的計費和支出警報也是一個好主意，以避免任何意外，例如，當忘記停止執行實例時。登入AWS帳戶後，轉到[console](http://console.aws.amazon.com/)並搜尋『Amazon SageMaker』（參見 :numref:`fig_sagemaker`），然後單擊它開啟SageMaker面板。

![搜尋並開啟SageMaker面板](../img/sagemaker.png)
:width:`300px`
:label:`fig_sagemaker`

## 建立SageMaker實例

接下來，讓我們建立一個notebook實例，如 :numref:`fig_sagemaker-create`所示。

![建立一個SageMaker實例](../img/sagemaker-create.png)
:width:`400px`
:label:`fig_sagemaker-create`

SageMaker提供多個具有不同計算能力和價格的[實例型別](https://aws.amazon.com/sagemaker/pricing/instance-types/)。建立notebook實例時，可以指定其名稱和型別。在 :numref:`fig_sagemaker-create-2`中，我們選擇`ml.p3.2xlarge`：使用一個Tesla V100 GPU和一個8核CPU，這個實例的效能足夠本書的大部分內容使用。

![選擇實例型別](../img/sagemaker-create-2.png)
:width:`400px`
:label:`fig_sagemaker-create-2`

:begin_tab:`mxnet`
用於與SageMaker一起執行的ipynb格式的整本書可從https://github.com/d2l-ai/d2l-en-sagemaker獲得。
我們可以指定此GitHub儲存庫URL（ :numref:`fig_sagemaker-create-3`），以允許SageMaker在建立實例時克隆它。
:end_tab:

:begin_tab:`pytorch`
用於與SageMaker一起執行的ipynb格式的整本書可從https://github.com/d2l-ai/d2l-pytorch-sagemaker獲得。
我們可以指定此GitHub儲存庫URL（ :numref:`fig_sagemaker-create-3`），以允許SageMaker在建立實例時克隆它。
:end_tab:

:begin_tab:`tensorflow`
用於與SageMaker一起執行的ipynb格式的整本書可從https://github.com/d2l-ai/d2l-tensorflow-sagemaker獲得。
我們可以指定此GitHub儲存庫URL（ :numref:`fig_sagemaker-create-3`），以允許SageMaker在建立實例時克隆它。
:end_tab:

![指定GitHub儲存庫](../img/sagemaker-create-3.png)
:width:`400px`
:label:`fig_sagemaker-create-3`

## 執行和停止實例

建立實例可能需要幾分鐘的時間。當實例準備就緒時，單擊它旁邊的『Open Jupyter』連結（ :numref:`fig_sagemaker-open`），以便你可以在此實例上編輯並執行本書的所有Jupyter Notebook（類似於 :numref:`sec_jupyter`中的步驟）。

![在建立的SageMaker實例上開啟Jupyter](../img/sagemaker-open.png)
:width:`400px`
:label:`fig_sagemaker-open`

完成工作後，不要忘記停止實例以避免進一步收費（ :numref:`fig_sagemaker-stop`）。

![停止SageMaker實例](../img/sagemaker-stop.png)
:width:`300px`
:label:`fig_sagemaker-stop`

## 更新Notebook

:begin_tab:`mxnet`
這本開源書的notebook將定期在GitHub上的[d2l-ai/d2l-en-sagemaker](https://github.com/d2l-ai/d2l-en-sagemaker)儲存庫中更新。要更新至最新版本，你可以在SageMaker實例（ :numref:`fig_sagemaker-terminal`）上開啟終端。
:end_tab:

:begin_tab:`pytorch`
這本開源書的notebook將定期在GitHub上的[d2l-ai/d2l-pytorch-sagemaker](https://github.com/d2l-ai/d2l-pytorch-sagemaker)儲存庫中更新。要更新至最新版本，你可以在SageMaker實例（ :numref:`fig_sagemaker-terminal`）上開啟終端。
:end_tab:

:begin_tab:`tensorflow`
這本開源書的notebook將定期在GitHub上的[d2l-ai/d2l-tensorflow-sagemaker](https://github.com/d2l-ai/d2l-tensorflow-sagemaker)儲存庫中更新。要更新至最新版本，你可以在SageMaker實例（ :numref:`fig_sagemaker-terminal`）上開啟終端。
:end_tab:

![在SageMaker實例上開啟終端](../img/sagemaker-terminal.png)
:width:`300px`
:label:`fig_sagemaker-terminal`

你可能希望在從遠端儲存函式庫提取更新之前送出本地更改。否則，只需在終端中使用以下指令放棄所有本地更改：

:begin_tab:`mxnet`

```bash
cd SageMaker/d2l-en-sagemaker/
git reset --hard
git pull
```


:end_tab:

:begin_tab:`pytorch`

```bash
cd SageMaker/d2l-pytorch-sagemaker/
git reset --hard
git pull
```


:end_tab:

:begin_tab:`tensorflow`

```bash
cd SageMaker/d2l-tensorflow-sagemaker/
git reset --hard
git pull
```


:end_tab:

## 小結

* 我們可以使用Amazon SageMaker建立一個GPU的notebook實例來執行本書的密集型程式碼。
* 我們可以透過Amazon SageMaker實例上的終端更新notebooks。

## 練習

1. 使用Amazon SageMaker編輯並執行任何需要GPU的部分。
1. 開啟終端以訪問儲存本書所有notebooks的本地目錄。

[Discussions](https://discuss.d2l.ai/t/5732)
