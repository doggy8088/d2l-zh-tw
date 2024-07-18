# 樣式規範

## 文字

* 章節
    * 每章開頭對全章做介紹
    * 結構標題一致
        * 小結
        * 練習
        * 掃碼直達討論區
        * 參考文獻（如有）
    * 引用
        * 在每節結尾處引用
* 字串
    * 使用中文雙引號
* 符號描述
    * 時刻t（不是t時刻）
	* 形狀使用英文標點
        * (10, 20) 不是 （10，20）
* 空格：
	* 文字中中文和英文、數字、數學公式、特殊字型等之間不要加空格
	* 非行首的`:numref:`、`:cite:`等前留有一個英文空格（否則網頁不渲染）
	* 程式碼註解同上
* 人稱
    * 第一人稱 → 我們
    * 第二人稱 → 讀者、你、大家
* 工具或部件
    * Gluon, MXNet, NumPy, spaCy, ResNet-18, Fashion-MNIST, matplotlib
        * 這些都作為詞，不要帶重音符
    * `backward`函式
        * 不是『`backward()`函式』 （不要帶括號）
    * `for`迴圈
* 術語
    * 統一使用
        * 函式（非方法）
        * 實例（非物件）
        * 區分：超引數和引數
        * 區分：小批次隨機梯度下降和隨機梯度下降
        * 權重、偏差、標籤
        * 模型訓練、模型預測（推斷）
        * 訓練資料集、驗證資料集、測試資料集
    * 中文優先於英文
        * 首次出現，註明原英文術語
            * 無須加粗
            * 無須加引號
    * 中英文對照統一標準
        * https://github.com/mli/gluon-tutorials-zh/blob/master/README.md

## 數學

* 數學符號樣式一致
    * https://github.com/goodfeli/dlbook_notation/blob/master/notation_example.pdf
* 書本頁寬限制
    * 每行長度
* 引用
    * 上式和下式
    * 以上N式，以下N式
* 公式末放英文標點
    * 逗號：,
    * 句號：.
* 賦值符號
    * \leftarrow

## 圖片

* 軟體
    * 使用OmniGraffle製圖，以100%的大小匯出pdf（infinite canvas），再使用pdf2svg轉成svg
* 樣式
    * 格式：
        * svg
        * png
            * export resolution: 144
    * 大小：
        * 橫向：不超過400畫素
        * 縱向：不超過200畫素
    * 粗細：
        * StickArrow
        * 1pt
		* arrow head size: 50%
    * 字型：
        * 英文：STIXGeneral, 9pt（下標和上標：6pt）
        * 中文：PingFang SC, 9pt
	* 下標和上標中的數字和括號不要斜體
    * 顏色：
        * 非填充深藍色（與黑相近）：
            * 5B7DAA
        * 填充藍色（與黑對比）
            * 深：66BFFF
            * 淡：B2D9FF
* 著作權
    * 不使用網路圖片
* 位置
    * 兩張圖不可以較鄰近
        * 兩張圖拼一下
* 引用
    * 手動引用（例如，圖7.1）
* matplotlib
    * 大小
    * 解析度

## 程式碼

* 使用utils.py封裝多次使用函式
    * 首次出現函式，書裡給出函式實現
* Python規範一致
    * PEP8
        * 二元運算子換行：運算子和後一元一起換行 (https://www.python.org/dev/peps/pep-0008/#should-a-line-break-before-or-after-a-binary-operator)
* 將相鄰賦值語句儘可能合併為同一行
	* 如 num_epochs, lr = 5, 0.1
* 變數名一致
    * num_epochs
        * 迭代週期
    * num_hiddens
        * 隱藏單元個數
    * num_inputs
        * 輸入個數
    * num_outputs
        * 輸出個數
    * net
        * 模型
    * lr
        * 學習率
    * acc
        * 準確率
    * 迭代中
        * 特徵：X
        * 標籤：y, y_hat 或 Y, Y_hat
        * for X, y in data_iter
    * 資料集：
        * 特徵：features或images
        * 標籤：labels
        * DataLoader實例：train_iter, test_iter, data_iter
* 註解
    * 中文
    * 句末不加句號
* 書本頁寬限制
    * 每行不超過78字元
        * In [X]: 79字元不會自動換行（X = 1, ..., 9）
    	* In [XX]: 78字元不會自動換行（XX = 10, 11, ..., 99）
    * 列印結果自動換行
* imports
    * import alphabetically
    * from mxnet.gluon import data as gdata, loss as gloss, nn, utils as gutils
* 列印名稱
    * epoch（從1開始計數）, lr, loss, train acc, time
    * 5行左右
* 列印變數
    * 程式碼區塊最後一行儘量不用print()語句，例如`x, y`而不是`print('x:', x, 'y:', y)`
* 字串
    * 使用單引號
* 其他
    * nd.f(x) → x.nd
    * random_normal → random.normal
    * multiple imports
    * .1 → 1.0
    * 1. → 1.0
    * remove namescope

## 超連結

* 內鏈格式
    * [『線性迴歸』](linear-reg.md)一節
* 外鏈
    * [層](http:bla)
    * 無須暴露URL

## 英翻漢的常見問題

* 遇到不確定的地方，可以翻閱中文版第一版的處理方法（即我們需要遵照的出版標準），以及查閱人工翻譯 http://www.jukuu.com/
* 建立中英文術語對照表，全書術語翻譯要完全一致。
* 語法要正確（如不能缺主語、謂語）、句子要通順（硬翻不妥就意譯）、不要漏內容。
* 程式碼註解要翻譯。注意：i) 每行不要超過78字元，註解末尾不用加句號。 ii) # 後要空一個半形字元（英文空格）。iii) 如果註解與程式碼同行，# 前要空兩個半形字元（英文空格）。iv）保留註解中的``符號（為了表示程式碼部分，如變數名、函式名等）。v）註解中中文和英文之間不要空格。vi）貪婪換行：只有當一行註解抵到78字元時再換行。
* 不要新加空行（這樣會另起一個自然段）。
* 術語要保留英文翻譯。現在很多地方漏了英文翻譯。格式：*術語*（terminology）
* 正文和程式碼註解均使用中文標點。例如，中文括號要用全形括號（），不是英文半形()。例外：所有表示形狀的括號和逗號（逗號後緊跟半形英文空格）用英文半形，例如『(批次大小, 詞數)』而不是『（批次大小，詞數）』
* 英文在標題裡或句首全部不要首字母大寫（即便在標題的第一個詞）。除非本身是首字母大寫的術語
* 不要客氣。『您』->『你』，去掉『請』

