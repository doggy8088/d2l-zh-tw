# 為本書做貢獻
:label:`sec_how_to_contribute`

讀者們的投稿大大幫助我們改進了本書的質量。
如果你發現筆誤、無效的連結、一些你認為我們遺漏了引文的地方，
程式碼看起來不優雅，或者解釋不清楚的地方，請回復我們以幫助讀者。
在常規書籍中，兩次印刷之間的間隔（即修訂筆誤的間隔）常常需要幾年，
但這本書的改進通常需要幾小時到幾天的時間。
由於版本控制和持續自動整合（CI）測試，這一切頗為高效。
為此，你需要向gihub儲存庫送出一個
[pull request](https://github.com/d2l-ai/d2l-en/pulls)。
當你的pull請求被作者合併到程式碼庫中時，
你將成為[貢獻者](https://github.com/d2l-ai/d2l-en/graphs/contributors)。

## 送出微小更改

最常見的貢獻是編輯一句話或修正筆誤。
我們建議你在[GitHub儲存庫](https://github.com/d2l-ai/d2l-en)
中查詢源檔案，以定位源檔案（一個markdown檔案）。
然後單擊右上角的『Edit this file』按鈕，在markdown檔案中進行更改。

![在Github上編輯檔案](../img/edit-file.png)
:width:`300px`
:label:`fig_edit_file`

完成後，在頁面底部的『Propose file change』（『送出檔案修改』）
面板中填寫更改說明，然後單擊『Propose file change』按鈕。
它會重導向到新頁面以檢視你的更改（ :numref:`fig_git_createpr`）。
如果一切正常，你可以透過點選『Create pull request』按鈕送出pull請求。

## 大量文字或程式碼修改

如果你計劃修改大量文字或程式碼，那麼你需要更多地瞭解本書使用的格式。
源檔案基於[markdown格式](https://daringfireball.net/projects/markdown/syntax)，
並透過[d2lbook](http://book.d2l.ai/user/markdown.html)套件提供了一組擴充套件，
例如引用公式、影象、章節和引文。
你可以使用任何markdown編輯器開啟這些檔案並進行更改。

如果你想要更改程式碼，我們建議你使用Jupyter Notebook開啟這些標記檔案，
如 :numref:`sec_jupyter`中所述。
這樣你就可以執行並測試你的更改。
請記住在送出更改之前清除所有輸出，我們的CI系統將執行你更新的部分以生成輸出。

某些部分可能支援多個框架實現。如果你新增的新程式碼區塊不是使用mxnet，
請使用`#@tab`來標記程式碼區塊的起始行。
例如`#@tab pytorch`用於一個PyTorch程式碼區塊，
`#@tab tensorflow`用於一個TensorFlow程式碼區塊，
`#@tab paddle`用於一個PaddlePaddle程式碼區塊，
或者`#@tab all`是所有實現的共享程式碼區塊。
你可以參考[d2lbook](http://book.d2l.ai/user/code_tabs.html)套件瞭解更多訊息。

## 送出主要更改

我們建議你使用標準的Git流程送出大量修改。
簡而言之，該過程的工作方式如 :numref:`fig_contribute`中所述。

![為這本書作貢獻](../img/contribute.svg)
:label:`fig_contribute`

我們將向你詳細介紹這些步驟。
如果你已經熟悉Git，可以跳過本部分。
在介紹時，我們假設貢獻者的使用者名為『astonzhang』。

### 安裝Git

Git開源書籍描述了[如何安裝git](https://git-scm.com/book/en/v2)。
這通常透過Ubuntu Linux上的`apt install git`，
在MacOS上安裝Xcode開發人員工具或使用gihub的
[桌面用戶端](https://desktop.github.com)來實現。
如果你沒有GitHub帳戶，則需要註冊一個帳戶。

### 登入GitHub

在瀏覽器中輸入本書程式碼儲存庫的[地址](https://github.com/d2l-ai/d2l-en/)。
單擊 :numref:`fig_git_fork`右上角紅色框中的`Fork`按鈕，以複製本書的儲存庫。
這將是你的副本，你可以隨心所欲地更改它。

![程式碼儲存庫頁面](../img/git-fork.png)
:width:`700px`
:label:`fig_git_fork`

現在，本書的程式碼庫將被分叉（即複製）到你的使用者名，
例如`astonzhang/d2l-en`顯示在 :numref:`fig_git_forked`的左上角。

![分叉程式碼儲存庫](../img/git-forked.png)
:width:`700px`
:label:`fig_git_forked`

### 克隆儲存庫

要克隆儲存庫（即製作本地副本），我們需要獲取其儲存庫地址。
點選 :numref:`fig_git_clone`中的綠色按鈕顯示此訊息。
如果你決定將此分支保留更長時間，請確保你的本地副本與主儲存庫保持最新。
現在，只需按照 :ref:`chap_installation`中的說明開始。
主要區別在於，你現在下載的是你自己的儲存庫分支。

![克隆儲存庫](../img/git-clone.png)
:width:`700px`
:label:`fig_git_clone`

```
# 將your_github_username替換為你的github使用者名
git clone https://github.com/your_github_username/d2l-en.git
```

### 編輯和推送

現在是編輯這本書的時候了。最好按照 :numref:`sec_jupyter`中的說明在Jupyter Notebook中編輯它。進行更改並檢查它們是否正常。假設我們已經修改了檔案`~/d2l-en/chapter_appendix_tools/how-to-contribute.md`中的一個拼寫錯誤。你可以檢查你更改了哪些檔案。

此時，Git將提示`chapter_appendix_tools/how-to-contribute.md`檔案已被修改。

```
mylaptop:d2l-en me$ git status
On branch master
Your branch is up-to-date with 'origin/master'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   chapter_appendix_tools/how-to-contribute.md
```

在確認這是你想要的之後，執行以下指令：

```
git add chapter_appendix_tools/how-to-contribute.md
git commit -m 'fix typo in git documentation'
git push
```

然後，更改後的程式碼將位於儲存庫的個人分支中。要請求新增更改，你必須為本書的官方儲存庫建立一個Pull請求。

### 送出Pull請求

如 :numref:`fig_git_newpr`所示，進入gihub上的儲存庫分支，選擇『New pull request』。這將開啟一個頁面，顯示你的編輯與本書主儲存庫中的當前內容之間的更改。

![新的Pull請求](../img/git-newpr.png)
:width:`700px`
:label:`fig_git_newpr`

最後，單擊按鈕送出Pull請求，如 :numref:`fig_git_createpr`所示。請務必描述你在Pull請求中所做的更改。這將使作者更容易審閱它，並將其與本書合併。根據更改的不同，這可能會立即被接受，也可能會被拒絕，或者更有可能的是，你會收到一些關於更改的反饋。一旦你把它們合併了，你就做完了。

![建立Pull請求](../img/git-createpr.png)
:width:`700px`
:label:`fig_git_createpr`

## 小結

* 你可以使用GitHub為本書做貢獻。
* 你可以直接在GitHub上編輯檔案以進行微小更改。
* 要進行重大更改，請分叉儲存庫，在本地編輯內容，並在準備好後再做出貢獻。
* 儘量不要送出巨大的Pull請求，因為這會使它們難以理解和合並。最好拆分為幾個小一點的。

## 練習

1. 啟動並分叉`d2l-ai/d2l-en`儲存庫。
1. 如果發現任何需要改進的地方（例如，缺少引用），請送出Pull請求。
1. 通常更好的做法是使用新分支建立Pull請求。學習如何用[Git分支](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell)來做這件事。

[Discussions](https://discuss.d2l.ai/t/5730)
