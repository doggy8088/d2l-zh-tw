## 編譯HTML版本

所有markdown檔案需要在送出前清除output，它們會在服務器上重新執行生成結果。所以需要保證每個notebook執行不要太久，目前限制是20min。

在本地可以如下build html（需要GPU支援）

```
conda env update -f build/env.yml
source activate d2l-zh-build
make html
```

生成的html會在`_build/html`。

如果沒有改動notebook裡面原始碼，所以不想執行notebook，可以使用

```
make html EVAL=0
```

但這樣生成的html將不含有輸出結果。

## 編譯PDF版本

編譯pdf版本需要xelatex、librsvg2-bin（svg圖片轉pdf）和思源字型。在Ubuntu可以這樣安裝。

```
sudo apt-get install texlive-full
sudo apt-get install librsvg2-bin
```

```
wget https://github.com/adobe-fonts/source-han-sans/releases/download/2.004R/SourceHanSansSC.zip
wget -O SourceHanSerifSC.zip https://github.com/adobe-fonts/source-han-serif/releases/download/2.001R/09_SourceHanSerifSC.zip

unzip SourceHanSansSC.zip -d SourceHanSansSC
unzip SourceHanSerifSC.zip -d SourceHanSerifSC

sudo mv SourceHanSansSC SourceHanSerifSC /usr/share/fonts/opentype/
sudo fc-cache -f -v
```


這時候可以透過 `fc-list :lang=zh` 來檢視安裝的中文字型。

同樣的去下載和安裝英文字型

```
wget -O source-serif-pro.zip https://www.fontsquirrel.com/fonts/download/source-serif-pro
unzip source-serif-pro -d source-serif-pro
sudo mv source-serif-pro /usr/share/fonts/opentype/

wget -O source-sans-pro.zip https://www.fontsquirrel.com/fonts/download/source-sans-pro
unzip source-sans-pro -d source-sans-pro
sudo mv source-sans-pro /usr/share/fonts/opentype/

wget -O source-code-pro.zip https://www.fontsquirrel.com/fonts/download/source-code-pro
unzip source-code-pro -d source-code-pro
sudo mv source-code-pro /usr/share/fonts/opentype/

sudo fc-cache -f -v
```

然後就可以編譯了。

```
make pdf
```

## 其他安裝

```
python -m spacy download en # 需已 pip install spacy
```

## 樣式規範

貢獻請遵照本教程的[樣式規範](STYLE_GUIDE.md)。

## 中英文術語對照

翻譯請參照[中英文術語對照](TERMINOLOGY.md)。
