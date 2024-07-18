# 編碼器-解碼器架構
:label:`sec_encoder-decoder`

正如我們在 :numref:`sec_machine_translation`中所討論的，
機器翻譯是序列轉換模型的一個核心問題，
其輸入和輸出都是長度可變的序列。
為了處理這種型別的輸入和輸出，
我們可以設計一個包含兩個主要元件的架構：
第一個元件是一個*編碼器*（encoder）：
它接受一個長度可變的序列作為輸入，
並將其轉換為具有固定形狀的編碼狀態。
第二個元件是*解碼器*（decoder）：
它將固定形狀的編碼狀態對映到長度可變的序列。
這被稱為*編碼器-解碼器*（encoder-decoder）架構，
如 :numref:`fig_encoder_decoder` 所示。

![編碼器-解碼器架構](../img/encoder-decoder.svg)
:label:`fig_encoder_decoder`

我們以英語到法語的機器翻譯為例：
給定一個英文的輸入序列：『They』『are』『watching』『.』。
首先，這種『編碼器－解碼器』架構將長度可變的輸入序列編碼成一個『狀態』，
然後對該狀態進行解碼，
一個詞元接著一個詞元地生成翻譯後的序列作為輸出：
『Ils』『regordent』『.』。
由於『編碼器－解碼器』架構是形成後續章節中不同序列轉換模型的基礎，
因此本節將把這個架構轉換為介面方便後面的程式碼實現。

## (**編碼器**)

在編碼器介面中，我們只指定長度可變的序列作為編碼器的輸入`X`。
任何繼承這個`Encoder`基底類別的模型將完成程式碼實現。

```{.python .input}
from mxnet.gluon import nn

#@save
class Encoder(nn.Block):
    """編碼器-解碼器架構的基本編碼器介面"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
#@tab pytorch
from torch import nn

#@save
class Encoder(nn.Module):
    """編碼器-解碼器架構的基本編碼器介面"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

#@save
class Encoder(tf.keras.layers.Layer):
    """編碼器-解碼器架構的基本編碼器介面"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def call(self, X, *args, **kwargs):
        raise NotImplementedError
```

```{.python .input}
#@tab paddle
import warnings
warnings.filterwarnings("ignore")
from paddle import nn

#@save
class Encoder(nn.Layer):
    """編碼器-解碼器架構的基本編碼器介面"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

## [**解碼器**]

在下面的解碼器介面中，我們新增一個`init_state`函式，
用於將編碼器的輸出（`enc_outputs`）轉換為編碼後的狀態。
注意，此步驟可能需要額外的輸入，例如：輸入序列的有效長度，
這在 :numref:`subsec_mt_data_loading`中進行瞭解釋。
為了逐個地生成長度可變的詞元序列，
解碼器在每個時間步都會將輸入
（例如：在前一時間步生成的詞元）和編碼後的狀態
對映成當前時間步的輸出詞元。

```{.python .input}
#@save
class Decoder(nn.Block):
    """編碼器-解碼器架構的基本解碼器介面"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
#@tab pytorch
#@save
class Decoder(nn.Module):
    """編碼器-解碼器架構的基本解碼器介面"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
#@tab tensorflow
#@save
class Decoder(tf.keras.layers.Layer):
    """編碼器-解碼器架構的基本解碼器介面"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def call(self, X, state, **kwargs):
        raise NotImplementedError
```

```{.python .input}
#@tab paddle
#@save
class Decoder(nn.Layer):
    """編碼器-解碼器架構的基本解碼器介面"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

## [**合併編碼器和解碼器**]

總而言之，『編碼器-解碼器』架構包含了一個編碼器和一個解碼器，
並且還擁有可選的額外的引數。
在前向傳播中，編碼器的輸出用於生成編碼狀態，
這個狀態又被解碼器作為其輸入的一部分。

```{.python .input}
#@save
class EncoderDecoder(nn.Block):
    """編碼器-解碼器架構的基底類別"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```

```{.python .input}
#@tab pytorch
#@save
class EncoderDecoder(nn.Module):
    """編碼器-解碼器架構的基底類別"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```

```{.python .input}
#@tab tensorflow
#@save
class EncoderDecoder(tf.keras.Model):
    """編碼器-解碼器架構的基底類別"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, enc_X, dec_X, *args, **kwargs):
        enc_outputs = self.encoder(enc_X, *args, **kwargs)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state, **kwargs)
```

```{.python .input}
#@tab paddle
#@save
class EncoderDecoder(nn.Layer):
    """編碼器-解碼器架構的基底類別"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```

『編碼器－解碼器』體系架構中的術語*狀態*
會啟發人們使用具有狀態的神經網路來實現該架構。
在下一節中，我們將學習如何應用迴圈神經網路，
來設計基於『編碼器－解碼器』架構的序列轉換模型。

## 小結

* 『編碼器－解碼器』架構可以將長度可變的序列作為輸入和輸出，因此適用於機器翻譯等序列轉換問題。
* 編碼器將長度可變的序列作為輸入，並將其轉換為具有固定形狀的編碼狀態。
* 解碼器將具有固定形狀的編碼狀態對映為長度可變的序列。

## 練習

1. 假設我們使用神經網路來實現『編碼器－解碼器』架構，那麼編碼器和解碼器必須是同一型別的神經網路嗎？
1. 除了機器翻譯，還有其它可以適用於』編碼器－解碼器『架構的應用嗎？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2780)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2779)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11837)
:end_tab: