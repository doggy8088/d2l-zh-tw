# 選擇伺服器和GPU
:label:`sec_buy_gpu`

深度學習訓練通常需要大量的計算。目前，GPU是深度學習最具成本效益的硬體加速器。與CPU相比，GPU更便宜，效能更高，通常超過一個數量級。此外，一台伺服器可以支援多個GPU，高端伺服器最多支援8個GPU。更典型的數字是工程工作站最多4個GPU，這是因為熱量、冷卻和電源需求會迅速增加，超出辦公樓所能支援的範圍。對於更大的部署，雲端運算（例如亞馬遜的[P3](https://aws.amazon.com/ec2/instance-types/p3/)和[G4](https://aws.amazon.com/blogs/aws/in-the-works-ec2-instances-g4-with-nvidia-t4-gpus/)實例）是一個更實用的解決方案。

## 選擇伺服器

通常不需要購買具有多個執行緒的高端CPU，因為大部分計算都發生在GPU上。這就是說，由於Python中的全域直譯器鎖（GIL），CPU的單線程效能在有4-8個GPU的情況下可能很重要。所有的條件都是一樣的，這意味著核數較少但時鐘頻率較高的CPU可能是更經濟的選擇。例如，當在6核4GHz和8核3.5GHz CPU之間進行選擇時，前者更可取，即使其聚合速度較低。一個重要的考慮因素是，GPU使用大量的電能，從而釋放大量的熱量。這需要非常好的冷卻和足夠大的機箱來容納GPU。如有可能，請遵循以下指南：

1. **電源**。GPU使用大量的電源。每個裝置預計高達350W（檢查顯卡的*峰值需求*而不是一般需求，因為高效程式碼可能會消耗大量能源）。如果電源不能滿足需求，系統會變得不穩定。
1. **機箱尺寸**。GPU很大，輔助電源連線器通常需要額外的空間。此外，大型機箱更容易冷卻。
1. **GPU散熱**。如果有大量的GPU，可能需要投資水冷。此外，即使風扇較少，也應以『公版設計』為目標，因為它們足夠薄，可以在裝置之間進氣。當使用多風扇GPU，安裝多個GPU時，它可能太厚而無法獲得足夠的空氣。
1. **PCIe插槽**。在GPU之間來回移動資料（以及在GPU之間交換資料）需要大量頻寬。建議使用16通道的PCIe 3.0插槽。當安裝了多個GPU時，請務必仔細閱讀主板說明，以確保在同時使用多個GPU時16$\times$頻寬仍然可用，並且使用的是PCIe3.0，而不是用於附加插槽的PCIe2.0。在安裝多個GPU的情況下，一些主板的頻寬降級到8$\times$甚至4$\times$。這部分是由於CPU提供的PCIe通道數量限制。

簡而言之，以下是建立深度學習伺服器的一些建議。

* **初學者**。購買低功耗的低端GPU（適合深度學習的廉價遊戲GPU，功耗150-200W）。如果幸運的話，大家現在常用的電腦將支援它。
* **1個GPU**。一個4核的低端CPU就足夠了，大多數主板也足夠了。以至少32 GB的DRAM為目標，投資SSD進行本地資料訪問。600W的電源應足夠。買一個有很多風扇的GPU。
* **2個GPU**。一個4-6核的低端CPU就足夠了。可以考慮64 GB的DRAM並投資於SSD。兩個高端GPU將需要1000瓦的功率。對於主板，請確保它們具有*兩個*PCIe 3.0 x16插槽。如果可以，請使用PCIe 3.0 x16插槽之間有兩個可用空間（60毫米間距）的主板，以提供額外的空氣。在這種情況下，購買兩個具有大量風扇的GPU。
* **4個GPU**。確保購買的CPU具有相對較快的單線程速度（即較高的時鐘頻率）。可能需要具有更多PCIe通道的CPU，例如AMD Threadripper。可能需要相對昂貴的主板才能獲得4個PCIe 3.0 x16插槽，因為它們可能需要一個PLX來多路複用PCIe通道。購買帶有公版設計的GPU，這些GPU很窄，並且讓空氣進入GPU之間。需要一個1600-2000W的電源，而辦公室的插座可能不支援。此伺服器可能在執行時*聲音很大，很熱*。不想把它放在桌子下面。建議使用128 GB的DRAM。獲取一個用於本地儲存的SSD（1-2 TB NVMe）和RAID設定的硬碟來儲存資料。
* **8 GPU**。需要購買帶有多個冗餘電源的專用多GPU伺服器機箱（例如，每個電源為1600W時為2+1）。這將需要雙插槽伺服器CPU、256 GB ECC DRAM、快速網絡卡（建議使用10 GBE），並且需要檢查伺服器是否支援GPU的*物理外形*。使用者GPU和伺服器GPU之間的氣流和布線位置存在顯著差異（例如RTX 2080和Tesla V100）。這意味著可能無法在服務器中安裝消費級GPU，因為電源線間隙不足或缺少合適的接線（本書一位合著者痛苦地發現了這一點）。

## 選擇GPU

目前，AMD和NVIDIA是專用GPU的兩大主要製造商。NVIDIA是第一個進入深度學習領域的公司，透過CUDA為深度學習框架提供更好的支援。因此，大多數買家選擇NVIDIA GPU。

NVIDIA提供兩種型別的GPU，針對個人用戶（例如，透過GTX和RTX系列）和企業使用者（透過其Tesla系列）。這兩種型別的GPU提供了相當的計算能力。但是，企業使用者GPU通常使用強制（被動）冷卻、更多記憶體和ECC（糾錯）記憶體。這些GPU更適用於資料中心，通常成本是消費者GPU的十倍。

如果是一個擁有100個伺服器的大公司，則應該考慮英偉達Tesla系列，或者在雲中使用GPU伺服器。對於實驗室或10+伺服器的中小型公司，英偉達RTX系列可能是最具成本效益的，可以購買超微或華碩機箱的預設定伺服器，這些伺服器可以有效地容納4-8個GPU。

GPU供應商通常每一到兩年發布一代，例如2017年發布的GTX 1000（Pascal）系列和2019年發布的RTX 2000（Turing）系列。每個系列都提供幾種不同的型號，提供不同的效能級別。GPU效能主要是以下三個引數的組合：

1. **計算能力**。通常大家會追求32位浮點計算能力。16位浮點訓練（FP16）也進入主流。如果只對預測感興趣，還可以使用8位整數。最新一代圖靈GPU提供4-bit加速。不幸的是，目前訓練低精度網路的演算法還沒有普及；
1. **記憶體大小**。隨著模型變大或訓練期間使用的批次變大，將需要更多的GPU記憶體。檢查HBM2（高頻寬記憶體）與GDDR6（圖形DDR）記憶體。HBM2速度更快，但成本更高；
1. **記憶體頻寬**。當有足夠的記憶體頻寬時，才能最大限度地利用計算能力。如果使用GDDR6，請追求寬記憶體匯流排。

對於大多數使用者，只需看看計算能力就足夠了。請注意，許多GPU提供不同型別的加速。例如，NVIDIA的Tensor Cores將運算子子集的速度提高了5$\times$。確保所使用的函式庫支援這一點。GPU記憶體應不小於4GB（8GB更好）。儘量避免將GPU也用於顯示GUI（改用內建顯卡）。如果無法避免，請新增額外的2GB RAM以確保安全。

:numref:`fig_flopsvsprice`比較了各種GTX 900、GTX 1000和RTX 2000系列的（GFlops）和價格（Price）。價格是維基百科上的建議價格。

![浮點計算能力和價格比較](../img/flopsvsprice.svg)
:label:`fig_flopsvsprice`

由上圖，可以看出很多事情：

1. 在每個系列中，價格和效能大致成比例。Titan因擁有大GPU記憶體而有相當的溢價。然而，透過比較980 Ti和1080 Ti可以看出，較新型號具有更好的成本效益。RTX 2000系列的價格似乎沒有多大提高。然而，它們提供了更優秀的低精度效能（FP16、INT8和INT4）；
2. GTX 1000系列的性價比大約是900系列的兩倍；
3. 對於RTX 2000系列，浮點計算能力是價格的『仿射』函式。

![浮點計算能力和能耗](../img/wattvsprice.svg)
:label:`fig_wattvsprice`

:numref:`fig_wattvsprice`顯示了能耗與計算量基本成線性關係。其次，後一代更有效率。這似乎與對應於RTX 2000系列的圖表相矛盾。然而，這是TensorCore不成比例的大能耗的結果。

## 小結

* 在建立伺服器時，請注意電源、PCIe匯流排通道、CPU單線程速度和散熱。
* 如果可能，應該購買最新一代的GPU。
* 使用雲進行大型部署。
* 高密度伺服器可能不與所有GPU相容。在購買之前，請檢查一下機械和散熱規格。
* 為提高效率，請使用FP16或更低的精度。
