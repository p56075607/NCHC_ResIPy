# ERT時序監測資料處理系統

## Time-lapse 2D ERT Data Processing System

一個完整的二維地電阻監測資料處理系統，支援STG、OHM、URF等多種資料格式，提供從原始資料到批次處理的完整解決方案。

## 功能特色

### 🔄 資料處理功能

- **多格式支援**: STG、OHM、URF資料格式自動識別與轉換
- **雜訊濾除**: 自動移除負電阻率和高互換誤差資料
- **網格建立**: 支援三角形和方形網格，自動優化
- **反演計算**: 加權最小二乘法反演

### 📊 視覺化功能

- **偽剖面繪製**: 原始資料品質評估
- **反演結果**: 高品質電阻率剖面圖
- **誤差分析**: 反演誤差分佈與收斂曲線

## 快速開始

### 1. 環境需求

```bash
# 必要套件
pip install resipy numpy pandas matplotlib pyyaml pillow
```

### 2. 創建範例專案

```bash
# 創建範例專案結構
python run_ert_analysis.py --create-sample ./my_project

# 進入專案目錄
cd my_project

# 將您的STG資料檔案複製到sample_data目錄
cp /path/to/your/data/*.stg sample_data/
```

### 3. 執行分析

```bash
# 執行完整分析
python run_ert_analysis.py --config sample_config.yaml

# 顯示詳細執行過程
python run_ert_analysis.py --config sample_config.yaml --verbose
```

## 檔案結構

```
project/
├── run.bat                          # 執行檔
├── config.yaml                      # 配置檔案
├── src/                             # 程式碼路徑
│   ├── ert_time_series_processor.py
│   └── run_ert_analysis.py
├── input_data/                      # 輸入資料目錄
│   ├── survey_01.stg
│   ├── survey_02.stg
│   └── ...
└── output_files/                    # 輸出結果目錄
    ├── pseudo_plots/                # 偽剖面圖
    ├── mesh_plots/                  # 網格圖
    ├── result_plots/                # 反演結果圖
    ├── error_plots/                 # 誤差分析圖
    ├── convergence_plots/           # 收斂曲線
    └── numerical_data/              # 數值資料

```

## 配置檔案說明

### 資料設定

```yaml
data:
  input_dir: "dataset_stg_once"     # 輸入資料目錄
  output_dir: "output"              # 輸出結果目錄
  supported_formats: [".stg", ".ohm", ".urf"]  # 支援格式
  terrain_file: null                # 地形檔案（可選）
```

### 篩選參數

```yaml
filter:
  rho_min: 0.0                      # 最小視電阻率
  reciprocal_error_max: 20.0        # 最大互換誤差(%)
```

### 網格參數

```yaml
mesh:
  type: "trian"                     # 網格類型: "trian" 或 "quad"
  cl: 0.75                          # 特徵長度
  cl_factor: 5                      # 特徵長度因子
```

### 反演參數

```yaml
inversion:
  tolerance: 1                      # 收斂容差
  max_iterations: 10                # 最大迭代次數
  remove_outliers: true             # 移除離群值
  outlier_threshold: 0.05           # 離群值門檻(5%)
```

## 輸出結果說明

### 1. 偽剖面圖 (pseudo_plots/)

- `pseudo_section_01.png`: 第1次測量偽剖面
- `pseudo_section_02.png`: 第2次測量偽剖面
- `error_model.png`: 誤差模型擬合圖

### 2. 反演結果 (result_plots/)

- `resistivity_01.png`: 第1次測量反演結果
- `resistivity_02.png`: 第2次測量反演結果

### 3. 數值資料 (numerical_data/)

- `resistivity_data_01.csv`: 第1次測量數值資料
- `processing_summary.yaml`: 處理摘要


## 進階使用

### 自定義處理流程

```python
from ert_time_series_processor import ERTTimeSeriesProcessor

# 創建處理器
processor = ERTTimeSeriesProcessor('config.yaml')

# 分步執行
data_files, file_format = processor.load_data()
processor.create_surveys(data_files, file_format)
processor.filter_data()
processor.plot_pseudo_sections()
processor.create_mesh()
processor.run_inversion()
processor.plot_results()

```

### 批次處理多個專案

```bash
# 批次處理腳本
for project in project1 project2 project3; do
    echo "處理專案: $project"
    cd $project
    python ../run_ert_analysis.py --config config.yaml
    cd ..
done
```

## 故障排除

### 常見問題

1. **找不到資料檔案**

   - 檢查 `config.yaml`中的 `input_dir`路徑
   - 確認資料檔案格式正確
2. **反演不收斂**

   - 增加 `max_iterations`
   - 調整 `tolerance`參數
   - 檢查資料品質
3. **記憶體不足**

   - 減少 `cl_factor`數值（粗化網格）

### 日誌分析

處理過程中會自動生成 `processing.log`檔案，包含詳細的執行記錄：

```bash
# 查看最新日誌
tail -f processing.log

# 搜尋錯誤訊息
grep -i "error\|warning" processing.log
```

## 技術支援

如需技術支援或回報問題，請提供：

1. 配置檔案內容
2. 錯誤訊息或日誌檔案
3. 輸入資料格式與大小
4. 運行環境資訊

## 更新日誌

### v1.0.0 (2025)

- 初始版本發布
- 支援STG、OHM、URF格式
- 完整的ERT反演流程
- 時序分析與動畫功能
- 自動化批次處理
- 綜合報告生成

## 授權資訊

本系統基於開源套件開發，遵循相應授權條款。

---

*ERT時序監測資料處理系統 v1.0.0*