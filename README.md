# FasterWhisperLocal

在 Windows 上用 Faster-Whisper `large-v3` 轉錄本機影音，支援單檔、資料夾批次及 TXT、SRT、JSON 輸出。預設使用 NVIDIA CUDA、`float16`、語音活動偵測（VAD）、自動語言辨識及台灣繁體轉換。也可以只用 CPU。

## 安裝

準備 Windows 和 Conda，在已啟用 Conda 的 PowerShell 中切換到專案根目錄。首次安裝需要網路。

```powershell
conda env create -f environment.yml
conda activate faster-whisper
.\download_model.ps1
```

之後開啟新的 PowerShell 時，先執行 `conda activate faster-whisper`。模型約 3.09 GB，下載位置是 `models/faster-whisper-large-v3/`。若要手動下載，請使用 [Systran 的 CTranslate2 模型](https://huggingface.co/Systran/faster-whisper-large-v3)，將模型檔直接放進該目錄。至少需要 `config.json`、`model.bin`、`tokenizer.json` 和 `preprocessor_config.json`。

### NVIDIA GPU 設定

安裝支援 CUDA 12 的 NVIDIA 驅動程式，從官方來源取得 CUDA 12 的 cuBLAS 和對應 CUDA 12 的 cuDNN 9。將套件提供的 `cublas*.dll` 和 `cudnn*.dll` 放入專案的 `runtime/cuda12/`。 [NVIDIA CUDA Toolkit 下載](https://developer.nvidia.com/cuda-toolkit-archive) · [NVIDIA cuDNN Windows 安裝](https://docs.nvidia.com/cudnn/latest/installation/backend/windows/)

```powershell
.\verify_install.ps1
```

診斷會列出套件版本、CUDA 裝置數量、找到的 DLL 數量，以及本機模型目錄是否存在。CPU 模式可略過這節。

## 轉錄

將範例中的路徑換成自己的影音檔或資料夾。

```powershell
# CPU 單檔
.\run.ps1 ".\example.mp4" --device cpu --compute-type int8

# NVIDIA GPU 單檔，預設 CUDA float16
.\run.ps1 ".\example.mp4"

# NVIDIA GPU 批次轉錄，產生三種格式
.\run.ps1 ".\input" --format txt srt json
```

預設將帶時間戳的 TXT 寫到 `output/`。批次處理會保留相對目錄，遇到同名而副檔名不同的來源會將來源副檔名保留在輸出檔名。可以用 `--output` 指定其他目錄。已有輸出的來源預設跳過，確定要取代時可加 `--overwrite`。如 GPU 顯存不足，可改用 `--compute-type int8_float16`。完整選項見 `.\run.ps1 --help`。

## 資料與隱私

模型下載會連線到 Hugging Face。安裝並下載完成後，轉錄所需的影音與模型可在本機離線處理。JSON 會保存逐字稿與輸入路徑，終端訊息也可能顯示檔案路徑。分享逐字稿、JSON 或日誌前請先確認內容，使用自訂輸出目錄時也要確認 Git 待提交清單。

模型、CUDA DLL 和預設輸出目錄的產物由 Git 忽略。CUDA 模式會載入 `runtime/cuda12/` 內的 DLL，請使用 NVIDIA 官方來源的檔案。

## 測試

```powershell
python -m pytest --ignore=tests/test_powershell_scripts.py
```

已完成 NVIDIA GPU 和模型設定時，可用 `python -m pytest` 一併執行 PowerShell 安裝診斷測試。
