🐣 Please follow me for new updates https://twitter.com/camenduru <br />
🔥 Please join our discord server https://discord.gg/k5BwmmvJJU <br />
🥳 Please join my patreon community https://patreon.com/camenduru <br />

## 🦒 Colab

| Colab | Info
| --- | --- |
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/Qwen-VL-Chat-colab/blob/main/Qwen_VL_Chat_colab.ipynb) | Qwen_VL_Chat_colab (4bit)

## Main Repo
https://github.com/QwenLM/Qwen-VL <br />
https://modelscope.cn/studios/qwen/Qwen-VL-Chat-Demo/files <br />

## Paper
https://arxiv.org/abs/2308.12966

## Output
![Screenshot 2023-09-07 074525](https://github.com/camenduru/Qwen-VL-Chat-colab/assets/54370274/a2a09ecb-15b6-4035-914e-aaee2cf608e8)

## 🚀 Local Installation Guide

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (NVIDIA recommended)
  - For NVIDIA GPUs: CUDA 11.8 or compatible drivers
- At least 8GB RAM (16GB recommended)
- At least 10GB free disk space for the model

### Step 1: Clone the Repository
```bash
git clone https://github.com/camenduru/Qwen-VL-Chat-colab.git
cd Qwen-VL-Chat-colab
```

### Step 2: Set Up Python Environment (Recommended)
```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install PyTorch with CUDA Support
For NVIDIA GPUs (recommended):
```bash
# Install PyTorch with CUDA 11.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only installation (not recommended for performance):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 4: Install Other Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Launching the Application

### Basic Usage
```bash
python app.py
```
The application will be available at `http://127.0.0.1:8000`

### Advanced Options

#### Run on CPU Only
```bash
python app.py --cpu-only
```

#### Make Accessible on Local Network
```bash
python app.py --server-name 0.0.0.0
```

#### Auto-Open in Browser
```bash
python app.py --inbrowser
```

#### Create Public Shareable Link (using ngrok)
```bash
python app.py --share
```

## 🛠️ Troubleshooting

### CUDA Compatibility Issues
If you encounter CUDA errors, try:
1. Verify CUDA installation:
   ```bash
   nvidia-smi
   nvcc --version
   ```
2. Update your conda environment:
   ```bash
   conda update --all
   conda clean --all
   ```
3. Reinstall PyTorch with specific CUDA version if needed

### Out of Memory Errors
- Try running with `--cpu-only` flag
- Clear conda cache: `conda clean --all`
- Reduce batch size in the application settings
- Close other memory-intensive applications

### Conda Environment Issues
- If you encounter dependency conflicts, create a fresh environment:
  ```bash
  conda deactivate
  conda env remove -n qwen_vl
  # Then start over from Step 2
  ```

## 📦 System Requirements

### Minimum
- 8GB RAM
- 10GB free disk space
- Any CPU (slower performance)

### Recommended
- NVIDIA GPU with 16GB+ VRAM (e.g., RTX 3090, A6000)
- 16GB+ RAM
- 20GB+ free disk space
- CUDA 11.8

### First Run Notes
- The first launch will download the model (several GB)
- This may take time depending on your internet connection
- Subsequent launches will be faster as the model is cached locally
