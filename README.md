# ZRID-Net
ZRID-Net: Zero-Reference Real-World Image  Dehazing Framework via Deep Self-Decoupling and  Reverse Knowledge Transfer (TCSVT'2025)

Authors: Shilong Wang, Wenqi Ren, Peng Gao, Jiguo Yu, Jianlei Liu

### Getting started

1. Clone this repo:
```bash
git clone https://github.com/wsl666/ZRID-Net.git
cd ZRID-Net/
```
2. Create a new conda environment and install dependencies:
```bash
conda create -n pytorch_1_13 python=3.9
conda activate pytorch_1_13
conda install pytorch==1.13.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

### Evaluation

1. Make sure the file structure is consistent with the following:
```bash
datasets/reals/*.png
```
2. Run the following script to evaluation the pre-trained model:

```bash
python test.py
```

### Contact
If you have any questions or suggestions, please feel free to contact me at cswangshilong@126.com.


