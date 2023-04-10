forked from https://github.com/uf-robopi/UDepth
with some file changes

#### :construction: SETUP
```
git clone https://github.com/uf-robopi/UDepth.git
conda create -n test_git python=3.7
conda install pytorch=1.13.1 torchvision=0.14.1 torchaudio=0.13.1 cpuonly=2.0 -c pytorch
conda install matlablib=3.5.2 scipy=1.7.3 pandas=1.3.5
```
download https://drive.google.com/file/d/188sybU9VU5rW2BH2Yzhko4w-G5sPp6yG/view and place in UDepth/CPD/

#### :rocket: TESTING
run check.py located in root (UDepth) folder
```
python check.py
```
