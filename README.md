forked from https://github.com/uf-robopi/UDepth
with some file changes

#### :construction: SETUP
##### With python
###### Git
```
git clone https://github.com/uf-robopi/UDepth.git
```
###### Conda
Create environment
```
conda create -n ENV_NAME python=3.7
```
Activate environment
```
conda activate ENV_NAME
```
Install required packages
```
conda install scikit-image=0.19.3 opencv=4.6.0 matplotlib=3.5.2 scipy=1.7.3 pandas=1.3.5 pytorch=1.13.1 torchvision=0.14.1 torchaudio=0.13.1 cpuonly=2.0 -c pytorch -c conda-forge
```
###### File requirements
download https://drive.google.com/file/d/188sybU9VU5rW2BH2Yzhko4w-G5sPp6yG/view and place in UDepth/CPD/
##### With docker
```
docker pull wintersnezh/unn_udepth_test
```
#### :rocket: TESTING
##### With python
run check.py located in the root (UDepth) folder
```
python check.py
```
##### With docker
```
docker run wintersnezh/unn_udepth_test
```
