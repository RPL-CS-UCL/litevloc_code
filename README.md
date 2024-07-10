### Installation

1. Download ```image-matching-methods``` and ```VPR-evaluation-methods```
```
git clone git@github.com:gogojjh/VPR-methods-evaluation.git
git clone git@github.com:gogojjh/image-matching-models.git --recursive
cd image-matching-models && python -m pip install -e .
```

2. Download ```pycpptools```
```
git clone git@github.com:gogojjh/pycpptools.git
cd pycpptools && python -m pip install -e .
```

3. Create conda environment
```
conda env create -f environment.yaml
conda activate topo_loc && pip install -r requirements.txt
```

4. (Not used) Check CUDA version and install correct torch [link](https://pytorch.org/get-started/previous-versions/) nad [link](https://pytorch.org/get-started/locally/)
```
nvcc -V
# CUDA 10.2
pip install torch==1.9.1+cu102 torchvision==0.10.1+cu102 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# CUDA 11.3
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

<!-- 1. ```export PYTHONPATH=$PYTHONPATH:~/robohike_ws/src``` -->