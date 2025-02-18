## LiteVLoc
### Requirements

Install ```image-matching-methods``` and ```VPR-evaluation-methods```
```bash
git clone git@github.com:gogojjh/VPR-methods-evaluation.git
git clone git@github.com:gogojjh/image-matching-models.git --recursive
cd image-matching-models && python -m pip install -e .
```
Install ```pycpptools```
```bash
git clone git@github.com:gogojjh/pycpptools.git
cd pycpptools && python -m pip install -e .
```
Create conda environment (NVIDIA GeForce RTX 4090 and CUDA 11.8)
```bash
conda env create -f environment.yaml
conda install pytorch=2.0.1 torchvision=0.15.2 pytorch-cuda=11.8 numpy=1.24.3 -c pytorch -c nvidia # use the correct version of cuda for your system
conda activate litevloc && pip install -r requirements.txt
```
Set this in the bash: 
```bash
export PYTHONPATH=$PYTHONPATH:~/robohike_ws/src
export TORCH_HOME=path_torch_hub
```
Build the ros package:
```bash
catkin build litevloc -DPYTHON_EXECUTABLE=$(which python)
```

### Instruction of Usage
1. [Instruction in Performing Map-free Benchmarking](doc/instruction_map_free_benchmark.md)
2. [Instruction in Running Visual Navigation with Simulated Matterport3d Environment](doc/instruction_vnav_simu_matterport3d.md)
<!-- 3. [Instruction in Performing Map Merging](doc/instruction_map_merging.md) -->

### Issues
Issue: ```cannot import name 'cache' from 'functools'```
> Replace the original code with [Link](https://stackoverflow.com/questions/66846743/importerror-cannot-import-name-cache-from-functools)
```bash
from functools import lru_cache
@lru_cache(maxsize=None)
    def xxx
```
Issue: ```/lib/aarch64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0``` using cv_bridge
> Change the ```.so```. Complete tutorial is shown [here](https://blog.csdn.net/qq_38606680/article/details/129118491)
```bash
rm /Rocket_ssd/miniconda3/envs/litevloc/lib/libffi.so.7
sudo ln -s /usr/lib/aarch64-linux-gnu/libffi.so.7 /Rocket_ssd/miniconda3/envs/litevloc/lib/libffi.so.7
```
Issue: ```ImportError: /lib/aarch64-linux-gnu/libgomp.so.1: cannot allocate memory in static TLS block```
> Set this in the bash file: ```export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1```