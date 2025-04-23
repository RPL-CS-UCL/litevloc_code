## LiteVloc: Map-Lite Visual Localization for Image-Goal Navigation
### Requirements
Create the workspace
```bash
mkdir -p catkin_ws/src/
cd catkin_ws/src/
```
Create conda environment (NVIDIA GeForce RTX 4090 and CUDA 11.8)
```bash
conda create --name litevloc python=3.8
conda activate litevloc
```
Install ```image-matching-methods```
```bash
git clone git@github.com:gogojjh/image-matching-models.git --recursive
cd image-matching-models && python -m pip install -e .
```
Install  ```VPR-evaluation-methods```
```bash
git clone git@github.com:gogojjh/VPR-methods-evaluation.git
```
Create conda environment (NVIDIA GeForce RTX 4090 and CUDA 11.8)
```bash
git clone https://github.com/RPL-CS-UCL/litevloc_code
conda install pytorch=2.0.1 torchvision=0.15.2 pytorch-cuda=11.8 numpy=1.24.3 -c pytorch -c nvidia # use the correct version of cuda for your system
pip install -r requirements.txt
```
Enter this code to check whether torch-related packages are installed
```bash
python test_torch_install.py
```
Build LiteVloc as the ROS package (optional)
```bash
catkin build litevloc -DPYTHON_EXECUTABLE=$(which python)
```


### We provide several usage of LiteVloc
1. [Instruction in Performing Map-free Benchmarking](doc/instruction_map_free_benchmark.md)
2. [Instruction in Running LiteVloc with Simulated Matterport3d Environment](doc/instruction_vnav_simu_matterport3d.md)
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
ln -s /usr/lib/aarch64-linux-gnu/libffi.so.7 /Rocket_ssd/miniconda3/envs/litevloc/lib/libffi.so.7
```
```bash
/Rocket_ssd/miniconda3/envs/litevloc/lib/libtiff.so.5
ln -s /usr/lib/x86_64-linux-gnu/libtiff.so.5 /Rocket_ssd/miniconda3/envs/litevloc/lib/libtiff.so.5
```
Issue: ```ImportError: /lib/aarch64-linux-gnu/libgomp.so.1: cannot allocate memory in static TLS block```
> Set this in the bash file: ```export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1```