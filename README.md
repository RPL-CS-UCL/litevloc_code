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
And modify ```PYTHONPATH```

3. Create conda environment
```
conda env create -f environment.yaml
conda activate topo_loc && pip install -r requirements.txt
```

4. Install Pytorch and Torchvision on the Jetson platform
Check ```RPL-RoboHike/docs/tutorial_setup_jetson_orin.md``` for details. Need to use python3.8 to be compatiable with pre-complied torch.

5. Download ```Map-free-reloc```

6. Set this in the bash: 
    ```
    export PYTHONPATH=$PYTHONPATH:~/robohike_ws/src
    export TORCH_HOME=path_torch_hub
    ```

7. Build the repo:
    ```
    catkin build topo_loc -DPYTHON_EXECUTABLE=$(which python)
    ```

### Issues
1. ```cannot import name 'cache' from 'functools'```
Replace the original code with [Link](https://stackoverflow.com/questions/66846743/importerror-cannot-import-name-cache-from-functools)
    ```shell script
    from functools import lru_cache
    @lru_cache(maxsize=None)
      def xxx
    ```

2. ```/lib/aarch64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0``` using cv_bridge
Change the ```.so```. Complete tutorial is shown [here](https://blog.csdn.net/qq_38606680/article/details/129118491)
    ```shell script
    rm /Rocket_ssd/miniconda3/envs/topo_loc/lib/libffi.so.7
    sudo ln -s /usr/lib/aarch64-linux-gnu/libffi.so.7 /Rocket_ssd/miniconda3/envs/topo_loc/lib/libffi.so.7
    ```

3. ```ImportError: /lib/aarch64-linux-gnu/libgomp.so.1: cannot allocate memory in static TLS block```
Set this in the bash file: ```export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1```