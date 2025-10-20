# BitLadder
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

BitLadder is a high-performance, GPU-optimized system
designed to accelerate long-context LLMs decoding with a low-bit KV
cache. Achieve **3-9x speedup** than Flash Attention v2.
![overview](imgs/overview.png)
![scheme](imgs/scheme.png)

## Benchmark
* Kernel Performance in RTX4090
![overview](imgs/4090.png)
* Kernel Performance in A100
![overview](imgs/a100.png)

## Installation
```
git clone --recursive https://github.com/DD-DuDa/BitLadder.git
conda create -n bitladder python=3.10
conda activate bitladder
pip install -r requirements.txt
python setup.py install
```

## Quick Start
1. See benchmark/bench_single_decode.ipynb
2. (Optional) Play with libtorch c++      
    ```
    # download libtorch 

    cd BitDecoding/csrc/bit_decode
    mkdir build && cd build
    cmake -DCMAKE_PREFIX_PATH=<libtorch_path> ..
    make -j12
    ```
3. End2end inference example, please see [e2e](https://github.com/DD-DuDa/BitDecoding/tree/e2e)


## Acknowledgement
BitDecoding is inspired by many open-source libraries, including (but not limited to) [flash-attention](https://github.com/Dao-AILab/flash-attention/tree/main), [flute](https://github.com/HanGuo97/flute), [Atom](https://github.com/efeslab/Atom), [omniserve](https://github.com/mit-han-lab/omniserve), [KIVI](https://github.com/jy-yuan/KIVI).
