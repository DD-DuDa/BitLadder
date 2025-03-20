# BitDecoding

## Quick Start
2. Run with libtorch c++
    ```
    cd libs/
    wget https://download.pytorch.org/libtorch /cu124/libtorch-shared-with-deps-2.5.1%2Bcu124.zip
    unzip libtorch-shared-with-deps-2.5.1+cu124.zip
    rm libtorch-shared-with-deps-2.5.1+cu124.zip

    cd BitDecoding/csrc/bit_decode
    mkdir build && cd build
    cmake -DCMAKE_PREFIX_PATH=<libtorch_path> ..
    make -j12
    ```