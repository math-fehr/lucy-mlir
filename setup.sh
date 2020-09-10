#!/bin/bash
mkdir -p build && cd build
cmake -G Ninja .. -DMLIR_DIR=/home/fehr/prog/llvm-project/build/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=/home/fehr/prog/llvm-project/build/bin/llvm-lit
