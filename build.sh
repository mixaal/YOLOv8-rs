#!/bin/bash -xe
#

[ -f "models/yolov8n.torchscript" ] || {
  cd models 
  ./yolo.sh 
  cd -
}

[ -d "libtorch" ] || {
  wget 'https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcpu.zip' 
  unzip libtorch-cxx11*.zip
  rm -f libtorch-cxx11*.zip
}

export LIBTORCH=$(pwd)/libtorch/
export LIBTORCH_INCLUDE=$(pwd)/libtorch/
export LIBTORCH_LIB=$(pwd)/libtorch/

export LD_LIBRARY_PATH="$LIBTORCH/lib/:$LD_LIBRARY_PATH"

cargo test
cargo build --release --examples

target/release/examples/yolo-predict


