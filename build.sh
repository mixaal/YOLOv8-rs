#!/bin/bash -xe
#

export CONDA_ENV_NAME=libtorch

conda --help || {
  echo "Please install conda(1) first"
  exit 1
}

[ -f "models/yolov8n.torchscript" ] || {
  cd models 
  ./yolo.sh 
  cd -
}

#export LIBTORCH=$(pwd)/libtorch/
#export LIBTORCH_INCLUDE=$(pwd)/libtorch/
#export LIBTORCH_LIB=$(pwd)/libtorch/

#export LD_LIBRARY_PATH="$LIBTORCH/lib/:$LD_LIBRARY_PATH"


echo ""
echo "[⚡] Activating environment..."
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}" || {
  echo "Please run create-env.sh first to get all necessary components in place..."
  exit 1
}

export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=true

cargo test
cargo build --release --examples


PYTORCH_PATH=$(python -c "import torch; print(torch.__path__[0])")
export LD_LIBRARY_PATH=$PYTORCH_PATH/lib:$LD_LIBRARY_PATH

target/release/examples/yolo-predict


