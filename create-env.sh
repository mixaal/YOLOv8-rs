#!/bin/bash -e

CONDA_ENV_NAME=${1:-libtorch}


conda --help &> /dev/null || {
	echo "[✘] Install conda(1) manager first ..."
        exit 1
}

[ -d "libtorch" ] || {
  echo "[💾] Downloading libtorch ..."
  wget https://download.pytorch.org/libtorch/cu130/libtorch-shared-with-deps-2.9.1%2Bcu130.zip
  unzip libtorch-shared*2.9.1*zip
  rm -f libtorch-shared*2.9.1*zip
}



PYTHON_VERSION="3.10"
CUDA_VERSION="13.0"
GPP_VERSION="11"
GCC_VERSION="11"

export CC="gcc-$GCC_VERSION"
export CXX="g++-$GCC_VERSION"

echo "=================================================="
echo "Libtorch Installation Script"
echo "=================================================="
echo "Conda environment: ${CONDA_ENV_NAME}"
echo "Python version: ${PYTHON_VERSION}"
echo "CUDA version: ${CUDA_VERSION}"
echo "g++ compiler: ${CXX}"
echo "gcc compiler: ${CC}"
echo "=================================================="

# Check for required compilers
echo "[🔍] Checking for required compilers..."
if ! command -v $CC &> /dev/null || ! command -v $CXX &> /dev/null; then
    echo "❌ ERROR: $CC and $CXX are required but not found!"
    echo "Install with: sudo apt install $CC $CXX"
    exit 1
fi

echo "✓ $CC found: $($CC --version | head -n1)"
echo "✓ $CXX found: $($CXX --version | head -n1)"

# Check if conda environment already exists
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo ""
    echo "[⚠️ ] WARNING: Conda environment '${CONDA_ENV_NAME}' already exists!"
    read -p "[💬] Do you want to remove it and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "[🗑]Removing existing environment..."
        conda deactivate 2>/dev/null || true
        conda env remove -n "${CONDA_ENV_NAME}" -y
    else
        echo "❌ Exiting. Please use a different environment name."
        exit 1
    fi
fi

# Create fresh conda environment
echo ""
echo "Creating conda environment: ${CONDA_ENV_NAME}..."
conda create -n "${CONDA_ENV_NAME}" python="${PYTHON_VERSION}" -y

# Activate environment
echo ""
echo "[⚡] Activating environment..."
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"


echo "[📦] Installing PyTorch with CUDA ${CUDA_VERSION}..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//.}

# Verify PyTorch CUDA installation
echo "[🔍] Verifying PyTorch CUDA installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo ""
    echo "❌ ERROR: CUDA is not available in PyTorch!"
    echo "Please check your NVIDIA driver installation."
    exit 1
fi


echo "[📦] Installing nvidia/label/cuda-${CUDA_VERSION}.0 cuda-toolkit ..."
conda install -c "nvidia/label/cuda-${CUDA_VERSION}.0" cuda-toolkit
nvcc --version && echo "[✓] cuda-toolkit installed"


echo "[☑️] Environment installed."


#conda activate $ENV_NAME || {
#	echo "[✘] Environment $ENV_NAME does not exists yet ..."
#	echo "[📦] Creating $ENV_NAME ..."
#}


