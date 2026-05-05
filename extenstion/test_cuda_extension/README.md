A PyTorch Extension based on CUDA/CUTLASS

## Requirements

Install Python dependencies:
```
pip install -r requirements.txt
```

Install C++ header dependency (nlohmann/json, required by PyTorch functorch headers):
```
conda install -c conda-forge nlohmann_json
```

## Build

```
CURRENT_DIR=$(pwd)

# Clone cutlass
cd torch_cuda_extension/csrc/include && git clone https://github.com/NVIDIA/cutlass.git && cd $CURRENT_DIR

# Build the package
pip install --no-build-isolation -e .
```

### Verified
```
PyTorch 2.11.0+cu130
CUDA 13.0
```

## Usage

Run all tests via pytest:
```
pytest tests/test_extended_add.py -v
```

Or run directly:
```
python tests/test_extended_add.py
```

### Hardware Requirements

| Feature | Minimum |
|---|---|
| TMA (`test_extended_add_one_tma`) | Hopper (sm90+) |
| All other tests | Ampere (sm80+) |

Tests that require unsupported hardware are automatically skipped.
