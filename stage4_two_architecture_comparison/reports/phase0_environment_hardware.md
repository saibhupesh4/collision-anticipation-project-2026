# Phase 0 Environment and Hardware Audit

**Date:** 2026-11-01

## Active project environment

The Python executable at `C:\Users\saibh\anaconda3\envs\collision_env\python.exe` ran successfully when invoked directly.

| Component | Observed version/status |
|---|---|
| Python | 3.10.22 |
| PyTorch | 2.11.0+cu128 |
| CUDA runtime reported by PyTorch | 12.8 |
| CUDA available | Yes |
| Ultralytics | 8.4.194 |
| OpenCV | 5.0.0 |
| XGBoost | 3.2.0 |
| scikit-learn | 1.7.2 |
| NumPy | 2.2.6 |
| Pandas | 2.3.3 |
| FilterPy | 2.0.0 |
| PyArrow | 25.0.1 |

The Conda executable and `conda.bat` could not start in this session (Windows DLL relocation / program launch errors). Direct environment Python execution works. No packages were installed or changed.

## Hardware and storage

- GPU: NVIDIA GeForce RTX 3050 A Laptop GPU.
- GPU memory: 4,292,345,856 bytes reported by PyTorch; 4,094 MiB reported by `nvidia-smi`.
- CPU: Intel family/model 154; 16 logical processors. System-management queries for the full CPU model were unavailable.
- RAM: 16,802,754,560 bytes total; approximately 4,172,922,880 bytes available at inspection time.
- C: drive free space: 291,611,488,256 bytes at inspection time.
- AWS CLI: 2.39.3.

The available GPU is consistent with the stated hardware constraint. Keep future perception runs sequential, use small batches, and monitor memory. No GPU workload was run during this audit.

## Access and reproducibility limitations

The S3 list request failed to connect to `https://s3.hf.co/sai-123456`; remote access remains unverified. This is an endpoint/network error, not evidence of invalid credentials. Credentials were neither displayed nor changed.

The package set is sufficient to proceed with Phase 1 and later CPU/GPU pipeline work once dataset timestamp semantics are established. The Conda launcher issue remains unresolved, but the existing environment's Python binary is usable directly.
