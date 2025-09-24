# 🌀 HelixML Architecture

## Overview

HelixML is a high-performance ML framework designed for **SSM/Hyena architectures** with **topological memory** and **reversible computations**. The framework prioritizes:

- **FLOP efficiency**: 10-20× reduction vs transformers
- **Memory efficiency**: 5-10× reduction in DRAM usage  
- **Long context**: 256k+ tokens (targeting 1M)
- **Reversible compute**: 50-70% VRAM reduction

## Core Design Principles

### 1. **No Self-Attention**
- Replaced with SSM (Mamba/RWKV-style) and Hyena/LongConv
- FFT-based convolutions for long sequences
- State-space models for efficient recurrence

### 2. **Topological Memory**
- **M0**: Motifs (short patterns)
- **M1**: Cycles (medium-term dependencies)  
- **M2**: Stable cores (long-term knowledge)
- **U/I/S Links**: Temporal/Intermediate/Stable connections

### 3. **Reversible Computing**
- RevNet-style reversible blocks
- Gradient checkpointing for memory efficiency
- Backward pass reconstruction from forward

## Architecture Layers

```
┌─────────────────────────────────────────┐
│              Applications               │
│  (ssm_byte_lm, hyena_span_infilling)   │
├─────────────────────────────────────────┤
│              Examples                   │
│  (Training loops, inference servers)   │
├─────────────────────────────────────────┤
│              High-Level APIs            │
│  (nn/, optim/, data/, serve/)          │
├─────────────────────────────────────────┤
│              Core Framework             │
│  (tensor-core/, autograd/, backends/)  │
├─────────────────────────────────────────┤
│              Hardware Layer             │
│  (CPU BLAS, CUDA, WGPU)                │
└─────────────────────────────────────────┘
```

## Key Components

### Tensor Core (`tensor-core/`)
- Generic tensor trait with shape/dtype/device
- Zero-copy operations where possible
- Backend-agnostic API

### Backends (`backend-cpu/`, `backend-cuda/`)
- CPU: BLAS/Accelerate optimized
- CUDA: cudarc + cuBLAS with fused kernels
- Future: WGPU for cross-platform GPU

### Neural Networks (`nn/`)
- **SSMBlock**: State-space model layers
- **HyenaBlock**: FFT-based long convolutions  
- **MoE**: Mixture of Experts with topological routing
- **Reversible**: RevNet-style reversible layers

### Topological Memory (`topo-memory/`)
- **Motif Detection**: Pattern recognition in sequences
- **Cycle Analysis**: Dependency graph construction
- **Stability Formula**: S = f(R, E, C, Φ, S)
- **KV-free Retrieval**: Geometric/ANN-based search

### Data Pipeline (`data/`)
- **Byte Streams**: UTF-8 tokenization
- **RVQ/VQ-VAE**: Vector quantization
- **Polars Integration**: Fast data processing

## Performance Targets

| Metric | Target | Baseline |
|--------|--------|----------|
| FLOPs/KB | -10× | Transformer-tiny |
| DRAM/KB | -5× | Transformer-tiny |
| Context Length | 256k+ | 4k (typical) |
| VRAM Reduction | 50-70% | Standard training |
| Latency p95 | <80ms | A100/3090 |

## Development Roadmap

### v0.1 (Weeks 0-4)
- [x] Workspace setup
- [ ] Tensor core implementation
- [ ] CPU backend with BLAS
- [ ] Basic autograd
- [ ] Linear/RMSNorm/SiLU layers
- [ ] AdamW optimizer
- [ ] SSM byte LM example

### v0.2 (Weeks 5-8)  
- [ ] CUDA backend with fused kernels
- [ ] Hyena FFT implementation
- [ ] Quantization (int8/fp8)
- [ ] MoE routing
- [ ] HTTP/gRPC serving
- [ ] Performance benchmarks

### v0.3 (Weeks 9-12)
- [ ] Topological memory v1
- [ ] GPU-accelerated pattern analysis
- [ ] Reversible compute
- [ ] ANN retrieval (HNSW)
- [ ] 1M context support

### v0.4+ (R&D)
- [ ] Twistor pre-encoder
- [ ] CDT scheduler  
- [ ] E8 symmetry tying
- [ ] MERA hierarchical access
