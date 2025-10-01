# 🌀 HelixML Architecture

## Overview

HelixML is a high-performance ML framework designed for **SSM/Hyena architectures** with **enhanced topological memory**, **adaptive scheduling**, and **synthetic data generation**. The framework prioritizes:

- **FLOP efficiency**: 10-20× reduction vs transformers
- **Memory efficiency**: 5-10× reduction in DRAM usage  
- **Long context**: 256k+ tokens (targeting 1M)
- **Multi-device orchestration**: CPU, CUDA adaptive scheduling
- **Synthetic data**: Multi-modal generation with verification
- **Enhanced memory**: Hierarchical processing with attention mechanisms

## Core Design Principles

### 1. **No Self-Attention**
- Replaced with SSM (Mamba/RWKV-style) and Hyena/LongConv
- FFT-based convolutions for long sequences
- State-space models for efficient recurrence

### 2. **Enhanced Topological Memory**
- **M0**: Motifs (short patterns) with hierarchical processing
- **M1**: Cycles (medium-term dependencies) with attention mechanisms
- **M2**: Stable cores (long-term knowledge) with adaptive consolidation
- **U/I/S Links**: Temporal/Intermediate/Stable connections with stability prediction
- **Enhanced Retrieval**: Multi-scale analysis and pattern synthesis
- **Phase Synchronization**: SSM core synchronization utilities

### 3. **Adaptive Multi-Device Orchestration**
- CPU, CUDA device management with automatic synchronization
- Load balancing with multiple strategies (Round Robin, Least Loaded, Adaptive)
- Resource monitoring and optimization
- Policy-based scheduling with energy, latency, and throughput considerations

### 4. **Synthetic Data Generation**
- Multi-modal generators (Sequences, Images, Graphs, Time Series, Text)
- Comprehensive verification and validation systems
- Pre-defined datasets for various ML tasks
- Performance benchmarking and optimization

### 5. **Reversible Computing**
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

### Enhanced Topological Memory (`topo-memory/`)
- **Motif Detection**: Pattern recognition with hierarchical processing
- **Cycle Analysis**: Dependency graph construction with attention mechanisms
- **Stability Formula**: S = f(R, E, C, Φ, S) with adaptive consolidation
- **Enhanced Retrieval**: Multi-scale analysis and pattern synthesis
- **Phase Synchronization**: SSM core synchronization utilities
- **Geometric Processing**: Twistor pre-encoder, E8 symmetry, MERA access

### Adaptive Scheduler (`adaptive-scheduler/`)
- **Multi-Device Orchestration**: CPU, CUDA device management
- **Load Balancing**: Multiple strategies with adaptive optimization
- **Resource Monitoring**: Memory, compute, bandwidth, storage tracking
- **Optimization Engine**: Genetic Algorithm, Simulated Annealing, Particle Swarm
- **Policy Management**: Resource, Load Balancing, Priority, Energy, Latency policies
- **Metrics Collection**: Comprehensive performance monitoring and analytics

### Synthetic Data Generation (`synthetic-data/`)
- **Multi-Modal Generators**: Sequences, Images, Graphs, Time Series, Text
- **Verification System**: Quality checks, statistical validation, cross-modal verification
- **Dataset Management**: Pre-defined datasets for various ML tasks
- **Benchmarking**: Performance testing and optimization

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
