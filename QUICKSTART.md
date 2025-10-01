# 🚀 HelixML Quick Start Guide

## Prerequisites

- Rust 1.70+ (install from [rustup.rs](https://rustup.rs/))
- CUDA toolkit (optional, for GPU acceleration)
- Git

## Installation

```bash
# Clone the repository
git clone https://github.com/nexusqi/helixml.git
cd helixml

# Build the project
cargo build --release
```

## Quick Examples

### 1. Basic Tensor Operations

```bash
cargo run -p simple_example
```

### 2. State-Space Model (SSM)

```bash
cargo run -p ssm_example
```

### 3. Hyena Architecture

```bash
cargo run -p hyena_example
```

### 4. Enhanced Topological Memory

```bash
cargo run -p enhanced_topo_memory_example
```

### 5. Adaptive Scheduler

```bash
cargo run -p adaptive_scheduler_example
```

### 6. Synthetic Data Generation

```bash
cargo run -p synthetic_data_example
```

### 7. CUDA Backend (if CUDA is available)

```bash
cargo run -p cuda_example
```

## Development

### Running Tests

```bash
# Run all tests
cargo test

# Run tests for specific crate
cargo test -p tensor-core
cargo test -p topo-memory
cargo test -p adaptive-scheduler
cargo test -p synthetic-data
```

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmarks
cargo bench -p tensor-core
cargo bench -p topo-memory
```

### Building Documentation

```bash
# Build documentation
cargo doc --open

# Build documentation for specific crate
cargo doc -p tensor-core --open
```

## Project Structure

```
helixml/
├── crates/
│   ├── tensor-core/          # Core tensor operations
│   ├── backend-cpu/          # CPU backend
│   ├── backend-cuda/         # CUDA backend
│   ├── autograd/             # Automatic differentiation
│   ├── nn/                   # Neural network layers
│   ├── optim/                # Optimizers
│   ├── training/             # Training system
│   ├── topo-memory/          # Enhanced topological memory
│   ├── adaptive-scheduler/   # Multi-device scheduling
│   └── synthetic-data/       # Synthetic data generation
├── examples/                 # Example applications
├── benches/                  # Performance benchmarks
└── docs/                     # Documentation
```

## Key Features

### 🧠 Enhanced Topological Memory
- **M0 (Motifs)**: Short pattern detection with hierarchical processing
- **M1 (Cycles)**: Medium-term dependency analysis with attention mechanisms
- **M2 (Stable Cores)**: Long-term knowledge extraction with adaptive consolidation
- **Phase Synchronization**: SSM core synchronization utilities

### 🎯 Adaptive Scheduling
- **Multi-Device Orchestration**: CPU, CUDA device management
- **Load Balancing**: Multiple strategies with adaptive optimization
- **Resource Monitoring**: Memory, compute, bandwidth tracking
- **Policy Management**: Flexible scheduling policies

### 🎲 Synthetic Data Generation
- **Multi-Modal Generators**: Sequences, Images, Graphs, Time Series, Text
- **Verification System**: Quality checks and statistical validation
- **Dataset Management**: Pre-defined datasets for ML tasks
- **Benchmarking**: Performance testing and optimization

## Performance

HelixML is designed for high performance:

- **FLOP Efficiency**: 10-20× reduction vs transformers
- **Memory Efficiency**: 5-10× reduction in DRAM usage
- **Long Context**: 256k+ tokens (targeting 1M)
- **Multi-Device**: Efficient CPU/CUDA orchestration

## Troubleshooting

### Common Issues

1. **CUDA not found**: Install CUDA toolkit or use CPU-only mode
2. **Memory issues**: Use gradient checkpointing for large models
3. **Compilation errors**: Ensure Rust 1.70+ is installed

### Getting Help

- Check the [documentation](docs/)
- Review [examples](examples/)
- Open an issue on GitHub

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

**HelixML** - Pushing the boundaries of machine learning with Rust 🦀
