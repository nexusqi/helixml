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

### 6. Multimodal Processing

```bash
cargo run -p multimodal_example
```

### 7. Synthetic Data Generation

```bash
cargo run -p synthetic_data_example
```

### 8. CUDA Backend (if CUDA is available)

```bash
cargo run -p cuda_example
cargo run -p cuda_backend_example
```

### 9. Advanced Autograd

```bash
cargo run -p advanced_autograd_example
cargo run -p complete_autograd_example
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
cargo test -p multimodal
cargo test -p meanings
cargo test -p geometry
cargo test -p data-pipeline
cargo test -p hal
```

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmarks
cargo bench -p tensor-core
cargo bench -p topo-memory
cargo bench -p meanings
cargo bench -p adaptive-scheduler
cargo bench -p synthetic-data
```

### Building Documentation

```bash
# Build documentation
cargo doc --open

# Build documentation for specific crate
cargo doc -p tensor-core --open
cargo doc -p topo-memory --open
cargo doc -p adaptive-scheduler --open
cargo doc -p multimodal --open
cargo doc -p meanings --open
```

## Project Structure

```
helixml/
├── crates/
│   ├── tensor-core/          # Core tensor operations
│   ├── hal/                  # Hardware Abstraction Layer
│   ├── backend-cpu/          # CPU backend
│   ├── backend-cuda/         # CUDA backend
│   ├── autograd/             # Automatic differentiation
│   ├── nn/                   # Neural network layers
│   ├── optim/                # Optimizers
│   ├── training/             # Training system
│   ├── topo-memory/          # Enhanced topological memory
│   ├── geometry/             # Geometric processing
│   ├── meanings/             # Semantic processing
│   ├── multimodal/           # Universal multimodal processing
│   ├── adaptive-scheduler/   # Multi-device scheduling
│   ├── synthetic-data/       # Synthetic data generation
│   ├── data-pipeline/        # Data loading and preprocessing
│   ├── scheduling/           # Advanced scheduling
│   └── serve/                # Model serving
├── examples/                 # Example applications
├── benches/                  # Performance benchmarks
├── docs/                     # Documentation
└── test_helix_project/       # Test project
```

## Key Features

### 🧠 Enhanced Topological Memory
- **M0 (Motifs)**: Short pattern detection with hierarchical processing
- **M1 (Cycles)**: Medium-term dependency analysis with attention mechanisms
- **M2 (Stable Cores)**: Long-term knowledge extraction with adaptive consolidation
- **Phase Synchronization**: SSM core synchronization utilities

### 🌐 Multimodal Processing
- **Universal Data Support**: Text, Images, Audio, Video, 3D Point Clouds
- **Auto-Modality Detection**: Automatic detection of data types and formats
- **Intelligent Processing**: Smart device selection and resource optimization
- **Cross-Modal Alignment**: Temporal, spatial, and semantic alignment

### 🎯 Intelligent Resource Management
- **Auto-Device Detection**: CPU, CUDA, OpenCL, Metal, Vulkan support
- **Smart Device Selection**: Optimal device selection based on workload
- **Resource Optimization**: 6 strategies for different optimization goals
- **Real-time Monitoring**: Performance metrics and alerting system

### 🎲 Synthetic Data Generation
- **Multi-Modal Generators**: Sequences, Images, Graphs, Time Series, Text
- **Verification System**: Quality checks and statistical validation
- **Dataset Management**: Pre-defined datasets for ML tasks
- **Benchmarking**: Performance testing and optimization

### 🧠 Meaning Induction Bootstrap
- **SIM/MIL Framework**: Semantic Induction and Meaning Induction Learning
- **U/I/S Links**: Temporal/Intermediate/Stable connections with stability prediction
- **Stability Analysis**: Advanced stability detection and analysis
- **Hierarchical Processing**: Multi-level semantic processing

### 🔧 Hardware Abstraction Layer
- **Universal Compute Interface**: Unified backend interface for all hardware
- **Device Management**: Automatic device detection and management
- **Resource Abstraction**: Hardware-agnostic resource management
- **Performance Optimization**: Hardware-specific optimizations

### 📊 Data Pipeline System
- **Async Data Loading**: High-performance async data loading and preprocessing
- **Universal Data Support**: Text, Images, Audio, Video, 3D Point Clouds
- **Auto-Modality Detection**: Automatic detection of data types and formats
- **Intelligent Processing**: Smart device selection and resource optimization

### 🎯 Advanced Scheduling
- **CDT Scheduler**: Causal Dependency Tree scheduling for advanced planning
- **Resource Optimization**: Multiple optimization strategies for different goals
- **Real-time Monitoring**: Performance metrics and alerting system
- **Auto-Adaptation**: Dynamic adaptation to workload changes

### 🚀 Model Serving
- **Production Deployment**: Production-ready model serving infrastructure
- **API Endpoints**: RESTful API for model inference
- **Load Balancing**: Intelligent request distribution
- **Monitoring**: Real-time performance and health monitoring

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
