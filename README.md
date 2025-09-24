# HelixML 🌀

**A high-performance Rust machine learning framework focused on post-transformer architectures**

[![Crates.io](https://img.shields.io/crates/v/helix-ml.svg)](https://crates.io/crates/helix-ml)
[![Documentation](https://docs.rs/helix-ml/badge.svg)](https://docs.rs/helix-ml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

### Core Components
- **Tensor Operations**: High-performance tensor operations with CPU backend
- **Automatic Differentiation**: Full autograd support with gradient checkpointing
- **Neural Networks**: Modern layers including RMSNorm, SiLU, GELU, Dropout
- **Optimizers**: AdamW, Lion, SGD with learning rate scheduling

### Post-Transformer Architectures
- **State-Space Models (SSM)**:
  - S4Block: Structured State Space Models
  - MambaBlock: Selective State Space Models
- **Hyena Blocks**: FFT-based long convolutions
- **HyenaOperator**: Advanced FFT convolution operations

### Experimental Components
- **Topological Memory System**:
  - M0 (Motifs): Short pattern detection
  - M1 (Cycles): Medium-term dependency analysis  
  - M2 (Stable Cores): Long-term knowledge extraction
  - U/I/S Links: Temporal/Intermediate/Stable connections
  - Stability Formula: S = f(R, E, C, Φ, S)

- **Geometric Processing**:
  - Twistor Pre-encoder: Geometric preprocessing
  - E8 Symmetry Tying: E8 group symmetries
  - MERA Hierarchical Access: Hierarchical memory access

- **Advanced Scheduling**:
  - CDT Scheduler: Causal Dynamical Triangulation
  - Operation optimization and parallelization

### Advanced Features
- **Mixed Precision**: FP16/INT8 support
- **Broadcasting**: Efficient tensor broadcasting
- **Gradient Checkpointing**: Memory-efficient training
- **Byte-level Language Modeling**: Token-free text processing

## 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
helix-ml = "0.1.0"
```

## 🎯 Quick Start

### Basic Usage

```rust
use helix_ml::*;

fn main() -> Result<()> {
    let device = Device::cpu();
    let input = CpuTensor::random_uniform(Shape::new(vec![10, 64]), -1.0, 1.0, &device)?;
    
    // Create a simple model
    let linear = Linear::<CpuTensor>::new(64, 32, &device)?;
    let activation = SiLU::<CpuTensor>::new(&device);
    
    // Forward pass
    let output = activation.forward(&linear.forward(&input)?)?;
    
    println!("Output shape: {:?}", output.shape());
    Ok(())
}
```

### SSM Model Example

```rust
use helix_ml::*;

fn main() -> Result<()> {
    let device = Device::cpu();
    let d_model = 64;
    let seq_len = 100;
    
    // Create SSM block
    let s4_block = S4Block::<CpuTensor>::new(d_model, 16, &device)?;
    
    // Create input sequence
    let input = CpuTensor::random_uniform(
        Shape::new(vec![seq_len, d_model]), 
        -1.0, 1.0, 
        &device
    )?;
    
    // Process through SSM
    let output = s4_block.forward(&input)?;
    
    println!("SSM output shape: {:?}", output.shape());
    Ok(())
}
```

### Hyena Model Example

```rust
use helix_ml::*;

fn main() -> Result<()> {
    let device = Device::cpu();
    let d_model = 128;
    let seq_len = 512;
    
    // Create Hyena block
    let hyena_block = HyenaBlock::<CpuTensor>::new(
        d_model, 
        256,  // max_length
        1024, // fft_size
        4,    // num_layers
        &device
    )?;
    
    // Create input sequence
    let input = CpuTensor::random_uniform(
        Shape::new(vec![seq_len, d_model]), 
        -1.0, 1.0, 
        &device
    )?;
    
    // Process through Hyena
    let output = hyena_block.forward(&input)?;
    
    println!("Hyena output shape: {:?}", output.shape());
    Ok(())
}
```

### Topological Memory Example

```rust
use helix_ml::*;

fn main() -> Result<()> {
    let device = Device::cpu();
    let d_model = 64;
    
    // Create topological memory system
    let mut topo_memory = TopologicalMemory::<CpuTensor>::new(
        d_model,
        5,    // max_motif_length
        0.7,  // cycle_detection_threshold
        0.8,  // stability_threshold
        &device
    )?;
    
    // Process sequence
    let sequence = CpuTensor::random_uniform(
        Shape::new(vec![50, d_model]), 
        -1.0, 1.0, 
        &device
    )?;
    
    let memory_output = topo_memory.process_sequence(&sequence)?;
    
    println!("Detected motifs: {}", memory_output.motifs.len());
    println!("Detected cycles: {}", memory_output.cycles.len());
    println!("Stable cores: {}", memory_output.stable_cores.len());
    
    Ok(())
}
```

## 🏗️ Architecture

### Core Crates
- `tensor-core`: Core tensor operations and types
- `backend-cpu`: CPU implementation using ndarray
- `autograd`: Automatic differentiation
- `nn`: Neural network layers and modules
- `optim`: Optimization algorithms

### Experimental Crates
- `topo-memory`: Topological memory system
- `geometry`: Geometric processing components
- `scheduling`: Advanced operation scheduling

### Utility Crates
- `data`: Data loading and preprocessing
- `io`: Input/output utilities
- `moe`: Mixture of Experts
- `quant`: Quantization support
- `rev`: Reverse mode autograd
- `serve`: Model serving
- `utils`: Common utilities

## 📚 Examples

The framework includes comprehensive examples:

```bash
# Basic examples
cargo run -p simple_example
cargo run -p advanced_example

# SSM examples
cargo run -p ssm_example
cargo run -p ssm_byte_lm train
cargo run -p ssm_byte_lm infer

# Hyena examples
cargo run -p hyena_example
cargo run -p hyena_span_infilling

# Advanced features
cargo run -p broadcasting_example
cargo run -p checkpointing_example
cargo run -p mixed_precision_example

# Experimental models
cargo run -p experimental_model
```

## 🔬 Research Applications

HelixML is designed for cutting-edge research in:

- **Long Sequence Modeling**: SSM and Hyena architectures
- **Topological Data Analysis**: Memory systems with geometric properties
- **Causal Inference**: CDT scheduling for causal dependencies
- **Geometric Deep Learning**: Twistor and E8 symmetry applications
- **Byte-level Processing**: Token-free language modeling

## 🚀 Performance

- **Memory Efficient**: Gradient checkpointing and mixed precision
- **Fast**: Optimized tensor operations with ndarray
- **Scalable**: Designed for large-scale models
- **Flexible**: Modular architecture for custom components

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by modern ML frameworks like PyTorch and JAX
- Built on the excellent Rust ecosystem
- Incorporates cutting-edge research in state-space models and geometric deep learning

## 📖 Documentation

- [API Documentation](https://docs.rs/helix-ml)
- [Architecture Guide](docs/ARCH.md)
- [Examples](examples/)

---

**HelixML** - Pushing the boundaries of machine learning with Rust 🦀