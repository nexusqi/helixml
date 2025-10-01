# HelixML 🌀

**A high-performance Rust machine learning framework focused on post-transformer architectures**

[![Crates.io](https://img.shields.io/crates/v/helix-ml.svg)](https://crates.io/crates/helix-ml)
[![Documentation](https://docs.rs/helix-ml/badge.svg)](https://docs.rs/helix-ml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

### Core Components
- **Tensor Operations**: High-performance tensor operations with CPU and CUDA backends
- **Automatic Differentiation**: Full autograd support with gradient checkpointing, accumulation, and clipping
- **Neural Networks**: Modern layers including RMSNorm, SiLU, GELU, Dropout, Linear
- **Optimizers**: AdamW, Lion, SGD with learning rate scheduling and mixed precision
- **Training System**: Comprehensive training with loss functions, metrics, and checkpointing
- **Multi-Device Support**: UnifiedTensor with automatic device synchronization

### Post-Transformer Architectures
- **State-Space Models (SSM)**:
  - S4Block: Structured State Space Models
  - MambaBlock: Selective State Space Models
- **Hyena Blocks**: FFT-based long convolutions
- **HyenaOperator**: Advanced FFT convolution operations

### Advanced Components
- **Enhanced Topological Memory System**:
  - M0 (Motifs): Short pattern detection with hierarchical processing
  - M1 (Cycles): Medium-term dependency analysis with attention mechanisms
  - M2 (Stable Cores): Long-term knowledge extraction with adaptive consolidation
  - U/I/S Links: Temporal/Intermediate/Stable connections with stability prediction
  - Enhanced Retrieval: Multi-scale analysis and pattern synthesis
  - Phase Synchronization: SSM core synchronization utilities

- **Geometric Processing**:
  - Twistor Pre-encoder: Geometric preprocessing with invariants
  - E8 Symmetry Tying: E8 group symmetries with breaking detection
  - MERA Hierarchical Access: Hierarchical memory access with isometries

- **Adaptive Scheduling System**:
  - Multi-Device Orchestration: CPU, CUDA device management
  - Load Balancing: Round Robin, Least Loaded, Weighted, Adaptive strategies
  - Resource Monitoring: Memory, compute, bandwidth, storage tracking
  - Optimization Engine: Genetic Algorithm, Simulated Annealing, Particle Swarm
  - Policy Management: Resource, Load Balancing, Priority, Energy, Latency policies
  - Metrics Collection: Comprehensive performance monitoring and analytics

- **Synthetic Data Generation**:
  - Multi-Modal Generators: Sequences, Images, Graphs, Time Series, Text
  - Verification System: Quality checks, statistical validation, cross-modal verification
  - Dataset Management: Pre-defined datasets for various ML tasks
  - Benchmarking: Performance testing and optimization

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

### Enhanced Topological Memory Example

```rust
use helix_ml::*;

fn main() -> Result<()> {
    let device = Device::cpu();
    let d_model = 64;
    
    // Create enhanced topological memory system
    let mut topo_memory = EnhancedTopologicalMemory::<CpuTensor>::new(
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
    println!("Hierarchical features: {}", memory_output.hierarchical_features.len());
    
    Ok(())
}
```

### Adaptive Scheduler Example

```rust
use helix_ml::*;

fn main() -> Result<()> {
    // Initialize scheduler configuration
    let config = SchedulerConfig::default();
    let mut scheduler = AdaptiveScheduler::new(config)?;
    
    // Start the scheduler
    scheduler.start()?;
    
    // Create and submit tasks
    let task = Task {
        operation: TaskOperation::TensorOperation {
            operation: TensorOp::MatrixMultiply,
            inputs: vec![],
            output_shape: Shape::new(vec![100, 100]),
        },
        priority: TaskPriority::High,
        resource_requirements: ResourceRequirements::default(),
        device_requirements: DeviceRequirements::default(),
        timeout: Duration::from_secs(30),
        retry_count: 0,
        max_retries: 3,
    };
    
    let task_id = scheduler.submit_task(task)?;
    println!("Submitted task: {}", task_id.id);
    
    // Monitor task execution
    let status = scheduler.get_task_status(&task_id)?;
    println!("Task status: {:?}", status);
    
    // Stop the scheduler
    scheduler.stop()?;
    
    Ok(())
}
```

### Synthetic Data Generation Example

```rust
use helix_ml::*;

fn main() -> Result<()> {
    let device = Device::cpu();
    
    // Create synthetic data system
    let config = SyntheticDataConfig::default();
    let mut synthetic_system = SyntheticDataSystem::new(config, &device)?;
    
    // Generate synthetic sequences
    let sequences = synthetic_system.generate_sequences(100)?;
    println!("Generated {} sequences", sequences.sequences.len());
    
    // Generate synthetic images
    let images = synthetic_system.generate_images(50)?;
    println!("Generated {} images", images.images.len());
    
    // Generate synthetic graphs
    let graphs = synthetic_system.generate_graphs(25)?;
    println!("Generated {} graphs", graphs.graphs.len());
    
    Ok(())
}
```

## 🏗️ Architecture

### Core Crates
- `tensor-core`: Core tensor operations and types
- `backend-cpu`: CPU implementation using ndarray
- `backend-cuda`: CUDA implementation with fused kernels
- `autograd`: Complete automatic differentiation system
- `nn`: Neural network layers and modules
- `optim`: Optimization algorithms
- `training`: Comprehensive training system

### Advanced Crates
- `topo-memory`: Enhanced topological memory system
- `geometry`: Geometric processing components
- `adaptive-scheduler`: Multi-device adaptive scheduling
- `synthetic-data`: Synthetic data generation and verification

### Utility Crates
- `data-pipeline`: Data loading and preprocessing
- `meanings`: Semantic processing
- `scheduling`: Advanced operation scheduling
- `serve`: Model serving

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
cargo run -p cuda_example

# Enhanced systems
cargo run -p enhanced_topo_memory_example
cargo run -p synthetic_data_example
cargo run -p adaptive_scheduler_example

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

## 📖 Documentation(soon)

- [API Documentation]
- [Architecture Guide]
- [Examples]

---

**HelixML** - Pushing the boundaries of machine learning with Rust 🦀
