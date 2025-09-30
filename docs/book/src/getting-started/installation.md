# Installation

This guide will help you install HelixML on your system.

## System Requirements

### **Minimum Requirements**
- **Rust**: 1.70+ (stable, beta, or nightly)
- **Memory**: 8GB RAM (16GB recommended)
- **Storage**: 2GB free space
- **OS**: Linux, macOS, or Windows

### **Recommended Requirements**
- **Rust**: Latest stable
- **Memory**: 32GB+ RAM for large models
- **Storage**: 10GB+ free space
- **GPU**: NVIDIA GPU with CUDA 11.8+ (optional)

## Installation Methods

### **Method 1: From Source (Recommended)**

```bash
# Clone the repository
git clone https://github.com/blackterrordistro/helix-ml.git
cd helix-ml

# Build the project
cargo build --release

# Run tests to verify installation
cargo test

# Run examples to verify functionality
cargo run --example simple_example
```

### **Method 2: Using Cargo (Future)**

```bash
# This will be available once published to crates.io
cargo install helix-ml
```

## CUDA Support (Optional)

### **Prerequisites**
- NVIDIA GPU with Compute Capability 6.0+
- CUDA Toolkit 11.8 or 12.0+
- cuDNN 8.6+
- cuBLAS (included with CUDA)

### **Installation Steps**

#### **1. Install CUDA Toolkit**
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

# macOS (using Homebrew)
brew install cuda

# Windows
# Download and install from NVIDIA website
```

#### **2. Set Environment Variables**
```bash
# Add to your ~/.bashrc or ~/.zshrc
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

#### **3. Build with CUDA Support**
```bash
# Build with CUDA features
cargo build --release --features cuda

# Test CUDA functionality
cargo run --example cuda_example --features cuda
```

## Docker Installation

### **CPU-only Docker**
```bash
# Build Docker image
docker build -t helix-ml:latest .

# Run container
docker run -it helix-ml:latest bash

# Run examples
docker run -it helix-ml:latest cargo run --example simple_example
```

### **GPU Docker (NVIDIA)**
```bash
# Build CUDA Docker image
docker build -f docker/Dockerfile.cuda -t helix-ml:cuda .

# Run with GPU support
docker run --gpus all -it helix-ml:cuda bash

# Test GPU functionality
docker run --gpus all -it helix-ml:cuda nvidia-smi
```

## Development Setup

### **Install Development Tools**
```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install additional tools
cargo install cargo-watch    # Auto-reload on changes
cargo install cargo-audit    # Security auditing
cargo install cargo-deny     # Dependency checking
cargo install cargo-tarpaulin # Code coverage
cargo install criterion      # Benchmarking
```

### **IDE Setup**

#### **VS Code**
```bash
# Install Rust extension
code --install-extension rust-lang.rust-analyzer

# Install additional extensions
code --install-extension tamasfe.even-better-toml
code --install-extension serayuzgur.crates
```

#### **IntelliJ IDEA / CLion**
1. Install the Rust plugin
2. Open the helix-ml directory
3. Wait for indexing to complete

### **Configure Git Hooks**
```bash
# Install pre-commit hooks
cargo install cargo-husky
cargo husky install

# Or manually install
cp .git/hooks/pre-commit.example .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

## Verification

### **Basic Verification**
```bash
# Check installation
cargo --version
rustc --version

# Run basic tests
cargo test --workspace

# Run examples
cargo run --example simple_example
cargo run --example ssm_example
cargo run --example hyena_example
```

### **Advanced Verification**
```bash
# Run integration tests
cargo test --workspace --test '*'

# Run benchmarks
cargo bench --workspace

# Check code coverage
cargo tarpaulin --out Html

# Security audit
cargo audit

# Dependency check
cargo deny check
```

### **CUDA Verification**
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Test CUDA backend
cargo run --example cuda_example --features cuda

# Run CUDA benchmarks
cargo bench --package backend-cuda --features cuda
```

## Troubleshooting

### **Common Issues**

#### **Build Failures**
```bash
# Clean and rebuild
cargo clean
cargo build --release

# Update dependencies
cargo update

# Check Rust version
rustup update
```

#### **CUDA Issues**
```bash
# Verify CUDA installation
nvcc --version
nvidia-smi

# Check environment variables
echo $CUDA_HOME
echo $LD_LIBRARY_PATH

# Rebuild with verbose output
cargo build --release --features cuda --verbose
```

#### **Memory Issues**
```bash
# Reduce parallelism
export CARGO_BUILD_JOBS=1

# Increase swap space
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### **Permission Issues**
```bash
# Fix cargo permissions
sudo chown -R $(whoami) ~/.cargo

# Fix rustup permissions
sudo chown -R $(whoami) ~/.rustup
```

### **Getting Help**

If you encounter issues:

1. **Check the logs**: Run with `RUST_BACKTRACE=1` for detailed error messages
2. **Search issues**: Look for similar problems in our [GitHub issues](https://github.com/blackterrordistro/helix-ml/issues)
3. **Create an issue**: Provide detailed information about your system and the error
4. **Join discussions**: Ask questions in our [GitHub discussions](https://github.com/blackterrordistro/helix-ml/discussions)

## Next Steps

Once you have HelixML installed:

1. **[Quick Start](quick-start.md)**: Learn the basics
2. **[First Example](first-example.md)**: Build your first model
3. **[Examples](../examples/basic-ssm.md)**: Explore more complex examples
4. **[Architecture](../architecture/overview.md)**: Understand the system design
