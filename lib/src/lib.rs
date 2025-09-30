// HelixML - High-performance ML framework for Rust
// Focus on post-transformer architectures (SSM, Hyena)

// Re-export commonly used types and traits from crates
pub use tensor_core::*;
pub use backend_cpu::CpuTensor;
pub use optim::*;

// Re-export autograd first to avoid conflicts
pub use autograd::*;

// Re-export nn modules (excluding AutogradContext)
pub use nn::{
    Module, CheckpointableModule,
    Linear, ReLU, GELU, SiLU,
    RMSNorm, Dropout, Sequential,
    S4Block, MambaBlock,
    HyenaBlock, HyenaOperator
};

// Re-export topological memory components
pub use topo_memory::*;

// Re-export meaning induction components
pub use meanings::*;

// Re-export geometric components
pub use geometry::*;

// Re-export scheduling components
pub use scheduling::*;