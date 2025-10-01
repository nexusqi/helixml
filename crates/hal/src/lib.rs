//! 🌀 HelixML Hardware Abstraction Layer (HAL)
//! 
//! Universal compute backend interface supporting ANY hardware:
//! - CPU (with BLAS optimization)
//! - CUDA (NVIDIA GPU)
//! - Metal (Apple GPU)
//! - ROCm (AMD GPU)
//! - NPU (Neural Processing Units)
//! - TPU (Tensor Processing Units)
//! - QPU (Quantum Processing Units)
//! - BCI (Brain-Computer Interface)
//! - Custom backends

pub mod backend;
pub mod device;
pub mod memory;
pub mod operations;
pub mod scheduler;
pub mod error;
pub mod capabilities;

// Re-exports
pub use backend::*;
pub use device::*;
pub use memory::*;
pub use operations::*;
pub use scheduler::*;
pub use error::*;
pub use capabilities::*;

/// HelixML HAL result type
pub type Result<T> = std::result::Result<T, HalError>;

