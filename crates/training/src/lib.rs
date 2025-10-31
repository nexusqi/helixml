//! 🌀 HelixML Training System
//! 
//! Advanced training system with loss functions, optimizers, schedulers, and monitoring.

#![recursion_limit = "1024"]

pub mod trainer;
pub mod loss;
pub mod optimizer;
pub mod scheduler;
pub mod metrics;
pub mod checkpoint;
pub mod monitor;
pub mod data_loader;
pub mod validation;

// Re-exports
pub use trainer::*;
pub use loss::*;
pub use optimizer::*;
pub use scheduler::*;
pub use metrics::*;
pub use checkpoint::*;
pub use monitor::*;
pub use data_loader::*;
pub use validation::*;

// Convenience type alias for CPU tensor trainer (backward compatibility)
pub use backend_cpu::CpuTensor;

/// Type alias for Trainer with CpuTensor (for convenience)
pub type CpuTrainer<M> = Trainer<M, CpuTensor>;
