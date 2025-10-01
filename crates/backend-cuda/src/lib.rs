//! 🌀 HelixML CUDA Backend
//! 
//! High-performance CUDA backend with fused kernels for SSM/Hyena architectures.

#![recursion_limit = "1024"]

pub mod cuda_backend;
pub mod cuda_kernels;
pub mod fused_ops;
pub mod memory_manager;

// Re-exports
pub use cuda_backend::*;
pub use cuda_kernels::*;
pub use fused_ops::*;
pub use memory_manager::*;