//! 🌀 HelixML Tensor Core
//! 
//! Core tensor trait and operations for high-performance ML computations.
//! Designed for SSM/Hyena architectures with topological memory.

pub mod tensor;
pub mod shape;
pub mod dtype;
pub mod device;
pub mod ops;
pub mod error;

// Re-exports
pub use tensor::Tensor;
pub use shape::Shape;
pub use dtype::DType;
pub use device::Device;
pub use error::TensorError;

/// HelixML tensor result type
pub type Result<T> = std::result::Result<T, TensorError>;