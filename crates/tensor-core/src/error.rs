//! Error types for tensor operations

use thiserror::Error;

/// Tensor operation errors
#[derive(Error, Debug)]
pub enum TensorError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },
    
    #[error("Invalid dimension: {dim} is out of bounds for shape {shape:?}")]
    InvalidDimension { dim: usize, shape: Vec<usize> },
    
    #[error("Device mismatch: expected {expected:?}, got {actual:?}")]
    DeviceMismatch { expected: String, actual: String },
    
    #[error("DType mismatch: expected {expected:?}, got {actual:?}")]
    DTypeMismatch { expected: String, actual: String },
    
    #[error("Operation not supported: {op}")]
    UnsupportedOperation { op: String },
    
    #[error("Memory allocation failed: {reason}")]
    AllocationFailed { reason: String },
    
    #[error("Backend error: {message}")]
    BackendError { message: String },
    
    #[error("Serialization error: {message}")]
    SerializationError { message: String },
}

/// Result type for tensor operations
pub type Result<T> = std::result::Result<T, TensorError>;
