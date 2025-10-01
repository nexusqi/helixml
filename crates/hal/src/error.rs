//! 🌀 HelixML HAL Error Types

use thiserror::Error;

/// HAL-specific error types
#[derive(Error, Debug)]
pub enum HalError {
    #[error("Device error: {message}")]
    DeviceError { message: String },
    
    #[error("Memory error: {message}")]
    MemoryError { message: String },
    
    #[error("Operation error: {message}")]
    OperationError { message: String },
    
    #[error("Scheduler error: {message}")]
    SchedulerError { message: String },
    
    #[error("Capability error: {message}")]
    CapabilityError { message: String },
    
    #[error("Backend error: {message}")]
    BackendError { message: String },
    
    #[error("Unsupported operation: {operation} on device {device}")]
    UnsupportedOperation { operation: String, device: String },
    
    #[error("Memory allocation failed: requested {requested} bytes, available {available} bytes")]
    MemoryAllocationFailed { requested: usize, available: usize },
    
    #[error("Device not found: {device_id}")]
    DeviceNotFound { device_id: String },
    
    #[error("Async operation failed: {message}")]
    AsyncError { message: String },
    
    #[error("Topological computation error: {message}")]
    TopologicalError { message: String },
}

impl From<std::io::Error> for HalError {
    fn from(err: std::io::Error) -> Self {
        HalError::BackendError {
            message: err.to_string(),
        }
    }
}

impl From<ndarray::ShapeError> for HalError {
    fn from(err: ndarray::ShapeError) -> Self {
        HalError::OperationError {
            message: err.to_string(),
        }
    }
}

