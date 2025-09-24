//! Tensor shape operations

use serde::{Deserialize, Serialize};
use std::fmt;

/// Tensor shape representation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Create a new shape from dimensions
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }
    
    /// Create a scalar shape (0 dimensions)
    pub fn scalar() -> Self {
        Self { dims: vec![] }
    }
    
    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }
    
    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }
    
    /// Get the dimensions as a slice
    pub fn as_slice(&self) -> &[usize] {
        &self.dims
    }
    
    /// Get a specific dimension
    pub fn dim(&self, idx: usize) -> Option<usize> {
        self.dims.get(idx).copied()
    }
    
    /// Set a specific dimension
    pub fn set_dim(&mut self, idx: usize, value: usize) -> Result<(), crate::TensorError> {
        if idx >= self.dims.len() {
            return Err(crate::TensorError::InvalidDimension {
                dim: idx,
                shape: self.dims.clone(),
            });
        }
        self.dims[idx] = value;
        Ok(())
    }
    
    /// Reshape to new dimensions
    pub fn reshape(&self, new_dims: Vec<usize>) -> Result<Self, crate::TensorError> {
        let new_numel: usize = new_dims.iter().product();
        if new_numel != self.numel() {
            return Err(crate::TensorError::ShapeMismatch {
                expected: vec![self.numel()],
                actual: vec![new_numel],
            });
        }
        Ok(Self::new(new_dims))
    }
    
    /// Broadcast to compatible shape
    pub fn broadcast_to(&self, target: &Shape) -> Result<Self, crate::TensorError> {
        if self.dims.len() > target.dims.len() {
            return Err(crate::TensorError::ShapeMismatch {
                expected: target.dims.clone(),
                actual: self.dims.clone(),
            });
        }
        
        let mut result = target.dims.clone();
        let offset = target.dims.len() - self.dims.len();
        
        for (i, &dim) in self.dims.iter().enumerate() {
            let target_dim = target.dims[offset + i];
            if dim != 1 && dim != target_dim {
                return Err(crate::TensorError::ShapeMismatch {
                    expected: vec![target_dim],
                    actual: vec![dim],
                });
            }
            result[offset + i] = target_dim;
        }
        
        Ok(Self::new(result))
    }
    
    /// Check if shapes are compatible for broadcasting
    pub fn is_broadcast_compatible(&self, other: &Shape) -> bool {
        self.broadcast_to(other).is_ok() || other.broadcast_to(self).is_ok()
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self::new(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self::new(dims.to_vec())
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.dims.is_empty() {
            write!(f, "()")
        } else {
            write!(f, "({})", self.dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
        }
    }
}
