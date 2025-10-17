//! Hammer Context - Core computation context

use tensor_core::{Tensor, Result};
use std::collections::HashMap;

/// Hammer tensor with gradient tracking
#[derive(Clone)]
pub struct HammerTensor<T: Tensor> {
    pub tensor: T,
    pub grad: Option<T>,
    pub requires_grad: bool,
    pub id: usize,
}

/// Hammer computation context
pub struct HammerContext<T: Tensor> {
    pub tensors: HashMap<usize, HammerTensor<T>>,
    next_id: usize,
}

impl<T: Tensor> HammerContext<T> {
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            next_id: 0,
        }
    }
    
    pub fn add_tensor(&mut self, tensor: T, requires_grad: bool) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        
        self.tensors.insert(id, HammerTensor {
            tensor,
            grad: None,
            requires_grad,
            id,
        });
        
        id
    }
    
    pub fn get_tensor(&self, id: usize) -> Option<&HammerTensor<T>> {
        self.tensors.get(&id)
    }
    
    pub fn get_tensor_mut(&mut self, id: usize) -> Option<&mut HammerTensor<T>> {
        self.tensors.get_mut(&id)
    }
}

impl<T: Tensor> Default for HammerContext<T> {
    fn default() -> Self {
        Self::new()
    }
}

