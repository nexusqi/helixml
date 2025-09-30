//! 🌀 HelixML Autograd
//! 
//! Automatic differentiation for SSM/Hyena architectures with topological memory.

use tensor_core::{Tensor, Result, TensorError};
use tensor_core::tensor::{TensorOps, TensorStats, TensorReduce};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Differentiable tensor with gradient tracking
#[derive(Debug, Clone)]
pub struct DiffTensor<T: Tensor> {
    tensor: T,
    requires_grad: bool,
    grad: Option<T>,
}

impl<T: Tensor> DiffTensor<T> {
    pub fn new(tensor: T, requires_grad: bool) -> Self {
        Self {
            tensor,
            requires_grad,
            grad: None,
        }
    }
    
    pub fn tensor(&self) -> &T {
        &self.tensor
    }
    
    pub fn tensor_mut(&mut self) -> &mut T {
        &mut self.tensor
    }
    
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }
    
    pub fn grad(&self) -> Option<&T> {
        self.grad.as_ref()
    }
    
    pub fn set_grad(&mut self, grad: T) {
        self.grad = Some(grad);
    }
}

impl<T: Tensor> std::ops::Deref for DiffTensor<T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        &self.tensor
    }
}

/// Checkpoint for gradient computation
#[derive(Debug, Clone)]
pub struct Checkpoint<T: Tensor> {
    tensor_id: usize,
    tensor: T,
}

/// Context for automatic differentiation with checkpointing
#[derive(Debug)]
pub struct AutogradContext<T: Tensor> {
    tensors: HashMap<usize, DiffTensor<T>>,
    next_tensor_id: usize,
    checkpoints: Vec<Checkpoint<T>>,
}

impl<T: Tensor> AutogradContext<T> {
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            next_tensor_id: 0,
            checkpoints: Vec::new(),
        }
    }
    
    pub fn tensor(&mut self, tensor: T, requires_grad: bool) -> usize {
        let id = self.next_tensor_id;
        self.next_tensor_id += 1;
        
        let diff_tensor = DiffTensor::new(tensor, requires_grad);
        self.tensors.insert(id, diff_tensor);
        id
    }
    
    pub fn get_tensor(&self, id: usize) -> Option<&DiffTensor<T>> {
        self.tensors.get(&id)
    }
    
    pub fn get_tensor_mut(&mut self, id: usize) -> Option<&mut DiffTensor<T>> {
        self.tensors.get_mut(&id)
    }
    
    /// Create a checkpoint for gradient computation
    /// This saves the tensor state and allows for memory-efficient backward pass
    pub fn checkpoint(&mut self, tensor_id: usize) -> Result<()> {
        if let Some(diff_tensor) = self.tensors.get(&tensor_id) {
            let checkpoint = Checkpoint {
                tensor_id,
                tensor: diff_tensor.tensor.clone(),
            };
            self.checkpoints.push(checkpoint);
            Ok(())
        } else {
            Err(TensorError::BackendError {
                message: format!("Tensor {} not found for checkpointing", tensor_id),
            })
        }
    }
    
    /// Restore from checkpoint and recompute forward pass
    /// This is used during backward pass to save memory
    pub fn restore_checkpoint<F>(&mut self, tensor_id: usize, forward_fn: F) -> Result<usize>
    where
        F: FnOnce(&mut AutogradContext<T>) -> Result<usize>,
    {
        // Find the checkpoint
        if let Some(checkpoint_idx) = self.checkpoints.iter().position(|cp| cp.tensor_id == tensor_id) {
            let checkpoint = self.checkpoints.remove(checkpoint_idx);
            
            // Restore the tensor
            if let Some(diff_tensor) = self.tensors.get_mut(&tensor_id) {
                diff_tensor.tensor = checkpoint.tensor;
            }
            
            // Recompute forward pass from this point
            forward_fn(self)
        } else {
            Err(TensorError::BackendError {
                message: format!("Checkpoint for tensor {} not found", tensor_id),
            })
        }
    }
    
    /// Clear all checkpoints (call after backward pass)
    pub fn clear_checkpoints(&mut self) {
        self.checkpoints.clear();
    }
    
    /// Get number of active checkpoints
    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }
}

/// Loss functions for training
pub mod losses {
    use super::*;
    
    /// Mean Squared Error loss
    pub fn mse_loss<T: Tensor + TensorOps + TensorReduce>(ctx: &mut AutogradContext<T>, predictions: usize, targets: usize) -> Result<usize> {
        let pred_tensor: &DiffTensor<T> = ctx.get_tensor(predictions)
            .ok_or_else(|| TensorError::BackendError {
                message: format!("Prediction tensor {} not found", predictions),
            })?;
        
        let target_tensor: &DiffTensor<T> = ctx.get_tensor(targets)
            .ok_or_else(|| TensorError::BackendError {
                message: format!("Target tensor {} not found", targets),
            })?;
        
        let diff = pred_tensor.tensor().sub(target_tensor.tensor())?;
        let squared_diff = diff.mul(&diff)?;
        let loss = squared_diff.mean(None, false)?;
        
        let loss_id = ctx.tensor(loss, true);
        Ok(loss_id)
    }
    
    /// Cross Entropy loss
    pub fn cross_entropy_loss<T: Tensor + TensorOps + TensorStats + TensorReduce>(ctx: &mut AutogradContext<T>, logits: usize, targets: usize) -> Result<usize> {
        let logits_tensor: &DiffTensor<T> = ctx.get_tensor(logits)
            .ok_or_else(|| TensorError::BackendError {
                message: format!("Logits tensor {} not found", logits),
            })?;
        
        let target_tensor: &DiffTensor<T> = ctx.get_tensor(targets)
            .ok_or_else(|| TensorError::BackendError {
                message: format!("Target tensor {} not found", targets),
            })?;
        
        let log_softmax = logits_tensor.tensor().log_softmax(logits_tensor.tensor().ndim() - 1)?;
        let loss = log_softmax.mul(target_tensor.tensor())?.sum(None, false)?;
        
        let loss_id = ctx.tensor(loss, true);
        Ok(loss_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend_cpu::CpuTensor;
    use tensor_core::{Shape, Device};
    use tensor_core::tensor::TensorRandom;
    
    #[test]
    fn test_simple_autograd() {
        let mut ctx = AutogradContext::<CpuTensor>::new();
        
        // Create tensors
        let x = CpuTensor::random_uniform(Shape::new(vec![2, 3]), 0.0, 1.0, &Device::cpu()).unwrap();
        let x_id = ctx.tensor(x, true);
        
        // Check that tensor was created
        assert!(ctx.get_tensor(x_id).is_some());
        assert!(ctx.get_tensor(x_id).unwrap().requires_grad());
    }
}