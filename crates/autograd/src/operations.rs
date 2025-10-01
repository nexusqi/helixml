//! 🌀 Autograd Operations
//! 
//! High-level operations with automatic differentiation support

use tensor_core::{Tensor, Result, TensorError, Shape};
use tensor_core::tensor::{TensorOps, TensorReduce, TensorStats, TensorBroadcast, TensorActivation, TensorRandom};
use super::{AutogradContext, DiffTensor};
use super::backward::{BackwardPass, Operation, ComputationNode};

/// High-level autograd operations
pub struct AutogradOps<T: Tensor> {
    ctx: AutogradContext<T>,
    backward_pass: BackwardPass<T>,
}

impl<T: Tensor + TensorOps + TensorReduce + TensorStats + TensorBroadcast + TensorActivation + TensorRandom + Clone> AutogradOps<T> {
    pub fn new() -> Self {
        Self {
            ctx: AutogradContext::new(),
            backward_pass: BackwardPass::new(),
        }
    }
    
    /// Create a tensor with gradient tracking
    pub fn tensor(&mut self, tensor: T, requires_grad: bool) -> usize {
        let id = self.ctx.tensor(tensor, requires_grad);
        id
    }
    
    /// Get a tensor by ID
    pub fn get_tensor(&self, id: usize) -> Option<&DiffTensor<T>> {
        self.ctx.get_tensor(id)
    }
    
    /// Get a mutable tensor by ID
    pub fn get_tensor_mut(&mut self, id: usize) -> Option<&mut DiffTensor<T>> {
        self.ctx.get_tensor_mut(id)
    }
    
    /// Negate a tensor
    pub fn neg(&mut self, input_id: usize) -> Result<usize> {
        let input_tensor = self.ctx.get_tensor(input_id).unwrap();
        let result = input_tensor.tensor().neg()?;
        let result_shape = result.shape().clone();
        let requires_grad = input_tensor.requires_grad();
        let result_id = self.ctx.tensor(result, requires_grad);
        
        if requires_grad {
            let node = ComputationNode {
                operation: Operation::Neg,
                inputs: vec![input_id],
                output_shape: result_shape,
                requires_grad: true,
            };
            self.backward_pass.add_node(result_id, node);
        }
        
        Ok(result_id)
    }
    
    /// Add two tensors
    pub fn add(&mut self, left_id: usize, right_id: usize) -> Result<usize> {
        let left_tensor = self.ctx.get_tensor(left_id).unwrap();
        let right_tensor = self.ctx.get_tensor(right_id).unwrap();
        let result = left_tensor.tensor().add(right_tensor.tensor())?;
        let result_shape = result.shape().clone();
        let requires_grad = left_tensor.requires_grad() || right_tensor.requires_grad();
        let result_id = self.ctx.tensor(result, requires_grad);
        
        if requires_grad {
            let node = ComputationNode {
                operation: Operation::Add,
                inputs: vec![left_id, right_id],
                output_shape: result_shape,
                requires_grad: true,
            };
            self.backward_pass.add_node(result_id, node);
        }
        
        Ok(result_id)
    }
    
    /// Subtract two tensors
    pub fn sub(&mut self, left_id: usize, right_id: usize) -> Result<usize> {
        let left_tensor = self.ctx.get_tensor(left_id).unwrap();
        let right_tensor = self.ctx.get_tensor(right_id).unwrap();
        let result = left_tensor.tensor().sub(right_tensor.tensor())?;
        let result_shape = result.shape().clone();
        let requires_grad = left_tensor.requires_grad() || right_tensor.requires_grad();
        let result_id = self.ctx.tensor(result, requires_grad);
        
        if requires_grad {
            let node = ComputationNode {
                operation: Operation::Sub,
                inputs: vec![left_id, right_id],
                output_shape: result_shape,
                requires_grad: true,
            };
            self.backward_pass.add_node(result_id, node);
        }
        
        Ok(result_id)
    }
    
    /// Multiply two tensors
    pub fn mul(&mut self, left_id: usize, right_id: usize) -> Result<usize> {
        let left_tensor = self.ctx.get_tensor(left_id).unwrap();
        let right_tensor = self.ctx.get_tensor(right_id).unwrap();
        let result = left_tensor.tensor().mul(right_tensor.tensor())?;
        let result_shape = result.shape().clone();
        let requires_grad = left_tensor.requires_grad() || right_tensor.requires_grad();
        let result_id = self.ctx.tensor(result, requires_grad);
        
        if requires_grad {
            let node = ComputationNode {
                operation: Operation::Mul,
                inputs: vec![left_id, right_id],
                output_shape: result_shape,
                requires_grad: true,
            };
            self.backward_pass.add_node(result_id, node);
        }
        
        Ok(result_id)
    }
    
    /// Divide two tensors
    pub fn div(&mut self, left_id: usize, right_id: usize) -> Result<usize> {
        let left_tensor = self.ctx.get_tensor(left_id).unwrap();
        let right_tensor = self.ctx.get_tensor(right_id).unwrap();
        let result = left_tensor.tensor().div(right_tensor.tensor())?;
        let result_shape = result.shape().clone();
        let requires_grad = left_tensor.requires_grad() || right_tensor.requires_grad();
        let result_id = self.ctx.tensor(result, requires_grad);
        
        if requires_grad {
            let node = ComputationNode {
                operation: Operation::Div,
                inputs: vec![left_id, right_id],
                output_shape: result_shape,
                requires_grad: true,
            };
            self.backward_pass.add_node(result_id, node);
        }
        
        Ok(result_id)
    }
    
    /// Matrix multiplication
    pub fn matmul(&mut self, left_id: usize, right_id: usize) -> Result<usize> {
        let left_tensor = self.ctx.get_tensor(left_id).unwrap();
        let right_tensor = self.ctx.get_tensor(right_id).unwrap();
        let result = left_tensor.tensor().matmul(right_tensor.tensor())?;
        let result_shape = result.shape().clone();
        let requires_grad = left_tensor.requires_grad() || right_tensor.requires_grad();
        let result_id = self.ctx.tensor(result, requires_grad);
        
        if requires_grad {
            let node = ComputationNode {
                operation: Operation::MatMul,
                inputs: vec![left_id, right_id],
                output_shape: result_shape,
                requires_grad: true,
            };
            self.backward_pass.add_node(result_id, node);
        }
        
        Ok(result_id)
    }
    
    /// Sum tensor along dimensions
    pub fn sum(&mut self, input_id: usize, dims: Option<Vec<usize>>, keep_dim: bool) -> Result<usize> {
        let input_tensor = self.ctx.get_tensor(input_id).unwrap();
        let result = input_tensor.tensor().sum(dims.as_ref(), keep_dim)?;
        let result_shape = result.shape().clone();
        let requires_grad = input_tensor.requires_grad();
        let result_id = self.ctx.tensor(result, requires_grad);
        
        if requires_grad {
            let node = ComputationNode {
                operation: Operation::Sum,
                inputs: vec![input_id],
                output_shape: result_shape,
                requires_grad: true,
            };
            self.backward_pass.add_node(result_id, node);
        }
        
        Ok(result_id)
    }
    
    /// Mean tensor along dimensions
    pub fn mean(&mut self, input_id: usize, dims: Option<Vec<usize>>, keep_dim: bool) -> Result<usize> {
        let input_tensor = self.ctx.get_tensor(input_id).unwrap();
        let result = input_tensor.tensor().mean(dims.as_ref(), keep_dim)?;
        let result_shape = result.shape().clone();
        let requires_grad = input_tensor.requires_grad();
        let result_id = self.ctx.tensor(result, requires_grad);
        
        if requires_grad {
            let node = ComputationNode {
                operation: Operation::Mean,
                inputs: vec![input_id],
                output_shape: result_shape,
                requires_grad: true,
            };
            self.backward_pass.add_node(result_id, node);
        }
        
        Ok(result_id)
    }
    
    /// ReLU activation
    pub fn relu(&mut self, input_id: usize) -> Result<usize> {
        let input_tensor = self.ctx.get_tensor(input_id).unwrap();
        let result = input_tensor.tensor().relu()?;
        let result_shape = result.shape().clone();
        let requires_grad = input_tensor.requires_grad();
        let result_id = self.ctx.tensor(result, requires_grad);
        
        if requires_grad {
            let node = ComputationNode {
                operation: Operation::Relu,
                inputs: vec![input_id],
                output_shape: result_shape,
                requires_grad: true,
            };
            self.backward_pass.add_node(result_id, node);
        }
        
        Ok(result_id)
    }
    
    /// GELU activation
    pub fn gelu(&mut self, input_id: usize) -> Result<usize> {
        let input_tensor = self.ctx.get_tensor(input_id).unwrap();
        let result = input_tensor.tensor().gelu()?;
        let result_shape = result.shape().clone();
        let requires_grad = input_tensor.requires_grad();
        let result_id = self.ctx.tensor(result, requires_grad);
        
        if requires_grad {
            let node = ComputationNode {
                operation: Operation::Gelu,
                inputs: vec![input_id],
                output_shape: result_shape,
                requires_grad: true,
            };
            self.backward_pass.add_node(result_id, node);
        }
        
        Ok(result_id)
    }
    
    /// SiLU activation
    pub fn silu(&mut self, input_id: usize) -> Result<usize> {
        let input_tensor = self.ctx.get_tensor(input_id).unwrap();
        let result = input_tensor.tensor().silu()?;
        let result_shape = result.shape().clone();
        let requires_grad = input_tensor.requires_grad();
        let result_id = self.ctx.tensor(result, requires_grad);
        
        if requires_grad {
            let node = ComputationNode {
                operation: Operation::Silu,
                inputs: vec![input_id],
                output_shape: result_shape,
                requires_grad: true,
            };
            self.backward_pass.add_node(result_id, node);
        }
        
        Ok(result_id)
    }
    
    /// Sigmoid activation
    pub fn sigmoid(&mut self, input_id: usize) -> Result<usize> {
        let input_tensor = self.ctx.get_tensor(input_id).unwrap();
        let result = input_tensor.tensor().sigmoid()?;
        let result_shape = result.shape().clone();
        let requires_grad = input_tensor.requires_grad();
        let result_id = self.ctx.tensor(result, requires_grad);
        
        if requires_grad {
            let node = ComputationNode {
                operation: Operation::Sigmoid,
                inputs: vec![input_id],
                output_shape: result_shape,
                requires_grad: true,
            };
            self.backward_pass.add_node(result_id, node);
        }
        
        Ok(result_id)
    }
    
    /// Tanh activation
    pub fn tanh(&mut self, input_id: usize) -> Result<usize> {
        let input_tensor = self.ctx.get_tensor(input_id).unwrap();
        let result = input_tensor.tensor().tanh()?;
        let result_shape = result.shape().clone();
        let requires_grad = input_tensor.requires_grad();
        let result_id = self.ctx.tensor(result, requires_grad);
        
        if requires_grad {
            let node = ComputationNode {
                operation: Operation::Tanh,
                inputs: vec![input_id],
                output_shape: result_shape,
                requires_grad: true,
            };
            self.backward_pass.add_node(result_id, node);
        }
        
        Ok(result_id)
    }
    
    /// Softmax activation
    pub fn softmax(&mut self, input_id: usize, dim: usize) -> Result<usize> {
        let input_tensor = self.ctx.get_tensor(input_id).unwrap();
        let result = input_tensor.tensor().softmax(dim)?;
        let result_shape = result.shape().clone();
        let requires_grad = input_tensor.requires_grad();
        let result_id = self.ctx.tensor(result, requires_grad);
        
        if requires_grad {
            let node = ComputationNode {
                operation: Operation::Softmax,
                inputs: vec![input_id],
                output_shape: result_shape,
                requires_grad: true,
            };
            self.backward_pass.add_node(result_id, node);
        }
        
        Ok(result_id)
    }
    
    /// Log softmax activation
    pub fn log_softmax(&mut self, input_id: usize, dim: usize) -> Result<usize> {
        let input_tensor = self.ctx.get_tensor(input_id).unwrap();
        let result = input_tensor.tensor().log_softmax(dim)?;
        let result_shape = result.shape().clone();
        let requires_grad = input_tensor.requires_grad();
        let result_id = self.ctx.tensor(result, requires_grad);
        
        if requires_grad {
            let node = ComputationNode {
                operation: Operation::LogSoftmax,
                inputs: vec![input_id],
                output_shape: result_shape,
                requires_grad: true,
            };
            self.backward_pass.add_node(result_id, node);
        }
        
        Ok(result_id)
    }
    
    /// Transpose tensor
    pub fn transpose(&mut self, input_id: usize) -> Result<usize> {
        let input_tensor = self.ctx.get_tensor(input_id).unwrap();
        let result = input_tensor.tensor().transpose(0, 1)?;
        let result_shape = result.shape().clone();
        let requires_grad = input_tensor.requires_grad();
        let result_id = self.ctx.tensor(result, requires_grad);
        
        if requires_grad {
            let node = ComputationNode {
                operation: Operation::Transpose,
                inputs: vec![input_id],
                output_shape: result_shape,
                requires_grad: true,
            };
            self.backward_pass.add_node(result_id, node);
        }
        
        Ok(result_id)
    }
    
    /// Execute backward pass
    pub fn backward(&mut self, output_id: usize) -> Result<()> {
        self.backward_pass.backward(&mut self.ctx, output_id)
    }
    
    /// Clear computation graph
    pub fn clear(&mut self) {
        self.backward_pass.clear();
        self.ctx.clear_checkpoints();
    }
    
    /// Get context
    pub fn context(&self) -> &AutogradContext<T> {
        &self.ctx
    }
    
    /// Get mutable context
    pub fn context_mut(&mut self) -> &mut AutogradContext<T> {
        &mut self.ctx
    }
}