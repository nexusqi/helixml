//! 🌀 Memory-Efficient Autograd
//! 
//! Memory optimization techniques for training large models

use tensor_core::{Tensor, Result, TensorError, Shape, DType, Device};
use tensor_core::tensor::{TensorOps, TensorReduce, TensorRandom, TensorBroadcast, TensorActivation};
use std::collections::{HashMap, VecDeque, HashSet};
use super::AutogradContext;

/// Memory pool for efficient tensor allocation
#[derive(Debug)]
pub struct TensorMemoryPool<T: Tensor> {
    available_tensors: HashMap<Shape, VecDeque<T>>,
    max_pool_size: usize,
    total_allocated: usize,
    peak_memory: usize,
}

impl<T: Tensor + TensorRandom> TensorMemoryPool<T> {
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            available_tensors: HashMap::new(),
            max_pool_size,
            total_allocated: 0,
            peak_memory: 0,
        }
    }
    
    /// Get a tensor from the pool or create a new one
    pub fn get_tensor(&mut self, shape: Shape, dtype: DType, device: &Device) -> Result<T> {
        if let Some(tensor_queue) = self.available_tensors.get_mut(&shape) {
            if let Some(tensor) = tensor_queue.pop_front() {
                return Ok(tensor);
            }
        }
        
        // Create new tensor
        let tensor = T::zeros(shape, dtype, device)?;
        self.total_allocated += 1;
        self.peak_memory = self.peak_memory.max(self.total_allocated);
        
        Ok(tensor)
    }
    
    /// Return a tensor to the pool
    pub fn return_tensor(&mut self, tensor: T) {
        if self.available_tensors.len() < self.max_pool_size {
            let shape = tensor.shape().clone();
            self.available_tensors
                .entry(shape)
                .or_insert_with(VecDeque::new)
                .push_back(tensor);
        }
    }
    
    /// Clear the pool
    pub fn clear(&mut self) {
        self.available_tensors.clear();
    }
    
    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            total_allocated: self.total_allocated,
            peak_memory: self.peak_memory,
            pool_size: self.available_tensors.values().map(|q| q.len()).sum(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub peak_memory: usize,
    pub pool_size: usize,
}

/// Memory-efficient gradient computation with lazy evaluation
#[derive(Debug)]
pub struct LazyGradientComputer<T: Tensor> {
    computation_graph: HashMap<usize, ComputationNode>,
    gradient_cache: HashMap<usize, T>,
    memory_pool: TensorMemoryPool<T>,
}

#[derive(Debug, Clone)]
pub enum ComputationNode {
    Leaf,
    Unary { op: UnaryOp, input: usize },
    Binary { op: BinaryOp, left: usize, right: usize },
    Reduce { op: ReduceOp, input: usize, dims: Option<Vec<usize>> },
}

#[derive(Debug, Clone)]
pub enum UnaryOp {
    Neg, Abs, Sqrt, Exp, Log, Sin, Cos, Tan,
    Relu, Gelu, Silu, Sigmoid, Tanh,
}

#[derive(Debug, Clone)]
pub enum BinaryOp {
    Add, Sub, Mul, Div, Pow, Max, Min,
}

#[derive(Debug, Clone)]
pub enum ReduceOp {
    Sum, Mean, Max, Min, Var, Std,
}

impl<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorReduce + TensorActivation> LazyGradientComputer<T> {
    pub fn new() -> Self {
        Self {
            computation_graph: HashMap::new(),
            gradient_cache: HashMap::new(),
            memory_pool: TensorMemoryPool::new(1000),
        }
    }
    
    /// Add a computation node to the graph
    pub fn add_node(&mut self, tensor_id: usize, node: ComputationNode) {
        self.computation_graph.insert(tensor_id, node);
    }
    
    /// Compute gradients lazily
    pub fn compute_gradients(&mut self, ctx: &mut AutogradContext<T>, output_id: usize) -> Result<()> {
        // Initialize output gradient
        if let Some(output_tensor) = ctx.get_tensor(output_id) {
            let output_shape = output_tensor.tensor().shape().clone();
            let output_dtype = output_tensor.tensor().dtype();
            let output_device = output_tensor.tensor().device();
            
            let ones = T::ones(output_shape, output_dtype, output_device)?;
            self.gradient_cache.insert(output_id, ones);
        }
        
        // Topological sort for gradient computation
        let sorted_ids = self.topological_sort()?;
        
        // Compute gradients in reverse order
        for tensor_id in sorted_ids.into_iter().rev() {
        if let Some(node) = self.computation_graph.get(&tensor_id).cloned() {
            self.compute_node_gradient(ctx, tensor_id, &node)?;
            }
        }
        
        Ok(())
    }
    
    fn compute_node_gradient(&mut self, ctx: &mut AutogradContext<T>, tensor_id: usize, node: &ComputationNode) -> Result<()> {
        if let Some(output_grad) = self.gradient_cache.get(&tensor_id) {
            match node {
                ComputationNode::Leaf => {
                    // Leaf node - accumulate gradient
                    if let Some(diff_tensor) = ctx.get_tensor_mut(tensor_id) {
                        if let Some(existing_grad) = &mut diff_tensor.grad {
                            *existing_grad = existing_grad.add(output_grad)?;
                        } else {
                            diff_tensor.grad = Some(output_grad.clone());
                        }
                    }
                }
                _ => {
                    // TODO: Implement gradient computation for unary, binary, and reduce operations
                    // For now, just skip these operations
                }
            }
        }
        Ok(())
    }
    
    // TODO: Implement unary gradient computation
    fn compute_unary_gradient(&self, _ctx: &AutogradContext<T>, _tensor_id: usize, _input_id: usize, _op: &UnaryOp, _output_grad: &T) -> Result<T> {
        Err(TensorError::UnsupportedOperation {
            op: "Unary gradient computation not implemented".to_string(),
        })
    }
    
    // TODO: Implement binary gradient computation
    fn compute_binary_gradient(&self, _ctx: &AutogradContext<T>, _tensor_id: usize, _left_id: usize, _right_id: usize, _op: &BinaryOp, _output_grad: &T) -> Result<(T, T)> {
        Err(TensorError::UnsupportedOperation {
            op: "Binary gradient computation not implemented".to_string(),
        })
    }
    
    // TODO: Implement reduce gradient computation
    fn compute_reduce_gradient(&self, _ctx: &AutogradContext<T>, _tensor_id: usize, _input_id: usize, _op: &ReduceOp, _dims: &Option<Vec<usize>>, _output_grad: &T) -> Result<T> {
        Err(TensorError::UnsupportedOperation {
            op: "Reduce gradient computation not implemented".to_string(),
        })
    }
    
    fn accumulate_gradient(&mut self, ctx: &mut AutogradContext<T>, tensor_id: usize, grad: T) -> Result<()> {
        if let Some(diff_tensor) = ctx.get_tensor_mut(tensor_id) {
            if let Some(existing_grad) = &mut diff_tensor.grad {
                *existing_grad = existing_grad.add(&grad)?;
            } else {
                diff_tensor.grad = Some(grad);
            }
        }
        Ok(())
    }
    
    fn topological_sort(&self) -> Result<Vec<usize>> {
        let mut visited = HashSet::new();
        let mut stack = Vec::new();
        
        for &tensor_id in self.computation_graph.keys() {
            if !visited.contains(&tensor_id) {
                self.dfs(tensor_id, &mut visited, &mut stack)?;
            }
        }
        
        Ok(stack)
    }
    
    fn dfs(&self, tensor_id: usize, visited: &mut HashSet<usize>, stack: &mut Vec<usize>) -> Result<()> {
        visited.insert(tensor_id);
        
        if let Some(node) = self.computation_graph.get(&tensor_id) {
            match node {
                ComputationNode::Unary { input, .. } => {
                    if !visited.contains(input) {
                        self.dfs(*input, visited, stack)?;
                    }
                }
                ComputationNode::Binary { left, right, .. } => {
                    if !visited.contains(left) {
                        self.dfs(*left, visited, stack)?;
                    }
                    if !visited.contains(right) {
                        self.dfs(*right, visited, stack)?;
                    }
                }
                ComputationNode::Reduce { input, .. } => {
                    if !visited.contains(input) {
                        self.dfs(*input, visited, stack)?;
                    }
                }
                ComputationNode::Leaf => {}
            }
        }
        
        stack.push(tensor_id);
        Ok(())
    }
}

/// Memory usage monitor
#[derive(Debug)]
pub struct MemoryMonitor {
    peak_usage: usize,
    current_usage: usize,
    allocations: HashMap<usize, usize>, // tensor_id -> size
}

impl MemoryMonitor {
    pub fn new() -> Self {
        Self {
            peak_usage: 0,
            current_usage: 0,
            allocations: HashMap::new(),
        }
    }
    
    pub fn track_allocation(&mut self, tensor_id: usize, size: usize) {
        self.current_usage += size;
        self.peak_usage = self.peak_usage.max(self.current_usage);
        self.allocations.insert(tensor_id, size);
    }
    
    pub fn track_deallocation(&mut self, tensor_id: usize) {
        if let Some(size) = self.allocations.remove(&tensor_id) {
            self.current_usage -= size;
        }
    }
    
    pub fn get_usage(&self) -> (usize, usize) {
        (self.current_usage, self.peak_usage)
    }
    
    pub fn get_allocations(&self) -> &HashMap<usize, usize> {
        &self.allocations
    }
}
