//! 🌀 Autograd-Optimizer Integration
//! 
//! Integration between autograd and optimizers for efficient training

use tensor_core::{Tensor, Result};
use tensor_core::tensor::{TensorOps, TensorRandom};
use std::collections::HashMap;
use super::{AutogradContext, DiffTensor};

/// Optimizer state for a parameter
#[derive(Debug, Clone)]
pub struct OptimizerState<T: Tensor> {
    pub momentum: Option<T>,
    pub variance: Option<T>,
    pub step_count: usize,
    pub lr_schedule: Option<f32>,
}

impl<T: Tensor> OptimizerState<T> {
    pub fn new() -> Self {
        Self {
            momentum: None,
            variance: None,
            step_count: 0,
            lr_schedule: None,
        }
    }
}

/// Integrated training step with autograd and optimizer
#[derive(Debug)]
pub struct IntegratedTrainer<T: Tensor> {
    autograd_ctx: AutogradContext<T>,
    optimizer_states: HashMap<usize, OptimizerState<T>>,
    learning_rate: f32,
    weight_decay: f32,
    momentum: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    step_count: usize,
}

impl<T: Tensor + TensorOps + TensorRandom> IntegratedTrainer<T> {
    pub fn new(learning_rate: f32, weight_decay: f32, momentum: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            autograd_ctx: AutogradContext::new(),
            optimizer_states: HashMap::new(),
            learning_rate,
            weight_decay,
            momentum,
            beta1,
            beta2,
            epsilon,
            step_count: 0,
        }
    }
    
    /// Add a parameter to the trainer
    pub fn add_parameter(&mut self, tensor: T, requires_grad: bool) -> usize {
        let param_id = self.autograd_ctx.tensor(tensor, requires_grad);
        self.optimizer_states.insert(param_id, OptimizerState::new());
        param_id
    }
    
    /// Get a parameter tensor
    pub fn get_parameter(&self, param_id: usize) -> Option<&DiffTensor<T>> {
        self.autograd_ctx.get_tensor(param_id)
    }
    
    /// Get a mutable parameter tensor
    pub fn get_parameter_mut(&mut self, param_id: usize) -> Option<&mut DiffTensor<T>> {
        self.autograd_ctx.get_tensor_mut(param_id)
    }
    
    /// Execute a forward pass
    pub fn forward<F>(&mut self, forward_fn: F) -> Result<usize>
    where
        F: FnOnce(&mut AutogradContext<T>) -> Result<usize>,
    {
        forward_fn(&mut self.autograd_ctx)
    }
    
    /// Execute a backward pass and update parameters
    pub fn backward_and_step(&mut self, loss_id: usize) -> Result<()> {
        // Compute gradients
        self.compute_gradients(loss_id)?;
        
        // Update parameters
        self.update_parameters()?;
        
        self.step_count += 1;
        Ok(())
    }
    
    /// Compute gradients for all parameters
    fn compute_gradients(&mut self, loss_id: usize) -> Result<()> {
        // Initialize loss gradient
        if let Some(loss_tensor) = self.autograd_ctx.get_tensor_mut(loss_id) {
            let loss_shape = loss_tensor.tensor().shape().clone();
            let loss_dtype = loss_tensor.tensor().dtype();
            let loss_device = loss_tensor.tensor().device();
            
            let ones = T::ones(loss_shape, loss_dtype, loss_device)?;
            loss_tensor.set_grad(ones);
        }
        
        // TODO: Implement proper gradient computation
        // This would involve traversing the computation graph
        
        Ok(())
    }
    
    /// Update parameters using optimizer
    fn update_parameters(&mut self) -> Result<()> {
        let param_ids: Vec<usize> = self.optimizer_states.keys().cloned().collect();
        for param_id in param_ids {
            if let Some(param_tensor) = self.autograd_ctx.get_tensor_mut(param_id) {
                if let Some(grad) = &param_tensor.grad {
                    // Apply weight decay
                    let decayed_grad = if self.weight_decay > 0.0 {
                        // TODO: Implement mul_scalar
                        // let weight_decay_grad = param_tensor.tensor().mul_scalar(self.weight_decay)?;
                        let weight_decay_grad = param_tensor.tensor().clone();
                        grad.add(&weight_decay_grad)?
                    } else {
                        grad.clone()
                    };
                    
                    // Update parameter based on optimizer type
                    if let Some(state) = self.optimizer_states.get_mut(&param_id) {
                        Self::update_parameter(param_tensor, &decayed_grad, state, self.learning_rate)?;
                    }
                }
            }
        }
        Ok(())
    }
    
    /// Update a single parameter
    fn update_parameter(param_tensor: &mut DiffTensor<T>, grad: &T, state: &mut OptimizerState<T>, learning_rate: f32) -> Result<()> {
        // This is a simplified update - in practice, you'd implement specific optimizers
        // TODO: Implement mul_scalar
        // let update = grad.mul_scalar(-learning_rate)?;
        let update = grad.clone();
        let new_param = param_tensor.tensor().add(&update)?;
        *param_tensor.tensor_mut() = new_param;
        
        state.step_count += 1;
        Ok(())
    }
    
    /// Get training statistics
    pub fn get_stats(&self) -> TrainingStats {
        let mut total_params = 0;
        let mut params_with_grad = 0;
        
        for (_, param_tensor) in self.autograd_ctx.tensors.iter() {
            total_params += 1;
            if param_tensor.grad.is_some() {
                params_with_grad += 1;
            }
        }
        
        TrainingStats {
            step_count: self.step_count,
            total_params,
            params_with_grad,
            learning_rate: self.learning_rate,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainingStats {
    pub step_count: usize,
    pub total_params: usize,
    pub params_with_grad: usize,
    pub learning_rate: f32,
}

/// Learning rate scheduler
#[derive(Debug)]
pub enum LRScheduler {
    Constant(f32),
    Step { initial_lr: f32, step_size: usize, gamma: f32 },
    Exponential { initial_lr: f32, gamma: f32 },
    CosineAnnealing { initial_lr: f32, max_steps: usize },
    WarmupCosine { initial_lr: f32, max_lr: f32, warmup_steps: usize, max_steps: usize },
}

impl LRScheduler {
    pub fn get_lr(&self, step: usize) -> f32 {
        match self {
            LRScheduler::Constant(lr) => *lr,
            LRScheduler::Step { initial_lr, step_size, gamma } => {
                let decay_steps = step / step_size;
                initial_lr * gamma.powi(decay_steps as i32)
            }
            LRScheduler::Exponential { initial_lr, gamma } => {
                initial_lr * gamma.powi(step as i32)
            }
            LRScheduler::CosineAnnealing { initial_lr, max_steps } => {
                if step >= *max_steps {
                    0.0
                } else {
                    let progress = step as f32 / *max_steps as f32;
                    initial_lr * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
                }
            }
            LRScheduler::WarmupCosine { initial_lr, max_lr, warmup_steps, max_steps } => {
                if step < *warmup_steps {
                    // Warmup phase
                    initial_lr + (max_lr - initial_lr) * (step as f32 / *warmup_steps as f32)
                } else if step >= *max_steps {
                    0.0
                } else {
                    // Cosine annealing phase
                    let progress = (step - warmup_steps) as f32 / (*max_steps - warmup_steps) as f32;
                    max_lr * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
                }
            }
        }
    }
}

/// Advanced training loop with all features
#[derive(Debug)]
pub struct AdvancedTrainer<T: Tensor> {
    integrated_trainer: IntegratedTrainer<T>,
    lr_scheduler: LRScheduler,
    gradient_clipper: Option<f32>, // max norm
    mixed_precision: bool,
    checkpoint_every: Option<usize>,
}

impl<T: Tensor + tensor_core::tensor::TensorOps + tensor_core::tensor::TensorRandom> AdvancedTrainer<T> {
    pub fn new(
        learning_rate: f32,
        weight_decay: f32,
        momentum: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        lr_scheduler: LRScheduler,
        gradient_clipper: Option<f32>,
        mixed_precision: bool,
        checkpoint_every: Option<usize>,
    ) -> Self {
        Self {
            integrated_trainer: IntegratedTrainer::new(learning_rate, weight_decay, momentum, beta1, beta2, epsilon),
            lr_scheduler,
            gradient_clipper,
            mixed_precision,
            checkpoint_every,
        }
    }
    
    /// Add a parameter to the trainer
    pub fn add_parameter(&mut self, tensor: T, requires_grad: bool) -> usize {
        self.integrated_trainer.add_parameter(tensor, requires_grad)
    }
    
    /// Execute a complete training step
    pub fn training_step<F, G>(&mut self, forward_fn: F, backward_fn: G) -> Result<TrainingStepResult>
    where
        F: FnOnce(&mut AutogradContext<T>) -> Result<usize>,
        G: FnOnce(&mut AutogradContext<T>, usize) -> Result<()>,
    {
        // Update learning rate
        let current_lr = self.lr_scheduler.get_lr(self.integrated_trainer.step_count);
        self.integrated_trainer.learning_rate = current_lr;
        
        // Forward pass
        let loss_id = self.integrated_trainer.forward(forward_fn)?;
        
        // Backward pass
        backward_fn(&mut self.integrated_trainer.autograd_ctx, loss_id)?;
        
        // Apply gradient clipping if enabled
        if let Some(max_norm) = self.gradient_clipper {
            self.clip_gradients(max_norm)?;
        }
        
        // Update parameters
        self.integrated_trainer.backward_and_step(loss_id)?;
        
        // Checkpoint if needed
        if let Some(every) = self.checkpoint_every {
            if self.integrated_trainer.step_count % every == 0 {
                self.save_checkpoint()?;
            }
        }
        
        Ok(TrainingStepResult {
            loss_id,
            learning_rate: current_lr,
            step_count: self.integrated_trainer.step_count,
            stats: self.integrated_trainer.get_stats(),
        })
    }
    
    /// Clip gradients to prevent exploding gradients
    fn clip_gradients(&mut self, max_norm: f32) -> Result<()> {
        let mut total_norm: f32 = 0.0;
        let mut param_count = 0;
        
        // Calculate total gradient norm
        for (_, param_tensor) in self.integrated_trainer.autograd_ctx.tensors.iter() {
            if let Some(grad) = &param_tensor.grad {
                // TODO: Calculate actual norm
                total_norm += 1.0; // Placeholder
                param_count += 1;
            }
        }
        
        if param_count == 0 {
            return Ok(());
        }
        
        total_norm = total_norm.sqrt();
        
        // Clip if necessary
        if total_norm > max_norm {
            let clip_coef = max_norm / (total_norm + 1e-6);
            
            for (_, param_tensor) in self.integrated_trainer.autograd_ctx.tensors.iter_mut() {
                if let Some(grad) = &mut param_tensor.grad {
                    // TODO: Apply clipping
                    // *grad = grad.mul_scalar(clip_coef)?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Save training checkpoint
    fn save_checkpoint(&self) -> Result<()> {
        // TODO: Implement checkpoint saving
        println!("💾 Saving checkpoint at step {}", self.integrated_trainer.step_count);
        Ok(())
    }
    
    /// Load training checkpoint
    pub fn load_checkpoint(&mut self, path: &str) -> Result<()> {
        // TODO: Implement checkpoint loading
        println!("📂 Loading checkpoint from {}", path);
        Ok(())
    }
    
    /// Get current training statistics
    pub fn get_stats(&self) -> TrainingStats {
        self.integrated_trainer.get_stats()
    }
}

#[derive(Debug, Clone)]
pub struct TrainingStepResult {
    pub loss_id: usize,
    pub learning_rate: f32,
    pub step_count: usize,
    pub stats: TrainingStats,
}
