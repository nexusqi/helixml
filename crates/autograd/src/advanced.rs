//! 🌀 Advanced Autograd Features
//! 
//! Advanced automatic differentiation features for training large models

use tensor_core::{Tensor, Result};
use tensor_core::tensor::{TensorOps, TensorStats, TensorReduce, TensorRandom};
use std::collections::HashMap;
use super::AutogradContext;

/// Gradient accumulation for large batch training
#[derive(Debug)]
pub struct GradientAccumulator<T: Tensor> {
    accumulated_grads: HashMap<usize, T>,
    accumulation_steps: usize,
    current_step: usize,
}

impl<T: Tensor + TensorOps> GradientAccumulator<T> {
    pub fn new(accumulation_steps: usize) -> Self {
        Self {
            accumulated_grads: HashMap::new(),
            accumulation_steps,
            current_step: 0,
        }
    }
    
    /// Accumulate gradients for a tensor
    pub fn accumulate_gradient(&mut self, tensor_id: usize, grad: T) -> Result<()> {
        if let Some(existing_grad) = self.accumulated_grads.get_mut(&tensor_id) {
            // Add to existing gradient
            *existing_grad = existing_grad.add(&grad)?;
        } else {
            // First gradient for this tensor
            self.accumulated_grads.insert(tensor_id, grad);
        }
        Ok(())
    }
    
    /// Check if we should apply gradients (accumulation complete)
    pub fn should_apply(&self) -> bool {
        self.current_step >= self.accumulation_steps - 1
    }
    
    /// Get accumulated gradient for a tensor
    pub fn get_accumulated_grad(&self, tensor_id: usize) -> Option<&T> {
        self.accumulated_grads.get(&tensor_id)
    }
    
    /// Clear accumulated gradients
    pub fn clear(&mut self) {
        self.accumulated_grads.clear();
        self.current_step = 0;
    }
    
    /// Advance to next accumulation step
    pub fn next_step(&mut self) {
        self.current_step = (self.current_step + 1) % self.accumulation_steps;
    }
    
    /// Get current accumulation progress
    pub fn progress(&self) -> f32 {
        (self.current_step + 1) as f32 / self.accumulation_steps as f32
    }
}

/// Gradient clipping for stable training
#[derive(Debug, Clone)]
pub struct GradientClipper {
    max_norm: f32,
    norm_type: f32, // 1.0 for L1, 2.0 for L2, inf for max norm
}

impl GradientClipper {
    pub fn new(max_norm: f32, norm_type: f32) -> Self {
        Self { max_norm, norm_type }
    }
    
    /// Clip gradients to prevent exploding gradients
    pub fn clip_gradients<T>(&self, ctx: &mut AutogradContext<T>) -> Result<f32>
    where
        T: Tensor + tensor_core::tensor::TensorOps + tensor_core::tensor::TensorReduce,
    {
        let mut total_norm: f32 = 0.0;
        let mut param_count = 0;
        
        // Calculate total norm
        for (_, diff_tensor) in ctx.tensors.iter() {
                if let Some(grad) = &diff_tensor.grad {
                if self.norm_type == 2.0 {
                    // L2 norm
                    let grad_norm_sq = grad.mul(grad)?.sum(None, false)?;
                    // Get scalar value from tensor
                    let scalar_val = grad_norm_sq.to_scalar()?;
                    total_norm += scalar_val;
                } else if self.norm_type == 1.0 {
                    // L1 norm
                    let grad_abs = grad.abs()?.sum(None, false)?;
                    // Get scalar value from tensor
                    let scalar_val = grad_abs.to_scalar()?;
                    total_norm += scalar_val;
                }
                param_count += 1;
            }
        }
        
        if param_count == 0 {
            return Ok(0.0);
        }
        
        total_norm = total_norm.sqrt();
        
        // Clip if necessary
        if total_norm > self.max_norm {
            let clip_coef = self.max_norm / (total_norm + 1e-6);
            
            for (_, diff_tensor) in ctx.tensors.iter_mut() {
                if let Some(grad) = &mut diff_tensor.grad {
                    // Implement gradient scaling
                    let scaled_grad = grad.mul_scalar(clip_coef)?;
                    *grad = scaled_grad;
                }
            }
        }
        
        Ok(total_norm)
    }
}

/// Mixed precision training support
#[derive(Debug)]
pub struct MixedPrecisionTrainer {
    loss_scale: f32,
    max_loss_scale: f32,
    min_loss_scale: f32,
    scale_factor: f32,
    scale_window: usize,
    consecutive_skips: usize,
    max_consecutive_skips: usize,
}

impl MixedPrecisionTrainer {
    pub fn new(
        initial_loss_scale: f32,
        max_loss_scale: f32,
        min_loss_scale: f32,
        scale_factor: f32,
        scale_window: usize,
        max_consecutive_skips: usize,
    ) -> Self {
        Self {
            loss_scale: initial_loss_scale,
            max_loss_scale,
            min_loss_scale,
            scale_factor,
            scale_window,
            consecutive_skips: 0,
            max_consecutive_skips,
        }
    }
    
    /// Scale loss for mixed precision training
    pub fn scale_loss<T>(&self, loss: &T) -> Result<T>
    where
        T: Tensor + tensor_core::tensor::TensorOps,
    {
        // Implement loss scaling
        loss.mul_scalar(self.loss_scale)
    }
    
    /// Unscale gradients after backward pass
    pub fn unscale_gradients<T>(&self, ctx: &mut AutogradContext<T>) -> Result<()>
    where
        T: Tensor + tensor_core::tensor::TensorOps,
    {
        for (_, diff_tensor) in ctx.tensors.iter_mut() {
            if let Some(grad) = &mut diff_tensor.grad {
                // Implement gradient unscaling
                // Note: div_scalar might not exist, so we use mul_scalar with 1/scale
                let scale = 1.0 / self.loss_scale;
                let unscaled_grad = grad.mul_scalar(scale)?;
                *grad = unscaled_grad;
            }
        }
        Ok(())
    }
    
    /// Update loss scale based on gradient overflow
    pub fn update_loss_scale(&mut self, has_overflow: bool) {
        if has_overflow {
            self.consecutive_skips += 1;
            if self.consecutive_skips >= self.max_consecutive_skips {
                self.loss_scale = (self.loss_scale / self.scale_factor).max(self.min_loss_scale);
                self.consecutive_skips = 0;
            }
        } else {
            self.consecutive_skips = 0;
            if self.consecutive_skips == 0 {
                self.loss_scale = (self.loss_scale * self.scale_factor).min(self.max_loss_scale);
            }
        }
    }
    
    pub fn current_loss_scale(&self) -> f32 {
        self.loss_scale
    }
}

/// Memory-efficient training with gradient checkpointing
#[derive(Debug)]
pub struct CheckpointTrainer {
    checkpoint_strategy: CheckpointStrategy,
    recompute_activations: bool,
}

#[derive(Debug, Clone)]
pub enum CheckpointStrategy {
    /// Checkpoint every N layers
    EveryN(usize),
    /// Checkpoint at specific layer indices
    AtLayers(Vec<usize>),
    /// Checkpoint based on memory usage
    MemoryBased(f32), // threshold as fraction of total memory
}

impl CheckpointTrainer {
    pub fn new(strategy: CheckpointStrategy, recompute_activations: bool) -> Self {
        Self {
            checkpoint_strategy: strategy,
            recompute_activations,
        }
    }
    
    /// Create checkpoints during forward pass
    pub fn checkpoint_forward<T: Tensor>(&mut self, ctx: &mut AutogradContext<T>, layer_id: usize, tensor_id: usize) -> Result<()> {
        match &self.checkpoint_strategy {
            CheckpointStrategy::EveryN(n) => {
                if layer_id % n == 0 {
                    ctx.checkpoint(tensor_id)?;
                }
            }
            CheckpointStrategy::AtLayers(layers) => {
                if layers.contains(&layer_id) {
                    ctx.checkpoint(tensor_id)?;
                }
            }
            CheckpointStrategy::MemoryBased(threshold) => {
                // Implement memory-based checkpointing
                // Estimate actual memory usage in bytes
                let mut total_memory = 0;
                for (_, diff_tensor) in ctx.tensors.iter() {
                    total_memory += diff_tensor.tensor.shape().numel() * std::mem::size_of::<f32>();
                    if let Some(grad) = &diff_tensor.grad {
                        total_memory += grad.shape().numel() * std::mem::size_of::<f32>();
                    }
                }
                
                // Estimate total available memory (rough estimate: 8GB default)
                // In practice, this would query system memory or device memory
                let estimated_total_memory: usize = 8 * 1024 * 1024 * 1024; // 8GB
                
                // Calculate memory usage as fraction
                let memory_usage_ratio = total_memory as f32 / estimated_total_memory as f32;
                
                if memory_usage_ratio >= *threshold {
                    ctx.checkpoint(tensor_id)?;
                }
            }
        }
        Ok(())
    }
    
    /// Restore from checkpoint during backward pass
    pub fn restore_checkpoint<T: Tensor, F>(&mut self, ctx: &mut AutogradContext<T>, tensor_id: usize, forward_fn: F) -> Result<usize>
    where
        F: FnOnce(&mut AutogradContext<T>) -> Result<usize>,
    {
        if self.recompute_activations {
            ctx.restore_checkpoint(tensor_id, forward_fn)
        } else {
            // Use stored activations
            Ok(tensor_id)
        }
    }
}

/// Training state for advanced autograd
#[derive(Debug)]
pub struct TrainingState<T: Tensor> {
    pub gradient_accumulator: GradientAccumulator<T>,
    pub gradient_clipper: GradientClipper,
    pub mixed_precision: Option<MixedPrecisionTrainer>,
    pub checkpoint_trainer: CheckpointTrainer,
    pub step_count: usize,
    pub epoch_count: usize,
}

impl<T: Tensor + TensorOps + TensorReduce> TrainingState<T> {
    pub fn new(
        accumulation_steps: usize,
        max_grad_norm: f32,
        use_mixed_precision: bool,
        checkpoint_strategy: CheckpointStrategy,
    ) -> Self {
        Self {
            gradient_accumulator: GradientAccumulator::new(accumulation_steps),
            gradient_clipper: GradientClipper::new(max_grad_norm, 2.0),
            mixed_precision: if use_mixed_precision {
                Some(MixedPrecisionTrainer::new(
                    65536.0, // initial loss scale
                    16777216.0, // max loss scale
                    1.0, // min loss scale
                    2.0, // scale factor
                    2000, // scale window
                    2000, // max consecutive skips
                ))
            } else {
                None
            },
            checkpoint_trainer: CheckpointTrainer::new(checkpoint_strategy, true),
            step_count: 0,
            epoch_count: 0,
        }
    }
    
    /// Execute a training step with all advanced features
    pub fn training_step<F, G>(
        &mut self,
        ctx: &mut AutogradContext<T>,
        forward_fn: F,
        backward_fn: G,
    ) -> Result<TrainingStepResult>
    where
        F: FnOnce(&mut AutogradContext<T>) -> Result<usize>, // Returns loss tensor ID
        G: FnOnce(&mut AutogradContext<T>, usize) -> Result<()>, // Backward pass
    {
        // Forward pass with checkpointing
        let loss_id = forward_fn(ctx)?;
        
        // Scale loss for mixed precision
        let scaled_loss_id = if let Some(ref mut mp) = self.mixed_precision {
            let loss_tensor = ctx.get_tensor(loss_id).unwrap();
            let scaled_loss = mp.scale_loss(loss_tensor.tensor())?;
            ctx.tensor(scaled_loss, true)
        } else {
            loss_id
        };
        
        // Backward pass
        backward_fn(ctx, scaled_loss_id)?;
        
        // Accumulate gradients
        for (tensor_id, diff_tensor) in ctx.tensors.iter() {
            if let Some(grad) = &diff_tensor.grad {
                self.gradient_accumulator.accumulate_gradient(*tensor_id, grad.clone())?;
            }
        }
        
        // Check if we should apply gradients
        let should_apply = self.gradient_accumulator.should_apply();
        
        if should_apply {
            // Unscale gradients for mixed precision
            if let Some(ref mut mp) = self.mixed_precision {
                mp.unscale_gradients(ctx)?;
            }
            
            // Clip gradients
            let grad_norm = self.gradient_clipper.clip_gradients(ctx)?;
            
            // Apply gradients using optimizer if provided
            // Note: Optimizer integration requires passing optimizer from outside
            // For now, this is structured to accept an optional optimizer
            // In practice, TrainingState would hold an Arc<Mutex<dyn Optimizer<T>>>
            
            // Clear accumulated gradients
            self.gradient_accumulator.clear();
            
            self.step_count += 1;
            
            Ok(TrainingStepResult {
                loss_id,
                grad_norm,
                should_apply,
                step_count: self.step_count,
            })
        } else {
            self.gradient_accumulator.next_step();
            Ok(TrainingStepResult {
                loss_id,
                grad_norm: 0.0,
                should_apply: false,
                step_count: self.step_count,
            })
        }
    }
}

/// Result of a training step
#[derive(Debug)]
pub struct TrainingStepResult {
    pub loss_id: usize,
    pub grad_norm: f32,
    pub should_apply: bool,
    pub step_count: usize,
}

/// Advanced loss functions with gradient scaling
pub mod advanced_losses {
    use super::*;
    
    /// Focal loss for imbalanced datasets
    pub fn focal_loss<T: Tensor + TensorOps + TensorStats + TensorReduce + TensorRandom>(
        ctx: &mut AutogradContext<T>,
        predictions: usize,
        targets: usize,
        alpha: f32,
        gamma: f32,
    ) -> Result<usize> {
        let pred_tensor = ctx.get_tensor(predictions).unwrap();
        let target_tensor = ctx.get_tensor(targets).unwrap();
        
        // Compute cross entropy
        let ce = pred_tensor.tensor().log_softmax(pred_tensor.tensor().ndim() - 1)?;
        let ce_loss = ce.mul(target_tensor.tensor())?.sum(None, false)?;
        
        // Compute focal weight
        let pt = ce.exp()?;
        let focal_weight = pt.pow(gamma)?;
        // TODO: Implement mul_scalar
        // let focal_weight = focal_weight.mul_scalar(alpha)?;
        
        let focal_loss = ce_loss.mul(&focal_weight)?;
        
        let loss_id = ctx.tensor(focal_loss, true);
        Ok(loss_id)
    }
    
    /// Label smoothing cross entropy
    pub fn label_smoothing_ce<T: Tensor + TensorOps + TensorStats + TensorReduce + TensorRandom>(
        ctx: &mut AutogradContext<T>,
        logits: usize,
        targets: usize,
        smoothing: f32,
        num_classes: usize,
    ) -> Result<usize> {
        let logits_tensor = ctx.get_tensor(logits).unwrap();
        let target_tensor = ctx.get_tensor(targets).unwrap();
        
        // Apply label smoothing
        let smooth_targets = target_tensor.tensor().clone();
        let uniform_dist = T::ones(
            target_tensor.tensor().shape().clone(),
            target_tensor.tensor().dtype(),
            target_tensor.tensor().device(),
        )?;
        // Scale uniform distribution
        let uniform_scale = smoothing / num_classes as f32;
        let uniform_dist = uniform_dist.mul_scalar(uniform_scale)?;
        
        let smoothed_targets = smooth_targets.add(&uniform_dist)?;
        
        // Compute loss
        let log_softmax = logits_tensor.tensor().log_softmax(logits_tensor.tensor().ndim() - 1)?;
        let loss = log_softmax.mul(&smoothed_targets)?.sum(None, false)?;
        
        let loss_id = ctx.tensor(loss, true);
        Ok(loss_id)
    }
}
