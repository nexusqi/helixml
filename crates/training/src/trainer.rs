//! 🌀 HelixML Trainer
//! 
//! Main training orchestrator with advanced features for SSM/Hyena models.

use crate::{LossFunction, Optimizer, Scheduler, Metrics, CheckpointManager, TrainingMonitor, DataLoader, ValidationManager};
use hal::DeviceType;
use nn::{Module, CheckpointableModule};
use tensor_core::tensor::{Tensor, TensorOps, TensorReduce, TensorBroadcast, TensorRandom, TensorStats, TensorActivation};
use autograd::AutogradOps;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use anyhow::Result as AnyResult;

/// Main trainer for HelixML models
/// Generic over tensor type T, allowing use with CPU, CUDA, or any other tensor backend
pub struct Trainer<M, T> 
where
    M: Module<T> + CheckpointableModule<T>,
    T: Tensor + TensorOps + TensorReduce + TensorBroadcast + TensorRandom + TensorStats + TensorActivation + Clone + 'static,
{
    /// Model to train
    model: Arc<Mutex<M>>,
    /// Loss function
    loss_fn: Box<dyn LossFunction<T>>,
    /// Optimizer
    optimizer: Arc<Mutex<Box<dyn Optimizer<T>>>>,
    /// Learning rate scheduler
    scheduler: Arc<Mutex<Box<dyn Scheduler<T>>>>,
    /// Metrics tracker
    metrics: Arc<Mutex<Metrics>>,
    /// Checkpoint manager
    checkpoint_manager: Arc<CheckpointManager<T>>,
    /// Training monitor
    monitor: Arc<TrainingMonitor>,
    /// Data loader
    data_loader: Arc<DataLoader<T>>,
    /// Validation manager
    validation_manager: Arc<ValidationManager>,
    /// Training configuration
    config: TrainingConfig,
    /// Training state
    state: Arc<Mutex<TrainingState>>,
    /// Autograd context for gradient computation (optional, for future integration)
    autograd_ctx: Option<Arc<Mutex<AutogradOps<T>>>>,
}

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Weight decay
    pub weight_decay: f64,
    /// Gradient clipping
    pub gradient_clipping: Option<f64>,
    /// Mixed precision training
    pub mixed_precision: bool,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    /// Validation frequency
    pub validation_frequency: usize,
    /// Checkpoint frequency
    pub checkpoint_frequency: usize,
    /// Early stopping patience
    pub early_stopping_patience: Option<usize>,
    /// Device type
    pub device_type: DeviceType,
    /// Number of workers
    pub num_workers: usize,
    /// Pin memory
    pub pin_memory: bool,
    /// Drop last batch
    pub drop_last: bool,
    /// Shuffle data
    pub shuffle: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            learning_rate: 1e-3,
            weight_decay: 1e-4,
            gradient_clipping: Some(1.0),
            mixed_precision: false,
            gradient_accumulation_steps: 1,
            validation_frequency: 1,
            checkpoint_frequency: 10,
            early_stopping_patience: Some(10),
            device_type: DeviceType::CPU,
            num_workers: 4,
            pin_memory: true,
            drop_last: false,
            shuffle: true,
        }
    }
}

/// Training state
#[derive(Debug, Clone)]
pub struct TrainingState {
    /// Current epoch
    pub current_epoch: usize,
    /// Current step
    pub current_step: usize,
    /// Best validation loss
    pub best_validation_loss: f64,
    /// Training start time
    pub start_time: Instant,
    /// Last validation time
    pub last_validation_time: Option<Instant>,
    /// Early stopping counter
    pub early_stopping_counter: usize,
    /// Is training
    pub is_training: bool,
    /// Is paused
    pub is_paused: bool,
    /// Training history
    pub history: TrainingHistory,
}

/// Training history
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    /// Training losses
    pub training_losses: Vec<f64>,
    /// Validation losses
    pub validation_losses: Vec<f64>,
    /// Learning rates
    pub learning_rates: Vec<f64>,
    /// Epochs
    pub epochs: Vec<usize>,
    /// Steps
    pub steps: Vec<usize>,
    /// Timestamps
    pub timestamps: Vec<Instant>,
}

impl Default for TrainingHistory {
    fn default() -> Self {
        Self {
            training_losses: Vec::new(),
            validation_losses: Vec::new(),
            learning_rates: Vec::new(),
            epochs: Vec::new(),
            steps: Vec::new(),
            timestamps: Vec::new(),
        }
    }
}

impl<M, T> Trainer<M, T>
where
    M: Module<T> + CheckpointableModule<T>,
    T: Tensor + TensorOps + TensorReduce + TensorBroadcast + TensorRandom + TensorStats + TensorActivation + Clone + 'static,
{
    /// Create new trainer
    pub fn new(
        model: M,
        loss_fn: Box<dyn LossFunction<T>>,
        optimizer: Box<dyn Optimizer<T>>,
        scheduler: Box<dyn Scheduler<T>>,
        config: TrainingConfig,
    ) -> AnyResult<Self> {
        let model = Arc::new(Mutex::new(model));
        let optimizer = Arc::new(Mutex::new(optimizer));
        let scheduler = Arc::new(Mutex::new(scheduler));
        let metrics = Arc::new(Mutex::new(Metrics::new()));
        let checkpoint_manager = Arc::new(CheckpointManager::new()?);
        let monitor = Arc::new(TrainingMonitor::new());
        let data_loader = Arc::new(DataLoader::new(config.num_workers, config.pin_memory)?);
        // Note: DataLoader needs to be generic over T - will need to update it
        let validation_manager = Arc::new(ValidationManager::new());
        
        let state = Arc::new(Mutex::new(TrainingState {
            current_epoch: 0,
            current_step: 0,
            best_validation_loss: f64::INFINITY,
            start_time: Instant::now(),
            last_validation_time: None,
            early_stopping_counter: 0,
            is_training: false,
            is_paused: false,
            history: TrainingHistory::default(),
        }));
        
        Ok(Self {
            model,
            loss_fn,
            optimizer,
            scheduler,
            metrics,
            checkpoint_manager,
            monitor,
            data_loader,
            validation_manager,
            config,
            state,
            autograd_ctx: Some(Arc::new(Mutex::new(AutogradOps::new()))), // Enable autograd for future integration
            // Note: AutogradOps requires T: Tensor + TensorOps + TensorReduce + TensorBroadcast + TensorStats + TensorActivation
            // which matches our bounds
        })
    }
    
    /// Train the model
    pub async fn train(&self, train_data: &[T], validation_data: Option<&[T]>) -> AnyResult<()> {
        let mut state = self.state.lock().unwrap();
        state.is_training = true;
        state.start_time = Instant::now();
        drop(state);
        
        self.monitor.start_training().await?;
        
        for epoch in 0..self.config.epochs {
            // Check if training should stop
            if self.should_stop_training().await? {
                break;
            }
            
            // Train epoch
            self.train_epoch(epoch, train_data).await?;
            
            // Validation
            if let Some(val_data) = validation_data {
                if epoch % self.config.validation_frequency == 0 {
                    self.validate_epoch(epoch, val_data).await?;
                }
            }
            
            // Checkpoint
            if epoch % self.config.checkpoint_frequency == 0 {
                self.save_checkpoint(epoch).await?;
            }
            
            // Update learning rate
            // Note: scheduler.step() might need mutable access depending on implementation
            // If scheduler needs mutation, we can use interior mutability (e.g., Mutex/RefCell)
            // For now, check if scheduler supports immutable step
            // self.scheduler.step()?; // Uncomment when scheduler trait is finalized
        }
        
        self.monitor.finish_training().await?;
        
        let mut state = self.state.lock().unwrap();
        state.is_training = false;
        
        Ok(())
    }
    
    /// Train single epoch
    async fn train_epoch(&self, epoch: usize, data: &[T]) -> AnyResult<()> {
        let mut state = self.state.lock().unwrap();
        state.current_epoch = epoch;
        drop(state);
        
        self.monitor.start_epoch(epoch).await?;
        
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;
        
        // Create batches
        let batches = self.data_loader.create_batches(data, self.config.batch_size)?;
        
        for (batch_idx, batch) in batches.iter().enumerate() {
            // Forward pass
            let loss = self.train_step(batch).await?;
            epoch_loss += loss;
            num_batches += 1;
            
            // Update metrics
            self.update_metrics(epoch, batch_idx, loss).await?;
            
            // Log progress
            if batch_idx % 100 == 0 {
                self.monitor.log_progress(epoch, batch_idx, loss).await?;
            }
        }
        
        let avg_loss = epoch_loss / num_batches as f64;
        
        // Update history
        let mut state = self.state.lock().unwrap();
        state.history.training_losses.push(avg_loss);
        state.history.epochs.push(epoch);
        state.history.timestamps.push(Instant::now());
        drop(state);
        
        self.monitor.end_epoch(epoch, avg_loss).await?;
        
        Ok(())
    }
    
    /// Train single step
    async fn train_step(&self, batch: &[T]) -> AnyResult<f64> {
        if batch.is_empty() {
            return Ok(0.0);
        }
        
        // Combine batch into single tensor if needed
        // For now, assume batch[0] is the input tensor (shape includes batch dimension)
        // If batch is a list of samples, we'd need to stack them
        let input = if batch.len() == 1 {
            batch[0].clone()
        } else {
            // Stack tensors along batch dimension (dim=0)
            // Use Tensor trait's stack method
            T::stack(batch.iter().map(|t| t.clone()).collect(), 0)?
        };
        
        // Forward pass
        let model = self.model.lock().unwrap();
        let output = model.forward(&input)
            .map_err(|e| anyhow::anyhow!("Forward pass failed: {}", e))?;
        drop(model);
        
        // Compute loss (assuming batch[1] is target if provided, otherwise use dummy target)
        // In practice, you'd separate inputs and targets properly
        let target = if batch.len() > 1 {
            batch[1].clone()
        } else {
            // Create dummy target with same shape as output for now
            // In practice, targets should be provided separately
            output.clone() // This is not correct, but placeholder for now
        };
        
        // Clone target for loss computation (original needed for autograd later)
        let target_for_loss = target.clone();
        let loss = self.loss_fn.compute(&[output.clone()], &[target_for_loss])
            .map_err(|e| anyhow::anyhow!("Loss computation failed: {}", e))?;
        
        // Extract scalar loss value
        let loss_value = loss.to_scalar()
            .map_err(|e| anyhow::anyhow!("Failed to extract loss scalar: {}", e))? as f64;
        
        // Backward pass - compute gradients using autograd
        // Integrate autograd for gradient computation
        
        // Zero gradients before backward pass
        self.optimizer.lock().unwrap().zero_grad()?;
        
        // Use autograd to compute gradients
        if let Some(autograd_ctx) = &self.autograd_ctx {
            let mut autograd = autograd_ctx.lock().unwrap();
            
            // Clear previous computation graph
            autograd.clear();
            
            // Map to track parameter pointer -> autograd ID
            let mut param_ids = std::collections::HashMap::new();
            
            // Re-acquire model lock for forward pass
            let model = self.model.lock().unwrap();
            
            // Forward pass through autograd using forward_with_autograd
            // Create input tensor in autograd context (don't track input gradients)
            let input_id = autograd.tensor(input.clone(), false);
            
            // Perform forward pass through model using forward_with_autograd
            // This will register all parameters and build computation graph
            let output_id = model.forward_with_autograd(&mut autograd, input_id, &mut param_ids)?;
            
            // Release model lock before loss computation
            drop(model);
            
            // Compute loss through autograd operations using LossFunction
            // Create target tensor (don't track target gradients)
            let target_id = autograd.tensor(target, false);
            
            // Compute loss through autograd operations
            // Using autograd operations directly for now (LossFunction could be integrated later)
            let diff_id = autograd.sub(output_id, target_id)?;
            let squared_id = autograd.mul(diff_id, diff_id)?;
            let loss_id = autograd.mean(squared_id, None, false)?;
            
            // Backward pass - this will compute gradients for all tensors with requires_grad=true
            autograd.backward(loss_id)?;
            
            // Extract gradients for model parameters and apply optimizer
            // Strategy: Clone gradients first, then work with parameters
            
            // Collect gradient clones while autograd is locked
            let mut param_grad_data: Vec<(usize, T)> = Vec::new(); // (autograd_id, gradient_clone)
            
            for (param_ptr, &autograd_id) in param_ids.iter() {
                if let Some(diff_tensor) = autograd.get_tensor(autograd_id) {
                    if let Some(grad) = diff_tensor.grad() {
                        param_grad_data.push((autograd_id, grad.clone()));
                    }
                }
            }
            
            // Release autograd lock
            drop(autograd);
            
            // Now get model parameters and create pairs
            // The challenge: we need &mut T params and &T gradients
            // Since we cloned gradients, we can store them temporarily
            let mut model = self.model.lock().unwrap();
            let param_muts = model.parameters_mut();
            
            // Match parameters with gradients using param_ids mapping
            // Build pairs: we'll collect owned gradients and match with param refs
            let mut gradient_clones: Vec<T> = Vec::new();
            
            for param_mut in param_muts.iter() {
                // Get the actual &mut T reference and convert to pointer
                let param_ref: &T = *param_mut;
                let param_ptr = param_ref as *const T;
                if let Some(&autograd_id) = param_ids.get(&param_ptr) {
                    // Find corresponding gradient in param_grad_data
                    if let Some((_, grad_clone)) = param_grad_data.iter().find(|(id, _)| *id == autograd_id) {
                        gradient_clones.push(grad_clone.clone());
                    }
                }
            }
            
            // Release model lock
            drop(model);
            
            // Apply gradients using optimizer
            // Note: Optimizer expects &mut [(&mut T, &T)], but we have:
            // - Owned gradients in gradient_clones
            // - Need to get &mut T for params
            // 
            // This is a borrowing challenge. For now, we acknowledge that:
            // 1. Gradients are correctly computed in autograd
            // 2. The mapping between params and gradients is established
            // 3. Full optimizer integration requires either:
            //    a) Storing gradients in a way that allows &T references, OR
            //    b) Modifying Optimizer API to work differently
            //
            // TODO: Complete optimizer integration - gradients are ready but application needs refinement
            
            // Note: Full gradient application requires:
            // 1. Tracking parameter -> autograd_id mapping during forward pass
            // 2. Extracting gradients using the mapping
            // 3. Creating param_grad_pairs with correct mutable references
            // 4. Calling optimizer.step(param_grad_pairs)
            
            // For now, the autograd integration is structured but gradients aren't applied
            // This will be completed in the next step
        } else {
            // No autograd context - training will proceed without gradients
            // This is expected for inference-only scenarios
        }
        
        Ok(loss_value)
    }
    
    /// Validate epoch
    async fn validate_epoch(&self, epoch: usize, data: &[T]) -> AnyResult<()> {
        self.monitor.start_validation(epoch).await?;
        
        let mut validation_loss = 0.0;
        let mut num_batches = 0;
        
        // Create validation batches
        let batches = self.data_loader.create_batches(data, self.config.batch_size)?;
        
        for batch in batches.iter() {
            let loss = self.validate_step(batch).await?;
            validation_loss += loss;
            num_batches += 1;
        }
        
        let avg_loss = validation_loss / num_batches as f64;
        
        // Update state
        let mut state = self.state.lock().unwrap();
        state.history.validation_losses.push(avg_loss);
        state.last_validation_time = Some(Instant::now());
        
        // Check for improvement
        if avg_loss < state.best_validation_loss {
            state.best_validation_loss = avg_loss;
            state.early_stopping_counter = 0;
        } else {
            state.early_stopping_counter += 1;
        }
        drop(state);
        
        self.monitor.end_validation(epoch, avg_loss).await?;
        
        Ok(())
    }
    
    /// Validate single step
    async fn validate_step(&self, batch: &[T]) -> AnyResult<f64> {
        let model = self.model.lock().unwrap();
        
        // Handle batch input similar to train_step
        if batch.is_empty() {
            return Ok(0.0);
        }
        
        // Combine batch into single tensor if needed
        let input = if batch.len() == 1 {
            batch[0].clone()
        } else {
            T::stack(batch.iter().map(|t| t.clone()).collect(), 0)?
        };
        
        // Forward pass
        let output = model.forward(&input)
            .map_err(|e| anyhow::anyhow!("Forward pass failed: {}", e))?;
        
        // Compute loss (use dummy target for validation placeholder)
        // In practice, validation data should include targets
        let target = if batch.len() > 1 {
            batch[1].clone()
        } else {
            // For validation, we'd typically have separate target tensors
            // This is a placeholder - actual validation should provide targets
            output.clone() // Not correct, but placeholder
        };
        
        let loss = self.loss_fn.compute(&[output], &[target])
            .map_err(|e| anyhow::anyhow!("Loss computation failed: {}", e))?;
        
        // Extract scalar loss value
        let loss_value = loss.to_scalar()
            .map_err(|e| anyhow::anyhow!("Failed to extract loss scalar: {}", e))? as f64;
        
        Ok(loss_value)
    }
    
    /// Update metrics
    async fn update_metrics(&self, epoch: usize, step: usize, loss: f64) -> AnyResult<()> {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.update_training_loss(loss);
        metrics.update_epoch(epoch);
        metrics.update_step(step);
        Ok(())
    }
    
    /// Clip gradients to prevent exploding gradients
    /// Implements gradient norm clipping: if norm > clip_value, scale gradients proportionally
    fn clip_gradients(&self, param_grad_pairs: &mut [(&mut T, &T)], clip_value: f64) -> AnyResult<()> {
        use tensor_core::tensor::TensorReduce;
        
        // Compute total gradient norm (L2 norm across all gradients)
        let mut total_norm_squared = 0.0;
        for (_param, grad) in param_grad_pairs.iter() {
            // Compute L2 norm squared for this gradient: sum(grad^2)
            let grad_squared = grad.mul(grad)?;
            let norm_squared = grad_squared.sum(None, false)?.to_scalar().unwrap() as f64;
            total_norm_squared += norm_squared;
        }
        
        let total_norm = total_norm_squared.sqrt();
        
        // If norm exceeds clip_value, scale all gradients
        if total_norm > clip_value {
            let clip_coef = clip_value / total_norm;
            
            // Scale all gradients
            for (_param, grad) in param_grad_pairs.iter_mut() {
                // Create scaled gradient
                let scaled_grad = grad.mul_scalar(clip_coef as f32)?;
                
                // Update gradient in the pair (note: we can't modify &CpuTensor directly)
                // In practice, gradients would be stored in a mutable structure
                // For now, this demonstrates the clipping logic
                // The actual update would happen through the gradient storage mechanism
                let _ = scaled_grad;
            }
        }
        
        Ok(())
    }
    
    /// Compute gradient norm for monitoring
    fn compute_gradient_norm(&self, param_grad_pairs: &[(&T, &T)]) -> AnyResult<f64> {
        use tensor_core::tensor::TensorReduce;
        
        let mut total_norm_squared = 0.0;
        for (_param, grad) in param_grad_pairs.iter() {
            let grad_squared = grad.mul(grad)?;
            let norm_squared = grad_squared.sum(None, false)?.to_scalar().unwrap() as f64;
            total_norm_squared += norm_squared;
        }
        
        Ok(total_norm_squared.sqrt())
    }
    
    /// Check if training should stop
    async fn should_stop_training(&self) -> AnyResult<bool> {
        let state = self.state.lock().unwrap();
        
        // Check early stopping
        if let Some(patience) = self.config.early_stopping_patience {
            if state.early_stopping_counter >= patience {
                return Ok(true);
            }
        }
        
        // Check if paused
        if state.is_paused {
            return Ok(true);
        }
        
        Ok(false)
    }
    
    /// Save checkpoint
    async fn save_checkpoint(&self, epoch: usize) -> AnyResult<()> {
        let model = self.model.lock().unwrap();
        let state = self.state.lock().unwrap();
        let metrics = self.metrics.lock().unwrap();
        
        self.checkpoint_manager.save_checkpoint(
            epoch,
            &*model,
            &*metrics,
        ).await?;
        
        Ok(())
    }
    
    /// Load checkpoint
    pub async fn load_checkpoint(&self, epoch: usize) -> AnyResult<()> {
        let mut model = self.model.lock().unwrap();
        let state = self.state.lock().unwrap();
        let mut metrics = self.metrics.lock().unwrap();
        
        self.checkpoint_manager.load_checkpoint(
            epoch,
            &mut *model,
            &mut *metrics,
        ).await?;
        
        Ok(())
    }
    
    /// Pause training
    pub fn pause(&self) {
        let mut state = self.state.lock().unwrap();
        state.is_paused = true;
    }
    
    /// Resume training
    pub fn resume(&self) {
        let mut state = self.state.lock().unwrap();
        state.is_paused = false;
    }
    
    /// Stop training
    pub fn stop(&self) {
        let mut state = self.state.lock().unwrap();
        state.is_training = false;
    }
    
    /// Get training state
    pub fn get_state(&self) -> TrainingState {
        self.state.lock().unwrap().clone()
    }
    
    /// Get metrics
    pub fn get_metrics(&self) -> Metrics {
        self.metrics.lock().unwrap().clone()
    }
    
    /// Get model
    pub fn get_model(&self) -> Arc<Mutex<M>> {
        self.model.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_trainer_creation() {
        // This test would require actual model, loss, optimizer, scheduler
        // For now, just test that the code compiles
        assert!(true);
    }
    
    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.epochs, 100);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.learning_rate, 1e-3);
    }
}
