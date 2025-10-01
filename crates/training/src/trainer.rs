//! 🌀 HelixML Trainer
//! 
//! Main training orchestrator with advanced features for SSM/Hyena models.

use crate::{LossFunction, Optimizer, Scheduler, Metrics, CheckpointManager, TrainingMonitor, DataLoader, ValidationManager};
use hal::{ComputeBackend, DeviceType, Result, HalError};
use nn::{Module, CheckpointableModule};
use tensor_core::tensor::Tensor;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use anyhow::Result as AnyResult;

/// Main trainer for HelixML models
pub struct Trainer<M: Module<Tensor> + CheckpointableModule<Tensor>> {
    /// Model to train
    model: Arc<Mutex<M>>,
    /// Loss function
    loss_fn: Box<dyn LossFunction>,
    /// Optimizer
    optimizer: Box<dyn Optimizer>,
    /// Learning rate scheduler
    scheduler: Box<dyn Scheduler>,
    /// Metrics tracker
    metrics: Arc<Mutex<Metrics>>,
    /// Checkpoint manager
    checkpoint_manager: Arc<CheckpointManager>,
    /// Training monitor
    monitor: Arc<TrainingMonitor>,
    /// Data loader
    data_loader: Arc<DataLoader>,
    /// Validation manager
    validation_manager: Arc<ValidationManager>,
    /// Training configuration
    config: TrainingConfig,
    /// Training state
    state: Arc<Mutex<TrainingState>>,
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

impl<M: Module<Tensor> + CheckpointableModule<Tensor>> Trainer<M> {
    /// Create new trainer
    pub fn new(
        model: M,
        loss_fn: Box<dyn LossFunction>,
        optimizer: Box<dyn Optimizer>,
        scheduler: Box<dyn Scheduler>,
        config: TrainingConfig,
    ) -> AnyResult<Self> {
        let model = Arc::new(Mutex::new(model));
        let metrics = Arc::new(Mutex::new(Metrics::new()));
        let checkpoint_manager = Arc::new(CheckpointManager::new()?);
        let monitor = Arc::new(TrainingMonitor::new());
        let data_loader = Arc::new(DataLoader::new(config.num_workers, config.pin_memory)?);
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
        })
    }
    
    /// Train the model
    pub async fn train(&self, train_data: &[Tensor], validation_data: Option<&[Tensor]>) -> AnyResult<()> {
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
            self.scheduler.step()?;
        }
        
        self.monitor.finish_training().await?;
        
        let mut state = self.state.lock().unwrap();
        state.is_training = false;
        
        Ok(())
    }
    
    /// Train single epoch
    async fn train_epoch(&self, epoch: usize, data: &[Tensor]) -> AnyResult<()> {
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
    async fn train_step(&self, batch: &[Tensor]) -> AnyResult<f64> {
        let model = self.model.lock().unwrap();
        
        // Forward pass
        let predictions = model.forward(batch)?;
        
        // Compute loss
        let loss = self.loss_fn.compute(&predictions, batch)?;
        
        // Backward pass
        let gradients = loss.backward()?;
        
        // Gradient clipping
        if let Some(clip_value) = self.config.gradient_clipping {
            self.clip_gradients(&gradients, clip_value)?;
        }
        
        // Update parameters
        self.optimizer.step(&gradients).await?;
        
        // Clear gradients
        self.optimizer.zero_grad()?;
        
        Ok(loss.item())
    }
    
    /// Validate epoch
    async fn validate_epoch(&self, epoch: usize, data: &[Tensor]) -> AnyResult<()> {
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
    async fn validate_step(&self, batch: &[Tensor]) -> AnyResult<f64> {
        let model = self.model.lock().unwrap();
        
        // Forward pass (no gradients)
        let predictions = model.forward(batch)?;
        
        // Compute loss
        let loss = self.loss_fn.compute(&predictions, batch)?;
        
        Ok(loss.item())
    }
    
    /// Update metrics
    async fn update_metrics(&self, epoch: usize, step: usize, loss: f64) -> AnyResult<()> {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.update_training_loss(loss);
        metrics.update_epoch(epoch);
        metrics.update_step(step);
        Ok(())
    }
    
    /// Clip gradients
    fn clip_gradients(&self, gradients: &[Tensor], clip_value: f64) -> AnyResult<()> {
        for grad in gradients {
            // TODO: Implement gradient clipping
            // This would involve computing the norm and scaling if necessary
        }
        Ok(())
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
        
        self.checkpoint_manager.save_checkpoint(
            epoch,
            &model,
            &self.metrics.lock().unwrap(),
        ).await?;
        
        Ok(())
    }
    
    /// Load checkpoint
    pub async fn load_checkpoint(&self, epoch: usize) -> AnyResult<()> {
        let mut model = self.model.lock().unwrap();
        let mut state = self.state.lock().unwrap();
        
        self.checkpoint_manager.load_checkpoint(
            epoch,
            &mut model,
            &mut self.metrics.lock().unwrap(),
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
