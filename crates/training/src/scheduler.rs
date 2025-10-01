//! 🌀 HelixML Learning Rate Schedulers
//! 
//! Advanced learning rate scheduling for optimal training.

use std::collections::HashMap;
use anyhow::Result as AnyResult;

/// Trait for learning rate schedulers
pub trait Scheduler: Send + Sync {
    /// Step the scheduler
    fn step(&mut self) -> AnyResult<()>;
    
    /// Get current learning rate
    fn get_learning_rate(&self) -> f64;
    
    /// Get scheduler name
    fn name(&self) -> &str;
    
    /// Get scheduler parameters
    fn parameters(&self) -> HashMap<String, f64>;
}

/// Step LR scheduler
pub struct StepLR {
    /// Initial learning rate
    initial_lr: f64,
    /// Current learning rate
    current_lr: f64,
    /// Step size
    step_size: usize,
    /// Gamma (decay factor)
    gamma: f64,
    /// Current step
    step: usize,
}

impl StepLR {
    /// Create new StepLR scheduler
    pub fn new(initial_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self {
            initial_lr,
            current_lr: initial_lr,
            step_size,
            gamma,
            step: 0,
        }
    }
}

impl Scheduler for StepLR {
    fn step(&mut self) -> AnyResult<()> {
        self.step += 1;
        
        if self.step % self.step_size == 0 {
            self.current_lr *= self.gamma;
        }
        
        Ok(())
    }
    
    fn get_learning_rate(&self) -> f64 {
        self.current_lr
    }
    
    fn name(&self) -> &str {
        "StepLR"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("initial_lr".to_string(), self.initial_lr);
        params.insert("step_size".to_string(), self.step_size as f64);
        params.insert("gamma".to_string(), self.gamma);
        params
    }
}

/// Exponential LR scheduler
pub struct ExponentialLR {
    /// Initial learning rate
    initial_lr: f64,
    /// Current learning rate
    current_lr: f64,
    /// Gamma (decay factor)
    gamma: f64,
    /// Current step
    step: usize,
}

impl ExponentialLR {
    /// Create new ExponentialLR scheduler
    pub fn new(initial_lr: f64, gamma: f64) -> Self {
        Self {
            initial_lr,
            current_lr: initial_lr,
            gamma,
            step: 0,
        }
    }
}

impl Scheduler for ExponentialLR {
    fn step(&mut self) -> AnyResult<()> {
        self.step += 1;
        self.current_lr = self.initial_lr * self.gamma.powi(self.step as i32);
        Ok(())
    }
    
    fn get_learning_rate(&self) -> f64 {
        self.current_lr
    }
    
    fn name(&self) -> &str {
        "ExponentialLR"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("initial_lr".to_string(), self.initial_lr);
        params.insert("gamma".to_string(), self.gamma);
        params
    }
}

/// Cosine Annealing LR scheduler
pub struct CosineAnnealingLR {
    /// Initial learning rate
    initial_lr: f64,
    /// Current learning rate
    current_lr: f64,
    /// Minimum learning rate
    min_lr: f64,
    /// Total number of steps
    total_steps: usize,
    /// Current step
    step: usize,
}

impl CosineAnnealingLR {
    /// Create new CosineAnnealingLR scheduler
    pub fn new(initial_lr: f64, total_steps: usize, min_lr: f64) -> Self {
        Self {
            initial_lr,
            current_lr: initial_lr,
            min_lr,
            total_steps,
            step: 0,
        }
    }
}

impl Scheduler for CosineAnnealingLR {
    fn step(&mut self) -> AnyResult<()> {
        self.step += 1;
        
        let progress = self.step as f64 / self.total_steps as f64;
        let cosine_factor = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
        
        self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_factor;
        
        Ok(())
    }
    
    fn get_learning_rate(&self) -> f64 {
        self.current_lr
    }
    
    fn name(&self) -> &str {
        "CosineAnnealingLR"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("initial_lr".to_string(), self.initial_lr);
        params.insert("min_lr".to_string(), self.min_lr);
        params.insert("total_steps".to_string(), self.total_steps as f64);
        params
    }
}

/// ReduceLROnPlateau scheduler
pub struct ReduceLROnPlateau {
    /// Initial learning rate
    initial_lr: f64,
    /// Current learning rate
    current_lr: f64,
    /// Factor to reduce LR
    factor: f64,
    /// Patience
    patience: usize,
    /// Minimum learning rate
    min_lr: f64,
    /// Current patience counter
    patience_counter: usize,
    /// Best metric value
    best_metric: f64,
    /// Mode (min or max)
    mode: PlateauMode,
}

/// Plateau mode
#[derive(Debug, Clone)]
pub enum PlateauMode {
    Min,
    Max,
}

impl ReduceLROnPlateau {
    /// Create new ReduceLROnPlateau scheduler
    pub fn new(initial_lr: f64, factor: f64, patience: usize, min_lr: f64, mode: PlateauMode) -> Self {
        Self {
            initial_lr,
            current_lr: initial_lr,
            factor,
            patience,
            min_lr,
            patience_counter: 0,
            best_metric: match mode {
                PlateauMode::Min => f64::INFINITY,
                PlateauMode::Max => f64::NEG_INFINITY,
            },
            mode,
        }
    }
    
    /// Update with metric value
    pub fn update(&mut self, metric: f64) -> AnyResult<()> {
        let is_better = match self.mode {
            PlateauMode::Min => metric < self.best_metric,
            PlateauMode::Max => metric > self.best_metric,
        };
        
        if is_better {
            self.best_metric = metric;
            self.patience_counter = 0;
        } else {
            self.patience_counter += 1;
        }
        
        if self.patience_counter >= self.patience {
            self.current_lr = (self.current_lr * self.factor).max(self.min_lr);
            self.patience_counter = 0;
        }
        
        Ok(())
    }
}

impl Scheduler for ReduceLROnPlateau {
    fn step(&mut self) -> AnyResult<()> {
        // This scheduler is updated with metric values, not steps
        Ok(())
    }
    
    fn get_learning_rate(&self) -> f64 {
        self.current_lr
    }
    
    fn name(&self) -> &str {
        "ReduceLROnPlateau"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("initial_lr".to_string(), self.initial_lr);
        params.insert("factor".to_string(), self.factor);
        params.insert("patience".to_string(), self.patience as f64);
        params.insert("min_lr".to_string(), self.min_lr);
        params.insert("mode".to_string(), match self.mode {
            PlateauMode::Min => 0.0,
            PlateauMode::Max => 1.0,
        });
        params
    }
}

/// Linear LR scheduler
pub struct LinearLR {
    /// Initial learning rate
    initial_lr: f64,
    /// Current learning rate
    current_lr: f64,
    /// Final learning rate
    final_lr: f64,
    /// Total number of steps
    total_steps: usize,
    /// Current step
    step: usize,
}

impl LinearLR {
    /// Create new LinearLR scheduler
    pub fn new(initial_lr: f64, final_lr: f64, total_steps: usize) -> Self {
        Self {
            initial_lr,
            current_lr: initial_lr,
            final_lr,
            total_steps,
            step: 0,
        }
    }
}

impl Scheduler for LinearLR {
    fn step(&mut self) -> AnyResult<()> {
        self.step += 1;
        
        let progress = self.step as f64 / self.total_steps as f64;
        self.current_lr = self.initial_lr + (self.final_lr - self.initial_lr) * progress;
        
        Ok(())
    }
    
    fn get_learning_rate(&self) -> f64 {
        self.current_lr
    }
    
    fn name(&self) -> &str {
        "LinearLR"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("initial_lr".to_string(), self.initial_lr);
        params.insert("final_lr".to_string(), self.final_lr);
        params.insert("total_steps".to_string(), self.total_steps as f64);
        params
    }
}

/// Warmup LR scheduler
pub struct WarmupLR {
    /// Base scheduler
    base_scheduler: Box<dyn Scheduler>,
    /// Warmup steps
    warmup_steps: usize,
    /// Current step
    step: usize,
    /// Initial learning rate
    initial_lr: f64,
}

impl WarmupLR {
    /// Create new WarmupLR scheduler
    pub fn new(base_scheduler: Box<dyn Scheduler>, warmup_steps: usize, initial_lr: f64) -> Self {
        Self {
            base_scheduler,
            warmup_steps,
            step: 0,
            initial_lr,
        }
    }
}

impl Scheduler for WarmupLR {
    fn step(&mut self) -> AnyResult<()> {
        self.step += 1;
        
        if self.step < self.warmup_steps {
            // Warmup phase
            let progress = self.step as f64 / self.warmup_steps as f64;
            // TODO: Set learning rate to initial_lr * progress
        } else {
            // Use base scheduler
            self.base_scheduler.step()?;
        }
        
        Ok(())
    }
    
    fn get_learning_rate(&self) -> f64 {
        if self.step < self.warmup_steps {
            let progress = self.step as f64 / self.warmup_steps as f64;
            self.initial_lr * progress
        } else {
            self.base_scheduler.get_learning_rate()
        }
    }
    
    fn name(&self) -> &str {
        "WarmupLR"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("warmup_steps".to_string(), self.warmup_steps as f64);
        params.insert("initial_lr".to_string(), self.initial_lr);
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_step_lr() {
        let scheduler = StepLR::new(0.1, 10, 0.5);
        assert_eq!(scheduler.name(), "StepLR");
        assert_eq!(scheduler.get_learning_rate(), 0.1);
    }
    
    #[test]
    fn test_cosine_annealing_lr() {
        let scheduler = CosineAnnealingLR::new(0.1, 100, 0.01);
        assert_eq!(scheduler.name(), "CosineAnnealingLR");
        assert_eq!(scheduler.get_learning_rate(), 0.1);
    }
    
    #[test]
    fn test_linear_lr() {
        let scheduler = LinearLR::new(0.1, 0.01, 100);
        assert_eq!(scheduler.name(), "LinearLR");
        assert_eq!(scheduler.get_learning_rate(), 0.1);
    }
}
