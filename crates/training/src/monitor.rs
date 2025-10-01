//! 🌀 HelixML Training Monitor
//! 
//! Real-time training monitoring and visualization.

use crate::metrics::Metrics;
use std::collections::HashMap;
use anyhow::Result as AnyResult;

/// Training monitor
pub struct TrainingMonitor {
    /// Monitor configuration
    config: MonitorConfig,
    /// Metrics history
    metrics_history: Vec<Metrics>,
    /// Custom callbacks
    callbacks: HashMap<String, Box<dyn Fn(&Metrics) -> AnyResult<()> + Send + Sync>>,
}

/// Monitor configuration
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Enable logging
    pub enable_logging: bool,
    /// Enable progress bars
    pub enable_progress_bars: bool,
    /// Enable metrics visualization
    pub enable_visualization: bool,
    /// Log frequency
    pub log_frequency: usize,
    /// Visualization frequency
    pub visualization_frequency: usize,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            enable_logging: true,
            enable_progress_bars: true,
            enable_visualization: true,
            log_frequency: 100,
            visualization_frequency: 10,
        }
    }
}

impl TrainingMonitor {
    /// Create new training monitor
    pub fn new() -> Self {
        Self {
            config: MonitorConfig::default(),
            metrics_history: Vec::new(),
            callbacks: HashMap::new(),
        }
    }
    
    /// Set configuration
    pub fn set_config(&mut self, config: MonitorConfig) {
        self.config = config;
    }
    
    /// Start training
    pub async fn start_training(&self) -> AnyResult<()> {
        if self.config.enable_logging {
            println!("🚀 Starting training...");
        }
        Ok(())
    }
    
    /// Finish training
    pub async fn finish_training(&self) -> AnyResult<()> {
        if self.config.enable_logging {
            println!("✅ Training completed!");
        }
        Ok(())
    }
    
    /// Start epoch
    pub async fn start_epoch(&self, epoch: usize) -> AnyResult<()> {
        if self.config.enable_logging {
            println!("📊 Epoch {}/{}", epoch + 1, 100); // Assuming 100 epochs
        }
        Ok(())
    }
    
    /// End epoch
    pub async fn end_epoch(&self, epoch: usize, loss: f64) -> AnyResult<()> {
        if self.config.enable_logging {
            println!("📈 Epoch {} - Loss: {:.6}", epoch + 1, loss);
        }
        Ok(())
    }
    
    /// Start validation
    pub async fn start_validation(&self, epoch: usize) -> AnyResult<()> {
        if self.config.enable_logging {
            println!("🔍 Validating epoch {}...", epoch + 1);
        }
        Ok(())
    }
    
    /// End validation
    pub async fn end_validation(&self, epoch: usize, loss: f64) -> AnyResult<()> {
        if self.config.enable_logging {
            println!("📊 Validation - Loss: {:.6}", loss);
        }
        Ok(())
    }
    
    /// Log progress
    pub async fn log_progress(&self, epoch: usize, step: usize, loss: f64) -> AnyResult<()> {
        if self.config.enable_logging && step % self.config.log_frequency == 0 {
            println!("🔄 Epoch {} - Step {} - Loss: {:.6}", epoch + 1, step, loss);
        }
        Ok(())
    }
    
    /// Update metrics
    pub fn update_metrics(&mut self, metrics: Metrics) -> AnyResult<()> {
        self.metrics_history.push(metrics.clone());
        
        // Execute callbacks
        for callback in self.callbacks.values() {
            callback(&metrics)?;
        }
        
        Ok(())
    }
    
    /// Add callback
    pub fn add_callback<F>(&mut self, name: &str, callback: F)
    where
        F: Fn(&Metrics) -> AnyResult<()> + Send + Sync + 'static,
    {
        self.callbacks.insert(name.to_string(), Box::new(callback));
    }
    
    /// Remove callback
    pub fn remove_callback(&mut self, name: &str) {
        self.callbacks.remove(name);
    }
    
    /// Get metrics history
    pub fn get_metrics_history(&self) -> &Vec<Metrics> {
        &self.metrics_history
    }
    
    /// Export metrics
    pub fn export_metrics(&self) -> AnyResult<String> {
        let json = serde_json::to_string_pretty(&self.metrics_history)?;
        Ok(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_monitor_creation() {
        let monitor = TrainingMonitor::new();
        assert!(monitor.config.enable_logging);
    }
    
    #[test]
    fn test_monitor_config() {
        let config = MonitorConfig::default();
        assert!(config.enable_logging);
        assert!(config.enable_progress_bars);
        assert!(config.enable_visualization);
    }
}
