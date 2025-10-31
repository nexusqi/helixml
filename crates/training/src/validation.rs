//! 🌀 HelixML Validation Manager
//! 
//! Advanced validation system for model evaluation.

use crate::metrics::{Accuracy, Precision, Recall, F1Score};
use backend_cpu::CpuTensor;
use tensor_core::tensor::TensorOps;
use std::collections::HashMap;
use anyhow::Result as AnyResult;

/// Validation manager (CPU tensor specialized)
pub struct ValidationManager {
    /// Validation metrics
    metrics: HashMap<String, Box<dyn ValidationMetric>>, 
    /// Validation history
    history: Vec<ValidationResult>,
}

/// Validation metric trait
pub trait ValidationMetric: Send + Sync {
    /// Update metric
    fn update(&mut self, predictions: &[f64], targets: &[f64]);
    
    /// Get metric value
    fn get_value(&self) -> f64;
    
    /// Reset metric
    fn reset(&mut self);
    
    /// Get metric name
    fn name(&self) -> &str;
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Epoch
    pub epoch: usize,
    /// Validation loss
    pub validation_loss: f64,
    /// Metrics
    pub metrics: HashMap<String, f64>,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
}

impl ValidationManager {
    /// Create new validation manager
    pub fn new() -> Self {
        Self { metrics: HashMap::new(), history: Vec::new() }
    }
    
    /// Add metric
    pub fn add_metric(&mut self, name: &str, metric: Box<dyn ValidationMetric>) {
        self.metrics.insert(name.to_string(), metric);
    }
    
    /// Validate model
    pub fn validate(&mut self, predictions: &[CpuTensor], targets: &[CpuTensor], epoch: usize) -> AnyResult<ValidationResult> {
        let mut result = ValidationResult {
            epoch,
            validation_loss: 0.0,
            metrics: HashMap::new(),
            timestamp: std::time::SystemTime::now(),
        };
        
        // Convert tensors to f64 vectors
        let pred_values: Vec<f64> = predictions.iter().map(|t| t.to_scalar().unwrap_or(0.0) as f64).collect();
        let target_values: Vec<f64> = targets.iter().map(|t| t.to_scalar().unwrap_or(0.0) as f64).collect();
        
        // Update metrics
        for (name, metric) in self.metrics.iter_mut() {
            metric.update(&pred_values, &target_values);
            result.metrics.insert(name.clone(), metric.get_value());
        }
        
        // Add to history
        self.history.push(result.clone());
        
        Ok(result)
    }
    
    /// Get validation history
    pub fn get_history(&self) -> &Vec<ValidationResult> {
        &self.history
    }
    
    /// Get best validation result
    pub fn get_best_result(&self) -> Option<&ValidationResult> {
        self.history.iter().min_by(|a, b| a.validation_loss.partial_cmp(&b.validation_loss).unwrap())
    }
    
    /// Clear history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}

/// Accuracy validation metric
pub struct AccuracyMetric {
    accuracy: Accuracy,
}

impl AccuracyMetric {
    /// Create new accuracy metric
    pub fn new() -> Self {
        Self {
            accuracy: Accuracy::new(),
        }
    }
}

impl ValidationMetric for AccuracyMetric {
    fn update(&mut self, predictions: &[f64], targets: &[f64]) {
        self.accuracy.update(predictions, targets);
    }
    
    fn get_value(&self) -> f64 {
        self.accuracy.get_accuracy()
    }
    
    fn reset(&mut self) {
        self.accuracy.reset();
    }
    
    fn name(&self) -> &str {
        "accuracy"
    }
}

/// Precision validation metric
pub struct PrecisionMetric {
    precision: Precision,
}

impl PrecisionMetric {
    /// Create new precision metric
    pub fn new() -> Self {
        Self {
            precision: Precision::new(),
        }
    }
}

impl ValidationMetric for PrecisionMetric {
    fn update(&mut self, predictions: &[f64], targets: &[f64]) {
        self.precision.update(predictions, targets);
    }
    
    fn get_value(&self) -> f64 {
        self.precision.get_precision()
    }
    
    fn reset(&mut self) {
        self.precision.reset();
    }
    
    fn name(&self) -> &str {
        "precision"
    }
}

/// Recall validation metric
pub struct RecallMetric {
    recall: Recall,
}

impl RecallMetric {
    /// Create new recall metric
    pub fn new() -> Self {
        Self {
            recall: Recall::new(),
        }
    }
}

impl ValidationMetric for RecallMetric {
    fn update(&mut self, predictions: &[f64], targets: &[f64]) {
        self.recall.update(predictions, targets);
    }
    
    fn get_value(&self) -> f64 {
        self.recall.get_recall()
    }
    
    fn reset(&mut self) {
        self.recall.reset();
    }
    
    fn name(&self) -> &str {
        "recall"
    }
}

/// F1 score validation metric
pub struct F1ScoreMetric {
    f1_score: F1Score,
}

impl F1ScoreMetric {
    /// Create new F1 score metric
    pub fn new() -> Self {
        Self {
            f1_score: F1Score::new(),
        }
    }
}

impl ValidationMetric for F1ScoreMetric {
    fn update(&mut self, predictions: &[f64], targets: &[f64]) {
        self.f1_score.update(predictions, targets);
    }
    
    fn get_value(&self) -> f64 {
        self.f1_score.get_f1_score()
    }
    
    fn reset(&mut self) {
        self.f1_score.reset();
    }
    
    fn name(&self) -> &str {
        "f1_score"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_validation_manager_creation() {
        let manager = ValidationManager::new();
        assert!(manager.metrics.is_empty());
        assert!(manager.history.is_empty());
    }
    
    #[test]
    fn test_accuracy_metric() {
        let mut metric = AccuracyMetric::new();
        let predictions = vec![1.0, 0.0, 1.0, 0.0];
        let targets = vec![1.0, 0.0, 0.0, 1.0];
        
        metric.update(&predictions, &targets);
        assert_eq!(metric.get_value(), 0.5);
    }
    
    #[test]
    fn test_f1_score_metric() {
        let mut metric = F1ScoreMetric::new();
        let predictions = vec![1.0, 0.0, 1.0, 0.0];
        let targets = vec![1.0, 0.0, 0.0, 1.0];
        
        metric.update(&predictions, &targets);
        assert!(metric.get_value() >= 0.0 && metric.get_value() <= 1.0);
    }
}
