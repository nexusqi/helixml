//! 🌀 HelixML Training Metrics
//! 
//! Comprehensive metrics tracking for training and validation.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use anyhow::Result as AnyResult;
use serde::{Serialize, Deserialize};

/// Training metrics tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    /// Training loss
    pub training_loss: f64,
    /// Validation loss
    pub validation_loss: f64,
    /// Current epoch
    pub epoch: usize,
    /// Current step
    pub step: usize,
    /// Training time (in seconds)
    #[serde(skip)]
    pub training_time: Duration,
    /// Validation time (in seconds)
    #[serde(skip)]
    pub validation_time: Duration,
    /// Learning rate
    pub learning_rate: f64,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
    /// History
    pub history: MetricsHistory,
}

/// Metrics history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsHistory {
    /// Training losses over time
    pub training_losses: Vec<f64>,
    /// Validation losses over time
    pub validation_losses: Vec<f64>,
    /// Learning rates over time
    pub learning_rates: Vec<f64>,
    /// Epochs
    pub epochs: Vec<usize>,
    /// Steps
    pub steps: Vec<usize>,
    /// Timestamps
    #[serde(skip)]
    pub timestamps: Vec<Instant>,
    /// Custom metrics over time
    pub custom_metrics_history: HashMap<String, Vec<f64>>,
}

impl Default for MetricsHistory {
    fn default() -> Self {
        Self {
            training_losses: Vec::new(),
            validation_losses: Vec::new(),
            learning_rates: Vec::new(),
            epochs: Vec::new(),
            steps: Vec::new(),
            timestamps: Vec::new(),
            custom_metrics_history: HashMap::new(),
        }
    }
}

impl Metrics {
    /// Create new metrics tracker
    pub fn new() -> Self {
        Self {
            training_loss: 0.0,
            validation_loss: 0.0,
            epoch: 0,
            step: 0,
            training_time: Duration::from_secs(0),
            validation_time: Duration::from_secs(0),
            learning_rate: 0.0,
            custom_metrics: HashMap::new(),
            history: MetricsHistory::default(),
        }
    }
    
    /// Update training loss
    pub fn update_training_loss(&mut self, loss: f64) {
        self.training_loss = loss;
        self.history.training_losses.push(loss);
    }
    
    /// Update validation loss
    pub fn update_validation_loss(&mut self, loss: f64) {
        self.validation_loss = loss;
        self.history.validation_losses.push(loss);
    }
    
    /// Update epoch
    pub fn update_epoch(&mut self, epoch: usize) {
        self.epoch = epoch;
        self.history.epochs.push(epoch);
    }
    
    /// Update step
    pub fn update_step(&mut self, step: usize) {
        self.step = step;
        self.history.steps.push(step);
    }
    
    /// Update learning rate
    pub fn update_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
        self.history.learning_rates.push(lr);
    }
    
    /// Update custom metric
    pub fn update_custom_metric(&mut self, name: &str, value: f64) {
        self.custom_metrics.insert(name.to_string(), value);
        
        if let Some(history) = self.history.custom_metrics_history.get_mut(name) {
            history.push(value);
        } else {
            self.history.custom_metrics_history.insert(name.to_string(), vec![value]);
        }
    }
    
    /// Update training time
    pub fn update_training_time(&mut self, time: Duration) {
        self.training_time = time;
    }
    
    /// Update validation time
    pub fn update_validation_time(&mut self, time: Duration) {
        self.validation_time = time;
    }
    
    /// Get average training loss
    pub fn get_avg_training_loss(&self) -> f64 {
        if self.history.training_losses.is_empty() {
            0.0
        } else {
            self.history.training_losses.iter().sum::<f64>() / self.history.training_losses.len() as f64
        }
    }
    
    /// Get average validation loss
    pub fn get_avg_validation_loss(&self) -> f64 {
        if self.history.validation_losses.is_empty() {
            0.0
        } else {
            self.history.validation_losses.iter().sum::<f64>() / self.history.validation_losses.len() as f64
        }
    }
    
    /// Get best training loss
    pub fn get_best_training_loss(&self) -> f64 {
        self.history.training_losses.iter().fold(f64::INFINITY, |a, &b| a.min(b))
    }
    
    /// Get best validation loss
    pub fn get_best_validation_loss(&self) -> f64 {
        self.history.validation_losses.iter().fold(f64::INFINITY, |a, &b| a.min(b))
    }
    
    /// Get training progress
    pub fn get_training_progress(&self) -> f64 {
        if self.history.epochs.is_empty() {
            0.0
        } else {
            *self.history.epochs.last().unwrap() as f64 / 100.0 // Assuming 100 epochs max
        }
    }
    
    /// Get custom metric
    pub fn get_custom_metric(&self, name: &str) -> Option<f64> {
        self.custom_metrics.get(name).copied()
    }
    
    /// Get custom metric history
    pub fn get_custom_metric_history(&self, name: &str) -> Option<&Vec<f64>> {
        self.history.custom_metrics_history.get(name)
    }
    
    /// Clear metrics
    pub fn clear(&mut self) {
        self.training_loss = 0.0;
        self.validation_loss = 0.0;
        self.epoch = 0;
        self.step = 0;
        self.training_time = Duration::from_secs(0);
        self.validation_time = Duration::from_secs(0);
        self.learning_rate = 0.0;
        self.custom_metrics.clear();
        self.history = MetricsHistory::default();
    }
    
    /// Export metrics to JSON
    pub fn export_json(&self) -> AnyResult<String> {
        let json = serde_json::to_string_pretty(self)?;
        Ok(json)
    }
    
    /// Import metrics from JSON
    pub fn import_json(&mut self, json: &str) -> AnyResult<()> {
        let metrics: Metrics = serde_json::from_str(json)?;
        *self = metrics;
        Ok(())
    }
}

/// Accuracy metric
pub struct Accuracy {
    /// Correct predictions
    correct: usize,
    /// Total predictions
    total: usize,
    /// History
    history: Vec<f64>,
}

impl Accuracy {
    /// Create new accuracy metric
    pub fn new() -> Self {
        Self {
            correct: 0,
            total: 0,
            history: Vec::new(),
        }
    }
    
    /// Update with predictions and targets
    pub fn update(&mut self, predictions: &[f64], targets: &[f64]) {
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            if (pred - target).abs() < 1e-6 {
                self.correct += 1;
            }
            self.total += 1;
        }
    }
    
    /// Get current accuracy
    pub fn get_accuracy(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f64 / self.total as f64
        }
    }
    
    /// Reset accuracy
    pub fn reset(&mut self) {
        self.correct = 0;
        self.total = 0;
        self.history.push(self.get_accuracy());
    }
}

/// Precision metric
pub struct Precision {
    /// True positives
    tp: usize,
    /// False positives
    fp: usize,
    /// History
    history: Vec<f64>,
}

impl Precision {
    /// Create new precision metric
    pub fn new() -> Self {
        Self {
            tp: 0,
            fp: 0,
            history: Vec::new(),
        }
    }
    
    /// Update with predictions and targets
    pub fn update(&mut self, predictions: &[f64], targets: &[f64]) {
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            if *pred > 0.5 && *target > 0.5 {
                self.tp += 1;
            } else if *pred > 0.5 && *target <= 0.5 {
                self.fp += 1;
            }
        }
    }
    
    /// Get current precision
    pub fn get_precision(&self) -> f64 {
        if self.tp + self.fp == 0 {
            0.0
        } else {
            self.tp as f64 / (self.tp + self.fp) as f64
        }
    }
    
    /// Reset precision
    pub fn reset(&mut self) {
        self.tp = 0;
        self.fp = 0;
        self.history.push(self.get_precision());
    }
}

/// Recall metric
pub struct Recall {
    /// True positives
    tp: usize,
    /// False negatives
    fn_: usize,
    /// History
    history: Vec<f64>,
}

impl Recall {
    /// Create new recall metric
    pub fn new() -> Self {
        Self {
            tp: 0,
            fn_: 0,
            history: Vec::new(),
        }
    }
    
    /// Update with predictions and targets
    pub fn update(&mut self, predictions: &[f64], targets: &[f64]) {
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            if *pred > 0.5 && *target > 0.5 {
                self.tp += 1;
            } else if *pred <= 0.5 && *target > 0.5 {
                self.fn_ += 1;
            }
        }
    }
    
    /// Get current recall
    pub fn get_recall(&self) -> f64 {
        if self.tp + self.fn_ == 0 {
            0.0
        } else {
            self.tp as f64 / (self.tp + self.fn_) as f64
        }
    }
    
    /// Reset recall
    pub fn reset(&mut self) {
        self.tp = 0;
        self.fn_ = 0;
        self.history.push(self.get_recall());
    }
}

/// F1 score metric
pub struct F1Score {
    /// Precision metric
    precision: Precision,
    /// Recall metric
    recall: Recall,
    /// History
    history: Vec<f64>,
}

impl F1Score {
    /// Create new F1 score metric
    pub fn new() -> Self {
        Self {
            precision: Precision::new(),
            recall: Recall::new(),
            history: Vec::new(),
        }
    }
    
    /// Update with predictions and targets
    pub fn update(&mut self, predictions: &[f64], targets: &[f64]) {
        self.precision.update(predictions, targets);
        self.recall.update(predictions, targets);
    }
    
    /// Get current F1 score
    pub fn get_f1_score(&self) -> f64 {
        let precision = self.precision.get_precision();
        let recall = self.recall.get_recall();
        
        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    }
    
    /// Reset F1 score
    pub fn reset(&mut self) {
        self.precision.reset();
        self.recall.reset();
        self.history.push(self.get_f1_score());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_creation() {
        let metrics = Metrics::new();
        assert_eq!(metrics.training_loss, 0.0);
        assert_eq!(metrics.epoch, 0);
        assert_eq!(metrics.step, 0);
    }
    
    #[test]
    fn test_accuracy_metric() {
        let mut accuracy = Accuracy::new();
        let predictions = vec![1.0, 0.0, 1.0, 0.0];
        let targets = vec![1.0, 0.0, 0.0, 1.0];
        
        accuracy.update(&predictions, &targets);
        assert_eq!(accuracy.get_accuracy(), 0.5);
    }
    
    #[test]
    fn test_f1_score_metric() {
        let mut f1 = F1Score::new();
        let predictions = vec![1.0, 0.0, 1.0, 0.0];
        let targets = vec![1.0, 0.0, 0.0, 1.0];
        
        f1.update(&predictions, &targets);
        assert!(f1.get_f1_score() >= 0.0 && f1.get_f1_score() <= 1.0);
    }
}
