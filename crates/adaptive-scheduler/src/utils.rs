//! 🛠️ Utility Functions
//! 
//! Utility functions for adaptive scheduling operations

use tensor_core::{Tensor, Device, Result, TensorError};
use tensor_core::tensor::{TensorOps, TensorRandom, TensorBroadcast, TensorMixedPrecision, TensorStats, TensorReduce};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::*;

/// Utility functions for adaptive scheduling
#[derive(Debug)]
pub struct SchedulingUtils<T: Tensor> {
    device: Device,
    cache: HashMap<String, T>,
}

impl<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision + TensorStats + TensorReduce> SchedulingUtils<T> {
    pub fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            cache: HashMap::new(),
        })
    }
    
    /// Calculate task priority score
    pub fn calculate_priority_score(&self, task: &Task) -> Result<f32> {
        let mut score: f32 = 0.0;
        
        // Priority weight
        match task.priority {
            TaskPriority::Low => score += 0.2,
            TaskPriority::Normal => score += 0.5,
            TaskPriority::High => score += 0.8,
            TaskPriority::Critical => score += 1.0,
        }
        
        // Resource requirements weight
        let memory_weight: f32 = if task.resource_requirements.memory > 1024 * 1024 * 1024 { // 1GB
            0.3
        } else {
            0.7
        };
        
        let compute_weight: f32 = if task.resource_requirements.compute > 10.0 {
            0.3
        } else {
            0.7
        };
        
        score += (memory_weight + compute_weight) / 2.0 * 0.3;
        
        // Timeout weight
        let timeout_weight: f32 = if task.timeout < Duration::from_secs(60) {
            0.8
        } else {
            0.4
        };
        
        score += timeout_weight * 0.2;
        
        Ok(score.min(1.0))
    }
    
    /// Calculate device suitability score
    pub fn calculate_device_suitability(&self, device: &Device, task: &Task) -> Result<f32> {
        let mut score: f32 = 0.0;
        
        // Device type compatibility
        let compatibility_score = if task.device_requirements.device_types.contains(device) {
            1.0
        } else {
            0.0
        };
        
        score += compatibility_score * 0.4;
        
        // Resource availability (simplified)
        let resource_score = if task.resource_requirements.memory < 1024 * 1024 * 1024 { // 1GB
            0.8
        } else {
            0.4
        };
        
        score += resource_score * 0.3;
        
        // Performance characteristics
        let performance_score = match device {
            Device::Cpu => 0.6,
            Device::Cuda(_) => 0.9,
            _ => 0.5, // Default for other devices
        };
        
        score += performance_score * 0.3;
        
        Ok(score.min(1.0))
    }
    
    /// Calculate load balancing score
    pub fn calculate_load_balance_score(&self, devices: &[Device], current_loads: &HashMap<Device, f32>) -> Result<f32> {
        if devices.is_empty() {
            return Ok(0.0);
        }
        
        let loads: Vec<f32> = devices.iter()
            .map(|device| current_loads.get(device).copied().unwrap_or(0.0))
            .collect();
        
        let average_load = loads.iter().sum::<f32>() / loads.len() as f32;
        let variance = loads.iter()
            .map(|&load| (load - average_load).powi(2))
            .sum::<f32>() / loads.len() as f32;
        
        let standard_deviation = variance.sqrt();
        let balance_score = 1.0 - (standard_deviation / (average_load + 0.001)); // Avoid division by zero
        
        Ok(balance_score.max(0.0).min(1.0))
    }
    
    /// Calculate energy efficiency score
    pub fn calculate_energy_efficiency(&self, device: &Device, task: &Task) -> Result<f32> {
        let base_efficiency = match device {
            Device::Cpu => 0.8,
            Device::Cuda(_) => 0.6,
            _ => 0.7, // Default for other devices
        };
        
        // Adjust based on task requirements
        let memory_factor = if task.resource_requirements.memory > 512 * 1024 * 1024 { // 512MB
            0.9
        } else {
            1.0
        };
        
        let compute_factor = if task.resource_requirements.compute > 5.0 {
            0.8
        } else {
            1.0
        };
        
        let efficiency: f32 = base_efficiency * memory_factor * compute_factor;
        Ok(efficiency.min(1.0))
    }
    
    /// Calculate latency score
    pub fn calculate_latency_score(&self, device: &Device, task: &Task) -> Result<f32> {
        let base_latency = match device {
            Device::Cpu => Duration::from_millis(100),
            Device::Cuda(_) => Duration::from_millis(50),
            _ => Duration::from_millis(75), // Default for other devices
        };
        
        // Adjust based on task complexity
        let complexity_factor = if task.resource_requirements.compute > 10.0 {
            1.5
        } else {
            1.0
        };
        
        let estimated_latency = base_latency.mul_f32(complexity_factor);
        let latency_score = if estimated_latency < Duration::from_millis(200) {
            1.0
        } else if estimated_latency < Duration::from_millis(500) {
            0.7
        } else {
            0.4
        };
        
        Ok(latency_score)
    }
    
    /// Calculate throughput score
    pub fn calculate_throughput_score(&self, device: &Device, task: &Task) -> Result<f32> {
        let base_throughput = match device {
            Device::Cpu => 100.0,
            Device::Cuda(_) => 1000.0,
            _ => 500.0, // Default for other devices
        };
        
        // Adjust based on task requirements
        let memory_factor = if task.resource_requirements.memory > 1024 * 1024 * 1024 { // 1GB
            0.8
        } else {
            1.0
        };
        
        let compute_factor = if task.resource_requirements.compute > 10.0 {
            0.9
        } else {
            1.0
        };
        
        let throughput = base_throughput * memory_factor * compute_factor;
        let throughput_score = if throughput > 500.0 {
            1.0
        } else if throughput > 200.0 {
            0.7
        } else {
            0.4
        };
        
        Ok(throughput_score)
    }
    
    /// Calculate overall scheduling score
    pub fn calculate_scheduling_score(
        &self,
        task: &Task,
        device: &Device,
        current_loads: &HashMap<Device, f32>,
    ) -> Result<f32> {
        let priority_score = self.calculate_priority_score(task)?;
        let suitability_score = self.calculate_device_suitability(device, task)?;
        let energy_score = self.calculate_energy_efficiency(device, task)?;
        let latency_score = self.calculate_latency_score(device, task)?;
        let throughput_score = self.calculate_throughput_score(device, task)?;
        
        // Weighted combination
        let total_score = priority_score * 0.3 +
                          suitability_score * 0.25 +
                          energy_score * 0.15 +
                          latency_score * 0.15 +
                          throughput_score * 0.15;
        
        Ok(total_score.min(1.0))
    }
    
    /// Generate scheduling recommendations
    pub fn generate_recommendations(
        &self,
        tasks: &[Task],
        devices: &[Device],
        current_loads: &HashMap<Device, f32>,
    ) -> Result<Vec<SchedulingRecommendation>> {
        let mut recommendations = Vec::new();
        
        // Analyze load distribution
        let load_balance_score = self.calculate_load_balance_score(devices, current_loads)?;
        if load_balance_score < 0.7 {
            recommendations.push(SchedulingRecommendation {
                recommendation_type: RecommendationType::LoadBalancing,
                priority: 0.8,
                description: "Load distribution is uneven, consider rebalancing".to_string(),
                suggested_actions: vec![
                    "Move tasks from overloaded devices".to_string(),
                    "Adjust load balancing strategy".to_string(),
                ],
            });
        }
        
        // Analyze resource utilization
        let total_memory_usage: f32 = current_loads.values().sum();
        let average_memory_usage = total_memory_usage / devices.len() as f32;
        
        if average_memory_usage > 0.9 {
            recommendations.push(SchedulingRecommendation {
                recommendation_type: RecommendationType::ResourceOptimization,
                priority: 0.9,
                description: "High memory usage detected, consider optimization".to_string(),
                suggested_actions: vec![
                    "Optimize memory allocation".to_string(),
                    "Consider adding more devices".to_string(),
                ],
            });
        }
        
        // Analyze task priorities
        let high_priority_tasks = tasks.iter()
            .filter(|task| task.priority == TaskPriority::High || task.priority == TaskPriority::Critical)
            .count();
        
        if high_priority_tasks > tasks.len() / 2 {
            recommendations.push(SchedulingRecommendation {
                recommendation_type: RecommendationType::PriorityOptimization,
                priority: 0.7,
                description: "Many high-priority tasks, consider priority-based scheduling".to_string(),
                suggested_actions: vec![
                    "Implement priority queues".to_string(),
                    "Preempt low-priority tasks".to_string(),
                ],
            });
        }
        
        Ok(recommendations)
    }
    
    /// Optimize task assignment
    pub fn optimize_task_assignment(
        &self,
        tasks: &[Task],
        devices: &[Device],
        current_loads: &HashMap<Device, f32>,
    ) -> Result<HashMap<TaskId, Device>> {
        let mut assignments = HashMap::new();
        let mut device_loads = current_loads.clone();
        
        // Sort tasks by priority
        let mut sorted_tasks = tasks.to_vec();
        sorted_tasks.sort_by(|a, b| b.priority.cmp(&a.priority));
        
        for task in sorted_tasks {
            let mut best_device = None;
            let mut best_score = 0.0;
            
            for device in devices {
                let score = self.calculate_scheduling_score(&task, device, &device_loads)?;
                if score > best_score {
                    best_score = score;
                    best_device = Some(device.clone());
                }
            }
            
            if let Some(device) = best_device {
                assignments.insert(TaskId::new(), device.clone());
                
                // Update device load
                let current_load = device_loads.get(&device).copied().unwrap_or(0.0);
                device_loads.insert(device, current_load + 0.1); // Simplified load update
            }
        }
        
        Ok(assignments)
    }
    
    /// Calculate system efficiency
    pub fn calculate_system_efficiency(
        &self,
        tasks: &[Task],
        devices: &[Device],
        current_loads: &HashMap<Device, f32>,
    ) -> Result<f32> {
        let mut total_efficiency = 0.0;
        let mut device_count = 0;
        
        for device in devices {
            let load = current_loads.get(device).copied().unwrap_or(0.0);
            let efficiency = if load > 0.0 && load < 1.0 {
                load * 0.8 + (1.0 - load) * 0.2 // Balance between utilization and availability
            } else {
                0.0
            };
            
            total_efficiency += efficiency;
            device_count += 1;
        }
        
        Ok(if device_count > 0 { total_efficiency / device_count as f32 } else { 0.0 })
    }
    
    /// Predict task execution time
    pub fn predict_execution_time(&self, task: &Task, device: &Device) -> Result<Duration> {
        let base_time = match device {
            Device::Cpu => Duration::from_millis(100),
            Device::Cuda(_) => Duration::from_millis(50),
            _ => Duration::from_millis(75), // Default for other devices
        };
        
        // Adjust based on task complexity
        let memory_factor = if task.resource_requirements.memory > 1024 * 1024 * 1024 { // 1GB
            1.5
        } else {
            1.0
        };
        
        let compute_factor = if task.resource_requirements.compute > 10.0 {
            2.0
        } else {
            1.0
        };
        
        let predicted_time = base_time.mul_f32(memory_factor * compute_factor);
        Ok(predicted_time)
    }
    
    /// Calculate resource utilization
    pub fn calculate_resource_utilization(
        &self,
        tasks: &[Task],
        devices: &[Device],
    ) -> Result<ResourceUtilization> {
        let total_memory_required: usize = tasks.iter()
            .map(|task| task.resource_requirements.memory)
            .sum();
        
        let total_compute_required: f32 = tasks.iter()
            .map(|task| task.resource_requirements.compute)
            .sum();
        
        let total_bandwidth_required: f32 = tasks.iter()
            .map(|task| task.resource_requirements.bandwidth)
            .sum();
        
        let total_storage_required: usize = tasks.iter()
            .map(|task| task.resource_requirements.storage)
            .sum();
        
        // Calculate available resources (simplified)
        let available_memory = devices.len() * 8 * 1024 * 1024 * 1024; // 8GB per device
        let available_compute = devices.len() as f32 * 100.0;
        let available_bandwidth = devices.len() as f32 * 100.0;
        let available_storage = devices.len() * 100 * 1024 * 1024 * 1024; // 100GB per device
        
        Ok(ResourceUtilization {
            memory_usage: if available_memory > 0 {
                total_memory_required as f32 / available_memory as f32
            } else {
                0.0
            },
            compute_usage: if available_compute > 0.0 {
                total_compute_required / available_compute
            } else {
                0.0
            },
            bandwidth_usage: if available_bandwidth > 0.0 {
                total_bandwidth_required / available_bandwidth
            } else {
                0.0
            },
            storage_usage: if available_storage > 0 {
                total_storage_required as f32 / available_storage as f32
            } else {
                0.0
            },
        })
    }
    
    /// Generate performance report
    pub fn generate_performance_report(
        &self,
        tasks: &[Task],
        devices: &[Device],
        current_loads: &HashMap<Device, f32>,
    ) -> Result<PerformanceReport> {
        let system_efficiency = self.calculate_system_efficiency(tasks, devices, current_loads)?;
        let resource_utilization = self.calculate_resource_utilization(tasks, devices)?;
        let load_balance_score = self.calculate_load_balance_score(devices, current_loads)?;
        let recommendations = self.generate_recommendations(tasks, devices, current_loads)?;
        
        Ok(PerformanceReport {
            system_efficiency,
            resource_utilization,
            load_balance_score,
            device_count: devices.len(),
            task_count: tasks.len(),
            recommendations,
            timestamp: Instant::now(),
        })
    }
}

/// Scheduling recommendation
#[derive(Debug, Clone)]
pub struct SchedulingRecommendation {
    pub recommendation_type: RecommendationType,
    pub priority: f32,
    pub description: String,
    pub suggested_actions: Vec<String>,
}

/// Recommendation types
#[derive(Debug, Clone)]
pub enum RecommendationType {
    LoadBalancing,
    ResourceOptimization,
    PriorityOptimization,
    EnergyOptimization,
    LatencyOptimization,
    ThroughputOptimization,
}

/// Performance report
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub system_efficiency: f32,
    pub resource_utilization: ResourceUtilization,
    pub load_balance_score: f32,
    pub device_count: usize,
    pub task_count: usize,
    pub recommendations: Vec<SchedulingRecommendation>,
    pub timestamp: Instant,
}

/// Configuration utilities
#[derive(Debug)]
pub struct ConfigUtils {
    default_configs: HashMap<String, serde_json::Value>,
}

impl ConfigUtils {
    pub fn new() -> Result<Self> {
        let mut default_configs = HashMap::new();
        
        // Add default configurations
        default_configs.insert("scheduler_config".to_string(), serde_json::json!({
            "max_concurrent_tasks": 100,
            "task_timeout": 300,
            "load_balancing_strategy": "RoundRobin",
            "optimization_strategy": "Performance",
            "monitoring_interval": 100,
            "adaptive_threshold": 0.8
        }));
        
        default_configs.insert("device_config".to_string(), serde_json::json!({
            "cpu_devices": 1,
            "cuda_devices": 4,
            "memory_limit": 32000000000i64,
            "compute_limit": 100.0
        }));
        
        default_configs.insert("optimization_config".to_string(), serde_json::json!({
            "learning_rate": 0.1,
            "exploration_rate": 0.1,
            "optimization_frequency": 10,
            "convergence_threshold": 0.01
        }));
        
        Ok(Self {
            default_configs,
        })
    }
    
    /// Get default configuration
    pub fn get_default_config(&self, config_name: &str) -> Option<&serde_json::Value> {
        self.default_configs.get(config_name)
    }
    
    /// Save configuration to file
    pub fn save_config(&self, config: &serde_json::Value, filename: &str) -> Result<()> {
        let file = std::fs::File::create(filename)
            .map_err(|e| TensorError::SerializationError { message: e.to_string() })?;
        let mut writer = std::io::BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, config)
            .map_err(|e| TensorError::SerializationError { message: e.to_string() })?;
        Ok(())
    }
    
    /// Load configuration from file
    pub fn load_config(&self, filename: &str) -> Result<serde_json::Value> {
        let file = std::fs::File::open(filename)
            .map_err(|e| TensorError::SerializationError { message: e.to_string() })?;
        let config: serde_json::Value = serde_json::from_reader(file)
            .map_err(|e| TensorError::SerializationError { message: e.to_string() })?;
        Ok(config)
    }
}

/// Performance monitoring utilities
#[derive(Debug)]
pub struct PerformanceMonitor {
    metrics: HashMap<String, f64>,
    timers: HashMap<String, Instant>,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            timers: HashMap::new(),
        }
    }
    
    /// Start timing an operation
    pub fn start_timer(&mut self, operation: &str) {
        self.timers.insert(operation.to_string(), Instant::now());
    }
    
    /// Stop timing an operation
    pub fn stop_timer(&mut self, operation: &str) -> Option<f64> {
        if let Some(start_time) = self.timers.remove(operation) {
            let duration = start_time.elapsed().as_secs_f64();
            self.metrics.insert(operation.to_string(), duration);
            Some(duration)
        } else {
            None
        }
    }
    
    /// Get performance metrics
    pub fn get_metrics(&self) -> &HashMap<String, f64> {
        &self.metrics
    }
    
    /// Clear all metrics
    pub fn clear_metrics(&mut self) {
        self.metrics.clear();
        self.timers.clear();
    }
    
    /// Generate performance report
    pub fn generate_report(&self) -> PerformanceReport {
        let total_time: f64 = self.metrics.values().sum();
        let average_time = if !self.metrics.is_empty() {
            total_time / self.metrics.len() as f64
        } else {
            0.0
        };
        
        PerformanceReport {
            system_efficiency: 0.8, // Placeholder
            resource_utilization: ResourceUtilization {
                memory_usage: 0.0,
                compute_usage: 0.0,
                bandwidth_usage: 0.0,
                storage_usage: 0.0,
            },
            load_balance_score: 0.8, // Placeholder
            device_count: 0,
            task_count: 0,
            recommendations: vec![],
            timestamp: Instant::now(),
        }
    }
}
