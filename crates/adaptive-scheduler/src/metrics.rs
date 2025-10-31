//! 📊 Metrics Collection
//! 
//! Comprehensive metrics collection and analysis for adaptive scheduling

use tensor_core::{Tensor, Device, Result};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

use super::*;

/// Metrics collector for adaptive scheduling
#[derive(Debug)]
pub struct MetricsCollector {
    scheduler_metrics: Arc<RwLock<SchedulerMetrics>>,
    device_metrics: Arc<RwLock<HashMap<Device, DeviceMetrics>>>,
    task_metrics: Arc<RwLock<HashMap<TaskId, TaskMetrics>>>,
    performance_history: Arc<RwLock<VecDeque<PerformanceSnapshot>>>,
    alert_thresholds: Arc<RwLock<AlertThresholds>>,
    metrics_aggregator: Arc<RwLock<MetricsAggregator>>,
    reporting_interval: Duration,
    last_report: Arc<Mutex<Instant>>,
    metrics_buffer: Arc<RwLock<VecDeque<MetricsEvent>>>,
}

/// Device metrics
#[derive(Debug, Clone)]
pub struct DeviceMetrics {
    pub device: Device,
    pub utilization: f32,
    pub throughput: f32,
    pub latency: Duration,
    pub energy_consumption: f32,
    pub temperature: f32,
    pub memory_usage: f32,
    pub compute_usage: f32,
    pub bandwidth_usage: f32,
    pub active_tasks: usize,
    pub completed_tasks: usize,
    pub failed_tasks: usize,
    pub average_execution_time: Duration,
    pub last_updated: Instant,
}

/// Task metrics
#[derive(Debug, Clone)]
pub struct TaskMetrics {
    pub task_id: TaskId,
    pub priority: TaskPriority,
    pub status: TaskStatus,
    pub created_at: Instant,
    pub started_at: Option<Instant>,
    pub completed_at: Option<Instant>,
    pub execution_time: Option<Duration>,
    pub wait_time: Duration,
    pub resource_usage: ResourceUsage,
    pub device: Option<Device>,
    pub retry_count: usize,
    pub error_count: usize,
}

/// Resource usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub memory_used: usize,
    pub compute_used: f32,
    pub bandwidth_used: f32,
    pub storage_used: usize,
    pub peak_memory: usize,
    pub peak_compute: f32,
}

/// Performance snapshot
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    pub timestamp: Instant,
    pub overall_throughput: f32,
    pub average_latency: Duration,
    pub system_utilization: f32,
    pub energy_efficiency: f32,
    pub load_balance: f32,
    pub task_completion_rate: f32,
    pub error_rate: f32,
}

/// Alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub high_utilization: f32,
    pub high_latency: Duration,
    pub low_throughput: f32,
    pub high_error_rate: f32,
    pub high_energy_consumption: f32,
    pub high_temperature: f32,
}

/// Metrics aggregator
#[derive(Debug, Clone)]
pub struct MetricsAggregator {
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub failed_tasks: usize,
    pub average_execution_time: Duration,
    pub average_wait_time: Duration,
    pub throughput: f32,
    pub utilization: f32,
    pub energy_efficiency: f32,
    pub load_balance: f32,
    pub error_rate: f32,
    pub last_updated: Instant,
}

/// Metrics event
#[derive(Debug, Clone)]
pub enum MetricsEvent {
    TaskCreated { task_id: TaskId, priority: TaskPriority, timestamp: Instant },
    TaskStarted { task_id: TaskId, device: Device, timestamp: Instant },
    TaskCompleted { task_id: TaskId, execution_time: Duration, timestamp: Instant },
    TaskFailed { task_id: TaskId, error: String, timestamp: Instant },
    DeviceUtilizationChanged { device: Device, utilization: f32, timestamp: Instant },
    PerformanceSnapshot { snapshot: PerformanceSnapshot },
    AlertTriggered { alert_type: AlertType, message: String, timestamp: Instant },
}

/// Alert types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    HighUtilization,
    HighLatency,
    LowThroughput,
    HighErrorRate,
    HighEnergyConsumption,
    HighTemperature,
    ResourceExhaustion,
    LoadImbalance,
}

impl MetricsCollector {
    pub fn new() -> Result<Self> {
        Ok(Self {
            scheduler_metrics: Arc::new(RwLock::new(SchedulerMetrics {
                total_tasks: 0,
                completed_tasks: 0,
                failed_tasks: 0,
                pending_tasks: 0,
                running_tasks: 0,
                average_execution_time: Duration::from_millis(0),
                throughput: 0.0,
                load_factor: 0.0,
                device_utilization: HashMap::new(),
                resource_utilization: ResourceUtilization {
                    memory_usage: 0.0,
                    compute_usage: 0.0,
                    bandwidth_usage: 0.0,
                    storage_usage: 0.0,
                },
            })),
            device_metrics: Arc::new(RwLock::new(HashMap::new())),
            task_metrics: Arc::new(RwLock::new(HashMap::new())),
            performance_history: Arc::new(RwLock::new(VecDeque::new())),
            alert_thresholds: Arc::new(RwLock::new(AlertThresholds {
                high_utilization: 0.9,
                high_latency: Duration::from_secs(5),
                low_throughput: 10.0,
                high_error_rate: 0.1,
                high_energy_consumption: 300.0,
                high_temperature: 80.0,
            })),
            metrics_aggregator: Arc::new(RwLock::new(MetricsAggregator {
                total_tasks: 0,
                completed_tasks: 0,
                failed_tasks: 0,
                average_execution_time: Duration::from_millis(0),
                average_wait_time: Duration::from_millis(0),
                throughput: 0.0,
                utilization: 0.0,
                energy_efficiency: 0.0,
                load_balance: 0.0,
                error_rate: 0.0,
                last_updated: Instant::now(),
            })),
            reporting_interval: Duration::from_secs(10),
            last_report: Arc::new(Mutex::new(Instant::now())),
            metrics_buffer: Arc::new(RwLock::new(VecDeque::new())),
        })
    }
    
    /// Record task execution
    pub fn record_task_execution(&self, task_id: &TaskId, execution_time: Duration, device: &Device) -> Result<()> {
        // Update task metrics
        {
            let mut task_metrics = self.task_metrics.write().unwrap();
            if let Some(metrics) = task_metrics.get_mut(task_id) {
                metrics.execution_time = Some(execution_time);
                metrics.completed_at = Some(Instant::now());
                metrics.status = TaskStatus::Completed;
            }
        }
        
        // Update device metrics
        {
            let mut device_metrics = self.device_metrics.write().unwrap();
            if let Some(metrics) = device_metrics.get_mut(device) {
                metrics.completed_tasks += 1;
                metrics.average_execution_time = (metrics.average_execution_time + execution_time) / 2;
                metrics.last_updated = Instant::now();
            }
        }
        
        // Update scheduler metrics
        {
            let mut scheduler_metrics = self.scheduler_metrics.write().unwrap();
            scheduler_metrics.completed_tasks += 1;
            scheduler_metrics.running_tasks -= 1;
            scheduler_metrics.average_execution_time = (scheduler_metrics.average_execution_time + execution_time) / 2;
        }
        
        // Record event
        self.record_event(MetricsEvent::TaskCompleted {
            task_id: task_id.clone(),
            execution_time,
            timestamp: Instant::now(),
        })?;
        
        // Update aggregator
        self.update_aggregator()?;
        
        Ok(())
    }
    
    /// Record task failure
    pub fn record_task_failure(&self, task_id: &TaskId, error: String, device: &Device) -> Result<()> {
        // Update task metrics
        {
            let mut task_metrics = self.task_metrics.write().unwrap();
            if let Some(metrics) = task_metrics.get_mut(task_id) {
                metrics.status = TaskStatus::Failed;
                metrics.error_count += 1;
                metrics.completed_at = Some(Instant::now());
            }
        }
        
        // Update device metrics
        {
            let mut device_metrics = self.device_metrics.write().unwrap();
            if let Some(metrics) = device_metrics.get_mut(device) {
                metrics.failed_tasks += 1;
                metrics.last_updated = Instant::now();
            }
        }
        
        // Update scheduler metrics
        {
            let mut scheduler_metrics = self.scheduler_metrics.write().unwrap();
            scheduler_metrics.failed_tasks += 1;
            scheduler_metrics.running_tasks -= 1;
        }
        
        // Record event
        self.record_event(MetricsEvent::TaskFailed {
            task_id: task_id.clone(),
            error,
            timestamp: Instant::now(),
        })?;
        
        // Update aggregator
        self.update_aggregator()?;
        
        Ok(())
    }
    
    /// Record device utilization
    pub fn record_device_utilization(&self, device: &Device, utilization: f32) -> Result<()> {
        // Update device metrics
        {
            let mut device_metrics = self.device_metrics.write().unwrap();
            if let Some(metrics) = device_metrics.get_mut(device) {
                metrics.utilization = utilization;
                metrics.last_updated = Instant::now();
            }
        }
        
        // Update scheduler metrics
        {
            let mut scheduler_metrics = self.scheduler_metrics.write().unwrap();
            scheduler_metrics.device_utilization.insert(device.clone(), utilization);
        }
        
        // Record event
        self.record_event(MetricsEvent::DeviceUtilizationChanged {
            device: device.clone(),
            utilization,
            timestamp: Instant::now(),
        })?;
        
        // Check for alerts
        self.check_alerts(device, utilization)?;
        
        Ok(())
    }
    
    /// Get scheduler metrics
    pub fn get_metrics(&self) -> Result<SchedulerMetrics> {
        let metrics = self.scheduler_metrics.read().unwrap();
        Ok(metrics.clone())
    }
    
    /// Get device metrics
    pub fn get_device_metrics(&self, device: &Device) -> Result<Option<DeviceMetrics>> {
        let device_metrics = self.device_metrics.read().unwrap();
        Ok(device_metrics.get(device).cloned())
    }
    
    /// Get task metrics
    pub fn get_task_metrics(&self, task_id: &TaskId) -> Result<Option<TaskMetrics>> {
        let task_metrics = self.task_metrics.read().unwrap();
        Ok(task_metrics.get(task_id).cloned())
    }
    
    /// Get performance history
    pub fn get_performance_history(&self) -> Result<Vec<PerformanceSnapshot>> {
        let history = self.performance_history.read().unwrap();
        Ok(history.clone().into())
    }
    
    /// Get metrics aggregator
    pub fn get_aggregator(&self) -> Result<MetricsAggregator> {
        let aggregator = self.metrics_aggregator.read().unwrap();
        Ok(aggregator.clone())
    }
    
    /// Set alert thresholds
    pub fn set_alert_thresholds(&self, thresholds: AlertThresholds) -> Result<()> {
        let mut alert_thresholds = self.alert_thresholds.write().unwrap();
        *alert_thresholds = thresholds;
        Ok(())
    }
    
    /// Generate performance report
    pub fn generate_performance_report(&self) -> Result<PerformanceReport> {
        let metrics = self.scheduler_metrics.read().unwrap();
        let device_metrics = self.device_metrics.read().unwrap();
        let aggregator = self.metrics_aggregator.read().unwrap();
        
        let mut device_summaries = Vec::new();
        for (device, device_metrics) in device_metrics.iter() {
            device_summaries.push(DeviceSummary {
                device: device.clone(),
                utilization: device_metrics.utilization,
                throughput: device_metrics.throughput,
                latency: device_metrics.latency,
                energy_consumption: device_metrics.energy_consumption,
                temperature: device_metrics.temperature,
                active_tasks: device_metrics.active_tasks,
                completed_tasks: device_metrics.completed_tasks,
                failed_tasks: device_metrics.failed_tasks,
            });
        }
        
        Ok(PerformanceReport {
            overall_metrics: metrics.clone(),
            device_summaries,
            aggregator: aggregator.clone(),
            recommendations: self.generate_recommendations()?,
            alerts: self.get_active_alerts()?,
        })
    }
    
    /// Record metrics event
    fn record_event(&self, event: MetricsEvent) -> Result<()> {
        let mut buffer = self.metrics_buffer.write().unwrap();
        buffer.push_back(event);
        
        // Keep only recent events
        if buffer.len() > 10000 {
            buffer.pop_front();
        }
        
        Ok(())
    }
    
    /// Update metrics aggregator
    fn update_aggregator(&self) -> Result<()> {
        let scheduler_metrics = self.scheduler_metrics.read().unwrap();
        let device_metrics = self.device_metrics.read().unwrap();
        
        let mut aggregator = self.metrics_aggregator.write().unwrap();
        
        aggregator.total_tasks = scheduler_metrics.total_tasks;
        aggregator.completed_tasks = scheduler_metrics.completed_tasks;
        aggregator.failed_tasks = scheduler_metrics.failed_tasks;
        aggregator.average_execution_time = scheduler_metrics.average_execution_time;
        aggregator.throughput = scheduler_metrics.throughput;
        aggregator.utilization = scheduler_metrics.resource_utilization.memory_usage;
        aggregator.error_rate = if aggregator.total_tasks > 0 {
            aggregator.failed_tasks as f32 / aggregator.total_tasks as f32
        } else {
            0.0
        };
        aggregator.last_updated = Instant::now();
        
        Ok(())
    }
    
    /// Check for alerts
    fn check_alerts(&self, device: &Device, utilization: f32) -> Result<()> {
        let thresholds = self.alert_thresholds.read().unwrap();
        
        if utilization > thresholds.high_utilization {
            self.record_event(MetricsEvent::AlertTriggered {
                alert_type: AlertType::HighUtilization,
                message: format!("High utilization on device {:?}: {:.2}%", device, utilization * 100.0),
                timestamp: Instant::now(),
            })?;
        }
        
        Ok(())
    }
    
    /// Generate recommendations
    fn generate_recommendations(&self) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();
        
        let aggregator = self.metrics_aggregator.read().unwrap();
        
        if aggregator.utilization > 0.9 {
            recommendations.push("Consider adding more devices or optimizing resource usage".to_string());
        }
        
        if aggregator.error_rate > 0.1 {
            recommendations.push("High error rate detected, investigate task failures".to_string());
        }
        
        if aggregator.throughput < 10.0 {
            recommendations.push("Low throughput detected, consider load balancing optimization".to_string());
        }
        
        Ok(recommendations)
    }
    
    /// Get active alerts
    fn get_active_alerts(&self) -> Result<Vec<Alert>> {
        let mut alerts = Vec::new();
        
        let aggregator = self.metrics_aggregator.read().unwrap();
        let thresholds = self.alert_thresholds.read().unwrap();
        
        if aggregator.utilization > thresholds.high_utilization {
            alerts.push(Alert {
                alert_type: AlertType::HighUtilization,
                message: format!("High system utilization: {:.2}%", aggregator.utilization * 100.0),
                severity: AlertSeverity::High,
                timestamp: Instant::now(),
            });
        }
        
        if aggregator.error_rate > thresholds.high_error_rate {
            alerts.push(Alert {
                alert_type: AlertType::HighErrorRate,
                message: format!("High error rate: {:.2}%", aggregator.error_rate * 100.0),
                severity: AlertSeverity::High,
                timestamp: Instant::now(),
            });
        }
        
        Ok(alerts)
    }
}

/// Device summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceSummary {
    pub device: Device,
    pub utilization: f32,
    pub throughput: f32,
    pub latency: Duration,
    pub energy_consumption: f32,
    pub temperature: f32,
    pub active_tasks: usize,
    pub completed_tasks: usize,
    pub failed_tasks: usize,
}

/// Performance report
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub overall_metrics: SchedulerMetrics,
    pub device_summaries: Vec<DeviceSummary>,
    pub aggregator: MetricsAggregator,
    pub recommendations: Vec<String>,
    pub alerts: Vec<Alert>,
}

/// Alert
#[derive(Debug, Clone)]
pub struct Alert {
    pub alert_type: AlertType,
    pub message: String,
    pub severity: AlertSeverity,
    pub timestamp: Instant,
}

/// Alert severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}
