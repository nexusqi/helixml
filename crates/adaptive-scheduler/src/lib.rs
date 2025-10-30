//! 🎯 Adaptive Scheduler for Multi-Device Orchestration
//! 
//! Advanced adaptive scheduling system for efficient multi-device
//! computation orchestration in HelixML

use tensor_core::{Tensor, Shape, DType, Device, Result, TensorError};
use tensor_core::tensor::{TensorOps, TensorRandom, TensorBroadcast, TensorMixedPrecision, TensorStats, TensorReduce};
use std::collections::{HashMap, VecDeque, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use anyhow::Context;
use uuid::Uuid;

pub mod scheduler;
pub mod device_manager;
pub mod task_queue;
pub mod load_balancer;
pub mod resource_monitor;
pub mod optimization;
pub mod policies;
pub mod metrics;
pub mod utils;

// Re-export main types (excluding ambiguous ones)
pub use scheduler::{CoreScheduler, SchedulerState, TaskNode, TaskDependency, SchedulerResourceAllocation, OptimizationResult};
pub use device_manager::*;
pub use task_queue::*;
pub use load_balancer::*;
pub use resource_monitor::*;
pub use optimization::{OptimizationEngine, OptimizationType as OptimizationStrategyType};
pub use policies::*;
pub use metrics::*;
pub use utils::*;

// Re-export Duration for convenience
pub use std::time::Duration as StdDuration;

/// Main adaptive scheduler for multi-device orchestration
pub struct AdaptiveScheduler {
    // Core components
    device_manager: Arc<DeviceManager>,
    task_queue: Arc<TaskQueue>,
    load_balancer: Arc<LoadBalancer>,
    resource_monitor: Arc<ResourceMonitor>,
    optimization_engine: Arc<OptimizationEngine>,
    policy_manager: Arc<PolicyManager>,
    metrics_collector: Arc<MetricsCollector>,
    
    // Configuration
    config: SchedulerConfig,
    is_running: Arc<Mutex<bool>>,
    scheduler_thread: Option<std::thread::JoinHandle<()>>,
}

/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_concurrent_tasks: usize,
    pub task_timeout: Duration,
    pub load_balancing_strategy: LoadBalancingStrategy,
    pub optimization_strategy: OptimizationStrategy,
    pub monitoring_interval: Duration,
    pub adaptive_threshold: f32,
    pub resource_limits: ResourceLimits,
    pub device_priorities: HashMap<Device, f32>,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: 100,
            task_timeout: Duration::from_secs(300),
            load_balancing_strategy: LoadBalancingStrategy::RoundRobin,
            optimization_strategy: OptimizationStrategy::Performance,
            monitoring_interval: Duration::from_millis(100),
            adaptive_threshold: 0.8,
            resource_limits: ResourceLimits::default(),
            device_priorities: HashMap::new(),
        }
    }
}

impl AdaptiveScheduler {
    pub fn new(config: SchedulerConfig) -> Result<Self> {
        let max_concurrent_tasks = config.max_concurrent_tasks;
        let load_balancing_strategy = config.load_balancing_strategy;
        let monitoring_interval = config.monitoring_interval;
        let optimization_strategy = config.optimization_strategy;
        
        let device_manager = Arc::new(DeviceManager::new()?);
        let task_queue = Arc::new(TaskQueue::new(max_concurrent_tasks)?);
        let load_balancer = Arc::new(LoadBalancer::new(load_balancing_strategy)?);
        let resource_monitor = Arc::new(ResourceMonitor::new(monitoring_interval)?);
        let optimization_engine = Arc::new(OptimizationEngine::new(optimization_strategy)?);
        let policy_manager = Arc::new(PolicyManager::new()?);
        let metrics_collector = Arc::new(MetricsCollector::new()?);
        
        Ok(Self {
            device_manager,
            task_queue,
            load_balancer,
            resource_monitor,
            optimization_engine,
            policy_manager,
            metrics_collector,
            config,
            is_running: Arc::new(Mutex::new(false)),
            scheduler_thread: None,
        })
    }
    
    /// Start the adaptive scheduler
    pub fn start(&mut self) -> Result<()> {
        let mut is_running = self.is_running.lock().unwrap();
        if *is_running {
            return Err(TensorError::InvalidInput { message: "Scheduler is already running".to_string() });
        }
        
        *is_running = true;
        
        // Start scheduler thread
        let scheduler_handle = self.start_scheduler_thread()?;
        self.scheduler_thread = Some(scheduler_handle);
        
        Ok(())
    }
    
    /// Stop the adaptive scheduler
    pub fn stop(&mut self) -> Result<()> {
        let mut is_running = self.is_running.lock().unwrap();
        if !*is_running {
            return Err(TensorError::InvalidInput { message: "Scheduler is not running".to_string() });
        }
        
        *is_running = false;
        
        // Wait for scheduler thread to finish
        if let Some(handle) = self.scheduler_thread.take() {
            handle.join().map_err(|_| TensorError::InvalidInput { message: "Failed to join scheduler thread".to_string() })?;
        }
        
        Ok(())
    }
    
    /// Submit a task for execution
    pub fn submit_task(&self, task: Task) -> Result<TaskId> {
        let task_id = TaskId::new();
        
        // Validate task
        self.validate_task(&task)?;
        
        // Add to task queue (clone task_id since enqueue may consume it)
        let task_id_clone = task_id.clone();
        self.task_queue.enqueue(task_id_clone, task)?;
        
        Ok(task_id)
    }
    
    /// Get task status
    pub fn get_task_status(&self, task_id: &TaskId) -> Result<TaskStatus> {
        self.task_queue.get_status(task_id)
    }
    
    /// Get task result
    pub fn get_task_result(&self, task_id: &TaskId) -> Result<Option<TaskResult>> {
        self.task_queue.get_result(task_id)
    }
    
    /// Cancel a task
    pub fn cancel_task(&self, task_id: &TaskId) -> Result<()> {
        self.task_queue.cancel(task_id)
    }
    
    /// Get scheduler metrics
    pub fn get_metrics(&self) -> Result<SchedulerMetrics> {
        self.metrics_collector.get_metrics()
    }
    
    /// Get device status
    pub fn get_device_status(&self) -> Result<HashMap<Device, DeviceStatus>> {
        self.device_manager.get_all_status()
    }
    
    /// Update scheduler configuration
    pub fn update_config(&mut self, config: SchedulerConfig) -> Result<()> {
        let load_balancing_strategy = config.load_balancing_strategy;
        let optimization_strategy = config.optimization_strategy;
        let monitoring_interval = config.monitoring_interval;
        
        self.config = config;
        
        // Update components with new configuration
        self.load_balancer.update_strategy(load_balancing_strategy)?;
        self.optimization_engine.update_strategy(optimization_strategy)?;
        self.resource_monitor.update_interval(monitoring_interval)?;
        
        Ok(())
    }
    
    fn start_scheduler_thread(&self) -> Result<std::thread::JoinHandle<()>> {
        let device_manager = Arc::clone(&self.device_manager);
        let task_queue = Arc::clone(&self.task_queue);
        let load_balancer = Arc::clone(&self.load_balancer);
        let resource_monitor = Arc::clone(&self.resource_monitor);
        let optimization_engine = Arc::clone(&self.optimization_engine);
        let policy_manager = Arc::clone(&self.policy_manager);
        let metrics_collector = Arc::clone(&self.metrics_collector);
        let is_running = Arc::clone(&self.is_running);
        let config = self.config.clone();
        
        let handle = std::thread::spawn(move || {
            let mut last_optimization = Instant::now();
            let optimization_interval = Duration::from_secs(1);
            
            while *is_running.lock().unwrap() {
                // Process tasks
                if let Err(e) = Self::process_tasks(
                    &device_manager,
                    &task_queue,
                    &load_balancer,
                    &resource_monitor,
                    &optimization_engine,
                    &policy_manager,
                    &metrics_collector,
                    &config,
                ) {
                    eprintln!("Error processing tasks: {}", e);
                }
                
                // Periodic optimization
                if last_optimization.elapsed() >= optimization_interval {
                    if let Err(e) = Self::optimize_scheduling(
                        &device_manager,
                        &task_queue,
                        &load_balancer,
                        &resource_monitor,
                        &optimization_engine,
                        &policy_manager,
                        &metrics_collector,
                        &config,
                    ) {
                        eprintln!("Error optimizing scheduling: {}", e);
                    }
                    last_optimization = Instant::now();
                }
                
                // Small delay to prevent busy waiting
                std::thread::sleep(Duration::from_millis(10));
            }
        });
        
        Ok(handle)
    }
    
    fn process_tasks(
        device_manager: &Arc<DeviceManager>,
        task_queue: &Arc<TaskQueue>,
        load_balancer: &Arc<LoadBalancer>,
        resource_monitor: &Arc<ResourceMonitor>,
        optimization_engine: &Arc<OptimizationEngine>,
        policy_manager: &Arc<PolicyManager>,
        metrics_collector: &Arc<MetricsCollector>,
        config: &SchedulerConfig,
    ) -> Result<()> {
        // Get available devices
        let available_devices = device_manager.get_available_devices()?;
        
        if available_devices.is_empty() {
            return Ok(());
        }
        
        // Get pending tasks
        let pending_tasks = task_queue.get_pending_tasks()?;
        
        for task_info in pending_tasks {
            // Find best device for task
            let best_device = load_balancer.select_device(&available_devices, &task_info.task)?;
            
            // Check resource requirements
            if !resource_monitor.can_allocate_resources(&best_device, &task_info.task)? {
                continue;
            }
            
            // Apply scheduling policies
            if !policy_manager.should_schedule(&task_info.task, &best_device)? {
                continue;
            }
            
            // Execute task
            if let Err(e) = Self::execute_task(
                device_manager,
                task_queue,
                resource_monitor,
                metrics_collector,
                &task_info.task_id,
                &task_info.task,
                &best_device,
            ) {
                eprintln!("Error executing task {:?}: {}", task_info.task_id, e);
            }
        }
        
        Ok(())
    }
    
    fn optimize_scheduling(
        device_manager: &Arc<DeviceManager>,
        task_queue: &Arc<TaskQueue>,
        load_balancer: &Arc<LoadBalancer>,
        resource_monitor: &Arc<ResourceMonitor>,
        optimization_engine: &Arc<OptimizationEngine>,
        policy_manager: &Arc<PolicyManager>,
        metrics_collector: &Arc<MetricsCollector>,
        config: &SchedulerConfig,
    ) -> Result<()> {
        // Collect current metrics
        let metrics = metrics_collector.get_metrics()?;
        
        // Check if optimization is needed
        if metrics.load_factor < config.adaptive_threshold {
            return Ok(());
        }
        
        // Get current scheduling state
        let device_status = device_manager.get_all_status()?;
        let task_status = task_queue.get_all_status()?;
        
        // Run optimization
        let optimization_result = optimization_engine.optimize(
            &device_status,
            &task_status,
            &metrics,
        )?;
        
        // Apply optimization results - fields are in critical_path, bottlenecks, optimizations
        // Skip automatic rescheduling for now
        
        Ok(())
    }
    
    fn execute_task(
        device_manager: &Arc<DeviceManager>,
        task_queue: &Arc<TaskQueue>,
        resource_monitor: &Arc<ResourceMonitor>,
        metrics_collector: &Arc<MetricsCollector>,
        task_id: &TaskId,
        task: &Task,
        device: &Device,
    ) -> Result<()> {
        // Mark task as running
        task_queue.set_status(task_id, TaskStatus::Running)?;
        
        // Allocate resources
        resource_monitor.allocate_resources(device, task)?;
        
        // Execute task on device
        let start_time = Instant::now();
        let result = device_manager.execute_task(device, task)?;
        let execution_time = start_time.elapsed();
        
        // Update metrics
        metrics_collector.record_task_execution(task_id, execution_time, device)?;
        
        // Free resources
        resource_monitor.free_resources(device, task)?;
        
        // Set task result (result is Vec<u8>, ignore for now)
        task_queue.set_result(task_id, TaskResult::Success)?;
        task_queue.set_status(task_id, TaskStatus::Completed)?;
        
        Ok(())
    }
    
    fn validate_task(&self, task: &Task) -> Result<()> {
        // Check task requirements
        if task.resource_requirements.memory > self.config.resource_limits.max_memory {
            return Err(TensorError::InvalidInput { message: "Task requires more memory than available".to_string() });
        }
        
        if task.resource_requirements.compute > self.config.resource_limits.max_compute {
            return Err(TensorError::InvalidInput { message: "Task requires more compute than available".to_string() });
        }
        
        // Check device compatibility
        let available_devices = self.device_manager.get_available_devices()?;
        let compatible_devices = available_devices.iter()
            .filter(|device| task.device_requirements.is_compatible(device))
            .count();
        
        if compatible_devices == 0 {
            return Err(TensorError::InvalidInput { message: "No compatible devices available for task".to_string() });
        }
        
        Ok(())
    }
}

/// Task identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TaskId {
    id: Uuid,
}

impl TaskId {
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
        }
    }
    
    pub fn id(&self) -> &Uuid {
        &self.id
    }
}

/// Task for execution
#[derive(Debug, Clone)]
pub struct Task {
    pub operation: TaskOperation,
    pub priority: TaskPriority,
    pub resource_requirements: ResourceRequirements,
    pub device_requirements: DeviceRequirements,
    pub timeout: Duration,
    pub retry_count: usize,
    pub max_retries: usize,
}

/// Task operation types
#[derive(Debug, Clone)]
pub enum TaskOperation {
    TensorOperation {
        operation: TensorOp,
        input_shapes: Vec<Shape>,
        output_shape: Shape,
    },
    ModelInference {
        model: String,
        input_shapes: Vec<Shape>,
    },
    TrainingStep {
        model: String,
        data_shapes: Vec<Shape>,
        label_shapes: Vec<Shape>,
    },
    Custom {
        function: String,
        parameters: HashMap<String, serde_json::Value>,
    },
}

/// Tensor operations
#[derive(Debug, Clone)]
pub enum TensorOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    MatrixMultiply,
    Convolution,
    Activation,
    Normalization,
    Pooling,
    Reshape,
    Transpose,
    Broadcast,
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TaskPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Resource requirements
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    pub memory: usize,
    pub compute: f32,
    pub bandwidth: f32,
    pub storage: usize,
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            memory: 1024 * 1024, // 1MB
            compute: 1.0,
            bandwidth: 1.0,
            storage: 0,
        }
    }
}

/// Device requirements
#[derive(Debug, Clone)]
pub struct DeviceRequirements {
    pub device_types: Vec<Device>,
    pub min_memory: usize,
    pub min_compute_capability: f32,
    pub special_features: Vec<String>,
}

impl DeviceRequirements {
    pub fn is_compatible(&self, device: &Device) -> bool {
        self.device_types.contains(device)
    }
}

impl Default for DeviceRequirements {
    fn default() -> Self {
        Self {
            device_types: vec![Device::Cpu],
            min_memory: 0,
            min_compute_capability: 0.0,
            special_features: vec![],
        }
    }
}

/// Resource limits
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    pub max_memory: usize,
    pub max_compute: f32,
    pub max_bandwidth: f32,
    pub max_storage: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory: 8 * 1024 * 1024 * 1024, // 8GB
            max_compute: 100.0,
            max_bandwidth: 100.0,
            max_storage: 100 * 1024 * 1024 * 1024, // 100GB
        }
    }
}

/// Task status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
    Timeout,
}

/// Task result
#[derive(Debug, Clone)]
pub enum TaskResult {
    Success,
    Error(String),
    Timeout,
    Cancelled,
}

/// Load balancing strategies
#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    WeightedRoundRobin,
    LeastConnections,
    ResourceBased,
    PerformanceBased,
    Adaptive,
}

/// Optimization strategies
#[derive(Debug, Clone, Copy)]
pub enum OptimizationStrategy {
    Performance,
    Throughput,
    Latency,
    ResourceUtilization,
    EnergyEfficiency,
    Balanced,
}

/// Scheduler metrics
#[derive(Debug, Clone)]
pub struct SchedulerMetrics {
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub failed_tasks: usize,
    pub pending_tasks: usize,
    pub running_tasks: usize,
    pub average_execution_time: Duration,
    pub throughput: f32,
    pub load_factor: f32,
    pub device_utilization: HashMap<Device, f32>,
    pub resource_utilization: ResourceUtilization,
}

/// Resource utilization
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    pub memory_usage: f32,
    pub compute_usage: f32,
    pub bandwidth_usage: f32,
    pub storage_usage: f32,
}

/// Device status
#[derive(Debug, Clone)]
pub struct DeviceStatus {
    pub device: Device,
    pub is_available: bool,
    pub current_load: f32,
    pub memory_usage: f32,
    pub compute_usage: f32,
    pub temperature: f32,
    pub power_consumption: f32,
    pub active_tasks: usize,
    pub queue_length: usize,
}
