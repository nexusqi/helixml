//! 🖥️ Device Manager
//! 
//! Device management and orchestration for multi-device scheduling

use tensor_core::{Tensor, Shape, DType, Device, Result, TensorError};
use tensor_core::tensor::{TensorOps, TensorRandom, TensorBroadcast, TensorMixedPrecision, TensorStats, TensorReduce};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use anyhow::Context;

use super::*;

/// Device manager for multi-device orchestration
#[derive(Debug)]
pub struct DeviceManager {
    devices: Arc<RwLock<HashMap<Device, DeviceInfo>>>,
    device_capabilities: Arc<RwLock<HashMap<Device, DeviceCapabilities>>>,
    device_utilization: Arc<RwLock<HashMap<Device, DeviceUtilization>>>,
    device_queues: Arc<RwLock<HashMap<Device, VecDeque<TaskId>>>>,
    device_monitors: Arc<RwLock<HashMap<Device, DeviceMonitor>>>,
}

/// Device information
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub device: Device,
    pub is_available: bool,
    pub current_load: f32,
    pub memory_usage: f32,
    pub compute_usage: f32,
    pub temperature: f32,
    pub power_consumption: f32,
    pub active_tasks: Vec<TaskId>,
    pub queue_length: usize,
    pub last_activity: Instant,
    pub performance_metrics: DevicePerformanceMetrics,
}

/// Device capabilities
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    pub max_memory: usize,
    pub max_compute: f32,
    pub max_bandwidth: f32,
    pub supported_operations: Vec<String>,
    pub special_features: Vec<String>,
    pub compute_capability: f32,
    pub memory_bandwidth: f32,
    pub cache_size: usize,
}

/// Device utilization
#[derive(Debug, Clone)]
pub struct DeviceUtilization {
    pub memory_utilization: f32,
    pub compute_utilization: f32,
    pub bandwidth_utilization: f32,
    pub queue_utilization: f32,
    pub overall_utilization: f32,
}

/// Device monitor
#[derive(Debug)]
pub struct DeviceMonitor {
    pub device: Device,
    pub monitoring_interval: Duration,
    pub last_update: Instant,
    pub metrics_history: VecDeque<DeviceMetrics>,
    pub alert_thresholds: AlertThresholds,
}

/// Device performance metrics
#[derive(Debug, Clone)]
pub struct DevicePerformanceMetrics {
    pub average_execution_time: Duration,
    pub throughput: f32,
    pub efficiency: f32,
    pub error_rate: f32,
    pub success_rate: f32,
}

/// Device metrics
#[derive(Debug, Clone)]
pub struct DeviceMetrics {
    pub timestamp: Instant,
    pub memory_usage: f32,
    pub compute_usage: f32,
    pub temperature: f32,
    pub power_consumption: f32,
    pub active_tasks: usize,
    pub queue_length: usize,
}

/// Alert thresholds
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub memory_threshold: f32,
    pub compute_threshold: f32,
    pub temperature_threshold: f32,
    pub power_threshold: f32,
}

impl DeviceManager {
    pub fn new() -> Result<Self> {
        let devices = Arc::new(RwLock::new(HashMap::new()));
        let device_capabilities = Arc::new(RwLock::new(HashMap::new()));
        let device_utilization = Arc::new(RwLock::new(HashMap::new()));
        let device_queues = Arc::new(RwLock::new(HashMap::new()));
        let device_monitors = Arc::new(RwLock::new(HashMap::new()));
        
        let manager = Self {
            devices,
            device_capabilities,
            device_utilization,
            device_queues,
            device_monitors,
        };
        
        // Initialize with default devices
        manager.initialize_default_devices()?;
        
        Ok(manager)
    }
    
    /// Initialize default devices
    fn initialize_default_devices(&self) -> Result<()> {
        // Add CPU device
        self.add_device(Device::Cpu)?;
        
        // Add CUDA devices if available
        for i in 0..4 { // Assume up to 4 CUDA devices
            let cuda_device = Device::Cuda(i);
            if self.is_device_available(&cuda_device)? {
                self.add_device(cuda_device)?;
            }
        }
        
        Ok(())
    }
    
    /// Add a device to the manager
    pub fn add_device(&self, device: Device) -> Result<()> {
        let device_info = DeviceInfo {
            device: device.clone(),
            is_available: true,
            current_load: 0.0,
            memory_usage: 0.0,
            compute_usage: 0.0,
            temperature: 25.0,
            power_consumption: 0.0,
            active_tasks: Vec::new(),
            queue_length: 0,
            last_activity: Instant::now(),
            performance_metrics: DevicePerformanceMetrics {
                average_execution_time: Duration::from_millis(100),
                throughput: 1.0,
                efficiency: 1.0,
                error_rate: 0.0,
                success_rate: 1.0,
            },
        };
        
        let capabilities = self.get_device_capabilities(&device)?;
        
        // Add to devices
        {
            let mut devices = self.devices.write().unwrap();
            devices.insert(device.clone(), device_info);
        }
        
        // Add capabilities
        {
            let mut caps = self.device_capabilities.write().unwrap();
            caps.insert(device.clone(), capabilities);
        }
        
        // Initialize utilization
        {
            let mut util = self.device_utilization.write().unwrap();
            util.insert(device.clone(), DeviceUtilization {
                memory_utilization: 0.0,
                compute_utilization: 0.0,
                bandwidth_utilization: 0.0,
                queue_utilization: 0.0,
                overall_utilization: 0.0,
            });
        }
        
        // Initialize queue
        {
            let mut queues = self.device_queues.write().unwrap();
            queues.insert(device.clone(), VecDeque::new());
        }
        
        // Initialize monitor
        {
            let mut monitors = self.device_monitors.write().unwrap();
            monitors.insert(device.clone(), DeviceMonitor {
                device: device.clone(),
                monitoring_interval: Duration::from_millis(100),
                last_update: Instant::now(),
                metrics_history: VecDeque::new(),
                alert_thresholds: AlertThresholds {
                    memory_threshold: 0.9,
                    compute_threshold: 0.9,
                    temperature_threshold: 80.0,
                    power_threshold: 300.0,
                },
            });
        }
        
        Ok(())
    }
    
    /// Remove a device from the manager
    pub fn remove_device(&self, device: &Device) -> Result<()> {
        // Move any active tasks to other devices
        self.migrate_active_tasks(device)?;
        
        // Remove from all collections
        {
            let mut devices = self.devices.write().unwrap();
            devices.remove(device);
        }
        
        {
            let mut caps = self.device_capabilities.write().unwrap();
            caps.remove(device);
        }
        
        {
            let mut util = self.device_utilization.write().unwrap();
            util.remove(device);
        }
        
        {
            let mut queues = self.device_queues.write().unwrap();
            queues.remove(device);
        }
        
        {
            let mut monitors = self.device_monitors.write().unwrap();
            monitors.remove(device);
        }
        
        Ok(())
    }
    
    /// Get available devices
    pub fn get_available_devices(&self) -> Result<Vec<Device>> {
        let devices = self.devices.read().unwrap();
        let available_devices: Vec<Device> = devices.values()
            .filter(|info| info.is_available && info.current_load < 0.9)
            .map(|info| info.device.clone())
            .collect();
        
        Ok(available_devices)
    }
    
    /// Get device status
    pub fn get_device_status(&self, device: &Device) -> Result<DeviceStatus> {
        let devices = self.devices.read().unwrap();
        if let Some(device_info) = devices.get(device) {
            Ok(DeviceStatus {
                device: device.clone(),
                is_available: device_info.is_available,
                current_load: device_info.current_load,
                memory_usage: device_info.memory_usage,
                compute_usage: device_info.compute_usage,
                temperature: device_info.temperature,
                power_consumption: device_info.power_consumption,
                active_tasks: device_info.active_tasks.len(),
                queue_length: device_info.queue_length,
            })
        } else {
            Err(TensorError::InvalidInput { message: "Device not found".to_string() })
        }
    }
    
    /// Get all device statuses
    pub fn get_all_status(&self) -> Result<HashMap<Device, DeviceStatus>> {
        let devices = self.devices.read().unwrap();
        let mut statuses = HashMap::new();
        
        for (device, info) in devices.iter() {
            statuses.insert(device.clone(), DeviceStatus {
                device: device.clone(),
                is_available: info.is_available,
                current_load: info.current_load,
                memory_usage: info.memory_usage,
                compute_usage: info.compute_usage,
                temperature: info.temperature,
                power_consumption: info.power_consumption,
                active_tasks: info.active_tasks.len(),
                queue_length: info.queue_length,
            });
        }
        
        Ok(statuses)
    }
    
    /// Execute a task on a device
    pub fn execute_task(&self, device: &Device, task: &Task) -> Result<Vec<u8>> {
        // Check if device is available
        if !self.is_device_available(device)? {
            return Err(TensorError::InvalidInput { message: "Device is not available".to_string() });
        }
        
        // Check resource requirements
        if !self.can_allocate_resources(device, task)? {
            return Err(TensorError::InvalidInput { message: "Insufficient resources on device".to_string() });
        }
        
        // Execute the task
        let start_time = Instant::now();
        let result = self.execute_task_operation(device, task)?;
        let execution_time = start_time.elapsed();
        
        // Update device metrics
        self.update_device_metrics(device, execution_time)?;
        
        Ok(result)
    }
    
    /// Execute task operation
    fn execute_task_operation(&self, device: &Device, task: &Task) -> Result<Vec<u8>> {
        match &task.operation {
            TaskOperation::TensorOperation { operation, input_shapes: _, output_shape: _ } => {
                self.execute_tensor_operation(device, operation)
            }
            TaskOperation::ModelInference { model, input_shapes: _ } => {
                self.execute_model_inference(device, model)
            }
            TaskOperation::TrainingStep { model, data_shapes: _, label_shapes: _ } => {
                self.execute_training_step(device, model)
            }
            TaskOperation::Custom { function, parameters } => {
                self.execute_custom_operation(device, function, parameters)
            }
        }
    }
    
    /// Execute tensor operation
    fn execute_tensor_operation(&self, device: &Device, operation: &TensorOp) -> Result<Vec<u8>> {
        // Placeholder for tensor operations
        // In practice, this would execute actual tensor operations
        Ok(vec![0u8])
    }
    
    /// Execute model inference
    fn execute_model_inference(&self, device: &Device, model: &str) -> Result<Vec<u8>> {
        // Placeholder for model inference
        // In practice, this would load the model and run inference
        Ok(vec![0u8])
    }
    
    /// Execute training step
    fn execute_training_step(&self, device: &Device, model: &str) -> Result<Vec<u8>> {
        // Placeholder for training step
        // In practice, this would run a training step
        Ok(vec![0u8])
    }
    
    /// Execute custom operation
    fn execute_custom_operation(&self, device: &Device, function: &str, parameters: &HashMap<String, serde_json::Value>) -> Result<Vec<u8>> {
        // Placeholder for custom operations
        // In practice, this would execute custom functions
        Ok(vec![])
    }
    
    /// Check if device is available
    fn is_device_available(&self, device: &Device) -> Result<bool> {
        let devices = self.devices.read().unwrap();
        if let Some(device_info) = devices.get(device) {
            Ok(device_info.is_available && device_info.current_load < 0.9)
        } else {
            Ok(false)
        }
    }
    
    /// Check if resources can be allocated
    fn can_allocate_resources(&self, device: &Device, task: &Task) -> Result<bool> {
        let devices = self.devices.read().unwrap();
        let capabilities = self.device_capabilities.read().unwrap();
        
        if let (Some(device_info), Some(caps)) = (devices.get(device), capabilities.get(device)) {
            let available_memory = (caps.max_memory as f32 * (1.0 - device_info.memory_usage)) as usize;
            let available_compute = caps.max_compute * (1.0 - device_info.compute_usage);
            
            Ok(task.resource_requirements.memory <= available_memory &&
                task.resource_requirements.compute <= available_compute)
        } else {
            Ok(false)
        }
    }
    
    /// Update device metrics
    fn update_device_metrics(&self, device: &Device, execution_time: Duration) -> Result<()> {
        let mut devices = self.devices.write().unwrap();
        if let Some(device_info) = devices.get_mut(device) {
            // Update performance metrics
            let total_time = device_info.performance_metrics.average_execution_time;
            let new_avg = (total_time + execution_time) / 2;
            device_info.performance_metrics.average_execution_time = new_avg;
            
            // Update throughput
            device_info.performance_metrics.throughput = 1.0 / execution_time.as_secs_f32();
            
            // Update last activity
            device_info.last_activity = Instant::now();
        }
        
        Ok(())
    }
    
    /// Get device capabilities
    fn get_device_capabilities(&self, device: &Device) -> Result<DeviceCapabilities> {
        match device {
            Device::Cpu => Ok(DeviceCapabilities {
                max_memory: 32 * 1024 * 1024 * 1024, // 32GB
                max_compute: 100.0,
                max_bandwidth: 100.0,
                supported_operations: vec![
                    "add".to_string(),
                    "subtract".to_string(),
                    "multiply".to_string(),
                    "divide".to_string(),
                    "matmul".to_string(),
                ],
                special_features: vec!["blas".to_string()],
                compute_capability: 1.0,
                memory_bandwidth: 100.0,
                cache_size: 32 * 1024 * 1024, // 32MB
            }),
            Device::Cuda(_) => Ok(DeviceCapabilities {
                max_memory: 16 * 1024 * 1024 * 1024, // 16GB
                max_compute: 1000.0,
                max_bandwidth: 1000.0,
                supported_operations: vec![
                    "add".to_string(),
                    "subtract".to_string(),
                    "multiply".to_string(),
                    "divide".to_string(),
                    "matmul".to_string(),
                    "convolution".to_string(),
                    "activation".to_string(),
                ],
                special_features: vec!["cuda".to_string(), "cublas".to_string()],
                compute_capability: 7.5,
                memory_bandwidth: 1000.0,
                cache_size: 6 * 1024 * 1024, // 6MB
            }),
            Device::Metal | Device::Wgpu | Device::Qpu(_) | Device::Npu(_) | Device::Tpu(_) | Device::Custom(_) => {
                // Default capabilities for other devices
                Ok(DeviceCapabilities {
                    max_memory: 8 * 1024 * 1024 * 1024, // 8GB
                    max_compute: 100.0,
                    max_bandwidth: 100.0,
                    supported_operations: vec!["add".to_string(), "multiply".to_string()],
                    special_features: vec![],
                    compute_capability: 1.0,
                    memory_bandwidth: 100.0,
                    cache_size: 8 * 1024 * 1024, // 8MB
                })
            },
        }
    }
    
    /// Migrate active tasks to other devices
    fn migrate_active_tasks(&self, device: &Device) -> Result<()> {
        let available_devices = self.get_available_devices()?;
        let devices_to_migrate: Vec<Device> = available_devices.into_iter()
            .filter(|d| d != device)
            .collect();
        
        if devices_to_migrate.is_empty() {
            return Err(TensorError::InvalidInput { message: "No available devices for migration".to_string() });
        }
        
        // Get active tasks on the device
        let active_tasks = {
            let devices = self.devices.read().unwrap();
            if let Some(device_info) = devices.get(device) {
                device_info.active_tasks.clone()
            } else {
                Vec::new()
            }
        };
        
        // Migrate tasks to other devices
        for task_id in active_tasks {
            // Find best alternative device
            let best_device = self.find_best_alternative_device(&devices_to_migrate, &task_id)?;
            
            // Migrate task
            self.migrate_task(&task_id, device, &best_device)?;
        }
        
        Ok(())
    }
    
    /// Find best alternative device
    fn find_best_alternative_device(&self, devices: &[Device], task_id: &TaskId) -> Result<Device> {
        // Simple strategy: choose device with lowest load
        let mut best_device = devices[0].clone();
        let mut best_load = f32::MAX;
        
        let device_statuses = self.get_all_status()?;
        
        for device in devices {
            if let Some(status) = device_statuses.get(device) {
                if status.current_load < best_load {
                    best_load = status.current_load;
                    best_device = device.clone();
                }
            }
        }
        
        Ok(best_device)
    }
    
    /// Migrate a task between devices
    fn migrate_task(&self, task_id: &TaskId, from_device: &Device, to_device: &Device) -> Result<()> {
        // Remove from source device
        {
            let mut devices = self.devices.write().unwrap();
            if let Some(device_info) = devices.get_mut(from_device) {
                device_info.active_tasks.retain(|id| id != task_id);
            }
        }
        
        // Add to destination device
        {
            let mut devices = self.devices.write().unwrap();
            if let Some(device_info) = devices.get_mut(to_device) {
                device_info.active_tasks.push(task_id.clone());
            }
        }
        
        Ok(())
    }
}
