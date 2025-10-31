//! 📊 Resource Monitor
//! 
//! Resource monitoring and management for multi-device scheduling

use tensor_core::{Tensor, Device, Result, TensorError};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use dashmap::DashMap;

use super::*;

/// Resource monitor for tracking and managing system resources
#[derive(Debug)]
pub struct ResourceMonitor {
    device_resources: Arc<DashMap<Device, DeviceResources>>,
    resource_allocations: Arc<DashMap<Device, Vec<ResourceAllocation>>>,
    resource_usage_history: Arc<RwLock<HashMap<Device, VecDeque<ResourceUsageSnapshot>>>>,
    monitoring_interval: Duration,
    last_update: Arc<Mutex<Instant>>,
    resource_limits: Arc<RwLock<ResourceLimits>>,
    alert_thresholds: Arc<RwLock<AlertThresholds>>,
    monitoring_thread: Option<std::thread::JoinHandle<()>>,
    is_monitoring: Arc<Mutex<bool>>,
}

/// Device resources
#[derive(Debug, Clone)]
pub struct DeviceResources {
    pub device: Device,
    pub total_memory: usize,
    pub available_memory: usize,
    pub total_compute: f32,
    pub available_compute: f32,
    pub total_bandwidth: f32,
    pub available_bandwidth: f32,
    pub total_storage: usize,
    pub available_storage: usize,
    pub utilization: ResourceUtilization,
    pub last_updated: Instant,
}

/// Resource allocation
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    pub task_id: TaskId,
    pub memory: usize,
    pub compute: f32,
    pub bandwidth: f32,
    pub storage: usize,
    pub allocated_at: Instant,
    pub estimated_duration: Duration,
}

/// Resource usage snapshot
#[derive(Debug, Clone)]
pub struct ResourceUsageSnapshot {
    pub timestamp: Instant,
    pub memory_usage: f32,
    pub compute_usage: f32,
    pub bandwidth_usage: f32,
    pub storage_usage: f32,
    pub active_allocations: usize,
}

/// Alert thresholds
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub memory_threshold: f32,
    pub compute_threshold: f32,
    pub bandwidth_threshold: f32,
    pub storage_threshold: f32,
    pub temperature_threshold: f32,
    pub power_threshold: f32,
}

impl ResourceMonitor {
    pub fn new(monitoring_interval: Duration) -> Result<Self> {
        let monitor = Self {
            device_resources: Arc::new(DashMap::new()),
            resource_allocations: Arc::new(DashMap::new()),
            resource_usage_history: Arc::new(RwLock::new(HashMap::new())),
            monitoring_interval,
            last_update: Arc::new(Mutex::new(Instant::now())),
            resource_limits: Arc::new(RwLock::new(ResourceLimits::default())),
            alert_thresholds: Arc::new(RwLock::new(AlertThresholds {
                memory_threshold: 0.9,
                compute_threshold: 0.9,
                bandwidth_threshold: 0.9,
                storage_threshold: 0.9,
                temperature_threshold: 80.0,
                power_threshold: 300.0,
            })),
            monitoring_thread: None,
            is_monitoring: Arc::new(Mutex::new(false)),
        };
        
        // Initialize with default devices
        monitor.initialize_default_devices()?;
        
        Ok(monitor)
    }
    
    /// Start resource monitoring
    pub fn start_monitoring(&mut self) -> Result<()> {
        let mut is_monitoring = self.is_monitoring.lock().unwrap();
        if *is_monitoring {
            return Err(TensorError::InvalidInput { message: "Monitoring is already running".to_string() });
        }
        
        *is_monitoring = true;
        
        // Start monitoring thread
        let device_resources = Arc::clone(&self.device_resources);
        let resource_allocations = Arc::clone(&self.resource_allocations);
        let resource_usage_history = Arc::clone(&self.resource_usage_history);
        let monitoring_interval = self.monitoring_interval;
        let is_monitoring = Arc::clone(&self.is_monitoring);
        
        let handle = std::thread::spawn(move || {
            while *is_monitoring.lock().unwrap() {
                // Update resource usage
                if let Err(e) = Self::update_resource_usage(
                    &device_resources,
                    &resource_allocations,
                    &resource_usage_history,
                ) {
                    eprintln!("Error updating resource usage: {}", e);
                }
                
                // Check for alerts
                if let Err(e) = Self::check_alerts(&device_resources) {
                    eprintln!("Error checking alerts: {}", e);
                }
                
                std::thread::sleep(monitoring_interval);
            }
        });
        
        self.monitoring_thread = Some(handle);
        Ok(())
    }
    
    /// Stop resource monitoring
    pub fn stop_monitoring(&mut self) -> Result<()> {
        let mut is_monitoring = self.is_monitoring.lock().unwrap();
        if !*is_monitoring {
            return Err(TensorError::InvalidInput { message: "Monitoring is not running".to_string() });
        }
        
        *is_monitoring = false;
        
        // Wait for monitoring thread to finish
        if let Some(handle) = self.monitoring_thread.take() {
            handle.join().map_err(|_| TensorError::InvalidInput { message: "Failed to join monitoring thread".to_string() })?;
        }
        
        Ok(())
    }
    
    /// Check if resources can be allocated
    pub fn can_allocate_resources(&self, device: &Device, task: &Task) -> Result<bool> {
        let device_resources = self.device_resources.get(device);
        if let Some(resources) = device_resources {
            let available_memory = resources.available_memory;
            let available_compute = resources.available_compute;
            let available_bandwidth = resources.available_bandwidth;
            let available_storage = resources.available_storage;
            
            Ok(task.resource_requirements.memory <= available_memory &&
                task.resource_requirements.compute <= available_compute &&
                task.resource_requirements.bandwidth <= available_bandwidth &&
                task.resource_requirements.storage <= available_storage)
        } else {
            Ok(false)
        }
    }
    
    /// Allocate resources for a task
    pub fn allocate_resources(&self, device: &Device, task: &Task) -> Result<ResourceAllocation> {
        if !self.can_allocate_resources(device, task)? {
            return Err(TensorError::InvalidInput { message: "Insufficient resources".to_string() });
        }
        
        let allocation = ResourceAllocation {
            task_id: TaskId::new(), // This should be passed from the caller
            memory: task.resource_requirements.memory,
            compute: task.resource_requirements.compute,
            bandwidth: task.resource_requirements.bandwidth,
            storage: task.resource_requirements.storage,
            allocated_at: Instant::now(),
            estimated_duration: Duration::from_secs(60), // Placeholder
        };
        
        // Update device resources
        if let Some(mut resources) = self.device_resources.get_mut(device) {
            resources.available_memory -= allocation.memory;
            resources.available_compute -= allocation.compute;
            resources.available_bandwidth -= allocation.bandwidth;
            resources.available_storage -= allocation.storage;
            resources.last_updated = Instant::now();
        }
        
        // Add to allocations
        self.resource_allocations.entry(device.clone())
            .or_insert_with(Vec::new)
            .push(allocation.clone());
        
        Ok(allocation)
    }
    
    /// Free resources for a task
    pub fn free_resources(&self, device: &Device, task: &Task) -> Result<()> {
        // Find and remove allocation
        if let Some(mut allocations) = self.resource_allocations.get_mut(device) {
            allocations.retain(|allocation| {
                // This is a simplified check - in practice, you'd match by task_id
                false // Remove all allocations for now
            });
        }
        
        // Update device resources
        if let Some(mut resources) = self.device_resources.get_mut(device) {
            // Restore resources (simplified)
            resources.available_memory += 1024 * 1024; // 1MB placeholder
            resources.available_compute += 1.0;
            resources.available_bandwidth += 1.0;
            resources.available_storage += 1024 * 1024; // 1MB placeholder
            resources.last_updated = Instant::now();
        }
        
        Ok(())
    }
    
    /// Get device resources
    pub fn get_device_resources(&self, device: &Device) -> Result<Option<DeviceResources>> {
        Ok(self.device_resources.get(device).map(|r| r.clone()))
    }
    
    /// Get resource utilization
    pub fn get_resource_utilization(&self, device: &Device) -> Result<Option<ResourceUtilization>> {
        if let Some(resources) = self.device_resources.get(device) {
            Ok(Some(resources.utilization.clone()))
        } else {
            Ok(None)
        }
    }
    
    /// Get resource usage history
    pub fn get_resource_usage_history(&self, device: &Device) -> Result<Vec<ResourceUsageSnapshot>> {
        let history = self.resource_usage_history.read().unwrap();
        Ok(history.get(device).map(|h| h.clone().into()).unwrap_or_default())
    }
    
    /// Update monitoring interval
    pub fn update_interval(&self, interval: Duration) -> Result<()> {
        // This would require restarting the monitoring thread
        // For now, just store the new interval
        Ok(())
    }
    
    /// Set resource limits
    pub fn set_resource_limits(&self, limits: ResourceLimits) -> Result<()> {
        let mut resource_limits = self.resource_limits.write().unwrap();
        *resource_limits = limits;
        Ok(())
    }
    
    /// Set alert thresholds
    pub fn set_alert_thresholds(&self, thresholds: AlertThresholds) -> Result<()> {
        let mut alert_thresholds = self.alert_thresholds.write().unwrap();
        *alert_thresholds = thresholds;
        Ok(())
    }
    
    /// Get resource statistics
    pub fn get_resource_statistics(&self) -> Result<ResourceStatistics> {
        let mut total_memory = 0;
        let mut total_compute = 0.0;
        let mut total_bandwidth = 0.0;
        let mut total_storage = 0;
        let mut available_memory = 0;
        let mut available_compute = 0.0;
        let mut available_bandwidth = 0.0;
        let mut available_storage = 0;
        let mut device_count = 0;
        
        for resources in self.device_resources.iter() {
            total_memory += resources.total_memory;
            total_compute += resources.total_compute;
            total_bandwidth += resources.total_bandwidth;
            total_storage += resources.total_storage;
            available_memory += resources.available_memory;
            available_compute += resources.available_compute;
            available_bandwidth += resources.available_bandwidth;
            available_storage += resources.available_storage;
            device_count += 1;
        }
        
        Ok(ResourceStatistics {
            total_memory,
            total_compute,
            total_bandwidth,
            total_storage,
            available_memory,
            available_compute,
            available_bandwidth,
            available_storage,
            device_count,
            memory_utilization: if total_memory > 0 {
                (total_memory - available_memory) as f32 / total_memory as f32
            } else {
                0.0
            },
            compute_utilization: if total_compute > 0.0 {
                (total_compute - available_compute) / total_compute
            } else {
                0.0
            },
            bandwidth_utilization: if total_bandwidth > 0.0 {
                (total_bandwidth - available_bandwidth) / total_bandwidth
            } else {
                0.0
            },
            storage_utilization: if total_storage > 0 {
                (total_storage - available_storage) as f32 / total_storage as f32
            } else {
                0.0
            },
        })
    }
    
    /// Initialize default devices
    fn initialize_default_devices(&self) -> Result<()> {
        // Add CPU device
        self.add_device(Device::Cpu)?;
        
        // Add CUDA devices if available
        for i in 0..4 {
            let cuda_device = Device::Cuda(i);
            if self.is_device_available(&cuda_device) {
                self.add_device(cuda_device)?;
            }
        }
        
        Ok(())
    }
    
    /// Add a device to monitoring
    fn add_device(&self, device: Device) -> Result<()> {
        let resources = self.get_device_resource_specs(&device)?;
        self.device_resources.insert(device.clone(), resources);
        
        // Initialize usage history
        {
            let mut history = self.resource_usage_history.write().unwrap();
            history.insert(device, VecDeque::new());
        }
        
        Ok(())
    }
    
    /// Check if device is available
    fn is_device_available(&self, device: &Device) -> bool {
        match device {
            Device::Cpu => true,
            Device::Cuda(_) => {
                // In practice, you'd check if CUDA is available
                true
            }
            _ => false, // Other devices not supported yet
        }
    }
    
    /// Get device resource specifications
    fn get_device_resource_specs(&self, device: &Device) -> Result<DeviceResources> {
        match device {
            Device::Cpu => Ok(DeviceResources {
                device: device.clone(),
                total_memory: 32 * 1024 * 1024 * 1024, // 32GB
                available_memory: 32 * 1024 * 1024 * 1024,
                total_compute: 100.0,
                available_compute: 100.0,
                total_bandwidth: 100.0,
                available_bandwidth: 100.0,
                total_storage: 100 * 1024 * 1024 * 1024, // 100GB
                available_storage: 100 * 1024 * 1024 * 1024,
                utilization: ResourceUtilization {
                    memory_usage: 0.0,
                    compute_usage: 0.0,
                    bandwidth_usage: 0.0,
                    storage_usage: 0.0,
                },
                last_updated: Instant::now(),
            }),
            Device::Cuda(_) => Ok(DeviceResources {
                device: device.clone(),
                total_memory: 16 * 1024 * 1024 * 1024, // 16GB
                available_memory: 16 * 1024 * 1024 * 1024,
                total_compute: 1000.0,
                available_compute: 1000.0,
                total_bandwidth: 1000.0,
                available_bandwidth: 1000.0,
                total_storage: 50 * 1024 * 1024 * 1024, // 50GB
                available_storage: 50 * 1024 * 1024 * 1024,
                utilization: ResourceUtilization {
                    memory_usage: 0.0,
                    compute_usage: 0.0,
                    bandwidth_usage: 0.0,
                    storage_usage: 0.0,
                },
                last_updated: Instant::now(),
            }),
            _ => Ok(DeviceResources {
                device: device.clone(),
                total_memory: 8 * 1024 * 1024 * 1024, // 8GB
                available_memory: 8 * 1024 * 1024 * 1024,
                total_compute: 100.0,
                available_compute: 100.0,
                total_bandwidth: 100.0,
                available_bandwidth: 100.0,
                total_storage: 20 * 1024 * 1024 * 1024, // 20GB
                available_storage: 20 * 1024 * 1024 * 1024,
                utilization: ResourceUtilization {
                    memory_usage: 0.0,
                    compute_usage: 0.0,
                    bandwidth_usage: 0.0,
                    storage_usage: 0.0,
                },
                last_updated: Instant::now(),
            }),
        }
    }
    
    /// Update resource usage
    fn update_resource_usage(
        device_resources: &Arc<DashMap<Device, DeviceResources>>,
        resource_allocations: &Arc<DashMap<Device, Vec<ResourceAllocation>>>,
        resource_usage_history: &Arc<RwLock<HashMap<Device, VecDeque<ResourceUsageSnapshot>>>>,
    ) -> Result<()> {
        for mut resources in device_resources.iter_mut() {
            let device = resources.key().clone();
            let resources = resources.value_mut();
            
            // Calculate current utilization
            let memory_usage = if resources.total_memory > 0 {
                (resources.total_memory - resources.available_memory) as f32 / resources.total_memory as f32
            } else {
                0.0
            };
            
            let compute_usage = if resources.total_compute > 0.0 {
                (resources.total_compute - resources.available_compute) / resources.total_compute
            } else {
                0.0
            };
            
            let bandwidth_usage = if resources.total_bandwidth > 0.0 {
                (resources.total_bandwidth - resources.available_bandwidth) / resources.total_bandwidth
            } else {
                0.0
            };
            
            let storage_usage = if resources.total_storage > 0 {
                (resources.total_storage - resources.available_storage) as f32 / resources.total_storage as f32
            } else {
                0.0
            };
            
            // Update utilization
            resources.utilization = ResourceUtilization {
                memory_usage,
                compute_usage,
                bandwidth_usage,
                storage_usage,
            };
            
            // Record usage snapshot
            let snapshot = ResourceUsageSnapshot {
                timestamp: Instant::now(),
                memory_usage,
                compute_usage,
                bandwidth_usage,
                storage_usage,
                active_allocations: resource_allocations.get(&device)
                    .map(|allocs| allocs.len())
                    .unwrap_or(0),
            };
            
            // Add to history
            {
                let mut history = resource_usage_history.write().unwrap();
                let device_history = history.entry(device.clone()).or_insert_with(VecDeque::new);
                device_history.push_back(snapshot);
                
                // Keep only recent history
                if device_history.len() > 1000 {
                    device_history.pop_front();
                }
            }
        }
        
        Ok(())
    }
    
    /// Check for resource alerts
    fn check_alerts(device_resources: &Arc<DashMap<Device, DeviceResources>>) -> Result<()> {
        for resources in device_resources.iter() {
            let utilization = &resources.utilization;
            
            // Check memory threshold
            if utilization.memory_usage > 0.9 {
                eprintln!("High memory usage on device {:?}: {:.2}%", 
                         resources.device, utilization.memory_usage * 100.0);
            }
            
            // Check compute threshold
            if utilization.compute_usage > 0.9 {
                eprintln!("High compute usage on device {:?}: {:.2}%", 
                         resources.device, utilization.compute_usage * 100.0);
            }
            
            // Check bandwidth threshold
            if utilization.bandwidth_usage > 0.9 {
                eprintln!("High bandwidth usage on device {:?}: {:.2}%", 
                         resources.device, utilization.bandwidth_usage * 100.0);
            }
        }
        
        Ok(())
    }
}

/// Resource statistics
#[derive(Debug, Clone)]
pub struct ResourceStatistics {
    pub total_memory: usize,
    pub total_compute: f32,
    pub total_bandwidth: f32,
    pub total_storage: usize,
    pub available_memory: usize,
    pub available_compute: f32,
    pub available_bandwidth: f32,
    pub available_storage: usize,
    pub device_count: usize,
    pub memory_utilization: f32,
    pub compute_utilization: f32,
    pub bandwidth_utilization: f32,
    pub storage_utilization: f32,
}
