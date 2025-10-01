//! 🌀 HelixML Memory Management
//! 
//! Unified memory system supporting multi-device operations with automatic synchronization.

use crate::{DeviceType, DataType, Result, HalError, ComputeBackend};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Memory handle for device-specific memory
#[derive(Debug, Clone)]
pub struct MemoryHandle {
    /// Unique memory ID
    pub id: u64,
    /// Device where memory is allocated
    pub device: DeviceType,
    /// Memory size in bytes
    pub size: usize,
    /// Data type
    pub dtype: DataType,
    /// Memory address (device-specific)
    pub address: usize,
    /// Reference count
    pub ref_count: Arc<Mutex<usize>>,
}

impl MemoryHandle {
    /// Create new memory handle
    pub fn new(id: u64, device: DeviceType, size: usize, dtype: DataType, address: usize) -> Self {
        Self {
            id,
            device,
            size,
            dtype,
            address,
            ref_count: Arc::new(Mutex::new(1)),
        }
    }
    
    /// Increment reference count
    pub fn add_ref(&self) {
        if let Ok(mut count) = self.ref_count.lock() {
            *count += 1;
        }
    }
    
    /// Decrement reference count
    pub fn remove_ref(&self) -> bool {
        if let Ok(mut count) = self.ref_count.lock() {
            *count = count.saturating_sub(1);
            *count == 0
        } else {
            false
        }
    }
    
    /// Get reference count
    pub fn ref_count(&self) -> usize {
        self.ref_count.lock().map(|c| *c).unwrap_or(0)
    }
}

/// Unified tensor supporting multi-device operations
pub struct UnifiedTensor {
    /// Data storage across devices
    locations: HashMap<DeviceType, MemoryHandle>,
    /// Primary device (where data is "canonical")
    primary_device: DeviceType,
    /// Tensor shape
    shape: Vec<usize>,
    /// Data type
    dtype: DataType,
    /// Version for synchronization
    version: u64,
    /// Dirty flags per device
    dirty_flags: HashMap<DeviceType, bool>,
    /// Topological metadata
    semantic_region: Option<String>,
    stability_score: Option<f64>,
    /// Synchronization locks per device
    sync_locks: HashMap<DeviceType, Arc<Mutex<bool>>>,
    /// Memory usage per device
    memory_usage: HashMap<DeviceType, usize>,
    /// Performance metrics per device
    performance_metrics: HashMap<DeviceType, DeviceMetrics>,
}

/// Device performance metrics
#[derive(Debug, Clone)]
pub struct DeviceMetrics {
    /// Last access time
    last_access: std::time::Instant,
    /// Access count
    access_count: u64,
    /// Transfer time
    transfer_time: std::time::Duration,
    /// Memory bandwidth
    bandwidth: f64,
}

impl UnifiedTensor {
    /// Create new unified tensor
    pub fn new(
        shape: Vec<usize>,
        dtype: DataType,
        primary_device: DeviceType,
        initial_handle: MemoryHandle,
    ) -> Self {
        let mut locations = HashMap::new();
        locations.insert(primary_device, initial_handle.clone());
        
        let mut dirty_flags = HashMap::new();
        dirty_flags.insert(primary_device, false);
        
        let mut sync_locks = HashMap::new();
        sync_locks.insert(primary_device, Arc::new(Mutex::new(false)));
        
        let mut memory_usage = HashMap::new();
        memory_usage.insert(primary_device, initial_handle.size);
        
        let mut performance_metrics = HashMap::new();
        performance_metrics.insert(primary_device, DeviceMetrics {
            last_access: std::time::Instant::now(),
            access_count: 1,
            transfer_time: std::time::Duration::ZERO,
            bandwidth: 0.0,
        });
        
        Self {
            locations,
            primary_device,
            shape,
            dtype,
            version: 0,
            dirty_flags,
            semantic_region: None,
            stability_score: None,
            sync_locks,
            memory_usage,
            performance_metrics,
        }
    }
    
    /// Get shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    /// Get data type
    pub fn dtype(&self) -> DataType {
        self.dtype
    }
    
    /// Get primary device
    pub fn primary_device(&self) -> DeviceType {
        self.primary_device
    }
    
    /// Get version
    pub fn version(&self) -> u64 {
        self.version
    }
    
    /// Check if tensor exists on device
    pub fn has_device(&self, device: DeviceType) -> bool {
        self.locations.contains_key(&device)
    }
    
    /// Get memory handle for device
    pub fn get_handle(&self, device: DeviceType) -> Option<&MemoryHandle> {
        self.locations.get(&device)
    }
    
    /// Add tensor to new device
    pub fn add_device(&mut self, device: DeviceType, handle: MemoryHandle) {
        let size = handle.size;
        self.locations.insert(device, handle);
        self.dirty_flags.insert(device, false);
        self.sync_locks.insert(device, Arc::new(Mutex::new(false)));
        self.memory_usage.insert(device, size);
        self.performance_metrics.insert(device, DeviceMetrics {
            last_access: std::time::Instant::now(),
            access_count: 0,
            transfer_time: std::time::Duration::ZERO,
            bandwidth: 0.0,
        });
    }
    
    /// Remove tensor from device
    pub fn remove_device(&mut self, device: DeviceType) -> Option<MemoryHandle> {
        self.dirty_flags.remove(&device);
        self.sync_locks.remove(&device);
        self.memory_usage.remove(&device);
        self.performance_metrics.remove(&device);
        self.locations.remove(&device)
    }
    
    /// Get all devices where tensor exists
    pub fn devices(&self) -> Vec<DeviceType> {
        self.locations.keys().cloned().collect()
    }
    
    /// Check if tensor is dirty on device
    pub fn is_dirty(&self, device: DeviceType) -> bool {
        self.dirty_flags.get(&device).copied().unwrap_or(false)
    }
    
    /// Mark tensor as dirty on device
    pub fn mark_dirty(&mut self, device: DeviceType) {
        self.dirty_flags.insert(device, true);
        self.version += 1;
    }
    
    /// Mark tensor as clean on device
    pub fn mark_clean(&mut self, device: DeviceType) {
        self.dirty_flags.insert(device, false);
    }
    
    /// Synchronize tensor across devices
    pub fn synchronize(&mut self, source_device: DeviceType, target_device: DeviceType) -> Result<()> {
        if !self.has_device(source_device) {
            return Err(HalError::OperationError {
                message: format!("Source device {:?} not found", source_device),
            });
        }
        
        if !self.has_device(target_device) {
            return Err(HalError::OperationError {
                message: format!("Target device {:?} not found", target_device),
            });
        }
        
        // Mark target as dirty since it will be updated
        self.mark_dirty(target_device);
        
        // Update performance metrics
        if let Some(metrics) = self.performance_metrics.get_mut(&target_device) {
            metrics.last_access = std::time::Instant::now();
            metrics.access_count += 1;
        }
        
        Ok(())
    }
    
    /// Get memory usage on device
    pub fn memory_usage(&self, device: DeviceType) -> usize {
        self.memory_usage.get(&device).copied().unwrap_or(0)
    }
    
    /// Get total memory usage across all devices
    pub fn total_memory_usage(&self) -> usize {
        self.memory_usage.values().sum()
    }
    
    /// Get performance metrics for device
    pub fn performance_metrics(&self, device: DeviceType) -> Option<&DeviceMetrics> {
        self.performance_metrics.get(&device)
    }
    
    /// Update performance metrics
    pub fn update_metrics(&mut self, device: DeviceType, transfer_time: std::time::Duration, bandwidth: f64) {
        if let Some(metrics) = self.performance_metrics.get_mut(&device) {
            metrics.transfer_time = transfer_time;
            metrics.bandwidth = bandwidth;
            metrics.last_access = std::time::Instant::now();
        }
    }
    
    /// Get optimal device for operation
    pub fn optimal_device(&self, operation: &str) -> DeviceType {
        // Simple heuristic: prefer GPU for compute-intensive operations
        match operation {
            "matmul" | "conv" | "fft" => {
                if self.has_device(DeviceType::CUDA) {
                    DeviceType::CUDA
                } else {
                    self.primary_device
                }
            },
            "add" | "sub" | "mul" | "div" => {
                // Prefer CPU for simple operations
                if self.has_device(DeviceType::CPU) {
                    DeviceType::CPU
                } else {
                    self.primary_device
                }
            },
            _ => self.primary_device,
        }
    }
    
    /// Set semantic region
    pub fn set_semantic_region(&mut self, region: Option<String>) {
        self.semantic_region = region;
    }
    
    /// Get semantic region
    pub fn semantic_region(&self) -> Option<&String> {
        self.semantic_region.as_ref()
    }
    
    /// Set stability score
    pub fn set_stability_score(&mut self, score: Option<f64>) {
        self.stability_score = score;
    }
    
    /// Get stability score
    pub fn stability_score(&self) -> Option<f64> {
        self.stability_score
    }
    
    /// Clone tensor to new device
    pub fn clone_to_device(&self, device: DeviceType, backend: &dyn ComputeBackend) -> Result<Self> {
        if !self.has_device(device) {
            return Err(HalError::OperationError {
                message: format!("Tensor not available on device {:?}", device),
            });
        }
        
        let source_handle = self.get_handle(device).unwrap();
        let new_handle = backend.allocate(source_handle.size, source_handle.dtype)?;
        
        let mut new_tensor = UnifiedTensor::new(
            self.shape.clone(),
            self.dtype,
            device,
            new_handle,
        );
        
        // Copy metadata
        new_tensor.semantic_region = self.semantic_region.clone();
        new_tensor.stability_score = self.stability_score;
        
        Ok(new_tensor)
    }
}

/// Memory pool for efficient allocation
pub struct MemoryPool {
    /// Available memory blocks
    free_blocks: HashMap<DeviceType, Vec<MemoryBlock>>,
    /// Allocated blocks
    allocated_blocks: HashMap<u64, MemoryBlock>,
    /// Next block ID
    next_id: u64,
}

/// Memory block for pooling
#[derive(Debug, Clone)]
struct MemoryBlock {
    id: u64,
    device: DeviceType,
    size: usize,
    dtype: DataType,
    address: usize,
    in_use: bool,
}

impl MemoryPool {
    /// Create new memory pool
    pub fn new() -> Self {
        Self {
            free_blocks: HashMap::new(),
            allocated_blocks: HashMap::new(),
            next_id: 0,
        }
    }
    
    /// Allocate memory block
    pub fn allocate(&mut self, device: DeviceType, size: usize, dtype: DataType) -> Result<MemoryHandle> {
        // Try to find existing free block
        if let Some(blocks) = self.free_blocks.get_mut(&device) {
            if let Some(pos) = blocks.iter().position(|b| b.size >= size && !b.in_use) {
                let block = blocks.remove(pos);
                let handle = MemoryHandle::new(block.id, device, size, dtype, block.address);
                self.allocated_blocks.insert(block.id, block);
                return Ok(handle);
            }
        }
        
        // Allocate new block
        let id = self.next_id;
        self.next_id += 1;
        
        let block = MemoryBlock {
            id,
            device,
            size,
            dtype,
            address: 0, // Would be set by actual backend
            in_use: true,
        };
        
        let handle = MemoryHandle::new(id, device, size, dtype, block.address);
        self.allocated_blocks.insert(id, block);
        
        Ok(handle)
    }
    
    /// Deallocate memory block
    pub fn deallocate(&mut self, handle: &MemoryHandle) -> Result<()> {
        if let Some(mut block) = self.allocated_blocks.remove(&handle.id) {
            block.in_use = false;
            self.free_blocks.entry(handle.device).or_insert_with(Vec::new).push(block);
            Ok(())
        } else {
            Err(HalError::MemoryError {
                message: format!("Memory block {} not found", handle.id),
            })
        }
    }
}

