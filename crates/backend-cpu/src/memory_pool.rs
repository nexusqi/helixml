//! 🌀 HelixML Memory Pool
//! 
//! Efficient memory management for CPU backend.

use hal::{Result, HalError, DeviceType, DataType};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::ptr::NonNull;

/// Memory pool for efficient allocation
pub struct MemoryPool {
    /// Pooled memory blocks
    blocks: HashMap<(usize, DataType), Vec<MemoryBlock>>,
    /// Total allocated memory
    total_allocated: usize,
    /// Maximum memory limit
    max_memory: usize,
    /// Memory statistics
    stats: MemoryStats,
}

/// Memory block
#[derive(Debug, Clone)]
struct MemoryBlock {
    /// Pointer to memory
    ptr: NonNull<u8>,
    /// Size in bytes
    size: usize,
    /// Data type
    dtype: DataType,
    /// Whether block is in use
    in_use: bool,
    /// Allocation timestamp
    timestamp: std::time::Instant,
}

/// Memory statistics
#[derive(Debug, Clone)]
struct MemoryStats {
    /// Total allocations
    total_allocations: usize,
    /// Total deallocations
    total_deallocations: usize,
    /// Peak memory usage
    peak_memory: usize,
    /// Current memory usage
    current_memory: usize,
    /// Memory fragmentation
    fragmentation: f64,
}

impl MemoryPool {
    /// Create new memory pool
    pub fn new() -> Self {
        Self {
            blocks: HashMap::new(),
            total_allocated: 0,
            max_memory: 32 * 1024 * 1024 * 1024, // 32GB default
            stats: MemoryStats {
                total_allocations: 0,
                total_deallocations: 0,
                peak_memory: 0,
                current_memory: 0,
                fragmentation: 0.0,
            },
        }
    }
    
    /// Set maximum memory limit
    pub fn set_max_memory(&mut self, max_memory: usize) {
        self.max_memory = max_memory;
    }
    
    /// Allocate memory block
    pub fn allocate(&mut self, device: DeviceType, size: usize, dtype: DataType) -> Result<hal::memory::MemoryHandle> {
        if device != DeviceType::CPU {
            return Err(HalError::OperationError {
                message: format!("Unsupported device: {:?}", device),
            });
        }
        
        // Check memory limit
        if self.total_allocated + size > self.max_memory {
            return Err(HalError::OperationError {
                message: format!("Out of memory: requested {}, available {}", size, self.max_memory - self.total_allocated),
            });
        }
        
        // Try to reuse existing block
        let key = (size, dtype);
        if let Some(blocks) = self.blocks.get_mut(&key) {
            for block in blocks.iter_mut() {
                if !block.in_use {
                    block.in_use = true;
                    block.timestamp = std::time::Instant::now();
                    self.stats.total_allocations += 1;
                    self.stats.current_memory += size;
                    self.stats.peak_memory = self.stats.peak_memory.max(self.stats.current_memory);
                    
                    return Ok(hal::memory::MemoryHandle::new(
                        block.ptr.as_ptr() as u64,
                        device,
                        size,
                        dtype,
                        block.ptr.as_ptr() as usize,
                    ));
                }
            }
        }
        
        // Allocate new block
        let layout = std::alloc::Layout::from_size_align(size, 8)
            .map_err(|_| HalError::OperationError {
                message: "Invalid memory layout".to_string(),
            })?;
        
        let ptr = unsafe {
            std::alloc::alloc(layout)
        };
        
        if ptr.is_null() {
            return Err(HalError::OperationError {
                message: "Failed to allocate memory".to_string(),
            });
        }
        
        let non_null_ptr = NonNull::new(ptr).unwrap();
        let block = MemoryBlock {
            ptr: non_null_ptr,
            size,
            dtype,
            in_use: true,
            timestamp: std::time::Instant::now(),
        };
        
        // Add to pool
        self.blocks.entry(key).or_insert_with(Vec::new).push(block);
        
        self.total_allocated += size;
        self.stats.total_allocations += 1;
        self.stats.current_memory += size;
        self.stats.peak_memory = self.stats.peak_memory.max(self.stats.current_memory);
        
        Ok(hal::memory::MemoryHandle::new(
            ptr as u64,
            device,
            size,
            dtype,
            ptr as usize,
        ))
    }
    
    /// Deallocate memory block
    pub fn deallocate(&mut self, handle: &hal::memory::MemoryHandle) -> Result<()> {
        if handle.device != DeviceType::CPU {
            return Err(HalError::OperationError {
                message: format!("Unsupported device: {:?}", handle.device),
            });
        }
        
        // Find and mark block as unused
        let key = (handle.size, handle.dtype);
        if let Some(blocks) = self.blocks.get_mut(&key) {
            for block in blocks.iter_mut() {
                if block.ptr.as_ptr() as usize == handle.address && block.in_use {
                    block.in_use = false;
                    self.stats.total_deallocations += 1;
                    self.stats.current_memory -= handle.size;
                    return Ok(());
                }
            }
        }
        
        Err(HalError::OperationError {
            message: format!("Invalid handle: {}", handle.id),
        })
    }
    
    /// Get memory statistics
    pub fn stats(&self) -> &MemoryStats {
        &self.stats
    }
    
    /// Get current memory usage
    pub fn current_memory(&self) -> usize {
        self.stats.current_memory
    }
    
    /// Get peak memory usage
    pub fn peak_memory(&self) -> usize {
        self.stats.peak_memory
    }
    
    /// Get memory fragmentation
    pub fn fragmentation(&self) -> f64 {
        self.stats.fragmentation
    }
    
    /// Cleanup unused blocks
    pub fn cleanup(&mut self) {
        let now = std::time::Instant::now();
        let cleanup_threshold = std::time::Duration::from_secs(60); // 1 minute
        
        for blocks in self.blocks.values_mut() {
            blocks.retain(|block| {
                if !block.in_use && now.duration_since(block.timestamp) > cleanup_threshold {
                    // Deallocate block
                    unsafe {
                        let layout = std::alloc::Layout::from_size_align(block.size, 8).unwrap();
                        std::alloc::dealloc(block.ptr.as_ptr(), layout);
                    }
                    false
                } else {
                    true
                }
            });
        }
    }
    
    /// Defragment memory
    pub fn defragment(&mut self) -> Result<()> {
        // TODO: Implement memory defragmentation
        // For now, just cleanup unused blocks
        self.cleanup();
        Ok(())
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        // Deallocate all remaining blocks
        for blocks in self.blocks.values() {
            for block in blocks {
                if block.in_use {
                    unsafe {
                        let layout = std::alloc::Layout::from_size_align(block.size, 8).unwrap();
                        std::alloc::dealloc(block.ptr.as_ptr(), layout);
                    }
                }
            }
        }
    }
}

/// Memory pool manager
pub struct MemoryPoolManager {
    /// CPU memory pool
    cpu_pool: Arc<Mutex<MemoryPool>>,
    /// Memory pool statistics
    global_stats: Arc<Mutex<GlobalMemoryStats>>,
}

/// Global memory statistics
#[derive(Debug, Clone)]
struct GlobalMemoryStats {
    /// Total memory allocated across all devices
    total_memory: usize,
    /// Peak memory usage
    peak_memory: usize,
    /// Memory efficiency
    efficiency: f64,
    /// Allocation patterns
    allocation_patterns: HashMap<String, usize>,
}

impl MemoryPoolManager {
    /// Create new memory pool manager
    pub fn new() -> Self {
        Self {
            cpu_pool: Arc::new(Mutex::new(MemoryPool::new())),
            global_stats: Arc::new(Mutex::new(GlobalMemoryStats {
                total_memory: 0,
                peak_memory: 0,
                efficiency: 1.0,
                allocation_patterns: HashMap::new(),
            })),
        }
    }
    
    /// Get CPU memory pool
    pub fn cpu_pool(&self) -> Arc<Mutex<MemoryPool>> {
        self.cpu_pool.clone()
    }
    
    /// Get global memory statistics
    pub fn global_stats(&self) -> Arc<Mutex<GlobalMemoryStats>> {
        self.global_stats.clone()
    }
    
    /// Update global statistics
    pub fn update_stats(&self) {
        let mut global_stats = self.global_stats.lock().unwrap();
        let cpu_pool = self.cpu_pool.lock().unwrap();
        
        global_stats.total_memory = cpu_pool.current_memory();
        global_stats.peak_memory = global_stats.peak_memory.max(global_stats.total_memory);
        
        // Calculate efficiency
        if global_stats.peak_memory > 0 {
            global_stats.efficiency = global_stats.total_memory as f64 / global_stats.peak_memory as f64;
        }
    }
    
    /// Cleanup all memory pools
    pub fn cleanup_all(&self) {
        let mut cpu_pool = self.cpu_pool.lock().unwrap();
        cpu_pool.cleanup();
    }
    
    /// Defragment all memory pools
    pub fn defragment_all(&self) -> Result<()> {
        let mut cpu_pool = self.cpu_pool.lock().unwrap();
        cpu_pool.defragment()?;
        Ok(())
    }
}

impl Default for MemoryPoolManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_pool_allocation() {
        let mut pool = MemoryPool::new();
        let handle = pool.allocate(DeviceType::CPU, 1024, DataType::F32).unwrap();
        
        assert_eq!(handle.size, 1024);
        assert_eq!(handle.dtype, DataType::F32);
        assert_eq!(handle.device, DeviceType::CPU);
        
        pool.deallocate(&handle).unwrap();
    }
    
    #[test]
    fn test_memory_pool_reuse() {
        let mut pool = MemoryPool::new();
        
        // Allocate and deallocate
        let handle1 = pool.allocate(DeviceType::CPU, 1024, DataType::F32).unwrap();
        pool.deallocate(&handle1).unwrap();
        
        // Allocate again - should reuse
        let handle2 = pool.allocate(DeviceType::CPU, 1024, DataType::F32).unwrap();
        assert_eq!(handle2.size, 1024);
        assert_eq!(handle2.dtype, DataType::F32);
        
        pool.deallocate(&handle2).unwrap();
    }
    
    #[test]
    fn test_memory_pool_stats() {
        let mut pool = MemoryPool::new();
        
        let handle = pool.allocate(DeviceType::CPU, 1024, DataType::F32).unwrap();
        let stats = pool.stats();
        
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.current_memory, 1024);
        
        pool.deallocate(&handle).unwrap();
        
        let stats = pool.stats();
        assert_eq!(stats.total_deallocations, 1);
        assert_eq!(stats.current_memory, 0);
    }
}
