//! 🌀 HelixML Compute Backend Trait
//! 
//! Core trait that EVERY compute backend must implement.
//! This is the foundation of hardware-agnostic computing.

use crate::{DeviceType, DeviceCapabilities, OperationType, DataType, Result};
use crate::memory::MemoryHandle;
use crate::operations::{Operation, ComputeGraph};
// use std::future::Future;
// use std::pin::Pin;

/// Core trait that every compute backend must implement
pub trait ComputeBackend: Send + Sync {
    /// Get device type
    fn device_type(&self) -> DeviceType;
    
    /// Get device identifier
    fn device_id(&self) -> String;
    
    /// Get device capabilities
    fn capabilities(&self) -> &DeviceCapabilities;
    
    /// Check if backend supports specific operation
    fn supports_operation(&self, op: OperationType) -> bool;
    
    /// Get optimal batch size for this device
    fn optimal_batch_size(&self) -> usize;
    
    /// Get supported data types
    fn supported_precisions(&self) -> Vec<DataType>;
    
    /// Get memory bandwidth in GB/s
    fn memory_bandwidth(&self) -> f64;
    
    /// Get number of compute units
    fn compute_units(&self) -> usize;
    
    /// Allocate memory on device
    fn allocate(&self, size: usize, dtype: DataType) -> Result<MemoryHandle>;
    
    /// Deallocate memory
    fn deallocate(&self, handle: MemoryHandle) -> Result<()>;
    
    /// Copy data to another device
    fn copy_to(&self, src: &MemoryHandle, dst_device: &dyn ComputeBackend) -> Result<MemoryHandle>;
    
    /// Execute single operation
    fn execute_op(&self, op: &Operation, inputs: &[&MemoryHandle]) -> Result<MemoryHandle>;
    
    /// Execute computation graph
    fn execute_graph(&self, graph: &ComputeGraph) -> Result<Vec<MemoryHandle>>;
    
    /// Compute topological features (for semantic understanding)
    fn compute_topological_features(&self, data: &MemoryHandle) -> Result<TopologicalFeatures>;
    
    /// Execute operation asynchronously
    fn execute_async(&self, op: &Operation, inputs: &[&MemoryHandle]) -> Result<AsyncHandle>;
    
    /// Wait for async operation completion
    fn wait(&self, handle: AsyncHandle) -> Result<MemoryHandle>;
    
    /// Check if device is available
    fn is_available(&self) -> bool;
    
    /// Get device utilization (0.0 to 1.0)
    fn utilization(&self) -> f64;
    
    /// Synchronize device (wait for all operations to complete)
    fn synchronize(&self) -> Result<()>;
}

/// Topological features extracted from data
#[derive(Debug, Clone)]
pub struct TopologicalFeatures {
    /// Motif signatures found
    pub motifs: Vec<MotifSignature>,
    /// Cycle patterns detected
    pub cycles: Vec<CyclePattern>,
    /// Stability score
    pub stability: f64,
    /// Semantic region
    pub semantic_region: Option<String>,
    /// Entropy measure
    pub entropy: f64,
}

/// Motif signature for pattern recognition
#[derive(Debug, Clone)]
pub struct MotifSignature {
    /// Pattern hash
    pub hash: u64,
    /// Frequency of occurrence
    pub frequency: usize,
    /// Stability score
    pub stability: f64,
    /// Position in sequence
    pub position: usize,
}

/// Cycle pattern for dependency analysis
#[derive(Debug, Clone)]
pub struct CyclePattern {
    /// Cycle nodes
    pub nodes: Vec<usize>,
    /// Cycle strength
    pub strength: f64,
    /// Period length
    pub period: Option<usize>,
    /// Stability score
    pub stability: f64,
}

/// Async operation handle
#[derive(Debug, Clone)]
pub struct AsyncHandle {
    /// Unique operation ID
    pub id: u64,
    /// Device that's executing
    pub device_id: String,
    /// Operation type
    pub operation: OperationType,
}

/// Backend registry for device discovery
pub struct BackendRegistry {
    backends: Vec<Box<dyn ComputeBackend>>,
}

impl BackendRegistry {
    /// Create new registry
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
        }
    }
    
    /// Register a backend
    pub fn register(&mut self, backend: Box<dyn ComputeBackend>) {
        self.backends.push(backend);
    }
    
    /// Get all available backends
    pub fn get_all(&self) -> &[Box<dyn ComputeBackend>] {
        &self.backends
    }
    
    /// Find backend by device type
    pub fn find_by_type(&self, device_type: DeviceType) -> Option<&dyn ComputeBackend> {
        self.backends.iter()
            .find(|b| b.device_type() == device_type)
            .map(|b| b.as_ref())
    }
    
    /// Find backend by device ID
    pub fn find_by_id(&self, device_id: &str) -> Option<&dyn ComputeBackend> {
        self.backends.iter()
            .find(|b| b.device_id() == device_id)
            .map(|b| b.as_ref())
    }
    
    /// Get backends that support specific operation
    pub fn find_supporting(&self, op: OperationType) -> Vec<&dyn ComputeBackend> {
        self.backends.iter()
            .filter(|b| b.supports_operation(op))
            .map(|b| b.as_ref())
            .collect()
    }
    
    /// Get optimal backend for operation
    pub fn get_optimal(&self, op: OperationType, _data_size: usize) -> Option<&dyn ComputeBackend> {
        let supporting: Vec<_> = self.find_supporting(op);
        
        if supporting.is_empty() {
            return None;
        }
        
        // Simple heuristic: prefer device with most available memory
        supporting.into_iter()
            .max_by_key(|b| b.capabilities().available_memory)
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}
