//! Device abstraction for tensor operations

use serde::{Deserialize, Serialize};
use std::fmt;

/// Device types for tensor computation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Device {
    /// CPU device
    Cpu,
    /// CUDA device with specific ID
    Cuda(usize),
    /// Metal device (macOS)
    Metal,
    /// WebGPU device
    Wgpu,
    /// Quantum Processing Unit
    Qpu(usize),
    /// Neural Processing Unit (Apple Neural Engine, etc.)
    Npu(usize),
    /// Tensor Processing Unit (Google TPU)
    Tpu(usize),
    /// Custom device
    Custom(String),
}

impl Device {
    /// Get the default CPU device
    pub fn cpu() -> Self {
        Self::Cpu
    }
    
    /// Get a CUDA device by ID
    pub fn cuda(id: usize) -> Self {
        Self::Cuda(id)
    }
    
    /// Get the default Metal device
    pub fn metal() -> Self {
        Self::Metal
    }
    
    /// Get a QPU device by ID
    pub fn qpu(id: usize) -> Self {
        Self::Qpu(id)
    }
    
    /// Get an NPU device by ID
    pub fn npu(id: usize) -> Self {
        Self::Npu(id)
    }
    
    /// Get a TPU device by ID
    pub fn tpu(id: usize) -> Self {
        Self::Tpu(id)
    }
    
    /// Check if this is a CPU device
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }
    
    /// Check if this is a GPU device
    pub fn is_gpu(&self) -> bool {
        matches!(self, Device::Cuda(_) | Device::Metal | Device::Wgpu)
    }
    
    /// Check if this is a specialized AI processor
    pub fn is_ai_processor(&self) -> bool {
        matches!(self, Device::Qpu(_) | Device::Npu(_) | Device::Tpu(_))
    }
    
    /// Check if this is a CUDA device
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }
    
    /// Check if this is a QPU device
    pub fn is_qpu(&self) -> bool {
        matches!(self, Device::Qpu(_))
    }
    
    /// Check if this is an NPU device
    pub fn is_npu(&self) -> bool {
        matches!(self, Device::Npu(_))
    }
    
    /// Check if this is a TPU device
    pub fn is_tpu(&self) -> bool {
        matches!(self, Device::Tpu(_))
    }
    
    /// Get CUDA device ID if applicable
    pub fn cuda_id(&self) -> Option<usize> {
        match self {
            Device::Cuda(id) => Some(*id),
            _ => None,
        }
    }
    
    /// Get QPU device ID if applicable
    pub fn qpu_id(&self) -> Option<usize> {
        match self {
            Device::Qpu(id) => Some(*id),
            _ => None,
        }
    }
    
    /// Get NPU device ID if applicable
    pub fn npu_id(&self) -> Option<usize> {
        match self {
            Device::Npu(id) => Some(*id),
            _ => None,
        }
    }
    
    /// Get TPU device ID if applicable
    pub fn tpu_id(&self) -> Option<usize> {
        match self {
            Device::Tpu(id) => Some(*id),
            _ => None,
        }
    }
    
    /// Get device name for display
    pub fn name(&self) -> String {
        match self {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(id) => format!("cuda:{}", id),
            Device::Metal => "metal".to_string(),
            Device::Wgpu => "wgpu".to_string(),
            Device::Qpu(id) => format!("qpu:{}", id),
            Device::Npu(id) => format!("npu:{}", id),
            Device::Tpu(id) => format!("tpu:{}", id),
            Device::Custom(name) => name.clone(),
        }
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::cpu()
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}
