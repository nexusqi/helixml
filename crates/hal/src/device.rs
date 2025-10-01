//! 🌀 HelixML Device Types and Capabilities

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported device types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceType {
    /// CPU (x86_64, ARM, RISC-V, etc.)
    CPU,
    /// NVIDIA CUDA GPU
    CUDA,
    /// Apple Metal GPU
    Metal,
    /// AMD ROCm GPU
    ROCm,
    /// Neural Processing Unit (Apple Neural Engine, Google Edge TPU, etc.)
    NPU,
    /// Tensor Processing Unit (Google TPU)
    TPU,
    /// Quantum Processing Unit
    QPU,
    /// Brain-Computer Interface
    BCI,
    /// Custom device type
    Custom,
}

impl DeviceType {
    /// Get human-readable name
    pub fn name(&self) -> &str {
        match self {
            DeviceType::CPU => "CPU",
            DeviceType::CUDA => "CUDA",
            DeviceType::Metal => "Metal",
            DeviceType::ROCm => "ROCm",
            DeviceType::NPU => "NPU",
            DeviceType::TPU => "TPU",
            DeviceType::QPU => "QPU",
            DeviceType::BCI => "BCI",
            DeviceType::Custom => "Custom",
        }
    }
    
    /// Check if device supports floating point operations
    pub fn supports_fp(&self) -> bool {
        match self {
            DeviceType::CPU | DeviceType::CUDA | DeviceType::Metal | DeviceType::ROCm => true,
            DeviceType::NPU | DeviceType::TPU => true,
            DeviceType::QPU | DeviceType::BCI => false, // Specialized operations
            DeviceType::Custom => true, // Assume support unless specified
        }
    }
    
    /// Check if device supports complex numbers
    pub fn supports_complex(&self) -> bool {
        match self {
            DeviceType::CPU | DeviceType::CUDA | DeviceType::Metal | DeviceType::ROCm => true,
            DeviceType::NPU | DeviceType::TPU => false, // Usually quantized
            DeviceType::QPU => true, // Quantum states are complex
            DeviceType::BCI => false,
            DeviceType::Custom => true,
        }
    }
}

/// Supported data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataType {
    /// 8-bit integer
    I8,
    /// 16-bit integer
    I16,
    /// 32-bit integer
    I32,
    /// 64-bit integer
    I64,
    /// 8-bit unsigned integer
    U8,
    /// 16-bit unsigned integer
    U16,
    /// 32-bit unsigned integer
    U32,
    /// 64-bit unsigned integer
    U64,
    /// 16-bit floating point
    F16,
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
    /// 8-bit brain float
    BF8,
    /// 16-bit brain float
    BF16,
    /// Complex 32-bit
    C32,
    /// Complex 64-bit
    C64,
    /// Boolean
    Bool,
}

impl DataType {
    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            DataType::I8 | DataType::U8 | DataType::Bool => 1,
            DataType::I16 | DataType::U16 | DataType::F16 | DataType::BF8 => 2,
            DataType::I32 | DataType::U32 | DataType::F32 | DataType::C32 => 4,
            DataType::I64 | DataType::U64 | DataType::F64 | DataType::C64 => 8,
            DataType::BF16 => 2,
        }
    }
    
    /// Check if type is floating point
    pub fn is_float(&self) -> bool {
        matches!(self, DataType::F16 | DataType::F32 | DataType::F64 | DataType::BF8 | DataType::BF16)
    }
    
    /// Check if type is complex
    pub fn is_complex(&self) -> bool {
        matches!(self, DataType::C32 | DataType::C64)
    }
}

/// Supported operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperationType {
    // Basic arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Sqrt,
    Abs,
    
    // Trigonometric
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    
    // Exponential/logarithmic
    Exp,
    Log,
    Log2,
    Log10,
    
    // Matrix operations
    MatMul,
    Transpose,
    Inverse,
    Determinant,
    Eigenvalues,
    SVD,
    
    // Convolution operations
    Conv1D,
    Conv2D,
    Conv3D,
    MaxPool,
    AvgPool,
    
    // FFT operations
    FFT,
    IFFT,
    RFFT,
    IRFFT,
    
    // Reduction operations
    Sum,
    Mean,
    Max,
    Min,
    Var,
    Std,
    
    // Activation functions
    ReLU,
    Sigmoid,
    Tanh,
    SiLU,
    GELU,
    Softmax,
    
    // State-space model operations
    SSMForward,
    SSMBackward,
    StateUpdate,
    
    // Topological operations
    TopologicalAnalysis,
    MotifDetection,
    CycleDetection,
    StabilityCalculation,
    
    // Quantum operations (for QPU)
    QuantumGate,
    QuantumMeasurement,
    QuantumEntanglement,
    
    // BCI operations
    SpikeEncoding,
    SpikeDecoding,
    SignalFiltering,
}

/// Device capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// Device type
    pub device_type: DeviceType,
    /// Device identifier
    pub device_id: String,
    /// Supported operations
    pub supported_operations: Vec<OperationType>,
    /// Supported data types
    pub supported_types: Vec<DataType>,
    /// Maximum memory in bytes
    pub max_memory: usize,
    /// Available memory in bytes
    pub available_memory: usize,
    /// Number of compute units
    pub compute_units: usize,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth: f64,
    /// Peak compute performance in FLOPS
    pub peak_flops: f64,
    /// Optimal batch size
    pub optimal_batch_size: usize,
    /// Supports async execution
    pub supports_async: bool,
    /// Supports kernel fusion
    pub supports_fusion: bool,
    /// Supports mixed precision
    pub supports_mixed_precision: bool,
    /// Supports topological operations
    pub supports_topological: bool,
    /// Device-specific metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl DeviceCapabilities {
    /// Check if device supports specific operation
    pub fn supports_operation(&self, op: OperationType) -> bool {
        self.supported_operations.contains(&op)
    }
    
    /// Check if device supports specific data type
    pub fn supports_type(&self, dtype: DataType) -> bool {
        self.supported_types.contains(&dtype)
    }
    
    /// Get device utilization score (0.0 to 1.0)
    pub fn utilization(&self) -> f64 {
        let used_memory = self.max_memory - self.available_memory;
        used_memory as f64 / self.max_memory as f64
    }
    
    /// Check if device has enough memory for operation
    pub fn has_memory(&self, required: usize) -> bool {
        self.available_memory >= required
    }
}
