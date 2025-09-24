//! Tensor operations module

use crate::{DType, Device, Shape, Result, TensorError};

/// Binary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Max,
    Min,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Unary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Abs,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Floor,
    Ceil,
    Round,
}

/// Reduction operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
    Var,
    Std,
    Norm,
}

/// Convolution operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvOp {
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
}

/// Pooling operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolOp {
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    AdaptiveMaxPool1d,
    AdaptiveMaxPool2d,
    AdaptiveMaxPool3d,
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
}

/// Padding modes for convolution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingMode {
    Zero,
    Reflect,
    Replicate,
    Circular,
}

/// Convolution parameters
#[derive(Debug, Clone)]
pub struct ConvParams {
    pub stride: Vec<usize>,
    pub padding: Vec<usize>,
    pub dilation: Vec<usize>,
    pub groups: usize,
    pub padding_mode: PaddingMode,
}

impl Default for ConvParams {
    fn default() -> Self {
        Self {
            stride: vec![1],
            padding: vec![0],
            dilation: vec![1],
            groups: 1,
            padding_mode: PaddingMode::Zero,
        }
    }
}

/// Pooling parameters
#[derive(Debug, Clone)]
pub struct PoolParams {
    pub kernel_size: Vec<usize>,
    pub stride: Option<Vec<usize>>,
    pub padding: Vec<usize>,
    pub dilation: Vec<usize>,
    pub ceil_mode: bool,
}

impl Default for PoolParams {
    fn default() -> Self {
        Self {
            kernel_size: vec![2],
            stride: None,
            padding: vec![0],
            dilation: vec![1],
            ceil_mode: false,
        }
    }
}

/// FFT operations for Hyena/LongConv
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FFTOp {
    FFT,
    IFFT,
    RFFT,
    IRFFT,
}

/// SSM operations for state-space models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SSMOp {
    Scan,           // Sequential scan operation
    Conv1d,         // 1D convolution for SSM
    Discretize,     // Discretize continuous SSM
    SelectiveScan,  // Selective scan (Mamba-style)
}

/// Topological memory operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopoOp {
    MotifDetect,    // Detect motifs in sequences
    CycleFind,      // Find cycles in dependency graphs
    StabilityCalc,  // Calculate stability S = f(R, E, C, Φ, S)
    GeodesicDist,   // Calculate geodesic distances
    HNSWSearch,     // Hierarchical Navigable Small World search
}

/// MoE operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoEOp {
    Router,         // Expert routing
    TopK,           // Top-K expert selection
    LoadBalance,    // Load balancing across experts
    Gating,         // Gating network
}

/// Quantization operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantOp {
    Quantize,       // Quantize to int8/fp8
    Dequantize,     // Dequantize back to fp32
    QAT,            // Quantization-aware training
    Calibration,    // Calibration for quantization
}

/// Reversible compute operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RevOp {
    RevNet,         // Reversible ResNet blocks
    RevSSM,         // Reversible SSM blocks
    RevHyena,       // Reversible Hyena blocks
    Checkpoint,     // Gradient checkpointing
    Rollback,       // Rollback to checkpoint
}

/// Operation result with metadata
#[derive(Debug, Clone)]
pub struct OpResult<T> {
    pub result: T,
    pub flops: u64,
    pub memory_used: usize,
    pub execution_time: std::time::Duration,
}

impl<T> OpResult<T> {
    pub fn new(result: T, flops: u64, memory_used: usize, execution_time: std::time::Duration) -> Self {
        Self {
            result,
            flops,
            memory_used,
            execution_time,
        }
    }
}
