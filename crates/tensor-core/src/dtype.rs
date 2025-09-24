//! Data types for tensors

use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported data types for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
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
    /// Boolean
    Bool,
    /// Complex 32-bit
    C32,
    /// Complex 64-bit
    C64,
}

impl DType {
    /// Get the size in bytes for this data type
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::I8 | DType::U8 | DType::Bool => 1,
            DType::I16 | DType::U16 | DType::F16 => 2,
            DType::I32 | DType::U32 | DType::F32 | DType::C32 => 4,
            DType::I64 | DType::U64 | DType::F64 | DType::C64 => 8,
        }
    }
    
    /// Check if this is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F16 | DType::F32 | DType::F64)
    }
    
    /// Check if this is an integer type
    pub fn is_int(&self) -> bool {
        matches!(self, DType::I8 | DType::I16 | DType::I32 | DType::I64 | 
                        DType::U8 | DType::U16 | DType::U32 | DType::U64)
    }
    
    /// Check if this is a complex type
    pub fn is_complex(&self) -> bool {
        matches!(self, DType::C32 | DType::C64)
    }
    
    /// Get the default floating point type
    pub fn default_float() -> Self {
        DType::F32
    }
    
    /// Get the default integer type
    pub fn default_int() -> Self {
        DType::I32
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::I8 => write!(f, "i8"),
            DType::I16 => write!(f, "i16"),
            DType::I32 => write!(f, "i32"),
            DType::I64 => write!(f, "i64"),
            DType::U8 => write!(f, "u8"),
            DType::U16 => write!(f, "u16"),
            DType::U32 => write!(f, "u32"),
            DType::U64 => write!(f, "u64"),
            DType::F16 => write!(f, "f16"),
            DType::F32 => write!(f, "f32"),
            DType::F64 => write!(f, "f64"),
            DType::Bool => write!(f, "bool"),
            DType::C32 => write!(f, "c32"),
            DType::C64 => write!(f, "c64"),
        }
    }
}
