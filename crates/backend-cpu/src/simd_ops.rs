//! 🌀 HelixML SIMD Operations
//! 
//! SIMD-optimized operations for CPU backend.

use hal::{Result, HalError};

/// SIMD operation result
pub type SimdResult = Result<()>;

/// SIMD-optimized vector addition
pub fn simd_add(a: &[f32], b: &[f32], result: &mut [f32]) -> SimdResult {
    if a.len() != b.len() || a.len() != result.len() {
        return Err(HalError::OperationError {
            message: "Vector lengths don't match".to_string(),
        });
    }
    
    // TODO: Implement actual SIMD operations
    // For now, use scalar operations with loop unrolling
    let len = a.len();
    let mut i = 0;
    
    // Process 4 elements at a time (SIMD width)
    while i + 4 <= len {
        result[i] = a[i] + b[i];
        result[i + 1] = a[i + 1] + b[i + 1];
        result[i + 2] = a[i + 2] + b[i + 2];
        result[i + 3] = a[i + 3] + b[i + 3];
        i += 4;
    }
    
    // Handle remaining elements
    while i < len {
        result[i] = a[i] + b[i];
        i += 1;
    }
    
    Ok(())
}

/// SIMD-optimized vector subtraction
pub fn simd_sub(a: &[f32], b: &[f32], result: &mut [f32]) -> SimdResult {
    if a.len() != b.len() || a.len() != result.len() {
        return Err(HalError::OperationError {
            message: "Vector lengths don't match".to_string(),
        });
    }
    
    let len = a.len();
    let mut i = 0;
    
    // Process 4 elements at a time
    while i + 4 <= len {
        result[i] = a[i] - b[i];
        result[i + 1] = a[i + 1] - b[i + 1];
        result[i + 2] = a[i + 2] - b[i + 2];
        result[i + 3] = a[i + 3] - b[i + 3];
        i += 4;
    }
    
    // Handle remaining elements
    while i < len {
        result[i] = a[i] - b[i];
        i += 1;
    }
    
    Ok(())
}

/// SIMD-optimized vector multiplication
pub fn simd_mul(a: &[f32], b: &[f32], result: &mut [f32]) -> SimdResult {
    if a.len() != b.len() || a.len() != result.len() {
        return Err(HalError::OperationError {
            message: "Vector lengths don't match".to_string(),
        });
    }
    
    let len = a.len();
    let mut i = 0;
    
    // Process 4 elements at a time
    while i + 4 <= len {
        result[i] = a[i] * b[i];
        result[i + 1] = a[i + 1] * b[i + 1];
        result[i + 2] = a[i + 2] * b[i + 2];
        result[i + 3] = a[i + 3] * b[i + 3];
        i += 4;
    }
    
    // Handle remaining elements
    while i < len {
        result[i] = a[i] * b[i];
        i += 1;
    }
    
    Ok(())
}

/// SIMD-optimized vector division
pub fn simd_div(a: &[f32], b: &[f32], result: &mut [f32]) -> SimdResult {
    if a.len() != b.len() || a.len() != result.len() {
        return Err(HalError::OperationError {
            message: "Vector lengths don't match".to_string(),
        });
    }
    
    let len = a.len();
    let mut i = 0;
    
    // Process 4 elements at a time
    while i + 4 <= len {
        result[i] = a[i] / b[i];
        result[i + 1] = a[i + 1] / b[i + 1];
        result[i + 2] = a[i + 2] / b[i + 2];
        result[i + 3] = a[i + 3] / b[i + 3];
        i += 4;
    }
    
    // Handle remaining elements
    while i < len {
        result[i] = a[i] / b[i];
        i += 1;
    }
    
    Ok(())
}

/// SIMD-optimized vector scaling
pub fn simd_scale(a: &[f32], scale: f32, result: &mut [f32]) -> SimdResult {
    if a.len() != result.len() {
        return Err(HalError::OperationError {
            message: "Vector lengths don't match".to_string(),
        });
    }
    
    let len = a.len();
    let mut i = 0;
    
    // Process 4 elements at a time
    while i + 4 <= len {
        result[i] = a[i] * scale;
        result[i + 1] = a[i + 1] * scale;
        result[i + 2] = a[i + 2] * scale;
        result[i + 3] = a[i + 3] * scale;
        i += 4;
    }
    
    // Handle remaining elements
    while i < len {
        result[i] = a[i] * scale;
        i += 1;
    }
    
    Ok(())
}

/// SIMD-optimized vector sum
pub fn simd_sum(a: &[f32]) -> Result<f32> {
    let len = a.len();
    let mut i = 0;
    let mut sum = 0.0;
    
    // Process 4 elements at a time
    while i + 4 <= len {
        sum += a[i] + a[i + 1] + a[i + 2] + a[i + 3];
        i += 4;
    }
    
    // Handle remaining elements
    while i < len {
        sum += a[i];
        i += 1;
    }
    
    Ok(sum)
}

/// SIMD-optimized vector dot product
pub fn simd_dot(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(HalError::OperationError {
            message: "Vector lengths don't match".to_string(),
        });
    }
    
    let len = a.len();
    let mut i = 0;
    let mut sum = 0.0;
    
    // Process 4 elements at a time
    while i + 4 <= len {
        sum += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
        i += 4;
    }
    
    // Handle remaining elements
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    
    Ok(sum)
}

/// SIMD-optimized vector norm
pub fn simd_norm(a: &[f32]) -> Result<f32> {
    let len = a.len();
    let mut i = 0;
    let mut sum = 0.0;
    
    // Process 4 elements at a time
    while i + 4 <= len {
        let val0 = a[i] * a[i];
        let val1 = a[i + 1] * a[i + 1];
        let val2 = a[i + 2] * a[i + 2];
        let val3 = a[i + 3] * a[i + 3];
        sum += val0 + val1 + val2 + val3;
        i += 4;
    }
    
    // Handle remaining elements
    while i < len {
        sum += a[i] * a[i];
        i += 1;
    }
    
    Ok(sum.sqrt())
}

/// SIMD-optimized vector maximum
pub fn simd_max(a: &[f32]) -> Result<f32> {
    if a.is_empty() {
        return Err(HalError::OperationError {
            message: "Cannot find max of empty vector".to_string(),
        });
    }
    
    let len = a.len();
    let mut i = 0;
    let mut max_val = a[0];
    
    // Process 4 elements at a time
    while i + 4 <= len {
        max_val = max_val.max(a[i]).max(a[i + 1]).max(a[i + 2]).max(a[i + 3]);
        i += 4;
    }
    
    // Handle remaining elements
    while i < len {
        max_val = max_val.max(a[i]);
        i += 1;
    }
    
    Ok(max_val)
}

/// SIMD-optimized vector minimum
pub fn simd_min(a: &[f32]) -> Result<f32> {
    if a.is_empty() {
        return Err(HalError::OperationError {
            message: "Cannot find min of empty vector".to_string(),
        });
    }
    
    let len = a.len();
    let mut i = 0;
    let mut min_val = a[0];
    
    // Process 4 elements at a time
    while i + 4 <= len {
        min_val = min_val.min(a[i]).min(a[i + 1]).min(a[i + 2]).min(a[i + 3]);
        i += 4;
    }
    
    // Handle remaining elements
    while i < len {
        min_val = min_val.min(a[i]);
        i += 1;
    }
    
    Ok(min_val)
}

/// SIMD-optimized ReLU activation
pub fn simd_relu(a: &[f32], result: &mut [f32]) -> SimdResult {
    if a.len() != result.len() {
        return Err(HalError::OperationError {
            message: "Vector lengths don't match".to_string(),
        });
    }
    
    let len = a.len();
    let mut i = 0;
    
    // Process 4 elements at a time
    while i + 4 <= len {
        result[i] = a[i].max(0.0);
        result[i + 1] = a[i + 1].max(0.0);
        result[i + 2] = a[i + 2].max(0.0);
        result[i + 3] = a[i + 3].max(0.0);
        i += 4;
    }
    
    // Handle remaining elements
    while i < len {
        result[i] = a[i].max(0.0);
        i += 1;
    }
    
    Ok(())
}

/// SIMD-optimized Sigmoid activation
pub fn simd_sigmoid(a: &[f32], result: &mut [f32]) -> SimdResult {
    if a.len() != result.len() {
        return Err(HalError::OperationError {
            message: "Vector lengths don't match".to_string(),
        });
    }
    
    let len = a.len();
    let mut i = 0;
    
    // Process 4 elements at a time
    while i + 4 <= len {
        result[i] = 1.0 / (1.0 + (-a[i]).exp());
        result[i + 1] = 1.0 / (1.0 + (-a[i + 1]).exp());
        result[i + 2] = 1.0 / (1.0 + (-a[i + 2]).exp());
        result[i + 3] = 1.0 / (1.0 + (-a[i + 3]).exp());
        i += 4;
    }
    
    // Handle remaining elements
    while i < len {
        result[i] = 1.0 / (1.0 + (-a[i]).exp());
        i += 1;
    }
    
    Ok(())
}

/// SIMD-optimized Tanh activation
pub fn simd_tanh(a: &[f32], result: &mut [f32]) -> SimdResult {
    if a.len() != result.len() {
        return Err(HalError::OperationError {
            message: "Vector lengths don't match".to_string(),
        });
    }
    
    let len = a.len();
    let mut i = 0;
    
    // Process 4 elements at a time
    while i + 4 <= len {
        result[i] = a[i].tanh();
        result[i + 1] = a[i + 1].tanh();
        result[i + 2] = a[i + 2].tanh();
        result[i + 3] = a[i + 3].tanh();
        i += 4;
    }
    
    // Handle remaining elements
    while i < len {
        result[i] = a[i].tanh();
        i += 1;
    }
    
    Ok(())
}

/// SIMD-optimized SiLU (Swish) activation: x * sigmoid(x)
pub fn simd_silu(a: &[f32], result: &mut [f32]) -> SimdResult {
    if a.len() != result.len() {
        return Err(HalError::OperationError {
            message: "Vector lengths don't match".to_string(),
        });
    }
    
    let len = a.len();
    let mut i = 0;
    
    // Process 4 elements at a time
    while i + 4 <= len {
        let sig0 = 1.0 / (1.0 + (-a[i]).exp());
        let sig1 = 1.0 / (1.0 + (-a[i + 1]).exp());
        let sig2 = 1.0 / (1.0 + (-a[i + 2]).exp());
        let sig3 = 1.0 / (1.0 + (-a[i + 3]).exp());
        result[i] = a[i] * sig0;
        result[i + 1] = a[i + 1] * sig1;
        result[i + 2] = a[i + 2] * sig2;
        result[i + 3] = a[i + 3] * sig3;
        i += 4;
    }
    
    // Handle remaining elements
    while i < len {
        let sig = 1.0 / (1.0 + (-a[i]).exp());
        result[i] = a[i] * sig;
        i += 1;
    }
    
    Ok(())
}

/// SIMD-optimized GELU activation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
pub fn simd_gelu(a: &[f32], result: &mut [f32]) -> SimdResult {
    if a.len() != result.len() {
        return Err(HalError::OperationError {
            message: "Vector lengths don't match".to_string(),
        });
    }
    
    const SQRT_2_OVER_PI: f32 = 0.7978845608;
    const GELU_COEF: f32 = 0.044715;
    
    let len = a.len();
    let mut i = 0;
    
    // Process 4 elements at a time
    while i + 4 <= len {
        let x0 = a[i];
        let x1 = a[i + 1];
        let x2 = a[i + 2];
        let x3 = a[i + 3];
        
        let x3_0 = x0 * x0 * x0;
        let x3_1 = x1 * x1 * x1;
        let x3_2 = x2 * x2 * x2;
        let x3_3 = x3 * x3 * x3;
        
        let arg0 = SQRT_2_OVER_PI * (x0 + GELU_COEF * x3_0);
        let arg1 = SQRT_2_OVER_PI * (x1 + GELU_COEF * x3_1);
        let arg2 = SQRT_2_OVER_PI * (x2 + GELU_COEF * x3_2);
        let arg3 = SQRT_2_OVER_PI * (x3 + GELU_COEF * x3_3);
        
        result[i] = 0.5 * x0 * (1.0 + arg0.tanh());
        result[i + 1] = 0.5 * x1 * (1.0 + arg1.tanh());
        result[i + 2] = 0.5 * x2 * (1.0 + arg2.tanh());
        result[i + 3] = 0.5 * x3 * (1.0 + arg3.tanh());
        i += 4;
    }
    
    // Handle remaining elements
    while i < len {
        let x = a[i];
        let x3 = x * x * x;
        let arg = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
        result[i] = 0.5 * x * (1.0 + arg.tanh());
        i += 1;
    }
    
    Ok(())
}

/// SIMD-optimized element-wise sqrt
pub fn simd_sqrt(a: &[f32], result: &mut [f32]) -> SimdResult {
    if a.len() != result.len() {
        return Err(HalError::OperationError {
            message: "Vector lengths don't match".to_string(),
        });
    }
    
    let len = a.len();
    let mut i = 0;
    
    // Process 4 elements at a time
    while i + 4 <= len {
        result[i] = a[i].sqrt();
        result[i + 1] = a[i + 1].sqrt();
        result[i + 2] = a[i + 2].sqrt();
        result[i + 3] = a[i + 3].sqrt();
        i += 4;
    }
    
    // Handle remaining elements
    while i < len {
        result[i] = a[i].sqrt();
        i += 1;
    }
    
    Ok(())
}

/// SIMD-optimized element-wise exp
pub fn simd_exp(a: &[f32], result: &mut [f32]) -> SimdResult {
    if a.len() != result.len() {
        return Err(HalError::OperationError {
            message: "Vector lengths don't match".to_string(),
        });
    }
    
    let len = a.len();
    let mut i = 0;
    
    // Process 4 elements at a time
    while i + 4 <= len {
        result[i] = a[i].exp();
        result[i + 1] = a[i + 1].exp();
        result[i + 2] = a[i + 2].exp();
        result[i + 3] = a[i + 3].exp();
        i += 4;
    }
    
    // Handle remaining elements
    while i < len {
        result[i] = a[i].exp();
        i += 1;
    }
    
    Ok(())
}

/// SIMD-optimized element-wise log
pub fn simd_log(a: &[f32], result: &mut [f32]) -> SimdResult {
    if a.len() != result.len() {
        return Err(HalError::OperationError {
            message: "Vector lengths don't match".to_string(),
        });
    }
    
    let len = a.len();
    let mut i = 0;
    
    // Process 4 elements at a time
    while i + 4 <= len {
        result[i] = a[i].ln();
        result[i + 1] = a[i + 1].ln();
        result[i + 2] = a[i + 2].ln();
        result[i + 3] = a[i + 3].ln();
        i += 4;
    }
    
    // Handle remaining elements
    while i < len {
        result[i] = a[i].ln();
        i += 1;
    }
    
    Ok(())
}

/// SIMD-optimized matrix multiplication
pub fn simd_matmul(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> SimdResult {
    if a.len() < m * k {
        return Err(HalError::OperationError {
            message: "Matrix A dimensions don't match".to_string(),
        });
    }
    
    if b.len() < k * n {
        return Err(HalError::OperationError {
            message: "Matrix B dimensions don't match".to_string(),
        });
    }
    
    if c.len() < m * n {
        return Err(HalError::OperationError {
            message: "Matrix C dimensions don't match".to_string(),
        });
    }
    
    // Initialize result matrix
    for i in 0..m * n {
        c[i] = 0.0;
    }
    
    // Optimized matrix multiplication with SIMD hints
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            let mut l = 0;
            
            // Process 4 elements at a time
            while l + 4 <= k {
                sum += a[i * k + l] * b[l * n + j];
                sum += a[i * k + l + 1] * b[(l + 1) * n + j];
                sum += a[i * k + l + 2] * b[(l + 2) * n + j];
                sum += a[i * k + l + 3] * b[(l + 3) * n + j];
                l += 4;
            }
            
            // Handle remaining elements
            while l < k {
                sum += a[i * k + l] * b[l * n + j];
                l += 1;
            }
            
            c[i * n + j] = sum;
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let mut result = vec![0.0; 4];
        
        simd_add(&a, &b, &mut result).unwrap();
        
        assert!((result[0] - 3.0).abs() < 1e-6);
        assert!((result[1] - 5.0).abs() < 1e-6);
        assert!((result[2] - 7.0).abs() < 1e-6);
        assert!((result[3] - 9.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_simd_dot() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        
        let result = simd_dot(&a, &b).unwrap();
        
        // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
        assert!((result - 40.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_simd_matmul() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let b = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let mut c = vec![0.0; 4]; // 2x2 result
        
        simd_matmul(&a, &b, &mut c, 2, 2, 2).unwrap();
        
        // Result should be A * I = A
        assert!((c[0] - 1.0).abs() < 1e-6);
        assert!((c[1] - 2.0).abs() < 1e-6);
        assert!((c[2] - 3.0).abs() < 1e-6);
        assert!((c[3] - 4.0).abs() < 1e-6);
    }
}
