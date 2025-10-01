//! 🌀 HelixML BLAS Operations
//! 
//! High-performance BLAS operations with multiple backend support.

use hal::{Result, HalError};

/// BLAS operation result
pub type BlasResult = Result<()>;

/// BLAS matrix multiplication
pub fn sgemm(
    trans_a: bool,
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) -> BlasResult {
    // TODO: Implement actual BLAS call
    // For now, use optimized generic implementation
    
    // Validate inputs
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    
    if a.len() < lda * (if trans_a { k } else { m }) {
        return Err(HalError::OperationError {
            message: "Matrix A dimensions don't match".to_string(),
        });
    }
    
    if b.len() < ldb * (if trans_b { n } else { k }) {
        return Err(HalError::OperationError {
            message: "Matrix B dimensions don't match".to_string(),
        });
    }
    
    if c.len() < ldc * m {
        return Err(HalError::OperationError {
            message: "Matrix C dimensions don't match".to_string(),
        });
    }
    
    // Optimized matrix multiplication
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                let a_idx = if trans_a { l * lda + i } else { i * lda + l };
                let b_idx = if trans_b { j * ldb + l } else { l * ldb + j };
                sum += a[a_idx] * b[b_idx];
            }
            c[i * ldc + j] = alpha * sum + beta * c[i * ldc + j];
        }
    }
    
    Ok(())
}

/// BLAS vector addition
pub fn saxpy(
    n: usize,
    alpha: f32,
    x: &[f32],
    incx: usize,
    y: &mut [f32],
    incy: usize,
) -> BlasResult {
    if n == 0 {
        return Ok(());
    }
    
    if x.len() < (n - 1) * incx + 1 {
        return Err(HalError::OperationError {
            message: "Vector X length insufficient".to_string(),
        });
    }
    
    if y.len() < (n - 1) * incy + 1 {
        return Err(HalError::OperationError {
            message: "Vector Y length insufficient".to_string(),
        });
    }
    
    for i in 0..n {
        y[i * incy] += alpha * x[i * incx];
    }
    
    Ok(())
}

/// BLAS dot product
pub fn sdot(
    n: usize,
    x: &[f32],
    incx: usize,
    y: &[f32],
    incy: usize,
) -> Result<f32> {
    if n == 0 {
        return Ok(0.0);
    }
    
    if x.len() < (n - 1) * incx + 1 {
        return Err(HalError::OperationError {
            message: "Vector X length insufficient".to_string(),
        });
    }
    
    if y.len() < (n - 1) * incy + 1 {
        return Err(HalError::OperationError {
            message: "Vector Y length insufficient".to_string(),
        });
    }
    
    let mut sum = 0.0;
    for i in 0..n {
        sum += x[i * incx] * y[i * incy];
    }
    
    Ok(sum)
}

/// BLAS vector scaling
pub fn sscal(n: usize, alpha: f32, x: &mut [f32], incx: usize) -> BlasResult {
    if n == 0 {
        return Ok(());
    }
    
    if x.len() < (n - 1) * incx + 1 {
        return Err(HalError::OperationError {
            message: "Vector X length insufficient".to_string(),
        });
    }
    
    for i in 0..n {
        x[i * incx] *= alpha;
    }
    
    Ok(())
}

/// BLAS vector copy
pub fn scopy(n: usize, x: &[f32], incx: usize, y: &mut [f32], incy: usize) -> BlasResult {
    if n == 0 {
        return Ok(());
    }
    
    if x.len() < (n - 1) * incx + 1 {
        return Err(HalError::OperationError {
            message: "Vector X length insufficient".to_string(),
        });
    }
    
    if y.len() < (n - 1) * incy + 1 {
        return Err(HalError::OperationError {
            message: "Vector Y length insufficient".to_string(),
        });
    }
    
    for i in 0..n {
        y[i * incy] = x[i * incx];
    }
    
    Ok(())
}

/// BLAS vector norm
pub fn snrm2(n: usize, x: &[f32], incx: usize) -> Result<f32> {
    if n == 0 {
        return Ok(0.0);
    }
    
    if x.len() < (n - 1) * incx + 1 {
        return Err(HalError::OperationError {
            message: "Vector X length insufficient".to_string(),
        });
    }
    
    let mut sum = 0.0;
    for i in 0..n {
        let val = x[i * incx];
        sum += val * val;
    }
    
    Ok(sum.sqrt())
}

/// BLAS vector sum of absolute values
pub fn sasum(n: usize, x: &[f32], incx: usize) -> Result<f32> {
    if n == 0 {
        return Ok(0.0);
    }
    
    if x.len() < (n - 1) * incx + 1 {
        return Err(HalError::OperationError {
            message: "Vector X length insufficient".to_string(),
        });
    }
    
    let mut sum = 0.0;
    for i in 0..n {
        sum += x[i * incx].abs();
    }
    
    Ok(sum)
}

/// BLAS matrix-vector multiplication
pub fn sgemv(
    trans: bool,
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    x: &[f32],
    incx: usize,
    beta: f32,
    y: &mut [f32],
    incy: usize,
) -> BlasResult {
    if m == 0 || n == 0 {
        return Ok(());
    }
    
    if a.len() < lda * (if trans { m } else { n }) {
        return Err(HalError::OperationError {
            message: "Matrix A dimensions don't match".to_string(),
        });
    }
    
    let x_len = if trans { m } else { n };
    if x.len() < (x_len - 1) * incx + 1 {
        return Err(HalError::OperationError {
            message: "Vector X length insufficient".to_string(),
        });
    }
    
    let y_len = if trans { n } else { m };
    if y.len() < (y_len - 1) * incy + 1 {
        return Err(HalError::OperationError {
            message: "Vector Y length insufficient".to_string(),
        });
    }
    
    // Scale y by beta
    if beta != 1.0 {
        for i in 0..y_len {
            y[i * incy] *= beta;
        }
    }
    
    // Perform matrix-vector multiplication
    if trans {
        // y = alpha * A^T * x + beta * y
        for j in 0..n {
            let mut sum = 0.0;
            for i in 0..m {
                sum += a[i * lda + j] * x[i * incx];
            }
            y[j * incy] += alpha * sum;
        }
    } else {
        // y = alpha * A * x + beta * y
        for i in 0..m {
            let mut sum = 0.0;
            for j in 0..n {
                sum += a[i * lda + j] * x[j * incx];
            }
            y[i * incy] += alpha * sum;
        }
    }
    
    Ok(())
}

/// BLAS rank-1 update
pub fn sger(
    m: usize,
    n: usize,
    alpha: f32,
    x: &[f32],
    incx: usize,
    y: &[f32],
    incy: usize,
    a: &mut [f32],
    lda: usize,
) -> BlasResult {
    if m == 0 || n == 0 {
        return Ok(());
    }
    
    if x.len() < (m - 1) * incx + 1 {
        return Err(HalError::OperationError {
            message: "Vector X length insufficient".to_string(),
        });
    }
    
    if y.len() < (n - 1) * incy + 1 {
        return Err(HalError::OperationError {
            message: "Vector Y length insufficient".to_string(),
        });
    }
    
    if a.len() < lda * n {
        return Err(HalError::OperationError {
            message: "Matrix A dimensions don't match".to_string(),
        });
    }
    
    // A = alpha * x * y^T + A
    for j in 0..n {
        for i in 0..m {
            a[i * lda + j] += alpha * x[i * incx] * y[j * incy];
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sgemm() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let b = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let mut c = vec![0.0; 4]; // 2x2 result
        
        sgemm(false, false, 2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2).unwrap();
        
        // Result should be A * I = A
        assert!((c[0] - 1.0).abs() < 1e-6);
        assert!((c[1] - 2.0).abs() < 1e-6);
        assert!((c[2] - 3.0).abs() < 1e-6);
        assert!((c[3] - 4.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_saxpy() {
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0, 0.0, 0.0];
        
        saxpy(3, 2.0, &x, 1, &mut y, 1).unwrap();
        
        assert!((y[0] - 2.0).abs() < 1e-6);
        assert!((y[1] - 4.0).abs() < 1e-6);
        assert!((y[2] - 6.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_sdot() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 3.0, 4.0];
        
        let result = sdot(3, &x, 1, &y, 1).unwrap();
        
        // 1*2 + 2*3 + 3*4 = 2 + 6 + 12 = 20
        assert!((result - 20.0).abs() < 1e-6);
    }
}
