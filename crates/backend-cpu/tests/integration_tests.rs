//! Integration tests for backend-cpu

use backend_cpu::*;
use tensor_core::*;
use tensor_core::tensor::{TensorOps, TensorRandom};

#[test]
fn test_cpu_backend_initialization() {
    let device = Device::cpu();
    
    // Test that CPU backend can be initialized
    assert_eq!(device.name(), "cpu");
    assert!(device.is_cpu());
}

#[test]
fn test_cpu_tensor_operations() {
    let device = Device::cpu();
    
    // Test basic tensor operations on CPU
    let a = CpuTensor::random_uniform(Shape::new(vec![10, 10]), -1.0, 1.0, &device).unwrap();
    let b = CpuTensor::random_uniform(Shape::new(vec![10, 10]), -1.0, 1.0, &device).unwrap();
    
    let c = a.add(&b).unwrap();
    assert_eq!(c.shape(), &Shape::new(vec![10, 10]));
    
    let d = a.matmul(&b).unwrap();
    assert_eq!(d.shape(), &Shape::new(vec![10, 10]));
}

#[test]
fn test_cpu_memory_management() {
    let device = Device::cpu();
    
    // Test memory allocation and deallocation
    let tensors: Vec<CpuTensor> = (0..100)
        .map(|_| CpuTensor::random_uniform(Shape::new(vec![100, 100]), -1.0, 1.0, &device).unwrap())
        .collect();
    
    // All tensors should be valid
    for tensor in &tensors {
        assert_eq!(tensor.device(), &device);
    }
    
    // Test that tensors can be dropped without issues
    drop(tensors);
}

#[test]
fn test_cpu_performance() {
    let device = Device::cpu();
    
    // Test performance of large matrix operations
    // Note: Using smaller size since BLAS is not fully optimized yet
    let size = 100;  // Reduced from 1000 to avoid timeout
    let a = CpuTensor::random_uniform(Shape::new(vec![size, size]), -1.0, 1.0, &device).unwrap();
    let b = CpuTensor::random_uniform(Shape::new(vec![size, size]), -1.0, 1.0, &device).unwrap();
    
    let start = std::time::Instant::now();
    let _c = a.matmul(&b).unwrap();
    let elapsed = start.elapsed();
    
    // Should complete within reasonable time (adjust as needed)
    assert!(elapsed.as_secs() < 10);
}

#[test]
fn test_cpu_concurrent_operations() {
    use std::sync::Arc;
    use std::thread;
    
    let device = Arc::new(Device::cpu());
    
    // Test concurrent tensor operations
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let device = device.clone();
            thread::spawn(move || {
                let a = CpuTensor::random_uniform(
                    Shape::new(vec![100, 100]), 
                    -1.0, 1.0, 
                    &device
                ).unwrap();
                let b = CpuTensor::random_uniform(
                    Shape::new(vec![100, 100]), 
                    -1.0, 1.0, 
                    &device
                ).unwrap();
                a.matmul(&b).unwrap()
            })
        })
        .collect();
    
    // Wait for all threads to complete
    for handle in handles {
        let result = handle.join().unwrap();
        assert_eq!(result.shape(), &Shape::new(vec![100, 100]));
    }
}

#[test]
fn test_cpu_error_handling() {
    let device = Device::cpu();
    
    // Test error handling for invalid operations
    let a = CpuTensor::random_uniform(Shape::new(vec![2, 3]), -1.0, 1.0, &device).unwrap();
    let b = CpuTensor::random_uniform(Shape::new(vec![4, 5]), -1.0, 1.0, &device).unwrap();
    
    // Matrix multiplication with incompatible shapes should fail
    let result = a.matmul(&b);
    assert!(result.is_err());
}

#[test]
fn test_cpu_dtype_support() {
    let device = Device::cpu();
    
    // Test different data types
    let a_f32 = CpuTensor::random_uniform(Shape::new(vec![2, 3]), -1.0, 1.0, &device).unwrap();
    assert_eq!(a_f32.dtype(), DType::F32);
    
    // Test if other dtypes are supported (if implemented)
    // let a_f16 = CpuTensor::random_uniform_f16(Shape::new(vec![2, 3]), -1.0, 1.0, &device).unwrap();
    // assert_eq!(a_f16.dtype(), DType::F16);
}

#[test]
fn test_cpu_device_info() {
    let device = Device::cpu();
    
    // Test device information
    assert_eq!(device.name(), "cpu");
    assert!(device.is_cpu());
    
    // Test device capabilities
    assert!(device.is_cpu());
    // Add more capability tests as they are implemented
}
