//! Integration tests for tensor-core

use tensor_core::*;
use tensor_core::tensor::{TensorOps, TensorRandom, TensorBroadcast, TensorMixedPrecision, TensorReduce};
use backend_cpu::CpuTensor;

#[test]
fn test_tensor_creation_and_basic_ops() {
    let device = Device::cpu();
    
    // Test tensor creation
    let a = CpuTensor::random_uniform(Shape::new(vec![2, 3]), -1.0, 1.0, &device).unwrap();
    let b = CpuTensor::random_uniform(Shape::new(vec![2, 3]), -1.0, 1.0, &device).unwrap();
    
    // Test basic operations
    let c = a.add(&b).unwrap();
    assert_eq!(c.shape(), &Shape::new(vec![2, 3]));
    
    let d = a.sub(&b).unwrap();
    assert_eq!(d.shape(), &Shape::new(vec![2, 3]));
    
    let e = a.mul(&b).unwrap();
    assert_eq!(e.shape(), &Shape::new(vec![2, 3]));
}

#[test]
fn test_tensor_broadcasting() {
    let device = Device::cpu();
    
    let a = CpuTensor::random_uniform(Shape::new(vec![2, 3]), -1.0, 1.0, &device).unwrap();
    let b = CpuTensor::random_uniform(Shape::new(vec![3]), -1.0, 1.0, &device).unwrap();
    
    // Test broadcasting
    let c = a.add(&b).unwrap();
    assert_eq!(c.shape(), &Shape::new(vec![2, 3]));
}

#[test]
fn test_tensor_reduce_operations() {
    let device = Device::cpu();
    
    let a = CpuTensor::random_uniform(Shape::new(vec![2, 3]), -1.0, 1.0, &device).unwrap();
    
    // Test reduce operations
    let sum = a.sum(None, false).unwrap();
    assert_eq!(sum.shape(), &Shape::new(vec![1]));
    
    let mean = a.mean(None, false).unwrap();
    assert_eq!(mean.shape(), &Shape::new(vec![1]));
}

#[test]
fn test_mixed_precision() {
    let device = Device::cpu();
    
    let a = CpuTensor::random_uniform(Shape::new(vec![2, 3]), -1.0, 1.0, &device).unwrap();
    
    // Test mixed precision operations
    let b = a.to_f16().unwrap();
    assert_eq!(b.dtype(), DType::F16);
    
    let c = b.to_f32().unwrap();
    assert_eq!(c.dtype(), DType::F32);
}

#[test]
fn test_tensor_shape_operations() {
    let device = Device::cpu();
    
    let a = CpuTensor::random_uniform(Shape::new(vec![2, 3, 4]), -1.0, 1.0, &device).unwrap();
    
    // Test reshape
    let b = a.reshape(Shape::new(vec![6, 4])).unwrap();
    assert_eq!(b.shape(), &Shape::new(vec![6, 4]));
    
    // Test transpose
    let c = a.transpose(0, 1).unwrap();
    assert_eq!(c.shape(), &Shape::new(vec![3, 2, 4]));
}

#[test]
fn test_device_operations() {
    let device = Device::cpu();
    
    let a = CpuTensor::random_uniform(Shape::new(vec![2, 3]), -1.0, 1.0, &device).unwrap();
    
    // Test device operations
    assert_eq!(a.device(), &device);
    
    let b = a.to_device(&device).unwrap();
    assert_eq!(b.device(), &device);
}

#[test]
fn test_error_handling() {
    let device = Device::cpu();
    
    let a = CpuTensor::random_uniform(Shape::new(vec![2, 3]), -1.0, 1.0, &device).unwrap();
    let b = CpuTensor::random_uniform(Shape::new(vec![4, 5]), -1.0, 1.0, &device).unwrap();
    
    // Test shape mismatch error
    let result = a.add(&b);
    assert!(result.is_err());
    
    // Test invalid dimension error
    let result = a.transpose(0, 10);
    assert!(result.is_err());
}

#[test]
fn test_tensor_persistence() {
    let device = Device::cpu();
    
    let a = CpuTensor::random_uniform(Shape::new(vec![2, 3]), -1.0, 1.0, &device).unwrap();
    
    // Test serialization (if implemented)
    // let serialized = serde_json::to_string(&a).unwrap();
    // let deserialized: CpuTensor = serde_json::from_str(&serialized).unwrap();
    // assert_eq!(a.shape(), deserialized.shape());
}
