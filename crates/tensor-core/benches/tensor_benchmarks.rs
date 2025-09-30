//! Benchmarks for tensor operations

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tensor_core::*;
use tensor_core::tensor::{TensorOps, TensorRandom};

fn bench_tensor_creation(c: &mut Criterion) {
    let device = Device::cpu();
    
    let mut group = c.benchmark_group("tensor_creation");
    
    for size in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("random_uniform", size), size, |b, &size| {
            b.iter(|| {
                let tensor = CpuTensor::random_uniform(
                    Shape::new(vec![size, size]), 
                    black_box(-1.0), 
                    black_box(1.0), 
                    &device
                ).unwrap();
                black_box(tensor)
            })
        });
    }
    
    group.finish();
}

fn bench_tensor_add(c: &mut Criterion) {
    let device = Device::cpu();
    
    let mut group = c.benchmark_group("tensor_add");
    
    for size in [10, 100, 1000].iter() {
        let a = CpuTensor::random_uniform(Shape::new(vec![size, size]), -1.0, 1.0, &device).unwrap();
        let b = CpuTensor::random_uniform(Shape::new(vec![size, size]), -1.0, 1.0, &device).unwrap();
        
        group.bench_with_input(BenchmarkId::new("add", size), size, |b, _| {
            b.iter(|| {
                let result = a.add(black_box(&b)).unwrap();
                black_box(result)
            })
        });
    }
    
    group.finish();
}

fn bench_tensor_matmul(c: &mut Criterion) {
    let device = Device::cpu();
    
    let mut group = c.benchmark_group("tensor_matmul");
    
    for size in [10, 50, 100, 500].iter() {
        let a = CpuTensor::random_uniform(Shape::new(vec![size, size]), -1.0, 1.0, &device).unwrap();
        let b = CpuTensor::random_uniform(Shape::new(vec![size, size]), -1.0, 1.0, &device).unwrap();
        
        group.bench_with_input(BenchmarkId::new("matmul", size), size, |b, _| {
            b.iter(|| {
                let result = a.matmul(black_box(&b)).unwrap();
                black_box(result)
            })
        });
    }
    
    group.finish();
}

fn bench_tensor_reduce_ops(c: &mut Criterion) {
    let device = Device::cpu();
    
    let mut group = c.benchmark_group("tensor_reduce");
    
    for size in [100, 1000, 10000].iter() {
        let tensor = CpuTensor::random_uniform(Shape::new(vec![size, size]), -1.0, 1.0, &device).unwrap();
        
        group.bench_with_input(BenchmarkId::new("sum", size), &tensor, |b, tensor| {
            b.iter(|| {
                let result = tensor.sum().unwrap();
                black_box(result)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("mean", size), &tensor, |b, tensor| {
            b.iter(|| {
                let result = tensor.mean().unwrap();
                black_box(result)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("max", size), &tensor, |b, tensor| {
            b.iter(|| {
                let result = tensor.max().unwrap();
                black_box(result)
            })
        });
    }
    
    group.finish();
}

fn bench_tensor_broadcasting(c: &mut Criterion) {
    let device = Device::cpu();
    
    let mut group = c.benchmark_group("tensor_broadcasting");
    
    for size in [100, 1000].iter() {
        let a = CpuTensor::random_uniform(Shape::new(vec![size, size]), -1.0, 1.0, &device).unwrap();
        let b = CpuTensor::random_uniform(Shape::new(vec![size]), -1.0, 1.0, &device).unwrap();
        
        group.bench_with_input(BenchmarkId::new("broadcast_add", size), size, |b, _| {
            b.iter(|| {
                let result = a.add(black_box(&b)).unwrap();
                black_box(result)
            })
        });
    }
    
    group.finish();
}

fn bench_tensor_reshape(c: &mut Criterion) {
    let device = Device::cpu();
    
    let mut group = c.benchmark_group("tensor_reshape");
    
    for size in [100, 1000, 10000].iter() {
        let tensor = CpuTensor::random_uniform(Shape::new(vec![size, size]), -1.0, 1.0, &device).unwrap();
        
        group.bench_with_input(BenchmarkId::new("reshape", size), &tensor, |b, tensor| {
            b.iter(|| {
                let result = tensor.reshape(Shape::new(vec![size * size, 1])).unwrap();
                black_box(result)
            })
        });
    }
    
    group.finish();
}

fn bench_tensor_memory_usage(c: &mut Criterion) {
    let device = Device::cpu();
    
    let mut group = c.benchmark_group("tensor_memory");
    
    // Test memory allocation patterns
    group.bench_function("allocate_deallocate", |b| {
        b.iter(|| {
            let tensors: Vec<CpuTensor> = (0..100)
                .map(|_| {
                    CpuTensor::random_uniform(Shape::new(vec![100, 100]), -1.0, 1.0, &device).unwrap()
                })
                .collect();
            black_box(tensors)
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_tensor_creation,
    bench_tensor_add,
    bench_tensor_matmul,
    bench_tensor_reduce_ops,
    bench_tensor_broadcasting,
    bench_tensor_reshape,
    bench_tensor_memory_usage
);

criterion_main!(benches);
