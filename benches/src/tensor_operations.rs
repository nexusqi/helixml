//! 🌀 HelixML Tensor Operations Benchmarks
//! 
//! Benchmarks for measuring performance of tensor operations

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tensor_core::*;
use tensor_core::tensor::{TensorOps, TensorRandom, TensorActivation, TensorBroadcast};
use backend_cpu::CpuTensor;

fn bench_matrix_multiplication(c: &mut Criterion) {
    let device = Device::cpu();
    let mut group = c.benchmark_group("Matrix Multiplication");
    
    for size in [32, 64, 128].iter() {
        let a = CpuTensor::random_uniform(
            Shape::new(vec![*size, *size]), 
            -1.0, 1.0, &device
        ).unwrap();
        let b = CpuTensor::random_uniform(
            Shape::new(vec![*size, *size]), 
            -1.0, 1.0, &device
        ).unwrap();
        
        group.bench_with_input(BenchmarkId::new("matmul", size), size, |bencher, _| {
            bencher.iter(|| {
                black_box(a.matmul(&b).unwrap())
            })
        });
    }
    group.finish();
}

fn bench_activation_functions(c: &mut Criterion) {
    let device = Device::cpu();
    let mut group = c.benchmark_group("Activation Functions");
    
    for size in [100, 1000, 10000].iter() {
        let tensor = CpuTensor::random_uniform(
            Shape::new(vec![*size]), 
            -1.0, 1.0, &device
        ).unwrap();
        
        group.bench_with_input(BenchmarkId::new("relu", size), size, |bencher, _| {
            bencher.iter(|| {
                black_box(tensor.relu().unwrap())
            })
        });
        
        group.bench_with_input(BenchmarkId::new("gelu", size), size, |bencher, _| {
            bencher.iter(|| {
                black_box(tensor.gelu().unwrap())
            })
        });
        
        group.bench_with_input(BenchmarkId::new("silu", size), size, |bencher, _| {
            bencher.iter(|| {
                black_box(tensor.silu().unwrap())
            })
        });
    }
    group.finish();
}

fn bench_elementwise_operations(c: &mut Criterion) {
    let device = Device::cpu();
    let mut group = c.benchmark_group("Elementwise Operations");
    
    for size in [100, 1000, 10000].iter() {
        let a = CpuTensor::random_uniform(
            Shape::new(vec![*size]), 
            -1.0, 1.0, &device
        ).unwrap();
        let b = CpuTensor::random_uniform(
            Shape::new(vec![*size]), 
            -1.0, 1.0, &device
        ).unwrap();
        
        group.bench_with_input(BenchmarkId::new("add", size), size, |bencher, _| {
            bencher.iter(|| {
                black_box(a.add(&b).unwrap())
            })
        });
        
        group.bench_with_input(BenchmarkId::new("mul", size), size, |bencher, _| {
            bencher.iter(|| {
                black_box(a.mul(&b).unwrap())
            })
        });
        
        group.bench_with_input(BenchmarkId::new("div", size), size, |bencher, _| {
            bencher.iter(|| {
                black_box(a.div(&b).unwrap())
            })
        });
    }
    group.finish();
}

fn bench_broadcasting_operations(c: &mut Criterion) {
    let device = Device::cpu();
    let mut group = c.benchmark_group("Broadcasting Operations");
    
    // Test different broadcasting scenarios
    let scenarios = vec![
        (vec![1000, 1], vec![1000, 1000]), // 1D to 2D
        (vec![100, 100], vec![10, 100, 100]), // 2D to 3D
        (vec![100, 1, 1], vec![100, 100, 100]), // 3D broadcasting
    ];
    
    for (i, (a_dims, b_dims)) in scenarios.iter().enumerate() {
        let a = CpuTensor::random_uniform(
            Shape::new(a_dims.clone()), 
            -1.0, 1.0, &device
        ).unwrap();
        let b = CpuTensor::random_uniform(
            Shape::new(b_dims.clone()), 
            -1.0, 1.0, &device
        ).unwrap();
        
        group.bench_with_input(BenchmarkId::new("broadcast", i), &i, |bencher, _| {
            bencher.iter(|| {
                black_box(a.broadcast_to(b.shape().clone()).unwrap())
            })
        });
    }
    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    let device = Device::cpu();
    let mut group = c.benchmark_group("Memory Usage");
    
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("allocation", size), size, |bencher, _| {
            bencher.iter(|| {
                let tensor = CpuTensor::random_uniform(
                    Shape::new(vec![*size, *size]), 
                    -1.0, 1.0, &device
                ).unwrap();
                black_box(tensor)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("clone", size), size, |bencher, _| {
            let tensor = CpuTensor::random_uniform(
                Shape::new(vec![*size, *size]), 
                -1.0, 1.0, &device
            ).unwrap();
            bencher.iter(|| {
                black_box(tensor.clone())
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_matrix_multiplication,
    bench_activation_functions,
    bench_elementwise_operations,
    bench_broadcasting_operations,
    bench_memory_usage
);
criterion_main!(benches);
