//! 🌀 HelixML Neural Network Benchmarks
//! 
//! Benchmarks for measuring performance of neural network components

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tensor_core::*;
use tensor_core::tensor::{TensorRandom, TensorOps};
use backend_cpu::CpuTensor;
use nn::{S4Block, MambaBlock, HyenaBlock, Linear, RMSNorm, Module};

fn bench_s4_block(c: &mut Criterion) {
    let device = Device::cpu();
    let mut group = c.benchmark_group("S4 Block");
    
    for seq_len in [64, 128, 256].iter() {
        let d_model = 512;
        let s4_block = S4Block::new(d_model, 64, &device).unwrap();
        let input = CpuTensor::random_uniform(
            Shape::new(vec![*seq_len, d_model]), 
            -1.0, 1.0, &device
        ).unwrap();
        
        group.bench_with_input(BenchmarkId::new("forward", seq_len), seq_len, |bencher, _| {
            bencher.iter(|| {
                black_box(s4_block.forward(&input).unwrap())
            })
        });
    }
    group.finish();
}

fn bench_mamba_block(c: &mut Criterion) {
    let device = Device::cpu();
    let mut group = c.benchmark_group("Mamba Block");
    
    for seq_len in [64, 128, 256].iter() {
        let d_model = 512;
        let mamba_block = MambaBlock::new(d_model, 64, 4, 2, &device).unwrap();
        let input = CpuTensor::random_uniform(
            Shape::new(vec![*seq_len, d_model]), 
            -1.0, 1.0, &device
        ).unwrap();
        
        group.bench_with_input(BenchmarkId::new("forward", seq_len), seq_len, |bencher, _| {
            bencher.iter(|| {
                black_box(mamba_block.forward(&input).unwrap())
            })
        });
    }
    group.finish();
}

fn bench_hyena_block(c: &mut Criterion) {
    let device = Device::cpu();
    let mut group = c.benchmark_group("Hyena Block");
    
    for seq_len in [64, 128, 256].iter() {
        let d_model = 512;
        let hyena_block = HyenaBlock::new(d_model, 64, 4, 2, &device).unwrap();
        let input = CpuTensor::random_uniform(
            Shape::new(vec![*seq_len, d_model]), 
            -1.0, 1.0, &device
        ).unwrap();
        
        group.bench_with_input(BenchmarkId::new("forward", seq_len), seq_len, |bencher, _| {
            bencher.iter(|| {
                black_box(hyena_block.forward(&input).unwrap())
            })
        });
    }
    group.finish();
}

fn bench_linear_layer(c: &mut Criterion) {
    let device = Device::cpu();
    let mut group = c.benchmark_group("Linear Layer");
    
    for (input_size, output_size) in [(512, 512), (1024, 1024), (2048, 2048)].iter() {
        let linear = Linear::new(*input_size, *output_size, &device).unwrap();
        let input = CpuTensor::random_uniform(
            Shape::new(vec![128, *input_size]), 
            -1.0, 1.0, &device
        ).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("forward", format!("{}x{}", input_size, output_size)), 
            &(*input_size, *output_size), 
            |bencher, _| {
                bencher.iter(|| {
                    black_box(linear.forward(&input).unwrap())
                })
            }
        );
    }
    group.finish();
}

fn bench_rms_norm(c: &mut Criterion) {
    let device = Device::cpu();
    let mut group = c.benchmark_group("RMS Norm");
    
    for seq_len in [64, 128, 256].iter() {
        let d_model = 512;
        let rms_norm = RMSNorm::new(d_model, 1e-5, &device).unwrap();
        let input = CpuTensor::random_uniform(
            Shape::new(vec![*seq_len, d_model]), 
            -1.0, 1.0, &device
        ).unwrap();
        
        group.bench_with_input(BenchmarkId::new("forward", seq_len), seq_len, |bencher, _| {
            bencher.iter(|| {
                black_box(rms_norm.forward(&input).unwrap())
            })
        });
    }
    group.finish();
}

fn bench_sequential_network(c: &mut Criterion) {
    let device = Device::cpu();
    let mut group = c.benchmark_group("Sequential Network");
    
    for seq_len in [128, 256, 512].iter() {
        let d_model = 512;
        let input = CpuTensor::random_uniform(
            Shape::new(vec![*seq_len, d_model]), 
            -1.0, 1.0, &device
        ).unwrap();
        
        // Create a simple sequential network
        let s4_block = S4Block::new(d_model, 64, &device).unwrap();
        let linear1 = Linear::new(d_model, d_model, &device).unwrap();
        let rms_norm = RMSNorm::new(d_model, 1e-5, &device).unwrap();
        let linear2 = Linear::new(d_model, d_model, &device).unwrap();
        
        group.bench_with_input(BenchmarkId::new("forward", seq_len), seq_len, |bencher, _| {
            bencher.iter(|| {
                let x = s4_block.forward(&input).unwrap();
                let x = linear1.forward(&x).unwrap();
                let x = rms_norm.forward(&x).unwrap();
                let x = linear2.forward(&x).unwrap();
                black_box(x)
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_s4_block,
    bench_mamba_block,
    bench_hyena_block,
    bench_linear_layer,
    bench_rms_norm,
    bench_sequential_network
);
criterion_main!(benches);
