//! Benchmarks for meaning induction bootstrap

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use meanings::*;
use meanings::bootstrap::*;
use topo_memory::*;
use tensor_core::*;

fn generate_test_data(size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(size);
    let mut rng = 42u64;
    
    for _ in 0..size {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        let byte = (rng % 256) as u8;
        data.push(byte);
    }
    
    data
}

fn bench_bootstrap_span(c: &mut Criterion) {
    let device = Device::cpu();
    let d_model = 64;
    
    let mut group = c.benchmark_group("bootstrap_span");
    
    for size in [100, 1000, 10000].iter() {
        let bytes = generate_test_data(*size);
        let mut topo_memory = TopologicalMemory::new(d_model, 5, 0.6, 0.8, &device).unwrap();
        let config = BootstrapCfg::default();
        
        group.bench_with_input(BenchmarkId::new("process_bytes", size), &bytes, |b, bytes| {
            b.iter(|| {
                let result = bootstrap_span(black_box(bytes), black_box(&config), black_box(&mut topo_memory)).unwrap();
                black_box(result)
            })
        });
    }
    
    group.finish();
}

fn bench_batch_processing(c: &mut Criterion) {
    let device = Device::cpu();
    let d_model = 64;
    
    let mut group = c.benchmark_group("batch_processing");
    
    for batch_size in [10, 100, 1000].iter() {
        let mut topo_memory = TopologicalMemory::new(d_model, 5, 0.6, 0.8, &device).unwrap();
        
        // Pre-populate with some links
        let bytes = generate_test_data(1000);
        bootstrap_span(&bytes, &BootstrapCfg::default(), &mut topo_memory).unwrap();
        
        let stats = BatchStats {
            repetition: 0.5,
            energy: 0.3,
            connectivity: 0.4,
            phase_sync: 0.6,
        };
        
        group.bench_with_input(BenchmarkId::new("observe_batch", batch_size), batch_size, |b, _| {
            b.iter(|| {
                for _ in 0..*batch_size {
                    observe_batch(black_box(stats), black_box(&mut topo_memory)).unwrap();
                }
            })
        });
    }
    
    group.finish();
}

fn bench_replay_operations(c: &mut Criterion) {
    let device = Device::cpu();
    let d_model = 64;
    
    let mut group = c.benchmark_group("replay_operations");
    
    for u_pool_size in [100, 1000, 10000].iter() {
        let mut topo_memory = TopologicalMemory::new(d_model, 5, 0.6, 0.8, &device).unwrap();
        
        // Pre-populate with many U-links
        for i in 0..*u_pool_size {
            let bytes = format!("Test data batch number {}", i).into_bytes();
            bootstrap_span(&bytes, &BootstrapCfg::default(), &mut topo_memory).unwrap();
        }
        
        let config = BootstrapCfg {
            u_pool_size: *u_pool_size,
            ..BootstrapCfg::default()
        };
        
        group.bench_with_input(BenchmarkId::new("maybe_replay", u_pool_size), u_pool_size, |b, _| {
            b.iter(|| {
                let result = maybe_replay(black_box(100), black_box(&config), black_box(&mut topo_memory)).unwrap();
                black_box(result)
            })
        });
    }
    
    group.finish();
}

fn bench_stability_calculations(c: &mut Criterion) {
    let device = Device::cpu();
    let d_model = 64;
    
    let mut group = c.benchmark_group("stability_calculations");
    
    for link_count in [100, 1000, 10000].iter() {
        let mut topo_memory = TopologicalMemory::new(d_model, 5, 0.6, 0.8, &device).unwrap();
        
        // Create many links
        for i in 0..*link_count {
            let bytes = format!("Stability test data {}", i).into_bytes();
            bootstrap_span(&bytes, &BootstrapCfg::default(), &mut topo_memory).unwrap();
        }
        
        let stats = BatchStats {
            repetition: 0.7,
            energy: 0.5,
            connectivity: 0.6,
            phase_sync: 0.8,
        };
        
        group.bench_with_input(BenchmarkId::new("update_stability", link_count), link_count, |b, _| {
            b.iter(|| {
                observe_batch(black_box(stats), black_box(&mut topo_memory)).unwrap();
            })
        });
    }
    
    group.finish();
}

fn bench_link_state_transitions(c: &mut Criterion) {
    let device = Device::cpu();
    let d_model = 64;
    
    let mut group = c.benchmark_group("link_state_transitions");
    
    for iterations in [10, 100, 1000].iter() {
        let mut topo_memory = TopologicalMemory::new(d_model, 5, 0.6, 0.8, &device).unwrap();
        
        // Create initial links
        let bytes = generate_test_data(1000);
        bootstrap_span(&bytes, &BootstrapCfg::default(), &mut topo_memory).unwrap();
        
        let high_quality_stats = BatchStats {
            repetition: 0.9,
            energy: 0.8,
            connectivity: 0.9,
            phase_sync: 0.9,
        };
        
        group.bench_with_input(BenchmarkId::new("state_transitions", iterations), iterations, |b, _| {
            b.iter(|| {
                for _ in 0..*iterations {
                    observe_batch(black_box(high_quality_stats), black_box(&mut topo_memory)).unwrap();
                }
            })
        });
    }
    
    group.finish();
}

fn bench_memory_retrieval(c: &mut Criterion) {
    let device = Device::cpu();
    let d_model = 64;
    
    let mut group = c.benchmark_group("memory_retrieval");
    
    for memory_size in [1000, 10000, 100000].iter() {
        let mut topo_memory = TopologicalMemory::new(d_model, 5, 0.6, 0.8, &device).unwrap();
        
        // Pre-populate memory
        for i in 0..*memory_size {
            let bytes = format!("Memory retrieval test {}", i).into_bytes();
            bootstrap_span(&bytes, &BootstrapCfg::default(), &mut topo_memory).unwrap();
        }
        
        // Apply signals to create I and S links
        let stats = BatchStats {
            repetition: 0.8,
            energy: 0.7,
            connectivity: 0.8,
            phase_sync: 0.8,
        };
        
        for _ in 0..100 {
            observe_batch(stats, &mut topo_memory).unwrap();
        }
        
        group.bench_with_input(BenchmarkId::new("retrieve_m0", memory_size), memory_size, |b, _| {
            b.iter(|| {
                let query = CpuTensor::random_uniform(Shape::new(vec![1, d_model]), -1.0, 1.0, &device).unwrap();
                let result = topo_memory.retrieve(black_box(&query), black_box(MemoryLevel::M0)).unwrap();
                black_box(result)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("retrieve_m1", memory_size), memory_size, |b, _| {
            b.iter(|| {
                let query = CpuTensor::random_uniform(Shape::new(vec![1, d_model]), -1.0, 1.0, &device).unwrap();
                let result = topo_memory.retrieve(black_box(&query), black_box(MemoryLevel::M1)).unwrap();
                black_box(result)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("retrieve_m2", memory_size), memory_size, |b, _| {
            b.iter(|| {
                let query = CpuTensor::random_uniform(Shape::new(vec![1, d_model]), -1.0, 1.0, &device).unwrap();
                let result = topo_memory.retrieve(black_box(&query), black_box(MemoryLevel::M2)).unwrap();
                black_box(result)
            })
        });
    }
    
    group.finish();
}

fn bench_phase_transitions(c: &mut Criterion) {
    let device = Device::cpu();
    let d_model = 64;
    
    let mut group = c.benchmark_group("phase_transitions");
    
    // Phase A: Bootstrap
    let phase_a_config = BootstrapCfg {
        enabled: true,
        pmi_threshold: 0.1,
        replay_period: 50,
        theta_low: 0.2,
        theta_high: 0.6,
        ..BootstrapCfg::default()
    };
    
    // Phase B: Consolidation
    let phase_b_config = BootstrapCfg {
        enabled: true,
        pmi_threshold: 0.15,
        replay_period: 75,
        theta_low: 0.3,
        theta_high: 0.7,
        ..BootstrapCfg::default()
    };
    
    // Phase C: Meaning-first
    let phase_c_config = BootstrapCfg {
        enabled: false,
        pmi_threshold: 0.2,
        replay_period: 100,
        theta_low: 0.4,
        theta_high: 0.8,
        ..BootstrapCfg::default()
    };
    
    group.bench_function("phase_a_bootstrap", |b| {
        b.iter(|| {
            let mut topo_memory = TopologicalMemory::new(d_model, 5, 0.6, 0.8, &device).unwrap();
            let bytes = generate_test_data(1000);
            bootstrap_span(black_box(&bytes), black_box(&phase_a_config), black_box(&mut topo_memory)).unwrap();
        })
    });
    
    group.bench_function("phase_b_consolidation", |b| {
        b.iter(|| {
            let mut topo_memory = TopologicalMemory::new(d_model, 5, 0.6, 0.8, &device).unwrap();
            let bytes = generate_test_data(1000);
            bootstrap_span(black_box(&bytes), black_box(&phase_b_config), black_box(&mut topo_memory)).unwrap();
        })
    });
    
    group.bench_function("phase_c_meaning_first", |b| {
        b.iter(|| {
            let mut topo_memory = TopologicalMemory::new(d_model, 5, 0.6, 0.8, &device).unwrap();
            let bytes = generate_test_data(1000);
            bootstrap_span(black_box(&bytes), black_box(&phase_c_config), black_box(&mut topo_memory)).unwrap();
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_bootstrap_span,
    bench_batch_processing,
    bench_replay_operations,
    bench_stability_calculations,
    bench_link_state_transitions,
    bench_memory_retrieval,
    bench_phase_transitions
);

criterion_main!(benches);
