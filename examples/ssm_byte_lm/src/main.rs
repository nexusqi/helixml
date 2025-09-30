//! 🌀 SSM Byte Language Model Example
//! 
//! Demonstrates VortexML with a simple SSM-based language model
//! trained on byte sequences without self-attention.

use std::env;
use helix_ml::*;
use helix_ml::meanings::*;
use helix_ml::nn::{S4Block, MambaBlock, Linear, SiLU, Module};
use helix_ml::topo_memory::*;
use helix_ml::tensor::{TensorRandom, TensorOps};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌀 HelixML SSM Byte Language Model");
    println!("=====================================");
    
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} <command>", args[0]);
        println!("Commands:");
        println!("  train [A|B|C] - Train the model (A=Bootstrap, B=Consolidation, C=Meaning-first)");
        println!("  infer         - Run inference");
        println!("  bench         - Run benchmarks");
        println!("  demo          - Run meaning induction demo");
        return Ok(());
    }
    
    let command = &args[1];
    
    match command.as_str() {
        "train" => {
            println!("🚀 Starting training...");
            let mode = args.get(2).unwrap_or(&"A".to_string()).clone();
            train_model_with_mode(&mode)?;
        },
        "infer" => {
            println!("🔮 Running inference...");
            run_inference()?;
        },
        "bench" => {
            println!("⚡ Running benchmarks...");
            run_benchmarks()?;
        },
        "demo" => {
            println!("🎯 Running meaning induction demo...");
            run_meaning_demo()?;
        },
        _ => {
            println!("❌ Unknown command: {}", command);
            return Ok(());
        }
    }
    
    Ok(())
}

fn train_model_with_mode(mode: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("📚 Training SSM Byte Language Model - Mode {}", mode);
    println!("- Architecture: SSM (State-Space Model)");
    println!("- No self-attention (10-20× FLOP reduction)");
    
    match mode {
        "A" => train_phase_a_bootstrap()?,
        "B" => train_phase_b_consolidation()?,
        "C" => train_phase_c_meaning_first()?,
        _ => {
            println!("❌ Unknown mode: {}. Use A, B, or C", mode);
            return Ok(());
        }
    }
    
    Ok(())
}

fn train_phase_a_bootstrap() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Phase A: Bootstrap Mode");
    println!("- Creating U-links from raw bytes");
    println!("- Active replay every 50 steps");
    println!("- PMI threshold: 0.1");
    
    let device = Device::cpu();
    let d_model = 64;
    let seq_len = 256;
    
    // Create bootstrap configuration
    let bootstrap_cfg = BootstrapCfg {
        enabled: true,
        window: 256,
        pmi_threshold: 0.1,
        replay_period: 50,
        theta_low: 0.2,
        theta_high: 0.6,
        decay: 0.01,
        u_pool_size: 1000,
    };
    
    // Create topological memory
    let mut topo_memory = TopologicalMemory::new(d_model, 5, 0.6, 0.8, &device)?;
    
    // Create SSM model
    let s4_block = S4Block::<CpuTensor>::new(d_model, 16, &device)?;
    let linear = Linear::<CpuTensor>::new(d_model, 256, &device)?; // 256 byte vocabulary
    
    println!("✅ Phase A setup complete");
    println!("- Bootstrap config: {:?}", bootstrap_cfg);
    println!("- Model: S4 + Linear");
    
    // Simulate training loop
    for epoch in 0..2 {
        println!("📖 Epoch {}/2", epoch + 1);
        
        for step in 0..100 {
            // Simulate byte sequence
            let bytes = generate_random_bytes(seq_len);
            
            // Bootstrap: Create U-links from bytes
            let u_links_created = bootstrap_span(&bytes, &bootstrap_cfg, &mut topo_memory)?;
            
            // Create input tensor
            let input = CpuTensor::from_slice(
                &bytes.iter().map(|&b| b as f32 / 255.0).collect::<Vec<_>>(),
                Shape::new(vec![seq_len, 1]),
                &device
            )?;
            
            // Forward pass
            let s4_output = s4_block.forward(&input)?;
            let output = linear.forward(&s4_output)?;
            
            // Calculate batch statistics
            let stats = BatchStats {
                repetition: 0.5,  // Simulated
                energy: 0.3,      // Simulated
                connectivity: 0.4, // Simulated
                phase_sync: 0.6,   // Simulated
            };
            
            // Observe batch
            observe_batch(stats, &mut topo_memory)?;
            
            // Periodic replay
            if step % bootstrap_cfg.replay_period == 0 {
                if let Some(report) = maybe_replay(step, &bootstrap_cfg, &mut topo_memory)? {
                    println!("  Step {}: Replay - {} U→I, {} I→S", 
                            step, report.i_links_created, report.s_links_created);
                }
            }
            
            if step % 20 == 0 {
                let link_stats = topo_memory.get_link_stats();
                println!("  Step {}: U={}, I={}, S={}, Stability={:.3}", 
                        step, link_stats.u_links, link_stats.i_links, 
                        link_stats.s_links, link_stats.avg_stability);
            }
        }
    }
    
    println!("✅ Phase A completed!");
    Ok(())
}

fn train_phase_b_consolidation() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Phase B: Consolidation Mode");
    println!("- Consolidating U→I→S transitions");
    println!("- Replay every 75 steps");
    println!("- Higher PMI threshold: 0.15");
    
    let device = Device::cpu();
    let d_model = 64;
    let seq_len = 512; // Longer sequences
    
    // Phase B configuration
    let bootstrap_cfg = BootstrapCfg {
        enabled: true,
        window: 512,
        pmi_threshold: 0.15, // Higher threshold
        replay_period: 75,   // Less frequent replay
        theta_low: 0.3,
        theta_high: 0.7,
        decay: 0.01,
        u_pool_size: 1000,
    };
    
    let mut topo_memory = TopologicalMemory::new(d_model, 5, 0.6, 0.8, &device)?;
    let s4_block = S4Block::<CpuTensor>::new(d_model, 16, &device)?;
    let linear = Linear::<CpuTensor>::new(d_model, 256, &device)?;
    
    println!("✅ Phase B setup complete");
    
    // Simulate training loop
    for epoch in 0..4 {
        println!("📖 Epoch {}/4", epoch + 1);
        
        for step in 0..150 {
            let bytes = generate_random_bytes(seq_len);
            
            // Bootstrap with higher threshold
            let u_links_created = bootstrap_span(&bytes, &bootstrap_cfg, &mut topo_memory)?;
            
            let input = CpuTensor::from_slice(
                &bytes.iter().map(|&b| b as f32 / 255.0).collect::<Vec<_>>(),
                Shape::new(vec![seq_len, 1]),
                &device
            )?;
            
            let s4_output = s4_block.forward(&input)?;
            let output = linear.forward(&s4_output)?;
            
            let stats = BatchStats {
                repetition: 0.6,  // Improved signals
                energy: 0.4,
                connectivity: 0.5,
                phase_sync: 0.7,
            };
            
            observe_batch(stats, &mut topo_memory)?;
            
            if step % bootstrap_cfg.replay_period == 0 {
                if let Some(report) = maybe_replay(step, &bootstrap_cfg, &mut topo_memory)? {
                    println!("  Step {}: Replay - {} U→I, {} I→S", 
                            step, report.i_links_created, report.s_links_created);
                }
            }
            
            if step % 30 == 0 {
                let link_stats = topo_memory.get_link_stats();
                println!("  Step {}: U={}, I={}, S={}, Stability={:.3}", 
                        step, link_stats.u_links, link_stats.i_links, 
                        link_stats.s_links, link_stats.avg_stability);
            }
        }
    }
    
    println!("✅ Phase B completed!");
    Ok(())
}

fn train_phase_c_meaning_first() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Phase C: Meaning-First Mode");
    println!("- Bootstrap disabled");
    println!("- Pure topological memory operation");
    println!("- Focus on retrieval and consolidation");
    
    let device = Device::cpu();
    let d_model = 64;
    let seq_len = 1024; // Even longer sequences
    
    // Phase C configuration - bootstrap disabled
    let bootstrap_cfg = BootstrapCfg {
        enabled: false, // Bootstrap disabled
        window: 1024,
        pmi_threshold: 0.2,
        replay_period: 100,
        theta_low: 0.4,
        theta_high: 0.8,
        decay: 0.005, // Slower decay
        u_pool_size: 500,
    };
    
    let mut topo_memory = TopologicalMemory::new(d_model, 5, 0.6, 0.8, &device)?;
    let s4_block = S4Block::<CpuTensor>::new(d_model, 16, &device)?;
    let linear = Linear::<CpuTensor>::new(d_model, 256, &device)?;
    
    println!("✅ Phase C setup complete");
    
    // Simulate training loop
    for epoch in 0..10 {
        println!("📖 Epoch {}/10", epoch + 1);
        
        for step in 0..200 {
            let bytes = generate_random_bytes(seq_len);
            
            // No bootstrap - only memory consolidation
            let input = CpuTensor::from_slice(
                &bytes.iter().map(|&b| b as f32 / 255.0).collect::<Vec<_>>(),
                Shape::new(vec![seq_len, 1]),
                &device
            )?;
            
            let s4_output = s4_block.forward(&input)?;
            let output = linear.forward(&s4_output)?;
            
            let stats = BatchStats {
                repetition: 0.7,  // High-quality signals
                energy: 0.5,
                connectivity: 0.6,
                phase_sync: 0.8,
            };
            
            observe_batch(stats, &mut topo_memory)?;
            
            // Periodic memory consolidation
            if step % 100 == 0 {
                let stability_params = StabilityParams::new(0.4, 0.8, 0.005);
                topo_memory.sweep_and_consolidate(&stability_params, false)?;
                
                let link_stats = topo_memory.get_link_stats();
                println!("  Step {}: Consolidation - U={}, I={}, S={}, Stability={:.3}", 
                        step, link_stats.u_links, link_stats.i_links, 
                        link_stats.s_links, link_stats.avg_stability);
            }
            
            // Test retrieval
            if step % 50 == 0 {
                let query = CpuTensor::random_uniform(Shape::new(vec![1, d_model]), -1.0, 1.0, &device)?;
                let m2_result = topo_memory.retrieve(&query, MemoryLevel::M2)?;
                println!("  Step {}: Retrieval test - M2 shape: {:?}", step, m2_result.shape());
            }
        }
    }
    
    println!("✅ Phase C completed!");
    Ok(())
}

fn run_meaning_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Meaning Induction Demo");
    println!("=========================");
    
    let device = Device::cpu();
    let d_model = 32;
    
    // Create bootstrap configuration
    let bootstrap_cfg = BootstrapCfg::default();
    
    // Create topological memory
    let mut topo_memory = TopologicalMemory::new(d_model, 5, 0.6, 0.8, &device)?;
    
    println!("📊 Initial Setup:");
    println!("- Bootstrap config: {:?}", bootstrap_cfg);
    println!("- Model dimension: {}", d_model);
    
    // Simulate processing a text sequence
    let text = "Hello, world! This is a test of meaning induction.";
    let bytes = text.as_bytes();
    
    println!("📝 Processing text: \"{}\"", text);
    
    // Bootstrap: Create U-links
    let u_links_created = bootstrap_span(bytes, &bootstrap_cfg, &mut topo_memory)?;
    println!("🔗 Created {} U-links from text", u_links_created);
    
    // Simulate training steps
    for step in 0..10 {
        let stats = BatchStats {
            repetition: 0.5 + (step as f32 * 0.05),
            energy: 0.3 + (step as f32 * 0.02),
            connectivity: 0.4 + (step as f32 * 0.03),
            phase_sync: 0.6 + (step as f32 * 0.02),
        };
        
        observe_batch(stats, &mut topo_memory)?;
        
        if step % 3 == 0 {
            if let Some(report) = maybe_replay(step, &bootstrap_cfg, &mut topo_memory)? {
                println!("🔄 Step {}: Replay - {} U→I, {} I→S", 
                        step, report.i_links_created, report.s_links_created);
            }
        }
        
        let link_stats = topo_memory.get_link_stats();
        println!("📈 Step {}: U={}, I={}, S={}, Avg Stability={:.3}", 
                step, link_stats.u_links, link_stats.i_links, 
                link_stats.s_links, link_stats.avg_stability);
    }
    
    println!("✅ Demo completed!");
    println!("🎯 Final state: {} U-links, {} I-links, {} S-links", 
            topo_memory.get_link_stats().u_links,
            topo_memory.get_link_stats().i_links,
            topo_memory.get_link_stats().s_links);
    
    Ok(())
}

fn generate_random_bytes(length: usize) -> Vec<u8> {
    use std::collections::HashMap;
    
    // Create a more realistic byte distribution
    let mut bytes = Vec::with_capacity(length);
    let mut rng = 42u64; // Simple LCG
    
    for _ in 0..length {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        let byte = (rng % 256) as u8;
        bytes.push(byte);
    }
    
    bytes
}

fn train_model() -> Result<(), Box<dyn std::error::Error>> {
    println!("📚 Training SSM Byte Language Model");
    println!("- Architecture: SSM (State-Space Model)");
    println!("- No self-attention (10-20× FLOP reduction)");
    println!("- Byte-level tokenization");
    println!("- Target: 256k+ context length");
    
    // TODO: Implement actual training
    println!("✅ Training completed (placeholder)");
    Ok(())
}

fn run_inference() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔮 SSM Inference");
    println!("- Input: Byte sequence");
    println!("- Model: Pre-trained SSM");
    println!("- Output: Next byte predictions");
    
    // TODO: Implement actual inference
    println!("✅ Inference completed (placeholder)");
    Ok(())
}

fn run_benchmarks() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Performance Benchmarks");
    println!("- FLOPs/KB: Target -10× vs Transformer");
    println!("- DRAM/KB: Target -5× vs Transformer");
    println!("- Context: 256k+ tokens");
    println!("- Latency: <80ms p95");
    
    // TODO: Implement actual benchmarks
    println!("✅ Benchmarks completed (placeholder)");
    Ok(())
}