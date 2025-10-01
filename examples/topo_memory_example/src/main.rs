//! 🌀 HelixML Topological Memory Example
//! 
//! Demonstration of topological memory and U/I/S link system.

use backend_cpu::CpuTensor;
use tensor_core::{Device, DType, Shape, Result, Tensor};
use tensor_core::tensor::{TensorRandom, TensorOps};
use topo_memory::{TopologicalMemory, LinkManager, StabilityParams, LinkState};
use meanings::bootstrap::{BootstrapCfg, bootstrap_span};

fn main() -> Result<()> {
    println!("🌀 HelixML Topological Memory Example");
    println!("=====================================");
    
    let device = Device::cpu();
    
    // Create topological memory
    println!("\nCreating topological memory...");
    let mut topo_memory = TopologicalMemory::new(
        64, // d_model
        10, // max_motif_length
        0.2, // cycle_threshold  
        0.3, // stability_threshold
        &device
    )?;
    
    // Create test sequence
    println!("\nCreating test sequence...");
    let sequence = CpuTensor::random_uniform(
        Shape::new(vec![10, 64]), // seq_len, d_model
        0.0,
        1.0,
        &device
    )?;
    
    println!("Sequence shape: {:?}", sequence.shape());
    
    // Process sequence
    println!("\nProcessing sequence through topological memory...");
    let result = topo_memory.process_sequence(&sequence)?;
    
    println!("Processing result shape: {:?}", result.stability.shape());
    
    // Get memory statistics
    println!("\nMemory statistics:");
    let stats = topo_memory.get_stats();
    println!("Motifs: {}", stats.motif_count);
    println!("Cycles: {}", stats.cycle_count);
    println!("Stable cores: {}", stats.stable_core_count);
    
    // Test U/I/S link system
    println!("\nTesting U/I/S link system...");
    let mut link_manager = LinkManager::new();
    
    // Add several U-links
    for i in 0..5 {
        let link = topo_memory::Link::new(i, i * 10, i * 10 + 1);
        link_manager.add_u_link(link)?;
    }
    
    println!("Added 5 U-links");
    
    // Get link statistics
    let link_stats = link_manager.get_stats();
    println!("Link stats: U={}, I={}, S={}, avg_stability={:.3}", 
             link_stats.u_links, link_stats.i_links, link_stats.s_links, link_stats.avg_stability);
    
    // Test stability
    println!("\nTesting stability calculation...");
    let stability_params = StabilityParams::new(0.1, 0.5, 0.01);
    
    // Update links with signals
    link_manager.update_links_with_signals(0.3, 0.4, 0.2, 0.1, &stability_params)?;
    
    let updated_stats = link_manager.get_stats();
    println!("Updated link stats: U={}, I={}, S={}, avg_stability={:.3}", 
             updated_stats.u_links, updated_stats.i_links, updated_stats.s_links, updated_stats.avg_stability);
    
    // Test consolidation
    println!("\nTesting consolidation...");
    link_manager.sweep_and_consolidate(&stability_params, true)?;
    
    let final_stats = link_manager.get_stats();
    println!("Final link stats: U={}, I={}, S={}, avg_stability={:.3}", 
             final_stats.u_links, final_stats.i_links, final_stats.s_links, final_stats.avg_stability);
    
    // Test meaning induction bootstrap
    println!("\nTesting Meaning Induction Bootstrap...");
    let bootstrap_cfg = BootstrapCfg {
        theta_low: 0.1,
        theta_high: 0.5,
        decay: 0.01,
        replay_boost: 0.1,
        max_u_links: 1000,
    };
    
    let batch_stats = bootstrap_span(&sequence, &bootstrap_cfg, &device)?;
    println!("Bootstrap stats: U={}, I={}, S={}, avg_stability={:.3}", 
             batch_stats.u_links, batch_stats.i_links, batch_stats.s_links, batch_stats.avg_stability);
    
    println!("\n✅ Topological memory example completed successfully!");
    println!("🧠 HelixML topological memory is working!");
    
    Ok(())
}
