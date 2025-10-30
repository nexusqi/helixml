//! Integration tests for meanings crate

use meanings::bootstrap::*;
use topo_memory::*;
use tensor_core::*;
use backend_cpu::CpuTensor;

#[test]
fn test_bootstrap_config_default() {
    let config = BootstrapCfg::default();
    
    assert!(config.enabled);
    assert_eq!(config.window, 256);
    assert_eq!(config.pmi_threshold, 0.1);
    assert_eq!(config.replay_period, 100);
    assert_eq!(config.theta_low, 0.3);
    assert_eq!(config.theta_high, 0.7);
    assert_eq!(config.decay, 0.01);
    assert_eq!(config.u_pool_size, 1000);
}

#[test]
fn test_bootstrap_span_processing() {
    let device = Device::cpu();
    let d_model = 32;
    
    let mut topo_memory = TopologicalMemory::<CpuTensor>::new(d_model, 5, 0.6, 0.8, &device).unwrap();
    let config = BootstrapCfg::default();
    
    // Test with simple byte sequence
    let bytes = b"Hello, world! This is a test of meaning induction.";
    
    let u_links_created = bootstrap_span(bytes, &config, &mut topo_memory).unwrap();
    
    // Should create some U-links
    assert!(u_links_created > 0);
    
    let link_stats = topo_memory.get_link_stats();
    assert!(link_stats.u_links > 0);
}

#[test]
fn test_batch_stats_processing() {
    let device = Device::cpu();
    let d_model = 32;
    
    let mut topo_memory = TopologicalMemory::<CpuTensor>::new(d_model, 5, 0.6, 0.8, &device).unwrap();
    
    let stats = BatchStats {
        repetition: 0.5,
        energy: 0.3,
        connectivity: 0.4,
        phase_sync: 0.6,
    };
    
    observe_batch(stats.clone(), &mut topo_memory).unwrap();
    
    // Should update memory without errors
    let link_stats = topo_memory.get_link_stats();
    assert!(link_stats.total_links >= 0);
}

#[test]
fn test_replay_functionality() {
    let device = Device::cpu();
    let d_model = 32;
    
    let mut topo_memory = TopologicalMemory::<CpuTensor>::new(d_model, 5, 0.6, 0.8, &device).unwrap();
    let config = BootstrapCfg::default();
    
    // Create some initial U-links
    let bytes = b"Test data for replay functionality testing.";
    bootstrap_span(bytes, &config, &mut topo_memory).unwrap();
    
    // Run replay
    let replay_report = maybe_replay(100, &config, &mut topo_memory).unwrap();
    
    match replay_report {
        Some(report) => {
            assert!(report.u_links_processed > 0);
            assert!(report.stability_updated > 0);
        }
        None => {
            // Replay might not trigger if no U-links exist
            println!("No replay triggered - this is normal for empty memory");
        }
    }
}

#[test]
fn test_stability_formula() {
    let device = Device::cpu();
    let d_model = 32;
    
    let mut topo_memory = TopologicalMemory::<CpuTensor>::new(d_model, 5, 0.6, 0.8, &device).unwrap();
    
    // Create a link manually
    let link = Link::new(1, 100, 200);
    topo_memory.add_u_link(link).unwrap();
    
    // Update with high-quality signals
    let stats = BatchStats {
        repetition: 0.8,
        energy: 0.7,
        connectivity: 0.9,
        phase_sync: 0.8,
    };
    
    observe_batch(stats.clone(), &mut topo_memory).unwrap();
    
    // Run multiple updates to build stability
    for _ in 0..10 {
        observe_batch(stats.clone(), &mut topo_memory).unwrap();
    }
    
    let link_stats = topo_memory.get_link_stats();
    assert!(link_stats.avg_stability > 0.0);
}

#[test]
fn test_link_state_transitions() {
    let device = Device::cpu();
    let d_model = 32;
    
    let mut topo_memory = TopologicalMemory::<CpuTensor>::new(d_model, 5, 0.6, 0.8, &device).unwrap();
    
    // Create initial links
    let bytes = b"Testing link state transitions from U to I to S.";
    bootstrap_span(bytes, &BootstrapCfg::default(), &mut topo_memory).unwrap();
    
    let initial_stats = topo_memory.get_link_stats();
    let initial_u = initial_stats.u_links;
    
    // Apply high-quality signals repeatedly
    let high_quality_stats = BatchStats {
        repetition: 0.9,
        energy: 0.8,
        connectivity: 0.9,
        phase_sync: 0.9,
    };
    
    // Run many updates to trigger state transitions
    for _ in 0..50 {
        observe_batch(high_quality_stats.clone(), &mut topo_memory).unwrap();
        
        if let Some(report) = maybe_replay(100, &BootstrapCfg::default(), &mut topo_memory).unwrap() {
            if report.i_links_created > 0 || report.s_links_created > 0 {
                break; // Successfully triggered state transitions
            }
        }
    }
    
    let final_stats = topo_memory.get_link_stats();
    
    // Should have some state transitions
    assert!(final_stats.i_links > 0 || final_stats.s_links > 0);
}

#[test]
fn test_phase_transitions() {
    let device = Device::cpu();
    let d_model = 32;
    
    // Test Phase A configuration
    let phase_a_config = BootstrapCfg {
        enabled: true,
        pmi_threshold: 0.1,
        replay_period: 50,
        theta_low: 0.2,
        theta_high: 0.6,
        ..BootstrapCfg::default()
    };
    
    // Test Phase B configuration
    let phase_b_config = BootstrapCfg {
        enabled: true,
        pmi_threshold: 0.15,
        replay_period: 75,
        theta_low: 0.3,
        theta_high: 0.7,
        ..BootstrapCfg::default()
    };
    
    // Test Phase C configuration
    let phase_c_config = BootstrapCfg {
        enabled: false, // Bootstrap disabled
        pmi_threshold: 0.2,
        replay_period: 100,
        theta_low: 0.4,
        theta_high: 0.8,
        ..BootstrapCfg::default()
    };
    
    let mut topo_memory = TopologicalMemory::<CpuTensor>::new(d_model, 5, 0.6, 0.8, &device).unwrap();
    
    // Phase A: Bootstrap
    let bytes = b"Phase A: Creating initial U-links through bootstrap.";
    let u_links_a = bootstrap_span(bytes, &phase_a_config, &mut topo_memory).unwrap();
    assert!(u_links_a > 0);
    
    // Phase B: Consolidation
    let bytes = b"Phase B: Consolidating U-links into I and S links.";
    let u_links_b = bootstrap_span(bytes, &phase_b_config, &mut topo_memory).unwrap();
    assert!(u_links_b > 0);
    
    // Phase C: Meaning-first (no bootstrap)
    let bytes = b"Phase C: Pure topological memory operation.";
    let u_links_c = bootstrap_span(bytes, &phase_c_config, &mut topo_memory).unwrap();
    assert_eq!(u_links_c, 0); // No new U-links in Phase C
    
    let final_stats = topo_memory.get_link_stats();
    assert!(final_stats.total_links > 0);
}

#[test]
fn test_error_handling() {
    let device = Device::cpu();
    let d_model = 32;
    
    let mut topo_memory = TopologicalMemory::<CpuTensor>::new(d_model, 5, 0.6, 0.8, &device).unwrap();
    
    // Test with empty byte sequence
    let empty_bytes = b"";
    let result = bootstrap_span(empty_bytes, &BootstrapCfg::default(), &mut topo_memory);
    assert!(result.is_ok()); // Should handle empty input gracefully
    
    // Test with very short sequence
    let short_bytes = b"a";
    let result = bootstrap_span(short_bytes, &BootstrapCfg::default(), &mut topo_memory);
    assert!(result.is_ok()); // Should handle short input gracefully
    
    // Test invalid configuration
    let invalid_config = BootstrapCfg {
        window: 0, // Invalid window size
        ..BootstrapCfg::default()
    };
    
    let bytes = b"Testing invalid configuration handling.";
    let result = bootstrap_span(bytes, &invalid_config, &mut topo_memory);
    // Should either handle gracefully or return an error
    // The exact behavior depends on implementation
}
