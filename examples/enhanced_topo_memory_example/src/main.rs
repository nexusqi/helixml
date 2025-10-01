//! 🌀 Enhanced Topological Memory Example
//! 
//! Demonstrates the enhanced topological memory system with hierarchical processing,
//! geometric transformations, and phase synchronization

use helix_ml::*;
use tensor_core::{Tensor, Shape, DType, Device};
use tensor_core::tensor::TensorRandom;
use backend_cpu::CpuTensor;
use topo_memory::*;
use anyhow::Result;
use tracing::{info, warn, error};

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    info!("🚀 Starting Enhanced Topological Memory Example");
    
    // Create device
    let device = Device::cpu();
    
    // Example 1: Enhanced topological memory
    enhanced_memory_example(&device)?;
    
    // Example 2: Geometric processing
    geometric_processing_example(&device)?;
    
    // Example 3: Phase synchronization
    phase_synchronization_example(&device)?;
    
    // Example 4: Hierarchical processing
    hierarchical_processing_example(&device)?;
    
    info!("✅ Enhanced Topological Memory Example finished successfully!");
    Ok(())
}

/// Enhanced topological memory example
fn enhanced_memory_example(device: &Device) -> Result<()> {
    info!("🧠 Running Enhanced Topological Memory Example");
    
    // Create enhanced memory configuration
    let config = EnhancedMemoryConfig {
        hierarchical_levels: 4,
        attention_heads: 8,
        compression_ratio: 0.5,
        temporal_window: 100,
        spatial_resolution: 32,
        adaptive_threshold: 0.7,
        synthesis_strength: 0.8,
        prediction_horizon: 50,
    };
    
    // Create enhanced topological memory
    let mut enhanced_memory = EnhancedTopologicalMemory::<CpuTensor>::new(
        128, // d_model
        config,
        device,
    )?;
    
    // Create input sequence
    let sequence = CpuTensor::random_uniform(
        Shape::new(vec![32, 128]), 0.0, 1.0, device
    )?;
    
    // Process sequence with enhanced features
    let output = enhanced_memory.process_enhanced(&sequence)?;
    
    info!("📊 Enhanced Memory Output:");
    info!("  - Hierarchical levels: {}", output.hierarchical_features.level_outputs.len());
    info!("  - Multi-scale features: {}", output.multi_scale_features.scale_outputs.len());
    info!("  - Temporal features: {:?}", output.temporal_features.temporal_encoding.shape());
    info!("  - Spatial features: {:?}", output.spatial_features.spatial_encoding.shape());
    info!("  - Synthesized patterns: {}", output.synthesized_patterns.patterns.len());
    
    // Get enhanced statistics
    let stats = enhanced_memory.get_enhanced_stats();
    info!("📈 Enhanced Memory Stats: {:?}", stats);
    
    info!("✅ Enhanced Topological Memory Example completed");
    Ok(())
}

/// Geometric processing example
fn geometric_processing_example(device: &Device) -> Result<()> {
    info!("🔺 Running Geometric Processing Example");
    
    // Create geometric configuration
    let config = GeometricConfig {
        twistor_dimension: 4,
        e8_root_system_size: 240,
        mera_layers: 6,
        symmetry_tolerance: 1e-6,
        topology_resolution: 64,
        geometric_precision: 1e-8,
    };
    
    // Create geometric processor
    let mut geometric_processor = GeometricProcessor::<CpuTensor>::new(
        128, // d_model
        config,
        device,
    )?;
    
    // Create input sequence
    let sequence = CpuTensor::random_uniform(
        Shape::new(vec![32, 128]), 0.0, 1.0, device
    )?;
    
    // Process sequence with geometric transformations
    let geometric_output = geometric_processor.process_geometric(&sequence)?;
    
    info!("🔺 Geometric Processing Output:");
    info!("  - Twistor features: {:?}", geometric_output.twistor_features.twistor_encoding.shape());
    info!("  - E8 features: {:?}", geometric_output.e8_features.symmetry_transformed.shape());
    info!("  - MERA features: {:?}", geometric_output.mera_features.final_features.shape());
    info!("  - Symmetry features: {} symmetries", geometric_output.symmetry_features.symmetries.len());
    info!("  - Topology features: {:?}", geometric_output.topology_features.curvature.shape());
    
    // Get geometric metadata
    let metadata = &geometric_output.geometric_metadata;
    info!("📊 Geometric Metadata:");
    info!("  - Curvature: {}", metadata.curvature);
    info!("  - Torsion: {}", metadata.torsion);
    info!("  - Symmetry count: {}", metadata.symmetry_count);
    info!("  - Topology complexity: {}", metadata.topology_complexity);
    info!("  - Geometric stability: {}", metadata.geometric_stability);
    
    info!("✅ Geometric Processing Example completed");
    Ok(())
}

/// Phase synchronization example
fn phase_synchronization_example(device: &Device) -> Result<()> {
    info!("🌊 Running Phase Synchronization Example");
    
    // Create phase synchronization configuration
    let config = PhaseSyncConfig {
        sampling_rate: 1000.0,
        window_size: 1024,
        overlap_ratio: 0.5,
        frequency_bands: vec![
            FrequencyBand { lower_freq: 0.1, upper_freq: 4.0, band_name: "Delta".to_string() },
            FrequencyBand { lower_freq: 4.0, upper_freq: 8.0, band_name: "Theta".to_string() },
            FrequencyBand { lower_freq: 8.0, upper_freq: 13.0, band_name: "Alpha".to_string() },
            FrequencyBand { lower_freq: 13.0, upper_freq: 30.0, band_name: "Beta".to_string() },
            FrequencyBand { lower_freq: 30.0, upper_freq: 100.0, band_name: "Gamma".to_string() },
        ],
        coherence_threshold: 0.7,
        synchronization_threshold: 0.8,
        phase_resolution: 360,
    };
    
    // Create phase synchronization analyzer
    let mut phase_analyzer = PhaseSynchronizationAnalyzer::<CpuTensor>::new(
        128, // d_model
        config,
        device,
    )?;
    
    // Create SSM cores
    let ssm_cores = create_ssm_cores(device)?;
    
    // Analyze phase synchronization
    let phase_analysis = phase_analyzer.analyze_phase_synchronization(&ssm_cores)?;
    
    info!("🌊 Phase Synchronization Analysis:");
    info!("  - Phase signals: {}", phase_analysis.phase_signals.len());
    info!("  - Instantaneous phases: {}", phase_analysis.instantaneous_phases.len());
    info!("  - Phase relationships: {}", phase_analysis.phase_relationships.len());
    info!("  - Phase sync index: {}", phase_analysis.phase_sync_index);
    
    // Analyze phase coherence
    let coherence = &phase_analysis.phase_coherence;
    info!("📊 Phase Coherence:");
    info!("  - Average coherence: {}", coherence.average_coherence);
    info!("  - Coherence stability: {}", coherence.coherence_stability);
    info!("  - Coherence matrix size: {}x{}", coherence.coherence_matrix.len(), coherence.coherence_matrix[0].len());
    
    // Analyze synchronization metrics
    let metrics = &phase_analysis.synchronization_metrics;
    info!("📈 Synchronization Metrics:");
    info!("  - Sync index: {}", metrics.sync_index);
    info!("  - Sync stability: {}", metrics.sync_stability);
    info!("  - Sync coherence: {}", metrics.sync_coherence);
    
    // Analyze phase relationships
    for (i, relationship) in phase_analysis.phase_relationships.iter().enumerate() {
        if i < 3 { // Show first 3 relationships
            info!("🔗 Phase Relationship {}: Core {} -> Core {} (lag: {:.3}, coupling: {:.3}, stability: {:.3})",
                i, relationship.core1_id, relationship.core2_id,
                relationship.phase_lag, relationship.phase_coupling, relationship.phase_stability);
        }
    }
    
    info!("✅ Phase Synchronization Example completed");
    Ok(())
}

/// Hierarchical processing example
fn hierarchical_processing_example(device: &Device) -> Result<()> {
    info!("🏗️ Running Hierarchical Processing Example");
    
    // Create enhanced memory with hierarchical processing
    let config = EnhancedMemoryConfig {
        hierarchical_levels: 6,
        attention_heads: 12,
        compression_ratio: 0.3,
        temporal_window: 200,
        spatial_resolution: 64,
        adaptive_threshold: 0.8,
        synthesis_strength: 0.9,
        prediction_horizon: 100,
    };
    
    let mut enhanced_memory = EnhancedTopologicalMemory::<CpuTensor>::new(
        256, // d_model
        config,
        device,
    )?;
    
    // Create complex input sequence
    let sequence = CpuTensor::random_uniform(
        Shape::new(vec![64, 256]), 0.0, 1.0, device
    )?;
    
    // Process sequence with hierarchical features
    let output = enhanced_memory.process_enhanced(&sequence)?;
    
    info!("🏗️ Hierarchical Processing Output:");
    info!("  - Hierarchical levels: {}", output.hierarchical_features.level_outputs.len());
    info!("  - Cross-level connections: {}", output.hierarchical_features.cross_level_connections.len());
    
    // Analyze hierarchical features
    for (i, level_output) in output.hierarchical_features.level_outputs.iter().enumerate() {
        info!("📊 Level {}: Features {:?}, Patterns {:?}, Stability {:?}",
            level_output.level_id,
            level_output.features.shape(),
            level_output.patterns.shape(),
            level_output.stability.shape()
        );
    }
    
    // Analyze cross-level connections
    for (i, connection) in output.hierarchical_features.cross_level_connections.iter().enumerate() {
        if i < 3 { // Show first 3 connections
            info!("🔗 Cross-Level Connection {}: Level {} -> Level {} (strength: {:.3})",
                i, connection.from_level, connection.to_level, connection.connection_strength);
        }
    }
    
    // Analyze multi-scale features
    info!("📏 Multi-Scale Features:");
    info!("  - Scale outputs: {}", output.multi_scale_features.scale_outputs.len());
    info!("  - Scale connections: {}", output.multi_scale_features.scale_connections.len());
    
    for (i, scale_output) in output.multi_scale_features.scale_outputs.iter().enumerate() {
        info!("📊 Scale {}: Features {:?}, Resolution: {:.3}",
            scale_output.scale_id,
            scale_output.features.shape(),
            scale_output.resolution
        );
    }
    
    // Analyze attention weights
    info!("🎯 Attention Analysis:");
    info!("  - Attention weights shape: {:?}", output.attention_weights.shape());
    
    // Analyze synthesized patterns
    info!("🎨 Pattern Synthesis:");
    info!("  - Synthesized patterns: {}", output.synthesized_patterns.patterns.len());
    info!("  - Synthesis weights: {:?}", output.synthesized_patterns.synthesis_weights.shape());
    info!("  - Pattern stability: {:?}", output.synthesized_patterns.pattern_stability.shape());
    
    // Analyze stability prediction
    info!("🔮 Stability Prediction:");
    info!("  - Predicted stability: {:?}", output.stability_prediction.predicted_stability.shape());
    info!("  - Prediction confidence: {:.3}", output.stability_prediction.prediction_confidence);
    info!("  - Prediction horizon: {}", output.stability_prediction.prediction_horizon);
    
    // Get enhanced statistics
    let stats = enhanced_memory.get_enhanced_stats();
    info!("📈 Enhanced Memory Statistics:");
    info!("  - Base stats: {:?}", stats.base_stats);
    info!("  - Hierarchical levels: {}", stats.hierarchical_levels);
    info!("  - Attention heads: {}", stats.attention_heads);
    info!("  - Compressed size: {}", stats.compressed_size);
    info!("  - Temporal features: {}", stats.temporal_features);
    info!("  - Spatial features: {}", stats.spatial_features);
    info!("  - Synthesized patterns: {}", stats.synthesized_patterns);
    info!("  - Prediction accuracy: {:.3}", stats.prediction_accuracy);
    
    info!("✅ Hierarchical Processing Example completed");
    Ok(())
}

/// Create SSM cores for phase synchronization example
fn create_ssm_cores(device: &Device) -> Result<Vec<SSMCore<CpuTensor>>> {
    let mut ssm_cores = Vec::new();
    
    for i in 0..5 {
        let state = CpuTensor::random_uniform(
            Shape::new(vec![32, 128]), 0.0, 1.0, device
        )?;
        
        let parameters = SSMParameters {
            a: 0.5 + i as f32 * 0.1,
            b: 0.3 + i as f32 * 0.05,
            c: 0.8 + i as f32 * 0.02,
            d: 0.2 + i as f32 * 0.03,
        };
        
        let core = SSMCore {
            core_id: i,
            state,
            parameters,
        };
        
        ssm_cores.push(core);
    }
    
    Ok(ssm_cores)
}
