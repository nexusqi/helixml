# 🌀 HelixML Meaning Induction Bootstrap (SIM/MIL)

## Overview

The Meaning Induction Bootstrap system implements a novel approach to training neural networks from scratch by building "meaning scaffolds" through topological memory systems. Unlike traditional approaches that rely on pre-trained features, this system enables models to bootstrap their own semantic understanding through the Stability formula and U/I/S link progression.

## Why Meanings, Not Features?

Traditional machine learning relies on **features** - predefined patterns extracted from data. These features are static and don't adapt to new domains or contexts. The HelixML approach focuses on **meanings** - dynamic, emergent semantic relationships that:

- **Self-organize** through the Stability formula
- **Evolve** from U → I → S link states
- **Consolidate** through replay and sleep cycles
- **Scale** to arbitrary context lengths

### Key Differences

| Aspect | Traditional Features | HelixML Meanings |
|--------|---------------------|------------------|
| **Source** | Pre-trained, static | Self-induced, dynamic |
| **Adaptation** | Limited | Continuous |
| **Context** | Fixed window | Arbitrary length |
| **Memory** | KV cache | Topological memory |
| **Scalability** | Quadratic | Linear/Log-linear |

## U/I/S Memory Model

The system implements a three-tier memory architecture:

### U-Links (Unstable)
- **Purpose**: Capture initial co-occurrence patterns
- **Creation**: PMI analysis from raw bytes/RVQ codes
- **Characteristics**: High noise, low stability
- **Lifecycle**: Created from bootstrap, evolve to I-links

### I-Links (Intermediate)  
- **Purpose**: Consolidate promising patterns
- **Creation**: U-links that exceed θ_low stability threshold
- **Characteristics**: Medium stability, active consolidation
- **Lifecycle**: Evolve to S-links or decay back to U-links

### S-Links (Stable)
- **Purpose**: Long-term semantic knowledge
- **Creation**: I-links that exceed θ_high stability threshold
- **Characteristics**: High stability, persistent knowledge
- **Lifecycle**: Permanent semantic anchors

## Stability Formula

The core mechanism driving link evolution is the **Stability Formula**:

```
S = (R + C).ln_1p() + E + Φ - decay
```

Where:
- **R** (Repetition): Signal from pattern recurrence
- **E** (Energy): Gradient/loss-based signal
- **C** (Connectivity): Attention/motif density signal  
- **Φ** (Phase): SSM state synchronization signal
- **decay**: Natural forgetting rate

### Signal Components

#### R - Repetition Signal
```rust
// Autocorrelation analysis of hidden states
let repetition = calculate_autocorrelation(hidden_states, max_lag=10);
```

#### E - Energy Signal
```rust
// Combined loss and gradient norm
let energy = loss.abs() + gradient_norm * 0.1;
```

#### C - Connectivity Signal
```rust
// Attention variance + motif density
let connectivity = attention_variance + motif_count / 100.0;
```

#### Φ - Phase Signal
```rust
// SSM state synchronization
let phase_sync = phase_synchronization_index(ssm_states);
```

## Training Pipeline: A → B → C

### Phase A: Bootstrap (Epochs 1-2)
**Goal**: Create initial U-links from raw data

```toml
[bootstrap.phase_a]
enabled = true
epochs = 2
pmi_threshold = 0.1
replay_period = 50
```

**Process**:
1. **Byte → RVQ**: Convert raw bytes to quantized codes
2. **PMI Analysis**: Find co-occurrence patterns above threshold
3. **U-Link Creation**: Initialize unstable links
4. **Active Replay**: Frequent consolidation every 50 steps

**Metrics**: U-link count, PMI distribution, replay effectiveness

### Phase B: Consolidation (Epochs 3-6)
**Goal**: Consolidate U → I → S transitions

```toml
[bootstrap.phase_b]
enabled = true
epochs = 4
pmi_threshold = 0.15  # Higher threshold
replay_period = 75    # Less frequent replay
moe_enabled = true    # Enable mixture of experts
```

**Process**:
1. **Stability Updates**: Continuous R/E/C/Φ signal integration
2. **State Transitions**: U → I → S based on thresholds
3. **MoE Integration**: Route through motif-based experts
4. **Moderate Replay**: Consolidation every 75 steps

**Metrics**: I/S link growth, transition rates, stability distribution

### Phase C: Meaning-First (Epochs 7+)
**Goal**: Pure topological memory operation

```toml
[bootstrap.phase_c]
enabled = false        # Bootstrap disabled
topo_memory_only = true
retrieval_mode = "topo_memory"
```

**Process**:
1. **Bootstrap Disabled**: No new U-link creation
2. **Topological Retrieval**: Query S/I-links for context
3. **Memory Consolidation**: Periodic sleep cycles
4. **Domain Adaptation**: Transfer to new domains

**Metrics**: Retrieval hit-rate, latency, memory efficiency

## Configuration

### Basic Setup
```toml
[bootstrap]
enabled = true
window = 256
pmi_threshold = 0.1
replay_period = 100

[stability]
theta_low = 0.3
theta_high = 0.7
decay = 0.01
```

### Domain-Specific Configuration
```toml
[domains.text]
bootstrap_enabled = true
context_length = 512000
pmi_threshold = 0.08
window = 512

[domains.code]
bootstrap_enabled = true
context_length = 128000
pmi_threshold = 0.12
window = 256
```

## Performance Metrics

### Primary Metrics

#### Link Evolution
- **U → I Transition Rate**: Fraction of U-links becoming intermediate
- **I → S Transition Rate**: Fraction of I-links becoming stable
- **S-Link Persistence**: Long-term retention of stable links

#### Memory Efficiency
- **FLOPs/KB**: Computational cost per kilobyte of context
- **DRAM/KB**: Memory usage per kilobyte of context
- **Latency p95**: 95th percentile response time

#### Retrieval Quality
- **Hit Rate**: Fraction of successful memory retrievals
- **Context Accuracy**: Semantic coherence of retrieved context
- **Transfer Performance**: Adaptation to new domains

### Target Benchmarks

| Metric | Phase A | Phase B | Phase C | Target |
|--------|---------|---------|---------|--------|
| **FLOPs/KB** | 1500 | 1200 | 1000 | < 1000 |
| **DRAM/KB** | 800 | 600 | 512 | < 512 |
| **Latency p95** | 80ms | 60ms | 50ms | < 50ms |
| **Context Length** | 256k | 512k | 1M | > 1M |

## Usage Examples

### Basic Bootstrap Training
```rust
use helix_ml::meanings::*;

let cfg = BootstrapCfg::default();
let mut topo = TopologicalMemory::new(64, 5, 0.7, 0.8, &device)?;

// Phase A: Bootstrap
for epoch in 0..2 {
    for batch in training_data {
        // Create U-links from raw bytes
        let u_links = bootstrap_span(&batch.bytes, &cfg, &mut topo)?;
        
        // Process through model
        let output = model.forward(&batch.input)?;
        
        // Calculate signals
        let stats = BatchStats {
            repetition: calculate_repetition(&output)?,
            energy: calculate_energy(&loss)?,
            connectivity: calculate_connectivity(&attention)?,
            phase_sync: calculate_phase_sync(&ssm_states)?,
        };
        
        // Update memory
        observe_batch(stats, &mut topo)?;
        
        // Periodic replay
        if step % cfg.replay_period == 0 {
            let report = maybe_replay(step, &cfg, &mut topo)?;
            println!("Replay: {} U→I, {} I→S", 
                    report.i_links_created, report.s_links_created);
        }
    }
}
```

### Phase Transition
```rust
// Transition from Phase A to B
let phase_b_cfg = BootstrapCfg {
    pmi_threshold: 0.15,  // Higher threshold
    replay_period: 75,    // Less frequent
    theta_high: 0.7,      // Stricter S-link criteria
    ..cfg
};

// Continue training with new configuration
```

### Memory Retrieval
```rust
// Query topological memory
let query = create_query_tensor(&input)?;
let m0_result = topo.retrieve(&query, MemoryLevel::M0)?;  // Motifs
let m1_result = topo.retrieve(&query, MemoryLevel::M1)?;  // Cycles  
let m2_result = topo.retrieve(&query, MemoryLevel::M2)?;  // Stable cores

// Combine results for context
let context = combine_retrieval_results(&[m0_result, m1_result, m2_result])?;
```

## Monitoring and Debugging

### Real-time Monitoring
```rust
// Get memory statistics
let stats = topo.get_link_stats();
println!("U: {}, I: {}, S: {}, Avg Stability: {:.3}",
         stats.u_links, stats.i_links, stats.s_links, stats.avg_stability);

// Monitor phase transitions
let transition_rate = stats.i_links as f32 / stats.u_links as f32;
if transition_rate > 0.3 {
    println!("High U→I transition rate: {:.3}", transition_rate);
}
```

### Performance Profiling
```rust
// Enable performance profiling
let profile = PerformanceProfile::new();
let start_time = std::time::Instant::now();

// ... training step ...

let elapsed = start_time.elapsed();
profile.record_step(elapsed, &stats);
```

## Troubleshooting

### Common Issues

#### Low U → I Transition Rate
- **Cause**: PMI threshold too high, insufficient replay
- **Solution**: Lower `pmi_threshold`, increase replay frequency
- **Monitor**: PMI distribution, replay effectiveness

#### High Memory Usage
- **Cause**: Too many U-links, insufficient decay
- **Solution**: Increase `decay` rate, limit `u_pool_size`
- **Monitor**: Link counts, memory statistics

#### Poor Retrieval Performance
- **Cause**: Insufficient S-links, weak stability signals
- **Solution**: Lower `theta_high`, improve signal quality
- **Monitor**: Hit rate, context coherence

### Debug Configuration
```toml
[monitoring]
verbose = true
metrics_interval = 10  # More frequent metrics
save_stats = true
profile_performance = true
```

## Research Directions

### Current Limitations
- **Signal Quality**: R/E/C/Φ signals may be noisy
- **Threshold Tuning**: Optimal θ_low/θ_high values domain-dependent
- **Replay Efficiency**: Optimal replay scheduling unclear

### Future Work
- **Adaptive Thresholds**: Dynamic threshold adjustment
- **Multi-scale Memory**: Hierarchical memory levels
- **Cross-domain Transfer**: Efficient domain adaptation
- **Theoretical Analysis**: Convergence guarantees, optimality

## References

- **Stability Formula**: Inspired by synaptic plasticity models
- **U/I/S States**: Based on memory consolidation theories
- **PMI Analysis**: Standard co-occurrence analysis
- **Phase Synchronization**: Adapted from neural oscillation research

---

*For technical implementation details, see the source code in `crates/meanings/` and `crates/topo-memory/`.*
