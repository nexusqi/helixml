use helix_ml::*;
use helix_ml::tensor::{TensorRandom, TensorOps};

fn main() -> Result<()> {
    println!("🌀 HelixML Hyena (FFT-based) Example");
    println!("====================================");

    let device = Device::cpu();

    // 1. HyenaBlock Example
    println!("\n1. Hyena Block:");
    let hyena_block = HyenaBlock::<CpuTensor>::new(64, 128, 1024, 4, &device)?;
    println!("  Hyena Block created: d_model={}, d_ff={}, max_length={}, num_fft_layers={}", 
             hyena_block.d_model(), hyena_block.d_ff(), hyena_block.max_length(), 4);
    
    // Create input sequence: [seq_len=20, d_model=64]
    let input_seq = CpuTensor::random_uniform(Shape::new(vec![20, 64]), -1.0, 1.0, &device)?;
    println!("  Input shape: {:?}", input_seq.shape());
    
    let hyena_output = hyena_block.forward(&input_seq)?;
    println!("  Hyena output shape: {:?}", hyena_output.shape());
    println!("  Hyena output sample: {:?}", hyena_output.data());

    // 2. HyenaOperator Example
    println!("\n2. Hyena Operator:");
    let hyena_op = HyenaOperator::<CpuTensor>::new(64, 512, 8, &device)?;
    println!("  Hyena Operator created: d_model=64, fft_size={}, num_filters={}", 
             hyena_op.fft_size(), hyena_op.num_filters());
    
    let op_output = hyena_op.forward(&input_seq)?;
    println!("  Operator output shape: {:?}", op_output.shape());

    // 3. Comparison: SSM vs Hyena vs Traditional
    println!("\n3. Architecture Comparison:");
    let s4_block = S4Block::<CpuTensor>::new(64, 16, &device)?;
    let mamba_block = MambaBlock::<CpuTensor>::new(64, 16, 4, 2, &device)?;
    let traditional_linear = Linear::<CpuTensor>::new(64, 64, &device)?;
    
    let test_input = CpuTensor::random_uniform(Shape::new(vec![50, 64]), -1.0, 1.0, &device)?;
    
    // Traditional Linear
    let linear_output = traditional_linear.forward(&test_input)?;
    println!("  Traditional Linear: {:?}", linear_output.shape());
    
    // S4 SSM
    let s4_output = s4_block.forward(&test_input)?;
    println!("  S4 SSM: {:?}", s4_output.shape());
    
    // Mamba SSM
    let mamba_output = mamba_block.forward(&test_input)?;
    println!("  Mamba SSM: {:?}", mamba_output.shape());
    
    // Hyena
    let hyena_output = hyena_block.forward(&test_input)?;
    println!("  Hyena FFT: {:?}", hyena_output.shape());

    // 4. Model Parameters Comparison
    println!("\n4. Model Parameters Comparison:");
    println!("  S4 parameters: {}", s4_block.parameters().len());
    println!("  Mamba parameters: {}", mamba_block.parameters().len());
    println!("  Hyena parameters: {}", hyena_block.parameters().len());
    println!("  HyenaOperator parameters: {}", hyena_op.parameters().len());

    // 5. Different Sequence Lengths
    println!("\n5. Variable Sequence Lengths:");
    let seq_lengths = vec![10, 50, 100, 500];
    
    for seq_len in seq_lengths {
        if seq_len <= hyena_block.max_length() {
            let test_input = CpuTensor::random_uniform(
                Shape::new(vec![seq_len, 64]), -1.0, 1.0, &device
            )?;
            
            let test_output = hyena_block.forward(&test_input)?;
            println!("  Seq length {}: {:?} -> {:?}", 
                     seq_len, test_input.shape(), test_output.shape());
        } else {
            println!("  Seq length {}: Exceeds max_length ({})", 
                     seq_len, hyena_block.max_length());
        }
    }

    // 6. FFT vs Traditional Performance Simulation
    println!("\n6. FFT vs Traditional Performance Simulation:");
    println!("  FFT-based Hyena advantages:");
    println!("    - O(n log n) complexity vs O(n²) for attention");
    println!("    - Better long-range dependencies");
    println!("    - More efficient for long sequences");
    println!("    - Parallelizable FFT operations");
    
    println!("  SSM advantages:");
    println!("    - Linear O(n) complexity");
    println!("    - Selective state updates (Mamba)");
    println!("    - Better for causal modeling");

    // 7. Hybrid Architecture Example
    println!("\n7. Hybrid Architecture Example:");
    let hybrid_input = CpuTensor::random_uniform(Shape::new(vec![30, 64]), -1.0, 1.0, &device)?;
    
    // Combine different architectures
    let s4_result = s4_block.forward(&hybrid_input)?;
    let hyena_result = hyena_block.forward(&hybrid_input)?;
    
    // Simple combination (in practice, would be more sophisticated)
    let combined = s4_result.add(&hyena_result)?;
    println!("  Hybrid (S4 + Hyena) output: {:?}", combined.shape());
    
    // Compare with individual results (simplified)
    let s4_max: f32 = s4_result.data().iter().fold(0.0, |acc, &x| acc.max(x.abs()));
    let hyena_max: f32 = hyena_result.data().iter().fold(0.0, |acc, &x| acc.max(x.abs()));
    let combined_max: f32 = combined.data().iter().fold(0.0, |acc, &x| acc.max(x.abs()));
    
    println!("  S4 max value: {:.6}", s4_max);
    println!("  Hyena max value: {:.6}", hyena_max);
    println!("  Combined max value: {:.6}", combined_max);

    println!("\n✅ Hyena example completed successfully!");
    println!("   FFT-based architectures are now available!");
    println!("   These provide efficient alternatives to self-attention");
    println!("   with O(n log n) complexity for long sequences.");

    Ok(())
}
