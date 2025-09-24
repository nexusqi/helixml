use helix_ml::*;
use helix_ml::tensor::{TensorRandom, TensorOps};

fn main() -> Result<()> {
    println!("🌀 HelixML SSM (State-Space Models) Example");
    println!("============================================");

    let device = Device::cpu();

    // 1. S4 Block Example
    println!("\n1. S4 (Structured State Space) Block:");
    let s4_block = S4Block::<CpuTensor>::new(64, 16, &device)?;
    println!("  S4 Block created: d_model={}, d_state={}", 
             s4_block.d_model(), s4_block.d_state());
    
    // Create input sequence: [seq_len=10, d_model=64] (2D for now)
    let input_seq = CpuTensor::random_uniform(Shape::new(vec![10, 64]), -1.0, 1.0, &device)?;
    println!("  Input shape: {:?}", input_seq.shape());
    
    let s4_output = s4_block.forward(&input_seq)?;
    println!("  S4 output shape: {:?}", s4_output.shape());
    println!("  S4 output sample: {:?}", s4_output.data());

    // 2. Mamba Block Example
    println!("\n2. Mamba Block:");
    let mamba_block = MambaBlock::<CpuTensor>::new(64, 16, 4, 2, &device)?;
    println!("  Mamba Block created: d_model=64, d_state=16, d_conv=4, expand=2");
    
    let mamba_output = mamba_block.forward(&input_seq)?;
    println!("  Mamba output shape: {:?}", mamba_output.shape());
    println!("  Mamba output sample: {:?}", mamba_output.data());

    // 3. Sequential SSM Model
    println!("\n3. Sequential SSM Model:");
    let ssm_model = Sequential::<CpuTensor>::new(&device);
    
    // Add layers: Input projection -> S4 -> Mamba -> Output projection
    let input_proj = Linear::<CpuTensor>::new(64, 128, &device)?;
    let s4_layer = S4Block::<CpuTensor>::new(128, 32, &device)?;
    let mamba_layer = MambaBlock::<CpuTensor>::new(128, 32, 4, 2, &device)?;
    let output_proj = Linear::<CpuTensor>::new(128, 64, &device)?;
    
    // Note: Sequential doesn't have add_layer method yet, so we'll use individual layers
    println!("  Created individual SSM layers");
    
    let proj_output = input_proj.forward(&input_seq)?;
    println!("  After input projection: {:?}", proj_output.shape());
    
    let s4_out = s4_layer.forward(&proj_output)?;
    println!("  After S4 layer: {:?}", s4_out.shape());
    
    let mamba_out = mamba_layer.forward(&s4_out)?;
    println!("  After Mamba layer: {:?}", mamba_out.shape());
    
    let final_output = output_proj.forward(&mamba_out)?;
    println!("  Final output: {:?}", final_output.shape());

    // 4. SSM Parameters
    println!("\n4. SSM Model Parameters:");
    println!("  S4 parameters: {}", s4_block.parameters().len());
    println!("  Mamba parameters: {}", mamba_block.parameters().len());
    println!("  Total parameters: {}", 
             s4_block.parameters().len() + mamba_block.parameters().len());

    // 5. Different Sequence Lengths
    println!("\n5. Variable Sequence Lengths:");
    let seq_lengths = vec![5, 20, 50];
    
    for seq_len in seq_lengths {
        let test_input = CpuTensor::random_uniform(
            Shape::new(vec![seq_len, 64]), -1.0, 1.0, &device
        )?;
        
        let test_output = s4_block.forward(&test_input)?;
        println!("  Seq length {}: {:?} -> {:?}", 
                 seq_len, test_input.shape(), test_output.shape());
    }

    // 6. SSM vs Traditional Comparison
    println!("\n6. SSM vs Traditional Linear:");
    let traditional_linear = Linear::<CpuTensor>::new(64, 64, &device)?;
    
    let test_input = CpuTensor::random_uniform(Shape::new(vec![100, 64]), -1.0, 1.0, &device)?;
    
    // Traditional linear (applied to each position independently)
    let linear_output = traditional_linear.forward(&test_input)?;
    println!("  Traditional Linear output: {:?}", linear_output.shape());
    
    // SSM (maintains state across sequence)
    let ssm_output = s4_block.forward(&test_input)?;
    println!("  S4 SSM output: {:?}", ssm_output.shape());
    
    // Compare outputs
    let output_diff = ssm_output.sub(&linear_output)?;
    let max_diff: f32 = output_diff.data().iter().fold(0.0, |acc, &x| acc.max(x.abs()));
    println!("  Max difference between SSM and Linear: {:.6}", max_diff);

    println!("\n✅ SSM example completed successfully!");
    println!("   State-Space Models are now available for sequence modeling!");
    println!("   These replace self-attention with more efficient state-space operations.");

    Ok(())
}
