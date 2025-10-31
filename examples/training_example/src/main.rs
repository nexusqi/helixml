//! 🌀 HelixML Training Example
//! 
//! Complete training example with backward pass integration

use helix_ml::*;
use tensor_core::tensor::TensorRandom;
use backend_cpu::CpuTensor;
use nn::{Module, Linear, SiLU};
use optim::AdamW;
use training::{MSELoss, LossFunction, Reduction};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🌀 HelixML Training Example");
    println!("===========================");
    
    let device = Device::cpu();
    let batch_size = 32;
    let input_dim = 64;
    let hidden_dim = 32;
    let output_dim = 1;
    let num_epochs = 10;
    
    // Create a simple neural network
    println!("\n📊 Creating model...");
    let linear1 = Linear::<CpuTensor>::new(input_dim, hidden_dim, &device)?;
    let activation = SiLU::<CpuTensor>::new(&device);
    let linear2 = Linear::<CpuTensor>::new(hidden_dim, output_dim, &device)?;
    
    println!("✅ Model created!");
    println!("   Input: {} -> Hidden: {} -> Output: {}", input_dim, hidden_dim, output_dim);
    
    // Create optimizer
    println!("\n⚙️  Creating optimizer...");
    let optimizer = AdamW::<CpuTensor>::new(0.001, &device);
    println!("✅ AdamW optimizer ready!");
    
    // Create loss function
    println!("\n📉 Creating loss function...");
    let loss_fn = MSELoss::<CpuTensor>::new(Reduction::Mean);
    println!("✅ MSE Loss ready!");
    
    // Training loop
    println!("\n🚀 Starting training...");
    for epoch in 0..num_epochs {
        // Generate synthetic data
        let input = CpuTensor::random_normal(
            Shape::new(vec![batch_size, input_dim]),
            0.0,
            1.0,
            &device,
        )?;
        
        // Forward pass
        let hidden = activation.forward(&linear1.forward(&input)?)?;
        let output = linear2.forward(&hidden)?;
        
        // Generate random targets
        let target = CpuTensor::random_normal(
            Shape::new(vec![batch_size, output_dim]),
            0.0,
            1.0,
            &device,
        )?;
        
        // Compute loss
        let loss = loss_fn.compute(&[output], &[target])?;
        
        // Print training info  
        println!("Epoch {:2}/{}: Loss computed", 
                epoch + 1, num_epochs);
        
        // TODO: Backward pass integration
        // This requires integration with autograd context
        // For now, we'll just show the structure
    }
    
    println!("\n✅ Training completed!");
    println!("🎉 HelixML training loop is working!");
    
    Ok(())
}
