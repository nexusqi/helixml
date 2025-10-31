//! 🌀 HelixML Broadcasting Example
//! 
//! Демонстрация broadcasting функциональности для правильного добавления bias.

use helix_ml::*;
use helix_ml::tensor::{TensorRandom, TensorBroadcast};

fn main() -> Result<()> {
    println!("🌀 HelixML Broadcasting Example");
    println!("===============================");
    
    let device = Device::cpu();
    
    // 1. Демонстрация broadcasting для bias
    println!("\n1. Linear Layer with Broadcasting Bias:");
    
    let linear = Linear::<CpuTensor>::new(4, 3, &device)?;
    let input = CpuTensor::random_uniform(Shape::new(vec![2, 4]), -1.0, 1.0, &device)?;
    
    println!("  Input shape: {:?}", input.shape());
    println!("  Bias shape: {:?}", linear.bias().unwrap().shape());
    
    let output = linear.forward(&input)?;
    println!("  Output shape: {:?}", output.shape());
    println!("  Output data: {:?}", output.data());
    
    // 2. Демонстрация unsqueeze
    println!("\n2. Unsqueeze Operation:");
    
    let tensor_1d = CpuTensor::random_uniform(Shape::new(vec![3]), 0.0, 1.0, &device)?;
    println!("  Original shape: {:?}", tensor_1d.shape());
    
    let tensor_2d = tensor_1d.unsqueeze(0)?;
    println!("  After unsqueeze(0): {:?}", tensor_2d.shape());
    
    let tensor_3d = tensor_2d.unsqueeze(2)?;
    println!("  After unsqueeze(2): {:?}", tensor_3d.shape());
    
    // 3. Демонстрация squeeze
    println!("\n3. Squeeze Operation:");
    
    let tensor_with_ones = CpuTensor::random_uniform(Shape::new(vec![1, 3, 1, 2]), 0.0, 1.0, &device)?;
    println!("  Original shape: {:?}", tensor_with_ones.shape());
    
    let squeezed = tensor_with_ones.squeeze(None)?;
    println!("  After squeeze(None): {:?}", squeezed.shape());
    
    // 4. Демонстрация broadcast_to
    println!("\n4. Broadcast To Operation:");
    
    let bias_1d = CpuTensor::random_uniform(Shape::new(vec![3]), 0.0, 1.0, &device)?;
    println!("  Bias shape: {:?}", bias_1d.shape());
    
    let bias_broadcast = bias_1d.broadcast_to(Shape::new(vec![2, 3]))?;
    println!("  Broadcasted shape: {:?}", bias_broadcast.shape());
    println!("  Broadcasted data: {:?}", bias_broadcast.data());
    
    // 5. Тест с разными размерами batch
    println!("\n5. Different Batch Sizes:");
    
    let linear_large = Linear::<CpuTensor>::new(5, 4, &device)?;
    
    // Batch size 1
    let input_batch1 = CpuTensor::random_uniform(Shape::new(vec![1, 5]), -1.0, 1.0, &device)?;
    let output_batch1 = linear_large.forward(&input_batch1)?;
    println!("  Batch 1: {:?} -> {:?}", input_batch1.shape(), output_batch1.shape());
    
    // Batch size 4
    let input_batch4 = CpuTensor::random_uniform(Shape::new(vec![4, 5]), -1.0, 1.0, &device)?;
    let output_batch4 = linear_large.forward(&input_batch4)?;
    println!("  Batch 4: {:?} -> {:?}", input_batch4.shape(), output_batch4.shape());
    
    // Batch size 8
    let input_batch8 = CpuTensor::random_uniform(Shape::new(vec![8, 5]), -1.0, 1.0, &device)?;
    let output_batch8 = linear_large.forward(&input_batch8)?;
    println!("  Batch 8: {:?} -> {:?}", input_batch8.shape(), output_batch8.shape());
    
    println!("\n✅ Broadcasting example completed successfully!");
    println!("   Bias is now properly added using broadcasting!");
    
    Ok(())
}
