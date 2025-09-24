//! 🌀 HelixML Gradient Checkpointing Example
//! 
//! Демонстрация gradient checkpointing для экономии памяти.

use helix_ml::*;
use helix_ml::tensor::{TensorRandom, TensorOps};

fn main() -> Result<()> {
    println!("🌀 HelixML Gradient Checkpointing Example");
    println!("==========================================");
    
    let device = Device::cpu();
    
    // 1. Создаем autograd контекст
    let mut ctx = AutogradContext::<CpuTensor>::new();
    
    println!("\n1. Creating AutogradContext:");
    println!("  Initial checkpoint count: {}", ctx.checkpoint_count());
    
    // 2. Создаем тензоры
    let input_data = CpuTensor::random_uniform(Shape::new(vec![2, 4]), -1.0, 1.0, &device)?;
    let input_id = ctx.tensor(input_data, true);
    
    let weight_data = CpuTensor::random_uniform(Shape::new(vec![3, 4]), -1.0, 1.0, &device)?;
    let weight_id = ctx.tensor(weight_data, true);
    
    println!("  Input tensor ID: {}", input_id);
    println!("  Weight tensor ID: {}", weight_id);
    
    // 3. Создаем checkpoint перед вычислениями
    println!("\n2. Creating Checkpoint:");
    ctx.checkpoint(input_id)?;
    println!("  Checkpoint count after checkpointing: {}", ctx.checkpoint_count());
    
    // 4. Выполняем forward pass (имитация)
    println!("\n3. Forward Pass Simulation:");
    
    // Получаем тензоры
    let input_tensor = ctx.get_tensor(input_id).unwrap();
    let weight_tensor = ctx.get_tensor(weight_id).unwrap();
    
    // Вычисляем output = input @ weight.T
    let output = input_tensor.matmul(&weight_tensor.transpose(0, 1)?)?;
    let output_id = ctx.tensor(output, true);
    
    println!("  Output tensor ID: {}", output_id);
    println!("  Output shape: {:?}", ctx.get_tensor(output_id).unwrap().shape());
    
    // 5. Создаем еще один checkpoint
    println!("\n4. Creating Second Checkpoint:");
    ctx.checkpoint(output_id)?;
    println!("  Checkpoint count: {}", ctx.checkpoint_count());
    
    // 6. Имитируем backward pass с восстановлением checkpoint
    println!("\n5. Backward Pass with Checkpoint Restoration:");
    
    // Восстанавливаем checkpoint для output
    let restored_output_id = ctx.restore_checkpoint(output_id, |ctx| {
        println!("    Recomputing forward pass from checkpoint...");
        
        // Получаем тензоры заново
        let input_tensor = ctx.get_tensor(input_id).unwrap();
        let weight_tensor = ctx.get_tensor(weight_id).unwrap();
        
        // Пересчитываем
        let output = input_tensor.matmul(&weight_tensor.transpose(0, 1)?)?;
        let output_id = ctx.tensor(output, true);
        
        Ok(output_id)
    })?;
    
    println!("  Restored output tensor ID: {}", restored_output_id);
    println!("  Checkpoint count after restoration: {}", ctx.checkpoint_count());
    
    // 7. Очищаем checkpoints
    println!("\n6. Cleaning Up:");
    ctx.clear_checkpoints();
    println!("  Checkpoint count after cleanup: {}", ctx.checkpoint_count());
    
    // 8. Демонстрация с отдельными модулями
    println!("\n7. Individual Modules with Checkpointing:");
    
    let linear1 = Linear::<CpuTensor>::new(4, 8, &device)?;
    let relu = ReLU::<CpuTensor>::new(&device);
    let linear2 = Linear::<CpuTensor>::new(8, 3, &device)?;
    
    let test_input = CpuTensor::random_uniform(Shape::new(vec![1, 4]), -1.0, 1.0, &device)?;
    println!("  Test input shape: {:?}", test_input.shape());
    
    let output1 = linear1.forward(&test_input)?;
    let output2 = relu.forward(&output1)?;
    let output3 = linear2.forward(&output2)?;
    
    println!("  After linear1: {:?}", output1.shape());
    println!("  After relu: {:?}", output2.shape());
    println!("  Final output: {:?}", output3.shape());
    println!("  Final data: {:?}", output3.data());
    
    println!("\n✅ Gradient checkpointing example completed successfully!");
    println!("   Memory-efficient training is now supported!");
    
    Ok(())
}
