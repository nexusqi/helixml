//! 🌀 HelixML Mixed Precision Example
//! 
//! Демонстрация mixed precision (FP16/INT8) функциональности.

use helix_ml::*;
use helix_ml::tensor::{TensorRandom, TensorMixedPrecision, TensorOps};

fn main() -> Result<()> {
    println!("🌀 HelixML Mixed Precision Example");
    println!("==================================");
    
    let device = Device::cpu();
    
    // 1. Создаем тензор в FP32
    println!("\n1. Creating FP32 Tensor:");
    let tensor_f32 = CpuTensor::random_uniform(Shape::new(vec![3, 4]), -1.0, 1.0, &device)?;
    println!("  Original tensor dtype: {:?}", tensor_f32.dtype());
    println!("  Original data: {:?}", tensor_f32.data());
    
    // 2. Конвертируем в FP16
    println!("\n2. Converting to FP16:");
    let tensor_f16 = tensor_f32.to_f16()?;
    println!("  FP16 tensor dtype: {:?}", tensor_f16.dtype());
    println!("  FP16 data: {:?}", tensor_f16.data());
    
    // 3. Конвертируем обратно в FP32
    println!("\n3. Converting back to FP32:");
    let tensor_f32_restored = tensor_f16.to_f32()?;
    println!("  Restored FP32 tensor dtype: {:?}", tensor_f32_restored.dtype());
    println!("  Restored data: {:?}", tensor_f32_restored.data());
    
    // 4. Конвертируем в INT8
    println!("\n4. Converting to INT8:");
    let tensor_i8 = tensor_f32.to_i8()?;
    println!("  INT8 tensor dtype: {:?}", tensor_i8.dtype());
    println!("  INT8 data: {:?}", tensor_i8.data());
    
    // 5. Конвертируем в INT32
    println!("\n5. Converting to INT32:");
    let tensor_i32 = tensor_f32.to_i32()?;
    println!("  INT32 tensor dtype: {:?}", tensor_i32.dtype());
    println!("  INT32 data: {:?}", tensor_i32.data());
    
    // 6. Демонстрация quantization
    println!("\n6. INT8 Quantization:");
    let scale = 0.1;
    let zero_point = 0;
    
    let quantized = tensor_f32.quantize_int8(scale, zero_point)?;
    println!("  Quantized dtype: {:?}", quantized.dtype());
    println!("  Quantized data: {:?}", quantized.data());
    
    // 7. Демонстрация dequantization
    println!("\n7. INT8 Dequantization:");
    let dequantized = quantized.dequantize_int8(scale, zero_point)?;
    println!("  Dequantized dtype: {:?}", dequantized.dtype());
    println!("  Dequantized data: {:?}", dequantized.data());
    
    // 8. Сравнение точности
    println!("\n8. Precision Comparison:");
    let original_max: f32 = tensor_f32.data().iter().fold(0.0, |acc, &x| acc.max(x.abs()));
    let restored_max: f32 = tensor_f32_restored.data().iter().fold(0.0, |acc, &x| acc.max(x.abs()));
    let dequantized_max: f32 = dequantized.data().iter().fold(0.0, |acc, &x| acc.max(x.abs()));
    
    println!("  Original max value: {:.6}", original_max);
    println!("  FP16->FP32 max value: {:.6}", restored_max);
    println!("  INT8->FP32 max value: {:.6}", dequantized_max);
    
    // 9. Демонстрация с Linear слоем
    println!("\n9. Mixed Precision with Linear Layer:");
    
    // Создаем Linear слой
    let linear = Linear::<CpuTensor>::new(4, 3, &device)?;
    let input = CpuTensor::random_uniform(Shape::new(vec![2, 4]), -1.0, 1.0, &device)?;
    
    println!("  Input dtype: {:?}", input.dtype());
    
    // Forward pass в FP32
    let output_f32 = linear.forward(&input)?;
    println!("  Output FP32 dtype: {:?}", output_f32.dtype());
    
    // Конвертируем input в FP16 и обратно
    let input_f16 = input.to_f16()?;
    let input_restored = input_f16.to_f32()?;
    
    // Forward pass с восстановленным input
    let output_restored = linear.forward(&input_restored)?;
    println!("  Output restored dtype: {:?}", output_restored.dtype());
    
    // Сравниваем результаты
    let diff = output_f32.sub(&output_restored)?;
    let max_diff: f32 = diff.data().iter().fold(0.0, |acc, &x| acc.max(x.abs()));
    println!("  Max difference between FP32 and FP16->FP32: {:.8}", max_diff);
    
    // 10. Проверка поддержки mixed precision
    println!("\n10. Mixed Precision Support:");
    println!("  Tensor supports mixed precision: {}", tensor_f32.supports_mixed_precision());
    println!("  FP16 tensor supports mixed precision: {}", tensor_f16.supports_mixed_precision());
    println!("  INT8 tensor supports mixed precision: {}", tensor_i8.supports_mixed_precision());
    
    println!("\n✅ Mixed precision example completed successfully!");
    println!("   FP16/INT8 support is now available!");
    
    Ok(())
}
