//! 🌀 HelixML Advanced Example
//! 
//! Продвинутый пример использования улучшенного фреймворка HelixML.

use helix_ml::*;
use helix_ml::tensor::{TensorRandom, TensorOps};

fn main() -> Result<()> {
    println!("🌀 HelixML Advanced Example");
    println!("===========================");
    
    let device = Device::cpu();
    
    // 1. Демонстрация улучшенного matrix multiplication
    println!("\n1. Enhanced Matrix Multiplication:");
    
    // 2D x 2D
    let a_2d = CpuTensor::random_uniform(Shape::new(vec![3, 4]), -1.0, 1.0, &device)?;
    let b_2d = CpuTensor::random_uniform(Shape::new(vec![4, 5]), -1.0, 1.0, &device)?;
    let c_2d = a_2d.matmul(&b_2d)?;
    println!("  2D x 2D: {:?} x {:?} = {:?}", a_2d.shape(), b_2d.shape(), c_2d.shape());
    
    // 1D x 2D (vector-matrix)
    let vec_1d = CpuTensor::random_uniform(Shape::new(vec![4]), -1.0, 1.0, &device)?;
    let mat_2d = CpuTensor::random_uniform(Shape::new(vec![4, 3]), -1.0, 1.0, &device)?;
    let result_1d = vec_1d.matmul(&mat_2d)?;
    println!("  1D x 2D: {:?} x {:?} = {:?}", vec_1d.shape(), mat_2d.shape(), result_1d.shape());
    
    // 2D x 1D (matrix-vector) - нужно правильные размеры
    let mat_2d_for_vec = CpuTensor::random_uniform(Shape::new(vec![3, 4]), -1.0, 1.0, &device)?;
    let result_vec = mat_2d_for_vec.matmul(&vec_1d)?;
    println!("  2D x 1D: {:?} x {:?} = {:?}", mat_2d_for_vec.shape(), vec_1d.shape(), result_vec.shape());
    
    // 2. Демонстрация slicing
    println!("\n2. Tensor Slicing:");
    let big_tensor = CpuTensor::random_uniform(Shape::new(vec![5, 6]), 0.0, 10.0, &device)?;
    println!("  Original tensor shape: {:?}", big_tensor.shape());
    
    let sliced = big_tensor.slice(vec![1, 2], vec![4, 5])?;
    println!("  Sliced tensor shape: {:?}", sliced.shape());
    println!("  Sliced data: {:?}", sliced.data());
    
    // 3. Демонстрация concatenation
    println!("\n3. Tensor Concatenation:");
    let tensor1 = CpuTensor::random_uniform(Shape::new(vec![3]), 0.0, 1.0, &device)?;
    let tensor2 = CpuTensor::random_uniform(Shape::new(vec![2]), 0.0, 1.0, &device)?;
    let tensor3 = CpuTensor::random_uniform(Shape::new(vec![4]), 0.0, 1.0, &device)?;
    
    let concatenated = CpuTensor::cat(vec![tensor1, tensor2, tensor3], 0)?;
    println!("  Concatenated shape: {:?}", concatenated.shape());
    println!("  Concatenated data: {:?}", concatenated.data());
    
    // 4. Демонстрация stacking
    println!("\n4. Tensor Stacking:");
    let stack_tensor1 = CpuTensor::random_uniform(Shape::new(vec![3]), 0.0, 1.0, &device)?;
    let stack_tensor2 = CpuTensor::random_uniform(Shape::new(vec![3]), 0.0, 1.0, &device)?;
    
    let stacked = CpuTensor::stack(vec![stack_tensor1, stack_tensor2], 0)?;
    println!("  Stacked shape: {:?}", stacked.shape());
    println!("  Stacked data: {:?}", stacked.data());
    
    // 5. Демонстрация sign операции
    println!("\n5. Sign Operation:");
    let mixed_tensor = CpuTensor::random_uniform(Shape::new(vec![5]), -2.0, 2.0, &device)?;
    println!("  Original data: {:?}", mixed_tensor.data());
    
    let sign_tensor = mixed_tensor.sign()?;
    println!("  Sign data: {:?}", sign_tensor.data());
    
    // 6. Демонстрация Linear слоя с bias
    println!("\n6. Linear Layer with Bias:");
    let linear = Linear::<CpuTensor>::new(4, 3, &device)?;
    let input = CpuTensor::random_uniform(Shape::new(vec![2, 4]), -1.0, 1.0, &device)?;
    
    println!("  Input shape: {:?}", input.shape());
    let output = linear.forward(&input)?;
    println!("  Output shape: {:?}", output.shape());
    println!("  Output data: {:?}", output.data());
    
    // 7. Демонстрация оптимизаторов
    println!("\n7. Optimizers with Sign Operation:");
    let lion_optimizer = Lion::<CpuTensor>::new(0.001, &device);
    println!("  Lion optimizer created with lr=0.001");
    
    let adamw_optimizer = AdamW::<CpuTensor>::new(0.001, &device);
    println!("  AdamW optimizer created with lr=0.001");
    
    println!("\n✅ Advanced example completed successfully!");
    println!("   All improvements are working correctly!");
    
    Ok(())
}
