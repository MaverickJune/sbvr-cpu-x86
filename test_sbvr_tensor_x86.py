# TODO: Use only GPU:0
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time
import sbvr
import copy
import torch

out_dir = "/home/wjbang/workspace//sbvr-cpu-x86/data"
os.makedirs(out_dir, exist_ok=True)

def r_str(s):
    return "\033[91m" + str(s) + "\033[0m"
def g_str(s):
    return "\033[92m" + str(s) + "\033[0m"
def y_str(s):
    return "\033[93m" + str(s) + "\033[0m"
def b_str(s):
    return "\033[94m" + str(s) + "\033[0m"

def print_tensor(tensor, name="Tensor"):
    print(b_str(name) + ": " 
          + g_str("shape: ") + str(tensor.shape))
    print(tensor)
    
def load_or_create_tensor(name, shape, device):
    shape_str = "_".join(map(str, shape))
    file_path = f"{out_dir}/{name}_[{shape_str}].pt"
    if os.path.exists(file_path):
        return torch.load(file_path, map_location=device).to(device)
    else:
        tensor = torch.randn(shape, device=device, dtype=torch.float16) * 0.3
        torch.save(tensor, file_path)
        return tensor

def load_or_create_sbvr(name, shape, device, num_sums, verbose_level=0, trans=False, cpu_kernel_x86=False, gpu_tensor_gen=False):
    shape_str = "_".join(map(str, shape))
    cpu_string = "cpu_" if cpu_kernel_x86 else ""
    file_path = f"{out_dir}/sbvr_{num_sums}_{cpu_string}{name}_[{shape_str}].pt"
    if os.path.exists(file_path):
        return sbvr.load(file_path, device=device, verbose_level=verbose_level, cpu_kernel_x86=cpu_kernel_x86)
    else:
        tensor = load_or_create_tensor(name, shape, device)
        sbvr_tensor = sbvr.sbvr(tensor, encoder_config={"num_sums": num_sums}, 
                                device=device, verbose_level=verbose_level,
                                trans=trans, cpu_kernel_x86=cpu_kernel_x86, gpu_tensor_gen=gpu_tensor_gen)
        sbvr_tensor.save(file_path)
        return sbvr_tensor

def float_to_fp4_e3m0(x):
    x_clamped = torch.clamp(x, -16.0, 16.0)  # Representable range
    sign = (x_clamped < 0).to(torch.uint8)

    # Prevent log2(0) by flooring to a small positive number
    x_abs = x_clamped.abs()
    x_abs = torch.clamp(x_abs, min=1e-8)

    # Compute exponent (bias = 3), round to nearest integer
    exp_unbiased = torch.round(torch.log2(x_abs))
    exp_clamped = exp_unbiased.clamp(-3, 4)
    exp_q = (exp_clamped + 3).to(torch.uint8)  # bias = 3 → encoded in 3 bits

    # Encode as 4-bit value: [sign | exponent (3 bits)]
    fp4 = (sign << 3) | exp_q
    return fp4.to(torch.uint8)

def fp4_e3m0_to_float(fp4):
    sign = (fp4 >> 3) & 0b1
    exp_q = fp4 & 0b111  # 3-bit exponent
    exponent = exp_q.to(torch.int32) - 3  # bias = 3
    value = 2.0 ** exponent
    return torch.where(sign.bool(), -value, value)

def float_to_fp4_e2m1(x):
    x_clamped = torch.clamp(x, -6.0, 6.0)  # Only representable range
    sign = (x_clamped < 0).to(torch.uint8)
    x_abs = x_clamped.abs()

    # Prevent log2(0) → set small floor
    x_abs = torch.clamp(x_abs, min=1e-8)

    # Compute exponent (bias = 1)
    exp_unbiased = torch.floor(torch.log2(x_abs))
    exp_clamped = exp_unbiased.clamp(-1, 2)
    exp_q = (exp_clamped + 1).to(torch.uint8)

    # Reconstruct base value (without mantissa)
    base = 2.0 ** exp_clamped

    # Decide mantissa: if closer to base * 1.5 than base, set mantissa = 1
    mantissa = ((x_abs >= base * 1.25)).to(torch.uint8)

    # Combine to 4-bit format: [sign | exponent (2) | mantissa]
    fp4 = (sign << 3) | (exp_q << 1) | mantissa
    return fp4.to(torch.uint8)

def fp4_e2m1_to_float(fp4):
    sign = (fp4 >> 3) & 0b1
    exp_q = (fp4 >> 1) & 0b11
    mantissa = fp4 & 0b1

    exponent = exp_q.to(torch.int32) - 1  # bias = 1
    base = 2.0 ** exponent
    value = base * (1.0 + 0.5 * mantissa)
    return torch.where(sign.bool(), -value, value)

def get_errors(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape")
    
    errors = tensor1 - tensor2
    mse = torch.mean(errors ** 2).item()
    max_error = torch.max(errors).item()
    min_error = torch.min(errors).item()
    std_dev = torch.std(errors).item()
    
    return errors, mse, max_error, min_error, std_dev
        
def print_errors(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError(f"Tensors must have the same shape: "
                         f"{tensor1.shape} vs {tensor2.shape}")
    print(g_str("Tensor 1: ") +
          y_str("Mean: ") + f"{torch.mean(tensor1):.4e}" + ", " +
          y_str("ABS Mean: ") + f"{torch.mean(tensor1.abs()):.4e}" + ", " +
          y_str("Max: ") + f"{torch.max(tensor1):.4e}" + ", " +
          y_str("Min: ") + f"{torch.min(tensor1):.4e}" + ", " +
          y_str("Std. Dev.: ") + f"{torch.std(tensor1):.4e}")
    print(g_str("Tensor 2: ") +
          y_str("Mean: ") + f"{torch.mean(tensor2):.4e}" + ", " +
          y_str("ABS Mean: ") + f"{torch.mean(tensor2.abs()):.4e}" + ", " +
          y_str("Max: ") + f"{torch.max(tensor2):.4e}" + ", " +
          y_str("Min: ") + f"{torch.min(tensor2):.4e}" + ", " +
          y_str("Std. Dev.: ") + f"{torch.std(tensor2):.4e}")
    errors, mse, max_error, min_error, std_dev = get_errors(tensor1, tensor2)
    print(r_str("Errors:   ") + 
          y_str("MSE:  ") + f"{mse:.4e}" + ", " +
          y_str("ABS Mean: ") + f"{torch.mean(errors.abs()):.4e}" + ", " +
          y_str("Max: ") + f"{max_error:.4e}" + ", " +
          y_str("Min: ") + f"{min_error:.4e}" + ", " +
          y_str("Std. Dev.: ") + f"{std_dev:.4e}")
    
def f64_matmul(mat_a, mat_b):
    if mat_a.shape[1] != mat_b.shape[0]:
        raise ValueError("Incompatible matrix shapes for multiplication")
    return (mat_a.to(torch.float64) @ mat_b.to(torch.float64)).to(torch.float64)

def generate_sbvr_tensors_pair(
    mat_len=512,
    l_num_sums=4,
    r_num_sums=4
):
    device = torch.device("cuda:0")
    # Run FP16 matmul for reference
    mat_a_size = (1, mat_len)
    mat_b_size = (mat_len, mat_len)
    mat_a = load_or_create_tensor(f"matrix_a_{mat_len}_{l_num_sums}_x86", mat_a_size, device)
    mat_b = load_or_create_tensor(f"matrix_b_{mat_len}_{r_num_sums}_x86", mat_b_size, device)
    # bias = torch.randn((mat_b.size(0),), dtype=torch.float16, device=device)*0.3
    bias = torch.zeros((mat_b.size(0),), dtype=torch.float16, device=device)
    
    # RUN SBVR v1 CPU kernel matmul
    mat_a_sbvr = load_or_create_sbvr(f"matrix_a_{mat_len}_{l_num_sums}_x86", mat_a.shape,
                                     device, l_num_sums, verbose_level=1, cpu_kernel_x86=True, gpu_tensor_gen=True)
    mat_b_sbvr = load_or_create_sbvr(f"matrix_b_{mat_len}_{r_num_sums}_x86", mat_b.shape,
                                     device, r_num_sums, verbose_level=1, cpu_kernel_x86=True, gpu_tensor_gen=True)
    
def generate_sbvr_tensors_single(
    is_matrix,
    mat_len=512,
    num_sums=4
):
    if is_matrix:
        shape = (mat_len, mat_len)
        name = f"matrix_b_{mat_len}_{num_sums}_x86"
    else:
        shape = (1, mat_len)
        name = f"matrix_a_{mat_len}_{num_sums}_x86"
    gen_matrix = load_or_create_tensor(name, shape, torch.device("cuda:0"))
    gen_sbvr = load_or_create_sbvr(name, shape, torch.device("cuda:0"), num_sums, verbose_level=1, cpu_kernel_x86=True, gpu_tensor_gen=True)
    
## Function to test the x86 kernel with a simple matmul
def sbvr_cpu_x86_matmul_time_test(
    mat_len=512,
    l_num_sums=4,
    r_num_sums=4,
    num_runs=1000
):
    device = torch.device("cpu")
    mat_a_size = (1, mat_len)
    mat_b_size = (mat_len, mat_len)
    
    mat_a = load_or_create_tensor(f"matrix_a_{mat_len}_{l_num_sums}_x86", mat_a_size, device)
    mat_b = load_or_create_tensor(f"matrix_b_{mat_len}_{r_num_sums}_x86", mat_b_size, device)
    bias = torch.zeros((mat_b.size(0),), dtype=torch.float16, device=device)
    
    # Run FP16 matmul for reference
    for i in range(10):
        f16_matmul = mat_a @ mat_b.T + bias
    time_start = time.perf_counter()
    for i in range(num_runs):
        f16_matmul = mat_a @ mat_b.T + bias
    f16_time = (time.perf_counter() - time_start) / num_runs
    
    # Run SBVR CPU x86 kernel matmul
    mat_a_sbvr = load_or_create_sbvr(f"matrix_a_{mat_len}_{l_num_sums}_x86", mat_a.shape,
                                     device, l_num_sums, verbose_level=0, cpu_kernel_x86=True, gpu_tensor_gen=False)
    mat_b_sbvr = load_or_create_sbvr(f"matrix_b_{mat_len}_{r_num_sums}_x86", mat_b.shape,
                                     device, r_num_sums, verbose_level=0, cpu_kernel_x86=True, gpu_tensor_gen=False)
    lhs_bvr = mat_a_sbvr.bvr
    lhs_coeff_idx = mat_a_sbvr.coeff_idx
    lhs_coeff_cache = mat_a_sbvr.coeff_cache
    rhs_bvr = mat_b_sbvr.bvr
    rhs_coeff_idx = mat_b_sbvr.coeff_idx
    rhs_coeff_cache = mat_b_sbvr.coeff_cache

    for _ in range(10):
        sbvr_matmul = sbvr._sbvr_cpu_mm_T(
                                lhs_bvr, lhs_coeff_idx, lhs_coeff_cache,
                                rhs_bvr, rhs_coeff_idx, rhs_coeff_cache,
                                bias)

    time_start = time.perf_counter()
    for _ in range(num_runs):
        sbvr_matmul = sbvr._sbvr_cpu_mm_T(
                                lhs_bvr, lhs_coeff_idx, lhs_coeff_cache,
                                rhs_bvr, rhs_coeff_idx, rhs_coeff_cache,
                                bias)
    sbvr_time = (time.perf_counter() - time_start) / num_runs
    
    # Printout results
    print(y_str("Matrix A Size: ") + str(mat_a_size) + ", " +
          y_str("Matrix B Size: ") + str(mat_b_size))
    print(b_str(f"fp16 matmul vs SBVR v1 {l_num_sums},{r_num_sums} bits"))
    print_errors(f16_matmul, sbvr_matmul)
    print(y_str("\tMatmul time taken: ")
          + f"{sbvr_time*10e6:.4f} usecs"
          + y_str(" vs ") + f"{f16_time*10e6:.4f} usecs")
    print(y_str("\tSpeedup: ") + f"{f16_time/sbvr_time:.4f}x")
    
    
    
if __name__ == "__main__":
    torch.manual_seed(0)
    
    # # Part 1: Generate sbvr quantized matrix
    # generate_sbvr_tensors_pair(mat_len=8192, l_num_sums=4, r_num_sums=4)
    # generate_sbvr_tensors_single(is_matrix=True, mat_len=8192, num_sums=4)
    # generate_sbvr_tensors_single(is_matrix=True, mat_len=8192, num_sums=8)
    # generate_sbvr_tensors_single(is_matrix=True, mat_len=8192, num_sums=6)
    # generate_sbvr_tensors_single(is_matrix=True, mat_len=4096, num_sums=4)
    # generate_sbvr_tensors_single(is_matrix=True, mat_len=4096, num_sums=8)
    # generate_sbvr_tensors_single(is_matrix=True, mat_len=4096, num_sums=6)
    # generate_sbvr_tensors_single(is_matrix=True, mat_len=2048, num_sums=8)
    # generate_sbvr_tensors_single(is_matrix=True, mat_len=2048, num_sums=6)
    # generate_sbvr_tensors_single(is_matrix=True, mat_len=2048, num_sums=4)
    # generate_sbvr_tensors_single(is_matrix=True, mat_len=1024, num_sums=8)
    # generate_sbvr_tensors_single(is_matrix=True, mat_len=1024, num_sums=6)
    # generate_sbvr_tensors_single(is_matrix=True, mat_len=1024, num_sums=4)
    
    # Part 2: Test matmul correctness and speed
    sbvr_cpu_x86_matmul_time_test(mat_len=1024, l_num_sums=4, r_num_sums=4, num_runs=1000)
    sbvr_cpu_x86_matmul_time_test(mat_len=1024, l_num_sums=8, r_num_sums=4, num_runs=1000)
    sbvr_cpu_x86_matmul_time_test(mat_len=1024, l_num_sums=8, r_num_sums=6, num_runs=1000)
    sbvr_cpu_x86_matmul_time_test(mat_len=1024, l_num_sums=8, r_num_sums=8, num_runs=1000)
    
    sbvr_cpu_x86_matmul_time_test(mat_len=2048, l_num_sums=4, r_num_sums=4, num_runs=1000)
    sbvr_cpu_x86_matmul_time_test(mat_len=2048, l_num_sums=8, r_num_sums=4, num_runs=1000)
    sbvr_cpu_x86_matmul_time_test(mat_len=2048, l_num_sums=8, r_num_sums=6, num_runs=1000)
    sbvr_cpu_x86_matmul_time_test(mat_len=2048, l_num_sums=8, r_num_sums=8, num_runs=1000)
    
    sbvr_cpu_x86_matmul_time_test(mat_len=4096, l_num_sums=4, r_num_sums=4, num_runs=1000)
    sbvr_cpu_x86_matmul_time_test(mat_len=4096, l_num_sums=8, r_num_sums=4, num_runs=1000)
    sbvr_cpu_x86_matmul_time_test(mat_len=4096, l_num_sums=8, r_num_sums=6, num_runs=1000)
    sbvr_cpu_x86_matmul_time_test(mat_len=4096, l_num_sums=8, r_num_sums=8, num_runs=1000)
    
    sbvr_cpu_x86_matmul_time_test(mat_len=8192, l_num_sums=4, r_num_sums=4, num_runs=1000)
    sbvr_cpu_x86_matmul_time_test(mat_len=8192, l_num_sums=8, r_num_sums=4, num_runs=1000)
    sbvr_cpu_x86_matmul_time_test(mat_len=8192, l_num_sums=8, r_num_sums=6, num_runs=1000)
    sbvr_cpu_x86_matmul_time_test(mat_len=8192, l_num_sums=8, r_num_sums=8, num_runs=1000)
    
    # Part 3: Individual extra speed measurement for verification
    # sbvr_cpu_x86_matmul_time_test(mat_len=8192, l_num_sums=8, r_num_sums=4, num_runs=1000)