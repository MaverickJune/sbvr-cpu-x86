# TODO: Use only GPU:0
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time
import sbvr
import copy
import torch

out_dir = "/home/wjbang/workspace/sbvr_cpu/sbvr-cpu-arm/data"
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

def load_or_create_sbvr(name, shape, device, num_sums, verbose_level=0, trans=False, cpu_kernel=False, gpu_tensor_gen=False):
    shape_str = "_".join(map(str, shape))
    cpu_string = "cpu_" if cpu_kernel else ""
    file_path = f"{out_dir}/sbvr_{num_sums}_{cpu_string}{name}_[{shape_str}].pt"
    if os.path.exists(file_path):
        return sbvr.load(file_path, device=device, verbose_level=verbose_level, cpu_kernel=cpu_kernel)
    else:
        tensor = load_or_create_tensor(name, shape, device)
        sbvr_tensor = sbvr.sbvr(tensor, encoder_config={"num_sums": num_sums}, 
                                device=device, verbose_level=verbose_level,
                                trans=trans, cpu_kernel=cpu_kernel, gpu_tensor_gen=gpu_tensor_gen)
        sbvr_tensor.save(file_path)
        return sbvr_tensor

# [v2] Helper to load or create SBVR objects with v2 (CUDA-style) layout for ARM CPU
def load_or_create_sbvr_v2(name, shape, device, num_sums, verbose_level=0, trans=False):
    shape_str = "_".join(map(str, shape))
    file_path = f"{out_dir}/sbvr_{num_sums}_cpuv2_{name}_[{shape_str}].pt"
    if os.path.exists(file_path):
        return sbvr.load(file_path, device=device, verbose_level=verbose_level, cpu_kernel_v2=True)
    else:
        tensor = load_or_create_tensor(name, shape, device)
        sbvr_tensor = sbvr.sbvr(tensor, encoder_config={"num_sums": num_sums},
                                device=device, verbose_level=verbose_level,
                                trans=trans, cpu_kernel_v2=True)
        sbvr_tensor.save(file_path)
        return sbvr_tensor
    
def create_sbvr(tensor, name, shape, device, num_sums, verbose_level=0):
    shape_str = "_".join(map(str, shape))
    file_path = f"{out_dir}/sbvr_{num_sums}_{name}_[{shape_str}].pt"
    sbvr_tensor = sbvr.sbvr(tensor, encoder_config={"num_sums": num_sums}, 
                            device=device, verbose_level=verbose_level)
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
    mat_a = load_or_create_tensor(f"matrix_a_{mat_len}_{l_num_sums}_v1", mat_a_size, device)
    mat_b = load_or_create_tensor(f"matrix_b_{mat_len}_{r_num_sums}_v1", mat_b_size, device)
    # bias = torch.randn((mat_b.size(0),), dtype=torch.float16, device=device)*0.3
    bias = torch.zeros((mat_b.size(0),), dtype=torch.float16, device=device)
    
    # RUN SBVR v1 CPU kernel matmul
    mat_a_sbvr = load_or_create_sbvr(f"matrix_a_{mat_len}_{l_num_sums}_v1", mat_a.shape,
                                     device, l_num_sums, verbose_level=1, cpu_kernel=True, gpu_tensor_gen=True)
    mat_b_sbvr = load_or_create_sbvr(f"matrix_b_{mat_len}_{r_num_sums}_v1", mat_b.shape,
                                     device, r_num_sums, verbose_level=1, cpu_kernel=True, gpu_tensor_gen=True)
    
def generate_sbvr_tensors_single(
    is_matrix,
    mat_len=512,
    num_sums=4
):
    if is_matrix:
        shape = (mat_len, mat_len)
        name = f"matrix_b_{mat_len}_{num_sums}_v1"
    else:
        shape = (1, mat_len)
        name = f"matrix_a_{mat_len}_{num_sums}_v1"
    gen_matrix = load_or_create_tensor(name, shape, torch.device("cuda:0"))
    gen_sbvr = load_or_create_sbvr(name, shape, torch.device("cuda:0"), num_sums, verbose_level=1, cpu_kernel=True, gpu_tensor_gen=True)
    
if __name__ == "__main__":
    # generate_sbvr_tensors_pair(mat_len=8192, l_num_sums=4, r_num_sums=4)
    # generate_sbvr_tensors_single(is_matrix=True, mat_len=8192, num_sums=8)
    generate_sbvr_tensors_single(is_matrix=True, mat_len=8192, num_sums=6)
    generate_sbvr_tensors_single(is_matrix=True, mat_len=4096, num_sums=8)
    generate_sbvr_tensors_single(is_matrix=True, mat_len=4096, num_sums=6)
    generate_sbvr_tensors_single(is_matrix=True, mat_len=2048, num_sums=8)
    generate_sbvr_tensors_single(is_matrix=True, mat_len=2048, num_sums=6)
    generate_sbvr_tensors_single(is_matrix=True, mat_len=2048, num_sums=4)
    generate_sbvr_tensors_single(is_matrix=True, mat_len=1024, num_sums=8)
    generate_sbvr_tensors_single(is_matrix=True, mat_len=1024, num_sums=6)
    generate_sbvr_tensors_single(is_matrix=True, mat_len=1024, num_sums=4)