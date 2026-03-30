import torch
import itertools
import math
import numpy as np
import ctypes
import os
from tqdm import tqdm

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def _r_str(s):
    return "\033[91m" + str(s) + "\033[0m"
def _g_str(s):
    return "\033[92m" + str(s) + "\033[0m"
def _y_str(s):
    return "\033[93m" + str(s) + "\033[0m"
def _b_str(s):
    return "\033[94m" + str(s) + "\033[0m"

def print_avg_min_max_std_histogram(tensor: torch.Tensor, 
                                    name: str, output_dir: str):
    """Print the average, minimum, maximum, and standard deviation of the tensor
    and save the histogram"""
    tensor_flat = tensor.flatten().to(torch.float64)
    num = tensor_flat.size(0)
    avg = torch.mean(tensor_flat)
    min_val = torch.min(tensor_flat)
    max_val = torch.max(tensor_flat)
    std = torch.std(tensor_flat)
    
    # Plot the histogram and save it
    plt.hist(tensor_flat.cpu().numpy(), bins=500)
    plt.title(name)
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # Plt text avg, min, max, std on the graph up to 6 decimal places
    if tensor_flat.shape[0] > 2**24:
        sample_idx = torch.randint(0, tensor_flat.shape[0], (2**24,))
        tensor_flat = tensor_flat[sample_idx]
    per_25 = torch.quantile(tensor_flat, 0.25)
    per_50 = torch.quantile(tensor_flat, 0.50)
    per_75 = torch.quantile(tensor_flat, 0.75)
    per_99 = torch.quantile(tensor_flat, 0.99)
    per_01 = torch.quantile(tensor_flat, 0.01)
    per_995 = torch.quantile(tensor_flat, 0.995)
    per_005 = torch.quantile(tensor_flat, 0.005)
    per_999 = torch.quantile(tensor_flat, 0.999)
    per_001 = torch.quantile(tensor_flat, 0.001)
    plt.text(0.6, 0.95, f"Num: {num:.0f}", ha='center', va='center', 
             transform=plt.gca().transAxes)
    plt.text(0.6, 0.9, f"Avg: {avg:.7f}", ha='center', va='center', 
             transform=plt.gca().transAxes)
    plt.text(0.6, 0.85, f"Min: {min_val:.7f}", ha='center', va='center', 
             transform=plt.gca().transAxes)
    plt.text(0.6, 0.8, f"Max: {max_val:.7f}", ha='center', va='center', 
             transform=plt.gca().transAxes)
    plt.text(0.6, 0.75, f"Std: {std:.7f}", ha='center', va='center', 
             transform=plt.gca().transAxes)
    plt.text(0.85, 0.95, f"0.1%: {per_001:.7f}", ha='center', va='center', 
             transform=plt.gca().transAxes)
    plt.text(0.85, 0.9, f"25%: {per_25:.7f}", ha='center', va='center', 
             transform=plt.gca().transAxes)
    plt.text(0.85, 0.85, f"50%: {per_50:.7f}", ha='center', va='center', 
             transform=plt.gca().transAxes)
    plt.text(0.85, 0.8, f"75%: {per_75:.7f}", ha='center', va='center', 
             transform=plt.gca().transAxes)
    plt.text(0.85, 0.75, f"99%: {per_99:.7f}", ha='center', va='center', 
             transform=plt.gca().transAxes)
    plt.text(0.85, 0.7, f"99.5%: {per_995:.7f}", ha='center', va='center', 
             transform=plt.gca().transAxes)
    plt.text(0.85, 0.65, f"99.9%: {per_999:.7f}", ha='center', va='center', 
             transform=plt.gca().transAxes)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}{name}.png", dpi=600)
    plt.show()
    plt.close()

@torch.inference_mode()
def get_bin_combs(num_sums, device, dtype):
    bin_combs = torch.tensor(
        list(itertools.product([0, 1], repeat=num_sums)),
        dtype=dtype, device=device
    )
    return bin_combs

@torch.inference_mode()
def get_coeff_search_space_from_lists(r_list, num_sums):
    exponents = torch.arange(num_sums, device=r_list.device) 
    search_space = r_list.unsqueeze(1) ** exponents.unsqueeze(0)
    if r_list[0].item() < 0:
        max = search_space[:,0::2].sum(dim=1)
        min = search_space[:,1::2].sum(dim=1)
    else:
        max = search_space.sum(dim=1)
        min = 0
    search_space =search_space / (max-min).unsqueeze(1)
    return search_space, r_list

@torch.inference_mode()
def get_coeff_search_space(data, num_sums):
    data_max = torch.max(data)
    data_avg = torch.mean(data)
    data_min = torch.min(data)
    data_90 = torch.quantile(data.to(torch.float), 0.7)
    
    r_search_num = 200
    
    r_max = math.pi*2/3
    r_min = math.pi/6
    r_gran = (r_max - r_min) / r_search_num 
        
    print(_b_str("\tNum_sums: ") + f"{num_sums}",
            ", " + _y_str("Data range: ") + 
            f"{data_min:.4e} to {data_max:.4e}" +
            ", " + _y_str("avg: ") + f"{data_avg:.4e}")
    print(_y_str("\t\tR search range: ") + 
            f"{r_min:.4e} to {r_max:.4e}, " +
        _y_str("search granularity: ") + f"{r_gran:.4e}")
    
    r_list = -torch.arange(r_min, r_max + r_gran, r_gran, 
                            device=data.device, dtype=data.dtype)
    print("R list: ", r_list)
    return get_coeff_search_space_from_lists(r_list, num_sums)


device = "cpu"
dtype = torch.float32
num_sums = 4

data = torch.randn(1024, device=device, dtype=dtype)
search_matrix, r_list = get_coeff_search_space(data, num_sums)

candidate_matrix = search_matrix @ get_bin_combs(num_sums, device, dtype).T

#Sorting the candidate matrix
sorted_indices = torch.argsort(candidate_matrix, dim=1)
candidate_matrix = candidate_matrix.gather(1, sorted_indices)

print("Candidate matrix shape:", candidate_matrix.shape)
print("Candidate matrix:", candidate_matrix)


# Dummy example (matrix of ints)
matrix = candidate_matrix

for row_idx in range(matrix.shape[0]):
    print_avg_min_max_std_histogram(
        matrix[row_idx], 
        f"histogram_r_{r_list[row_idx].item()}", 
        output_dir="./histograms/"
    )

# Move to CPU if needed
matrix = matrix.cpu()
print(matrix.min(), matrix.max())

# Define bins
num_bins = 16
bin_edges = torch.linspace(0, 1, num_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

dim0, dim1 = matrix.shape

# Create histogram for each row
counts = []
for i in range(dim0):
    c = torch.histc(matrix[i], bins=num_bins, min=0.0, max=1.0)
    counts.append(c)

counts = torch.stack(counts)  # (dim0, num_bins)

# Now, prepare 3D bar plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

dx = 0.05  # bar width (x)
dy = 0.05   # bar width (y)

# Build bar positions
_x = bin_centers.repeat(dim0)
_y = torch.arange(dim0).repeat_interleave(num_bins)
_z = torch.zeros_like(_x)

dz = counts.flatten()

ax.bar3d(
    _x.numpy(), _y.numpy(), _z.numpy(),
    dx, dy, dz.numpy(), shade=True
)

ax.set_xlabel('Value (bin centers)')
ax.set_ylabel('Row index (dim0)')
ax.set_zlabel('Frequency across dim1')

plt.savefig('./histograms/3d_histogram.png')
plt.show()

num_bins = 16
value_bins = torch.linspace(0, 1, num_bins + 1)  # value bins
dim1_bins = torch.linspace(0, matrix.shape[1], 50)  # index bins

dim0 = matrix.shape[0]

