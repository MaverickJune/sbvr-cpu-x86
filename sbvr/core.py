import torch
import itertools
import math
import numpy as np
import ctypes
from tqdm import tqdm
# from sbvr.sbvr_cuda import _sbvr_mm_T
# from sbvr.sbvr_cuda import _sbvr_row_deq_mm_T

def _r_str(s):
    return "\033[91m" + str(s) + "\033[0m"
def _g_str(s):
    return "\033[92m" + str(s) + "\033[0m"
def _y_str(s):
    return "\033[93m" + str(s) + "\033[0m"
def _b_str(s):
    return "\033[94m" + str(s) + "\033[0m"

class _sbvr_enc_conf():
    def __init__(self, **kwargs):
        self.num_sums = kwargs.get("num_sums", 4)
        self.r_search_num = kwargs.get("r_search_num", 64)
        self.b_search_num = kwargs.get("b_search_num", 40)
        self.s_search_num = kwargs.get("s_search_num", 40)
        self.error_function = kwargs.get("error_function", "data_diff_mse")
        self.mse_window_size = kwargs.get("mse_window_size", 20)
        self.search_extend_ratio = kwargs.get("search_extend_ratio", 1.2)
        self.coeff_cache = kwargs.get("coeff_cache", None)
        self.cache_warmup_num = kwargs.get("cache_warmup_num", 10)
        self.acceptable_mse = kwargs.get("acceptable_mse", 10**-12)
        self.mse_history = kwargs.get("mse_history", [])
        self.search_batch_size = kwargs.get("search_batch_size", 0)
        self.group_idx = kwargs.get("group_idx", 0)
        self.cache_hits = kwargs.get("cache_hits", 0)
        self.num_coeff_cache_lines = kwargs.get("num_coeff_cache_lines", 0)
        self.extend_ratio = kwargs.get("extend_ratio", 1.2)
        self.input_tensor = kwargs.get("input_tensor", None)
        self.input_vectors = kwargs.get("input_vectors", None)
        
    def _get_conf_str(self):
        conf_str = _g_str("SBVR Encoder Config:") + \
            _y_str("\n\tNumber of Summations: ") + str(self.num_sums) + \
            _y_str("\n\tR search num: ") + str(self.r_search_num) + \
            _y_str("\n\tB search num: ") + str(self.b_search_num) + \
            _y_str("\n\tS search num: ") + str(self.s_search_num) + \
            _y_str("\n\tMSE window size: ") + str(self.mse_window_size) + \
            _y_str("\n\tSearch extend ratio: ") + \
            str(self.search_extend_ratio) + \
            _y_str("\n\tCache warmup num: ") +str(self.cache_warmup_num) + \
            _y_str("\n\tAcceptable MSE: ") + str(self.acceptable_mse) + \
            _y_str("\n\tSearch batch size: ") + str(self.search_batch_size) + \
            _y_str("\n\tExtend ratio: ") + str(self.extend_ratio)
        
        return conf_str
            
    def _get_result_str(self):
        result_str = _y_str("\tCache hits: ") + str(self.cache_hits) + \
            _y_str("\n\tNum coeff cache lines: ") + \
                str(self.num_coeff_cache_lines)
        return result_str
        
class _sbvr_serialized():
    def __init__(self, 
                 num_sums: int,
                 bvr_len: int,
                 compute_dtype: torch.dtype,
                 bvr_dtype: torch.dtype,
                 original_dtype: torch.dtype,
                 original_data_shape: tuple,
                 bvr: torch.Tensor,
                 coeff_idx: torch.Tensor,
                 coeff_cache: torch.Tensor,
                 input_num_sums: int,
                 input_coeff: torch.Tensor):
        # Save base parameters
        self.num_sums = num_sums
        self.bvr_len = bvr_len
        self.compute_dtype = compute_dtype
        self.bvr_dtype = bvr_dtype
        bvr_num_bits = \
            torch.tensor(0, dtype=self.bvr_dtype).element_size() * 8
        if num_sums > 11 and self.compute_dtype == torch.float16:
            raise UserWarning(
                _r_str("Warning: compute_dtype float16 does not have sufficient"
                      " precision for num_sums > 11."))
        if self.bvr_len % bvr_num_bits != 0:
            raise ValueError(
                _r_str("BVR length must be a multiple of ") +
                      f"{bvr_num_bits}")
            
        self.original_dtype = original_dtype
        self.original_data_shape = original_data_shape
        self.padded_data_shape = list(self.original_data_shape)
        self.padded_data_shape[-1] = \
            (self.original_data_shape[-1] + self.bvr_len - 1) // \
            self.bvr_len * self.bvr_len
            
        if bvr.dtype != self.bvr_dtype:
            raise ValueError(
                _r_str(f"The BVR data type does not match - expected type " +
                      f"{self.bvr_dtype} but got {bvr.dtype}"))
        # if bvr.shape[2] != num_sums:
        #     raise ValueError(
        #         _r_str("The number of summations does not match the BVR, " +
        #               f"expected {num_sums} but got " + 
        #               f"{bvr.shape[2]}"))
        # if bvr.shape[0] * bvr_num_bits != self.padded_data_shape[-1]:
        #     raise ValueError(
        #         _r_str("The BVR inner dimension does not match the padded "+
        #               f"data shape, expected {self.padded_data_shape[-1]} " +
        #               f"but got {bvr.shape[0] * bvr_num_bits}"))
        self.bvr = self._serialize_tensor(bvr)
        self.bvr_shape = bvr.shape
        self.bvr_dtype = bvr.dtype
        
        if coeff_cache.shape[0] <= 256:
            if coeff_idx.dtype != torch.uint8:
                raise ValueError(
                    _r_str("The coefficient index data type must be uint8 " +
                            f"but got {coeff_idx.dtype}, " +
                            f"number of cache lines: {coeff_cache.shape[0]}"))
        elif coeff_cache.shape[0] <= 65536 and coeff_idx.dtype != torch.uint16:
            raise ValueError(
                _r_str("The coefficient index data type must be uint16 " +
                        f"but got {coeff_idx.dtype}, " +
                        f"number of cache lines: {coeff_cache.shape[0]}"))
        elif coeff_cache.shape[0] > 65536:
            raise ValueError(
                _r_str("Unsupported number of cache lines, " +
                        f"{coeff_cache.shape[0]}"))
        self.coeff_idx = self._serialize_tensor(coeff_idx)
        self.coeff_idx_shape = coeff_idx.shape
        self.coeff_idx_dtype = coeff_idx.dtype
        
        if coeff_cache.dtype != self.compute_dtype:
            raise ValueError(
                _r_str("The coefficient cache data type does not match - "
                        f"expected type {self.compute_dtype} but got " +
                        f"{coeff_cache.dtype}"))
        self.coeff_cache = self._serialize_tensor(coeff_cache)
        self.coeff_cache_shape = coeff_cache.shape
        self.coeff_cache_dtype = coeff_cache.dtype
        
        self.input_num_sums = input_num_sums
        if input_coeff is not None:
            if input_coeff.dtype != self.compute_dtype:
                raise ValueError(
                    _r_str("The input coefficient data type does not match - "
                            f"expected type {self.compute_dtype} but got " +
                            f"{input_coeff.dtype}"))
            self.input_coeff = self._serialize_tensor(input_coeff)
            self.input_coeff_shape = input_coeff.shape
            self.input_coeff_dtype = input_coeff.dtype
        else:
            self.input_coeff = None
            self.input_coeff_shape = None
            self.input_coeff_dtype = None
        
    def _serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        raw_bytes = tensor.detach().cpu().numpy().tobytes()
        nd_array = np.frombuffer(raw_bytes, dtype=np.int8).copy()
        torch_tensor = torch.from_numpy(nd_array)
        return torch_tensor
    
    def _deserialize_tensor(self, serialized_data: torch.Tensor,
                            shape, dtype) -> torch.Tensor:
        serialized_data = serialized_data.detach().cpu().numpy()
        dtype_map = {
            "torch.uint8": np.uint8,
            "torch.uint16": np.uint16,
            "torch.uint32": np.uint32,
            "torch.int32": np.int32,
            "torch.float16": np.float16,
            "torch.float32": np.float32,
        }
        np_dtype = dtype_map.get(str(dtype), None)
        if np_dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype}")
        array = \
            np.frombuffer(serialized_data, 
                          dtype=np_dtype).reshape(shape).copy()
        return torch.from_numpy(array).to(dtype=dtype).contiguous()
    
    def deserialize_sbvr(self):
        bvr = self._deserialize_tensor(self.bvr, self.bvr_shape, self.bvr_dtype)
        coeff_idx = self._deserialize_tensor(self.coeff_idx, 
                                             self.coeff_idx_shape, 
                                             self.coeff_idx_dtype)
        coeff_cache = self._deserialize_tensor(self.coeff_cache, 
                                               self.coeff_cache_shape, 
                                               self.coeff_cache_dtype)
        if self.input_coeff is not None:
            input_coeff = self._deserialize_tensor(self.input_coeff, 
                                                   self.input_coeff_shape, 
                                                   self.input_coeff_dtype)
        else:
            input_coeff = None
            
        return self.num_sums, self.bvr_len, self.compute_dtype, \
            self.bvr_dtype, self.original_dtype, self.original_data_shape, \
            bvr, coeff_idx, coeff_cache, self.input_num_sums, input_coeff

        
        
class sbvr(torch.nn.Module):
    def __init__(self, 
                 data: torch.Tensor = None, 
                 encoder_config: dict = None,
                 device: torch.device = None,
                 sbvr_serialized: _sbvr_serialized = None,
                 verbose_level: int = 1,
                 trans: bool = False,
                 cpu_kernel_x86 : bool = False,
                 gpu_tensor_gen : bool = False):
        super(sbvr, self).__init__()
        _device = device if device is not None else \
            torch.device("cuda") if torch.cuda.is_available() \
                else torch.device("cpu")
        self.verbose_level = verbose_level
        
        enforce_bvr_len = 256
        enforce_compute_dtype = torch.float16

        # Added support for CPU kernel
        if not cpu_kernel_x86:
            enforce_bvr_dtype = torch.uint32
        else:
            enforce_bvr_dtype = torch.uint8
            if gpu_tensor_gen:
                _device = torch.device("cuda:0")
            else:
                _device = torch.device("cpu")
        
        if data is not None and sbvr_serialized is not None:
            raise ValueError(
                _r_str("Cannot provide both data and serialized SBVR"))
        elif data is None and sbvr_serialized is None:
            raise ValueError(
                _r_str("Must provide either data or serialized SBVR"))
        
        if data is not None:
            if trans:
                data = data.transpose(0, 1).contiguous()
            if not isinstance(data, torch.Tensor):
                raise ValueError(
                    _r_str("Data must be a torch tensor"))
            if encoder_config is None:
                encoder_config = {} 
            enc_conf = _sbvr_enc_conf(**encoder_config)
            self.num_sums = enc_conf.num_sums
            self.bvr_len = enforce_bvr_len \
                if enforce_bvr_len is not None else 256
            self.compute_dtype = enforce_compute_dtype \
                if enforce_compute_dtype is not None else torch.float16
            self.bvr_dtype = enforce_bvr_dtype \
                if enforce_bvr_dtype is not None else torch.uint32
            if self.num_sums > 11 and self.compute_dtype == torch.float16:
                raise UserWarning(
                    _r_str("Warning: compute_dtype float16 does not have "
                           "sufficient precision for num_sums > 11."))
            if self.bvr_len % self._get_bvr_num_bits() != 0:
                raise ValueError(
                    _r_str("BVR length must be a multiple of ") +
                        f"{self._get_bvr_num_bits()}")
                
            self.original_dtype = data.dtype
            self.original_data_shape = data.shape
            
            self.input_num_sums = -1
        
            self.bvr = None
            self.coeff_idx = None
            self.coeff_cache = None
            self.input_coeff = None
            self._encode_to_sbvr(data.to(_device).to(self.compute_dtype), 
                                 enc_conf, cpu_kernel_x86)
        else:
            if not isinstance(sbvr_serialized, _sbvr_serialized):
                raise ValueError(
                    _r_str("Serialized SBVR object is not valid"))
            self.num_sums, self.bvr_len, self.compute_dtype, \
                self.bvr_dtype, self.original_dtype, self.original_data_shape, \
                bvr, coeff_idx, coeff_cache, self.input_num_sums, \
                                input_coeff = sbvr_serialized.deserialize_sbvr()
                
            if enforce_bvr_len is not None and \
                self.bvr_len != enforce_bvr_len:
                raise ValueError(
                    _r_str("bvr length does not match the enforced value, " +
                          f"expected {enforce_bvr_len} but got " +
                          f"{self.bvr_len}"))
            if enforce_compute_dtype is not None and \
                self.compute_dtype != enforce_compute_dtype:
                raise ValueError(
                    _r_str("Compute data type does not match the enforced " +
                          f"value, expected {enforce_compute_dtype} but got " +
                          f"{self.compute_dtype}"))
            if enforce_bvr_dtype is not None and \
                self.bvr_dtype != enforce_bvr_dtype:
                raise ValueError(
                    _r_str("BVR data type does not match the enforced value, " +
                          f"expected {enforce_bvr_dtype} but got " +
                          f"{self.bvr_dtype}"))
                
            self.bvr = torch.nn.Parameter(bvr.to(_device), requires_grad=False)
            self.coeff_idx = torch.nn.Parameter(coeff_idx.to(_device), 
                                                requires_grad=False)
            self.coeff_cache = torch.nn.Parameter(coeff_cache.to(_device),
                                                  requires_grad=False)
            if input_coeff is not None:
                self.input_coeff = torch.nn.Parameter(
                    input_coeff.to(_device), requires_grad=False)
            else:
                self.input_coeff = None
    
    def _get_bvr_num_bits(self):
        if not hasattr(self, 'bvr_num_bits'):
            self.bvr_num_bits = \
                torch.tensor(0, dtype=self.bvr_dtype).element_size() * 8
        return self.bvr_num_bits
    
    def _get_padded_data_shape(self):
        if not hasattr(self, 'padded_data_shape'):
            if self.original_data_shape is not None:
                self.padded_data_shape = list(self.original_data_shape)
                self.padded_data_shape[-1] = \
                    (self.original_data_shape[-1] + self.bvr_len - 1) // \
                    self.bvr_len * self.bvr_len
                self.padded_data_shape = \
                    torch.Size(list(self.padded_data_shape))
            else:
                self.padded_data_shape = None
        return self.padded_data_shape
    
    def _get_padded_input_shape(self, input):
        padded_input_shape = list(input.shape)
        padded_input_shape[-1] = \
            (input.shape[-1] + self.bvr_len - 1) // self.bvr_len * self.bvr_len
        padded_input_shape = torch.Size(list(padded_input_shape))
        return padded_input_shape
        
    def _get_bin_combs(self):
        if not hasattr(self, 'bin_combs'):
            self.bin_combs = torch.tensor(
                list(itertools.product([0, 1], repeat=self.num_sums)),
                dtype=self.compute_dtype, device=self.coeff_cache.device
            )
        return self.bin_combs
    
    def _get_dummy_bias(self):
        if not hasattr(self, 'dummy_bias'):
            self.dummy_bias = torch.zeros([0],
                                          dtype=self.compute_dtype,
                                          device=self.coeff_cache.device)
        return self.dummy_bias
        
    def _check_coeff_cache_full(self, enc_conf):
        if enc_conf.num_coeff_cache_lines >= enc_conf.coeff_cache.shape[0]:
            return True
        return False
        
    def _get_all_points(self, coeff: torch.tensor):
        return self._get_bin_combs() @ coeff
    
    def _get_additional_search_space(self, data, enc_conf, extended=False):
        search_budget = enc_conf.r_search_num * enc_conf.b_search_num * \
            enc_conf.s_search_num * 1.4
        if extended:
            search_budget *= enc_conf.search_extend_ratio**3
        search_num_per_dim = int(search_budget**(1/self.num_sums))
        
        data_max = torch.max(data)
        data_min = torch.min(data)
        
        dim_edges = torch.linspace(data_min, data_max, self.num_sums + 1,
                                    device=data.device, dtype=data.dtype)
        search_space = []
        for i in range (self.num_sums):
            search_range_i = \
                torch.linspace(dim_edges[i] - abs(dim_edges[i])*0.8, 
                               dim_edges[i+1] + abs(dim_edges[i])*0.8, 
                                search_num_per_dim + 1, device=data.device,
                                dtype=data.dtype)
            search_space.append(search_range_i)
        search_space = torch.cartesian_prod(*search_space)

        return search_space
    
    def _get_coeff_search_space_from_lists(self, r_list, b_list, s_list):
        exponents = torch.arange(self.num_sums, device=r_list.device) 
        search_space = r_list.unsqueeze(1) ** exponents.unsqueeze(0)
        all_vals = search_space @ self._get_bin_combs().T
        max = all_vals.max(dim=1)[0]
        min = all_vals.min(dim=1)[0]
        search_space = search_space / (max-min).unsqueeze(1)
        search_space = s_list.view(-1, 1, 1) * search_space.unsqueeze(0)
        search_space = b_list.view(-1, 1, 1, 1) + search_space.unsqueeze(0)
        search_space = search_space.view(-1, self.num_sums)
        
        return search_space
    
    def _get_coeff_search_space(self, data, enc_conf, extended=False):

        data_max = torch.max(data)
        data_avg = torch.mean(data)
        data_min = torch.min(data)
        data_95 = torch.quantile(data.to(torch.float), 0.95)

        r0_min = math.pi/6
        r0_max = 0.94
        r0_gran = (r0_max - r0_min) / (enc_conf.r_search_num / 2) 

        r1_max = math.pi*2/3
        r1_min = 1.06
        r1_gran = (r1_max - r1_min) / (enc_conf.r_search_num / 2) 
        b_max = abs(data_avg) * 2.0 / self.num_sums 
        if b_max < 0.3:
            b_max = 0.3
        b_min = -b_max
        b_gran = (b_max - b_min) / enc_conf.b_search_num 
        s_max = (data_max - data_min) * 1.1 
        s_min = 2 * data_95
        s_gran = (s_max - s_min) / enc_conf.s_search_num 
        
        if extended:
            if self.verbose_level > 1:
                print (_r_str("\tUsing extended search space..."))
            r0_gran /= enc_conf.extend_ratio
            r1_gran /= enc_conf.extend_ratio
            b_gran /= enc_conf.extend_ratio
            s_gran /= enc_conf.extend_ratio
            
        if self.verbose_level > 2:
            print(_b_str("\tNum_sums: ") + f"{self.num_sums}",
                    ", " + _y_str("Data range: ") + 
                    f"{data_min:.4e} to {data_max:.4e}" +
                    ", " + _y_str("avg: ") + f"{data_avg:.4e}")
            print(_y_str("\t\tR0 search range: ") + 
                  f"{r0_min:.4e} to {r0_max:.4e}, " +
                _y_str("search granularity: ") + f"{r0_gran:.4e}")
            print(_y_str("\t\tR1 search range: ") + 
                  f"{r1_min:.4e} to {r1_max:.4e}, " +
                _y_str("search granularity: ") + f"{r1_gran:.4e}")
            print(_y_str("\t\tBias search range: ") + 
                f"{b_min:.4e} to {b_max:.4e}, " +
                _y_str("search granularity: ") + f"{b_gran:.4e}")
            print(_y_str("\t\tScale search range: ") + 
                f"{s_min:.4e} to {s_max:.4e}, " +
                _y_str("search granularity: ") + f"{s_gran:.4e}")
        
        r0_list = -torch.arange(r0_min + r0_gran, r0_max + r0_gran, r0_gran, 
                              device=data.device, dtype=data.dtype)
        r1_list = -torch.arange(r1_min + r1_gran, r1_max + r1_gran, r1_gran, 
                              device=data.device, dtype=data.dtype)
        r_list = torch.cat((r0_list, r1_list))
        if s_gran != 0:
            s_list = torch.arange(s_min + s_gran, s_max + s_gran, s_gran, 
                                device=data.device, dtype=data.dtype)
        else:
            s_list = torch.tensor([s_min], device=data.device, dtype=data.dtype)
        if b_gran != 0:
            b_list = torch.arange(b_min, b_max, b_gran, 
                                  device=data.device, dtype=data.dtype)
        else:
            b_list = torch.tensor([b_min], device=data.device, dtype=data.dtype)
            
        search_space = \
            self._get_coeff_search_space_from_lists(r_list, b_list, s_list)
        org_search_space_len = search_space.shape[0]
        
        if self.num_sums <= 6:
            additional_search_space = \
                self._get_additional_search_space(data, enc_conf, extended)
            search_space = torch.cat((search_space, additional_search_space), 
                                     dim=0)
        
        _, indices = torch.sort(search_space.abs(), dim=1)
        search_space = torch.gather(search_space, dim=1, index=indices)
            
        return search_space, r_list, b_list, s_list, org_search_space_len
    
    def _data_diff_min_mse(self, data, candidate_matrix):
        n_ss_row = candidate_matrix.shape[0]
        n_ss_col = candidate_matrix.shape[1]
        
        data = data.view(1, -1, 1)
        candidate_matrix = candidate_matrix.view(n_ss_row, 1, n_ss_col) 
        
        diff = (data - candidate_matrix)**2

        diff_selected, coeff_comb_indices = diff.min(dim=-1) 
        mse = diff_selected.to(torch.float32).mean(dim=-1)
        
        min_idx = mse.argmin()
        coeff_comb_sel = coeff_comb_indices[min_idx]
        min_mse = mse[min_idx].item()
        
        return min_mse, min_idx, coeff_comb_sel
    
    def _input_inner_min_mse(self, data, candidate_matrix, enc_conf):
        n_ss_row = candidate_matrix.shape[0]
        n_ss_col = candidate_matrix.shape[1]
        
        data_inner_prd = data @ enc_conf.input_vectors.T
        candidate_matrix = candidate_matrix.view(n_ss_row, 1, n_ss_col) 
        
        diff = (data_zeros + candidate_matrix)

        diff_selected, coeff_comb_indices = diff.min(dim=-1) 
        mse = diff_selected.to(torch.float32).mean(dim=-1)
        
        min_idx = mse.argmin()
        coeff_comb_sel = coeff_comb_indices[min_idx]
        min_mse = mse[min_idx].item()
        
        return min_mse, min_idx, coeff_comb_sel
    
    def _get_min_mse_coeff(self, data, search_matrix, enc_conf):
        candidate_matrix = search_matrix @ self._get_bin_combs().T
        
        if enc_conf.error_function == "data_diff_mse":
            min_mse, min_idx, coeff_comb_sel = \
                self._data_diff_min_mse(data, candidate_matrix)

        return min_mse, min_idx, coeff_comb_sel
    
    def _search_coeff_bias_space(self, coeff_search_space, data, cutoff_mse,
                                 enc_conf):
        min_mse = float("inf")
        len_search_space = coeff_search_space.shape[0]
        best_coeff_idx = -1
        best_coeff_sel = -1
        # Loop over the bias values
        for search_start in range(0, len_search_space, 
                                  enc_conf.search_batch_size):
            torch.cuda.empty_cache()
            search_end = \
                min(search_start + enc_conf.search_batch_size, len_search_space)
            coeff_list = coeff_search_space[search_start:search_end]
            # Call a method to get the index and MSE among these coefficients
            mse, min_idx, coeff_comb_sel = \
                self._get_min_mse_coeff(data, coeff_list, enc_conf)
            search_space_idx = search_start + min_idx
            if mse < min_mse:
                min_mse = mse
                best_coeff_idx = search_space_idx
                best_coeff_sel = coeff_comb_sel
                if min_mse < cutoff_mse:
                    break
        return min_mse, best_coeff_idx, best_coeff_sel
    
    def _encode_data(self, data, enc_conf):
        min_mse = float("inf")
        enc_conf.group_idx += 1
        do_warmup = enc_conf.num_coeff_cache_lines < enc_conf.cache_warmup_num
        # Check cached search space
        if not do_warmup:
            # Setup the search space
            coeff_search_space = \
                enc_conf.coeff_cache[:enc_conf.num_coeff_cache_lines]
            # Setup the cutoff MSE 
            window_size = min(len(enc_conf.mse_history), 
                              enc_conf.mse_window_size)
            mse_window = enc_conf.mse_history[-window_size:]
            cutoff_mse = (sum(mse_window) / len(mse_window))*0.99
            if cutoff_mse < enc_conf.acceptable_mse:
                cutoff_mse = enc_conf.acceptable_mse

            # Search the cache for the best coeff and bias
            min_mse, best_coeff_idx, best_coeff_sel = \
                self._search_coeff_bias_space(coeff_search_space, 
                                              data, cutoff_mse, 
                                              enc_conf) 
            if min_mse < cutoff_mse:
                enc_conf.cache_hits += 1
                return best_coeff_idx, best_coeff_sel
            else:
                if self.verbose_level > 1:
                    best_coeff_str = ['%.4f' % elem for elem in 
                              coeff_search_space[best_coeff_idx].tolist()]
                    hitrate = enc_conf.cache_hits / enc_conf.group_idx
                    print (_b_str("\n\tGroup ") + f"{enc_conf.group_idx}: " 
                        + _r_str("Cache Miss ") +
                        f"(Hitrate: {hitrate:.2f}) - " +
                        _y_str("Coeff cache: ") +
                        f"{enc_conf.num_coeff_cache_lines}/" +
                        f"{enc_conf.coeff_cache.shape[0]}" +
                        _y_str("\n\t\tCutoff MSE: ") + f"{cutoff_mse:.4e}" +
                        ", " + _y_str("Best MSE: ") + f"{min_mse:.4e}" +
                        _y_str("\n\t\tCoeff: ") + str(best_coeff_str))
        else:
            if self.verbose_level > 1:
                print(_b_str("\n\tRun ") + f"{enc_conf.group_idx}: " +
                    _r_str("Warming up cache... "))

        if not self._check_coeff_cache_full(enc_conf):
            hitrate = enc_conf.cache_hits / enc_conf.group_idx
            coeff_search_space, r_list, b_list, s_list, org_search_space_len = \
                self._get_coeff_search_space(data, enc_conf, 
                                             hitrate > 0.6 or do_warmup)
            
            # Search the cache for the best coeff and bias  
            new_mse, new_coeff_idx, new_coeff_sel = \
                    self._search_coeff_bias_space(coeff_search_space, data, 
                                                  enc_conf.acceptable_mse,
                                                  enc_conf)
                    
            if new_coeff_idx < org_search_space_len:
                new_b = b_list[new_coeff_idx // (len(s_list) * len(r_list))]
                new_s = s_list[new_coeff_idx // len(r_list) % len(s_list)]
                new_r = r_list[new_coeff_idx % len(r_list)]
            else:
                new_b = -1
                new_s = -1
                new_r = -1
                    
            if self.verbose_level > 1:
                new_coeff_str = ['%.4f' % elem for elem in 
                             coeff_search_space[new_coeff_idx].tolist()]
                print(_g_str("\tNew MSE: ") + f"{new_mse:.4e}" +
                    ", " + _y_str("(r, b, s): ") +
                    f"{new_r:.4e}, {new_b:.4e}, {new_s:.4e}" +
                    _y_str("\n\t\tCoeff: ") + str(new_coeff_str))
            if new_mse >= min_mse:
                # If the new search space is NOT better than the cached one:
                if self.verbose_level > 1:
                    print(_r_str("\t\tNo better coeff found: ") +
                          f"{new_mse:.4e} >= {min_mse:.4e}")
            else:
                # If the new search space is better than the cached one:
                # Cache the results
                coeff_diff = enc_conf.coeff_cache - \
                    coeff_search_space[new_coeff_idx].unsqueeze(0)
                avg_abs_coeff = coeff_search_space[new_coeff_idx].abs().sum(-1)
                mask = coeff_diff.abs().sum(-1) < avg_abs_coeff*0.0001
                if mask.any():
                    # If the coeff is already in the cache, use it
                    nonzero_idx = mask.nonzero(as_tuple=True)[0]
                    best_coeff_idx = nonzero_idx[0]
                else:
                    enc_conf.coeff_cache[enc_conf.num_coeff_cache_lines] =\
                        coeff_search_space[new_coeff_idx]
                    best_coeff_idx = enc_conf.num_coeff_cache_lines
                    enc_conf.num_coeff_cache_lines += 1

                # If caching was successful, update the output
                best_coeff_sel = new_coeff_sel
                enc_conf.mse_history.append(new_mse)
 
        return best_coeff_idx, best_coeff_sel
    
    @torch.inference_mode()
    def _encode_to_sbvr(self, data, enc_conf, cpu_kernel_x86):
        if data.device.type == 'cuda':
            elem_size = torch.tensor(0, dtype=self.compute_dtype).element_size()
            diff_mat_size = 3 * enc_conf.extend_ratio * (2**self.num_sums) \
                                * self.bvr_len * elem_size
            total_mem = torch.cuda.mem_get_info(data.device)[0]
            enc_conf.search_batch_size = int(total_mem * 0.8 / diff_mat_size)
        else:
            enc_conf.search_batch_size = 1024

        # Pad the data to the nearest multiple of bvr_len
        if self.original_data_shape != self._get_padded_data_shape():
            data_padded = torch.zeros(self._get_padded_data_shape(), 
                                      dtype=data.dtype, device=data.device)
            slices = tuple(slice(0, s) for s in data.shape)
            data_padded[slices] = data
        else:
            data_padded = data
        data_padded = data_padded.view(-1, 
                        self._get_padded_data_shape()[-1] // self.bvr_len,
                        self.bvr_len)
            
        if enc_conf.input_tensor is not None:
            if enc_conf.input_tensor.shape[-1] != self.data.shape[-1]:
                raise ValueError(
                    _r_str("Input tensor shape does not match data shape, " +
                          f"expected {self.data.shape[-1]} but got " +
                          f"{enc_conf.input_tensor.shape[-1]}"))
            input_padded = torch.zeros(self._get_padded_input_shape(input),
                                        dtype=data.dtype, device=data.device)
            slices = tuple(slice(0, s) for s in enc_conf.input_tensor.shape)
            input_padded[slices] = enc_conf.input_tensor.to(data.device)
            enc_conf.input_tensor = input_padded.view(-1,
                        self._get_padded_input_shape()[-1] // self.bvr_len,
                        self.bvr_len).permute(1, 0, 2).contiguous()
        
        data_num = data_padded.numel()
        num_bvr = data_padded.numel() // self.bvr_len
        self.coeff_idx = torch.empty((num_bvr), dtype=torch.uint16, 
                                     device=data.device)
        self.coeff_cache = torch.zeros((2**16, self.num_sums), 
                        dtype=self.compute_dtype, device=data.device)
        out_coeff_sel = torch.empty((data_num), dtype=torch.int32,
                                     device=data.device)
        enc_conf.coeff_cache = self.coeff_cache
        
        if self.verbose_level > 0:
            print(enc_conf._get_conf_str())
        
        if self.verbose_level > -1:
            group_iter = tqdm(range(num_bvr), ncols=80, 
                      desc=_b_str("Encoding SBVR groups"), unit="g")
        else:
            group_iter = range(num_bvr)
        
        for i in group_iter:
            torch.cuda.empty_cache()
            group_start = i * self.bvr_len
            group_end = \
            min(group_start + self.bvr_len, data_num)
            group_data = data_padded.flatten()[group_start:group_end]
            if enc_conf.input_tensor is not None:
                vector_idx = i % enc_conf.input_tensor.shape[0]
                enc_conf.input_vectors = enc_conf.input_tensor[vector_idx]
            g_coeff_idx, coeff_sel = self._encode_data(group_data, enc_conf)
            self.coeff_idx[i] = g_coeff_idx
            out_coeff_sel[group_start:group_end] = coeff_sel
    
        bvr_raw = self._change_coeff_sel_to_bvr(out_coeff_sel)

        bvr = bvr_raw.view(
                self.num_sums,
                -1,
                self._get_padded_data_shape()[-1] // self._get_bvr_num_bits()
            )


        # bvr = bvr.view(self.num_sums, -1, self._get_padded_data_shape()[-1] // \
        #                                         self._get_bvr_num_bits()) # (num_sums, num_bvr, bvr_len // num_bits)
        # bvr = bvr.permute(2, 1, 0).contiguous()
        # print(f"[DEBUG] _get_padded_data_shape()={self._get_padded_data_shape()}")
        # print(f"[DEBUG] _get_bvr_num_bits()={self._get_bvr_num_bits()}")
        # print(f"[DEBUG] bvr.shape={bvr.shape}, bvr_len={self.bvr_len}, " +
        #       f"bvr_num_bits={self._get_bvr_num_bits()}")
        # self.bvr = torch.nn.Parameter(bvr, requires_grad=False)
        
        # self.coeff_cache = \
        #     self.coeff_cache[:enc_conf.num_coeff_cache_lines].contiguous()
        # if enc_conf.num_coeff_cache_lines <= 256:
        #     self.coeff_idx = self.coeff_idx.to(torch.uint8)
        # self.coeff_idx = \
        #     self.coeff_idx.view(-1, self.bvr.shape[0] // \
        #                         (self.bvr_len // self._get_bvr_num_bits()))
        # self.coeff_idx = self.coeff_idx.transpose(0, 1).contiguous()
            
        

        if cpu_kernel_x86 and bvr.shape[1] > 1: 
            # ---------- ① BVR 16-lane pack --------------------------
            bits_per_bvr = self._get_bvr_num_bits() # 8          
            K_total      = self._get_padded_data_shape()[-1] # K_total
            N_LANE       = 32
            # K_PER_BVR     = self.bvr_len // bits_per_bvr # 256 // 8 = 32


            bvr_pk = (bvr
                    .view(self.num_sums, -1, N_LANE, K_total // self._get_bvr_num_bits())   # (num_sums, num_bvr, N_LANE, bvr_len // num_bits)
                    .permute(1, 3, 0, 2).contiguous())  # (num_sums, bvr_len // num_bits, num_sums, N_LANE)
                    # .view(self.num_sums, -1, K_total // self.bvr_len, K_PER_BVR)   # (num_sums, N, K_total // bvr_len, K_PER_BVR)
                    # .permute(1, 2, 0, 3)  # (N, K_total // bvr_len, num_sums, K_PER_BVR).contiguous()

            self.bvr = torch.nn.Parameter(bvr_pk, requires_grad=False)

            self.coeff_cache = self.coeff_cache[:enc_conf.num_coeff_cache_lines].contiguous()
            if enc_conf.num_coeff_cache_lines <= 256:
                self.coeff_idx = self.coeff_idx.to(torch.uint8)

            coeff_idx = self.coeff_idx.view((-1, K_total // self.bvr_len))

            # if bvr_pk.shape[0] == 1:
            #     coeff_idx = coeff_idx.transpose(0, 1).contiguous() ##########

            self.coeff_idx = torch.nn.Parameter(coeff_idx,
                                                requires_grad=False)
            
        else:                              # <-- 원본 경로 ❷
            bvr = bvr.permute(2, 1, 0).contiguous()

            print(f"[DEBUG] _get_padded_data_shape()={self._get_padded_data_shape()}")
            print(f"[DEBUG] _get_bvr_num_bits()={self._get_bvr_num_bits()}")
            print(f"[DEBUG] bvr.shape={bvr.shape}, bvr_len={self.bvr_len}, " +
                f"bvr_num_bits={self._get_bvr_num_bits()}")

            self.bvr = torch.nn.Parameter(bvr, requires_grad=False)

            self.coeff_cache = \
                self.coeff_cache[:enc_conf.num_coeff_cache_lines].contiguous()
            if enc_conf.num_coeff_cache_lines <= 256:
                self.coeff_idx = self.coeff_idx.to(torch.uint8)

            self.coeff_idx = self.coeff_idx.view(
                -1,
                self.bvr.shape[0] // (self.bvr_len // self._get_bvr_num_bits())
            ).transpose(0, 1).contiguous()


        self.coeff_idx = torch.nn.Parameter(self.coeff_idx, requires_grad=False)
        self.coeff_cache = torch.nn.Parameter(self.coeff_cache,
                                              requires_grad=False)



        
        if self.verbose_level > 0:
            print(_b_str("Encoding complete."))
            print(enc_conf._get_result_str())
            print(self.get_sbvr_info())            
            
    def _dec2bin(self, x, bits):
        # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

    def _bin2dec(self, b, bits):
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
        return torch.sum(mask * b, -1)
            
    # def _change_coeff_sel_to_bvr(self, coeff_sel):
    #     coeff_sel_len = self._get_padded_data_shape().numel()
    #     num_bits = self._get_bvr_num_bits()
    #     bvr = torch.zeros((self.num_sums, (coeff_sel_len // num_bits)),
    #             dtype=self.bvr_dtype, device=self.coeff_cache.device)
    #     powers = 2 ** torch.arange(num_bits, dtype=torch.int64, 
    #                                device=self.coeff_cache.device)
    #     iter_size = 65536
    #     for i in range(0, coeff_sel_len, iter_size):
    #         max_i = min(i + iter_size, coeff_sel_len)
    #         coeff_sel_i = coeff_sel[i:max_i]
    #         bin_vec = self._dec2bin(coeff_sel_i, self.num_sums).to(torch.int64)
    #         bin_vec = \
    #             bin_vec.transpose(0, 1).reshape(self.num_sums, -1, num_bits)
    #         bvr_i = torch.sum(bin_vec * powers.unsqueeze(0), dim=2)
    #         bvr[:, i//32:max_i//32] = bvr_i
        
    #     return bvr
    def _change_coeff_sel_to_bvr(self, coeff_sel):
        coeff_sel_len = self._get_padded_data_shape().numel()
        num_bits = self._get_bvr_num_bits()
        bvr = torch.zeros(
            (self.num_sums, (coeff_sel_len // num_bits)),
            dtype=self.bvr_dtype,
            device=self.coeff_cache.device
        )

        
        powers = 2 ** torch.arange(
            num_bits,
            dtype=torch.int64,
            device=self.coeff_cache.device
        )
        
        iter_size = 65536
        for i in range(0, coeff_sel_len, iter_size):
            max_i = min(i + iter_size, coeff_sel_len)
            coeff_sel_i = coeff_sel[i:max_i]
            
            bin_vec = self._dec2bin(coeff_sel_i, self.num_sums).to(torch.int64)
            
            bin_vec = bin_vec.transpose(0, 1)
            
            bin_vec = bin_vec.reshape(self.num_sums, -1, num_bits)
            
            bvr_i = torch.sum(bin_vec * powers.unsqueeze(0), dim=2)
            
            start = i // num_bits
            end = max_i // num_bits
            
            bvr[:, start:end] = bvr_i

        return bvr

     
    def _change_bvr_to_coeff_sel(self):
        coeff_sel_len = self._get_padded_data_shape().numel()
        bvr = self.bvr.permute(2, 1, 0).contiguous().view(self.num_sums, -1)
        bvr = bvr.view(self.num_sums, -1)
        num_bits = self._get_bvr_num_bits()
        powers = 2 ** torch.arange(num_bits, 
                                   dtype=torch.int64, 
                                   device=self.coeff_cache.device)
        coeff_sel = torch.empty((bvr.shape[1] * num_bits),
                               dtype=torch.int32, 
                               device=self.coeff_cache.device)
        iter_size = 2048
        for i in range(0, bvr.shape[1], iter_size):
            max_i = min(i + iter_size, bvr.shape[1])
            bvr_i = bvr[:, i:max_i].to(torch.int64)
            bin_vec = ((bvr_i.unsqueeze(-1) & powers) != 0).to(torch.int32)
            bin_vec = bin_vec.view(self.num_sums, -1)
            max_coeff_i = min(max_i*num_bits, coeff_sel.shape[0])
            bin_vec_trunc = bin_vec[:, :max_coeff_i].transpose(0, 1)
            coeff_sel_i = self._bin2dec(bin_vec_trunc, self.num_sums)
            coeff_sel[i*num_bits:max_coeff_i] = coeff_sel_i.view(-1)

        return coeff_sel[:coeff_sel_len]
    
    def _serialize(self):
        return _sbvr_serialized(
            self.num_sums,
            self.bvr_len,
            self.compute_dtype,
            self.bvr_dtype,
            self.original_dtype,
            self.original_data_shape,
            self.bvr,
            self.coeff_idx,
            self.coeff_cache,
            self.input_num_sums,
            self.input_coeff
        )
    
    @torch.inference_mode()
    def save(self, filename):   
        if self.verbose_level > 0:
            print(_b_str("Saving SBVR object to: ") + filename)
            print(self.get_sbvr_info()) 
        serialized_sbvr = self._serialize()
        torch.save(serialized_sbvr, filename)
            
    @torch.inference_mode()
    def decode(self):
        decoded_tensor = torch.empty(self._get_padded_data_shape(),
                                      dtype=self.original_dtype,
                                      device=self.coeff_cache.device)
        num_bvr = self.coeff_idx.numel()
        coeff_sel = self._change_bvr_to_coeff_sel()
        coeff_idx = self.coeff_idx.transpose(0, 1).contiguous().flatten()
        for i in range(num_bvr):
            group_start = i * self.bvr_len
            group_end = \
                min(group_start + self.bvr_len, decoded_tensor.numel())
            group_coeff = self.coeff_cache[coeff_idx[i].item()]
            group_coeff_sel = coeff_sel[group_start:group_end]
            group_all_points = self._get_all_points(group_coeff)
            group_data = group_all_points[group_coeff_sel]
            decoded_tensor.flatten()[group_start:group_end] = group_data
            
        # Truncate the tensor to the original shape
        if self.original_data_shape != self._get_padded_data_shape():
            slices = tuple(slice(0, s) for s in self.original_data_shape)
            decoded_tensor = decoded_tensor[slices]
        
        return decoded_tensor
    
    def get_sbvr_info(self):
        info_str = _g_str("SBVR Info:") + \
        _y_str("\n\tNumber of Summations: ") + str(self.num_sums) + \
        _y_str("\n\tBVR Length: ") + str(self.bvr_len) + \
        _y_str("\n\tCompute Data Type: ") + str(self.compute_dtype) + \
        _y_str("\n\tBVR Data Type: ") + str(self.bvr_dtype) + \
        _y_str("\n\tOriginal Data Type: ") + str(self.original_dtype) + \
        _y_str("\n\tOriginal Data Shape: ") + str(self.original_data_shape) + \
        _y_str("\n\tNumber of Coefficient Cache Lines: ") + \
        _y_str("\n\tBVR Tensor Shape: ") + str(self.bvr.shape) + \
        _y_str("\n\tCoefficient Index Shape: ") + str(self.coeff_idx.shape) + \
        _y_str("\n\tCoefficient Cache Shape: ") + \
            str(self.coeff_cache.shape) + \
        _y_str("\n\tInput Number of Summations: ") + \
            str(self.input_num_sums) + \
        _y_str("\n\tInput Coefficient Shape: ") + \
            str(self.input_coeff.shape if self.input_coeff is not None 
                else "Input Coefficient not set") 
        return info_str
    
    @torch.inference_mode()
    def profile_input(self, input, encoder_config = None):
        if input.shape[-1] != self.original_data_shape[-1]:
            raise ValueError(
                _r_str("Inner dimension shape of the input does not match " + 
                       "the original data shape, expected " +
                       f"{self.original_data_shape[-1]} but " +
                       f"got {input.size(-1)}"))
        
        if encoder_config is None:
            encoder_config = {"num_sums": 8} 
        enc_conf = _sbvr_enc_conf(**encoder_config)
        self.input_num_sums = enc_conf.num_sums
        
        # Pad the input to the nearest multiple of bvr_len
        if input.shape != self._get_padded_input_shape(input):
            input_padded = torch.zeros(self._get_padded_input_shape(input), 
                                      dtype=input.dtype, device=input.device)
            slices = tuple(slice(0, s) for s in input.shape)
            input_padded[slices] = input
        else:
            input_padded = input.copy()
            
        inner_dim = input_padded.shape[-1]
        num_bvr = inner_dim // self.bvr_len
        input_padded = input_padded.view(-1, num_bvr, self.bvr_len)
        input_padded = input_padded.permute(1, 0, 2).contiguous()
        input_padded = input_padded.view(num_bvr, -1).to(self.bvr.device)
        enc_conf.cache_warmup_num = num_bvr
        input_coeff = torch.empty((num_bvr, self.input_num_sums), 
                                  dtype=self.compute_dtype,
                                  device=self.bvr.device)
        enc_conf.coeff_cache = input_coeff
            
        if input.device.type == 'cuda':
            elem_size = input_coeff.element_size()
            diff_mat_size = 3 * enc_conf.extend_ratio * input_padded.shape[1] \
                                * (2**self.input_num_sums) * elem_size
            total_mem = torch.cuda.mem_get_info(input.device)[0]
            enc_conf.search_batch_size = int(total_mem * 0.8 / diff_mat_size)
        else:
            enc_conf.search_batch_size = 1024

        for i in range(num_bvr):
            torch.cuda.empty_cache()
            group_data = input_padded[i]
            coeff_idx, coeff_sel = self._encode_data(group_data, enc_conf)
            
        self.input_coeff = torch.nn.Parameter(input_coeff, requires_grad=False)
    
    def _online_tranfrom(self, input):
        return None
    
    def online_mm_T(self, rhs, bias=None):
        if bias is None:
            bias = self._get_dummy_bias()
        return None
    
def mm_T(lhs, rhs, bias):
    lhs_bvr = lhs.bvr
    lhs_coeff_idx = lhs.coeff_idx
    lhs_coeff_cache = lhs.coeff_cache
    rhs_bvr = rhs.bvr
    rhs_coeff_idx = rhs.coeff_idx
    rhs_coeff_cache = rhs.coeff_cache
    if bias is None:
        bias = lhs._get_dummy_bias()
    return _sbvr_mm_T(lhs_bvr, lhs_coeff_idx, lhs_coeff_cache,
                rhs_bvr, rhs_coeff_idx, rhs_coeff_cache, bias)
    
def load(filename, device=None, verbose_level=1, cpu_kernel_x86=False) -> sbvr:
    serialized_sbvr = torch.load(filename)
    sbvr_obj = sbvr(sbvr_serialized=serialized_sbvr, 
                    verbose_level=verbose_level, device=device, 
                    cpu_kernel_x86=cpu_kernel_x86)
    sbvr_obj.verbose_level = verbose_level
    if verbose_level > 0:
        print(_b_str("Loaded SBVR object from: ") + filename)
        print(sbvr_obj.get_sbvr_info())
    return sbvr_obj
