from .core import (sbvr, _sbvr_serialized, load, mm_T)
from sbvr.sbvr_cpu_x86 import _sbvr_init_pool, _sbvr_finalize_pool, _sbvr_x86_test, _sbvr_cpu_mm_T
import torch, atexit

torch.serialization.add_safe_globals([_sbvr_serialized])
# _sbvr_cuda_init()

_sbvr_init_pool(16)
atexit.register(_sbvr_finalize_pool)

