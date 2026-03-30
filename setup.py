import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# Read Python package dependencies
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Paths
this_dir = os.path.dirname(os.path.abspath(__file__))
# cutlass_dir = os.path.join(this_dir, 'cutlass', 'include')
sbvr_include_dir = os.path.join(this_dir, 'sbvr', 'include')

setup(
    name='sbvr',
    packages=['sbvr', 'sbvr.kernels'],
    ext_modules=[
        CppExtension(
            name='sbvr.sbvr_cpu_x86',
            sources=[
                'sbvr/kernels/sbvr_kernel_x86.cpp',
            ],
            include_dirs=[
                sbvr_include_dir,
            ],
            extra_compile_args={
                'cxx': [
                    '-O3',
                    '-march=native',
                    '-ffast-math',
                    '-mavx2',
                    '-mfma',
                    '-mf16c',
                    '-mavx512f',
                    '-mavx512bw',
                    '-mavx512vl',
                    '-mavx512bitalg',
                    '-mavx512vpopcntdq',
                ],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=requirements,
)