from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='warp_bitnet_cuda',
    version='2.0.0',
    ext_modules=[
        CUDAExtension(
            name='warp_bitnet_cuda',
            sources=[
                'cuda/bitnet_lite.cu',
            ],
            extra_compile_args={
                'cxx': ['/O2'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-arch=sm_89',
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    description='Warp BitNet Lite - Open Source 1.58-bit inference',
)
