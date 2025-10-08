from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fused_attention",
    ext_modules=[
        CUDAExtension("fused_attention", ["fused_attention.cu"]),
    ],
    cmdclass={"build_ext": BuildExtension},
)
