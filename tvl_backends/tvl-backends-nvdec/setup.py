from pathlib import Path
from shutil import copy

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig

version = Path(__file__).absolute().parent.joinpath('VERSION').read_text('utf-8').strip()


class build_ext(build_ext_orig):
    def run(self):
        # Build extensions
        super().run()

        # Copy generated Python file into the source tree
        copy('ext/tvlnv.py', 'src/tvlnv.py')


setup(
    name='tvl-backends-nvdec',
    version=version,
    author='Aiden Nibali',
    license='Apache Software License 2.0',
    packages=['tvl_backends.nvdec'],
    package_dir={'': 'src'},
    include_package_data=True,
    py_modules=['tvlnv'],
    ext_modules=[
        Extension(
            '_tvlnv',
            sources=['ext/tvlnv.i', 'ext/nvidia/NvDecoder/NvDecoder.cpp',
                     'ext/nvidia/MemManager.cpp', 'ext/TvlnvFrameReader.cpp'],
            swig_opts=['-c++'],
            extra_compile_args=['-std=c++11'],
            include_dirs=['/usr/local/cuda/include', 'ext/nvidia', 'ext'],
            library_dirs=['ext/nvidia/Lib/linux/stubs/x86_64'],
            libraries=['avcodec', 'avutil', 'avformat', 'cuda', 'nvcuvid']
        )
    ],
    cmdclass={
        'build_ext': build_ext,
    },
    install_requires=[
        'tvl==' + version,
        'numpy',
        'torch',
    ],
)
