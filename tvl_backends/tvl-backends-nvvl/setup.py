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
        copy('ext/tvlnvvl.py', 'src/tvlnvvl.py')


setup(
    name='tvl-backends-nvvl',
    version=version,
    author='Aiden Nibali',
    license='Apache Software License 2.0',
    packages=['tvl_backends.nvvl'],
    package_dir={'': 'src'},
    include_package_data=True,
    py_modules=['tvlnvvl'],
    ext_modules=[
        Extension(
            '_tvlnvvl',
            sources=['ext/tvlnvvl.i', 'ext/VideoInfo.cpp'],
            swig_opts=['-c++'],
            extra_compile_args=['-std=c++11'],
            include_dirs=['ext', 'ext/nvvl', '/usr/local/cuda/include'],
            library_dirs=['ext/nvidia/Lib/linux/stubs/x86_64', '/usr/local/lib'],
            libraries=['avcodec', 'avutil', 'avformat', 'cuda', 'nvcuvid', 'nvvl']
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
