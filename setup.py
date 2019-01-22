from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig
from shutil import copy


class build_ext(build_ext_orig):
    def run(self):
        # Build extensions
        super().run()

        # Copy generated Python file into the source tree
        copy('ext/tvlnv.py', 'src/tvlnv.py')


setup(
    name='tvl',
    version='0.1.0a0',
    author='Aiden Nibali',
    license='Apache Software License 2.0',
    packages=find_packages('src'),
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
        'av',
        'numpy',
        'torch',
    ],
)
