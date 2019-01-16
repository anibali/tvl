from setuptools import setup, find_packages, Extension


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
            sources=['ext/NvCodec/NvDecoder/NvDecoder.cpp', 'ext/tvlnv.i',
                     'ext/TvlnvFrameReader.cpp', 'ext/MemManager.cpp'],
            swig_opts=['-c++'],
            extra_compile_args=['-std=c++11'],
            include_dirs=['/usr/local/cuda/include', 'ext/NvCodec'],
            library_dirs=['ext/NvCodec/Lib/linux/stubs/x86_64'],
            libraries=['avcodec', 'avutil', 'avformat', 'cuda', 'nvcuvid']
        )
    ],
)
