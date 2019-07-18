from pathlib import Path

from skbuild import setup

version = Path(__file__).absolute().parent.joinpath('VERSION').read_text('utf-8').strip()

setup(
    name='tvl-backends-nvdec',
    version=version,
    author='Aiden Nibali',
    license='Apache Software License 2.0',
    packages=['tvl_backends.nvdec'],
    package_dir={'': 'src'},
    include_package_data=True,
    py_modules=['tvlnv'],
    install_requires=[
        'tvl==' + version,
        'numpy',
        'torch',
    ],
)
