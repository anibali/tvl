from pathlib import Path

from setuptools import setup, find_packages

version = Path(__file__).absolute().parent.joinpath('VERSION').read_text('utf-8').strip()


setup(
    name='tvl',
    version=version,
    author='Aiden Nibali',
    license='Apache Software License 2.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'numpy',
        'torch',
    ],
    extras_require={
        'FffrBackend': ['tvl-backends-fffr==' + version],
        'NvdecBackend': ['tvl-backends-nvdec==' + version],
        'PyAvBackend': ['tvl-backends-pyav==' + version],
        'OpenCvBackend': ['tvl-backends-opencv==' + version],
    },
)
