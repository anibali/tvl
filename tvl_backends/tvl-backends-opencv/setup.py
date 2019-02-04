from pathlib import Path

from setuptools import setup

version = Path(__file__).absolute().parent.joinpath('VERSION').read_text('utf-8').strip()


setup(
    name='tvl-backends-opencv',
    version=version,
    author='Aiden Nibali',
    license='Apache Software License 2.0',
    packages=['tvl_backends.opencv'],
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'tvl==' + version,
        'numpy',
        'torch',
        'opencv-python',
    ],
)
