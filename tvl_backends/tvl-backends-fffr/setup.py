from pathlib import Path

from skbuild import setup

version = Path(__file__).absolute().parent.joinpath('VERSION').read_text('utf-8').strip()

setup(
    name='tvl-backends-fffr',
    version=version,
    author='Aiden Nibali, Matthew Oliver',
    license='Apache Software License 2.0',
    packages=['tvl_backends.fffr'],
    package_dir={'': 'src'},
    include_package_data=True,
    py_modules=['pyfffr'],
    install_requires=[
        'tvl==' + version,
        'numpy',
        'torch',
    ],
)
