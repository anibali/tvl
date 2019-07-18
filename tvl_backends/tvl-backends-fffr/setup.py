import sys
from pathlib import Path

if 'sdist' in sys.argv:
    # The sdist that comes with skbuild doesn't handle the symlinked VERSION file properly.
    # Related issue: https://github.com/scikit-build/scikit-build/issues/401
    # TODO: Remove this hack once issue scikit-build#401 is resolved.
    from setuptools import setup
else:
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
    install_requires=[
        'tvl==' + version,
        'numpy',
        'torch',
    ],
)
