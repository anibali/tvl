import os
from distutils import log
from pathlib import Path

from skbuild import setup
from skbuild.command.clean import clean

version = Path(__file__).absolute().parent.joinpath('VERSION').read_text('utf-8').strip()


class CustomClean(clean):
    FILES_TO_REMOVE = ['MANIFEST']

    def run(self):
        super().run()
        for filename in self.FILES_TO_REMOVE:
            if os.path.isfile(filename):
                log.info("removing '%s'", filename)
            else:
                log.debug("'%s' does not exist -- can't clean it", filename)
            if not self.dry_run and os.path.exists(filename):
                os.remove(filename)


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
    cmdclass={'clean': CustomClean},
)
