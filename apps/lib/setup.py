from setuptools import setup


setup(
    name='dlcjob',
    version='0.3',
    license='Apache-2.0 license',
    author="Zhuangwei Kang",
    author_email='zhuangwei.kang@vanderbilt.edu',
    description="DLCache Job Dataset and DataLoader",
    package_dir={'': 'src'},
    py_modules=["DLCJob"],             # Name of the python package
    python_requires='>=3.6',              # Minimum version requirement of the package
    url='https://github.com/VUZhuangweiKang/DLCache/tree/main/src/dlcjob',
    keywords='DLCache',
    install_requires=[
          'torch',
          'pickle-mixin',
          'grpcio',
          'protobuf'
      ],
)