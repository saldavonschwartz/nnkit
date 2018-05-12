from setuptools import setup

setup(
    name='nnkit',
    version='0.1a',
    description='NNKit: A dynamic neural network framework.',
    url='http://github.com/saldavonschwartz/nnkit',
    author='Federico Saldarini',
    author_email='fede@0xfede.io',
    license='MIT',
    packages=['nnkit'],
    install_requires=[
          'numpy'
    ],
    zip_safe=False
)
