from setuptools import setup
from codecs import open
from os import path

root = path.abspath(path.dirname(__file__))
with open(path.join(root, 'README.md'), encoding='utf-8') as readme:
    long_description = readme.read()

setup(
    name='nnkit',
    version='1.2',
    description='NNKit: A dynamic neural network framework.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='neural networks ai AI',
    url='http://github.com/saldavonschwartz/nnkit.git',
    author='Federico Saldarini',
    author_email='fede@0xfede.io',
    license='MIT',
    packages=['nnkit'],
    install_requires=['numpy'],
    zip_safe=False
)

