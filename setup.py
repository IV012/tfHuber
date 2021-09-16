from setuptools import setup, Extension
import numpy

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

tfHuber_module = Extension('tfHuber',
                           include_dirs = ['eigen-3.4.0', numpy.get_include()],
                           sources = ['src/module.cpp']
                           )

setup(
    name='tfHuber',
    version='0.1.1',
    description='Python Package for tuning-free Huber Regression',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IV012/tfHuber",
    author= 'Yifan Dai & Qiang Sun',
    author_email = 'yifandai@yeah.net, qsun.ustc@gmail.com',
    license='GPL (ver 3)',
    ext_modules=[tfHuber_module],
    python_requires='>=3.5'
)