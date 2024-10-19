from setuptools import setup, Extension
import numpy

module = Extension('discrepancies',
                   sources=['discrepancies.c'],
                   include_dirs=[numpy.get_include()])

setup(name='discrepancies',
      version='1.0',
      description='Calculate discrepancies',
      ext_modules=[module])
