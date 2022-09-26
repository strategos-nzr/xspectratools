# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 15:12:59 2022

@author: Nick
"""

from setuptools import setup,find_packages

setup(name='xspectratools',
      version = '0.0.8',
      description='Read and Process ALS Beamline Data',
      author = "Nick Russo",
      author_email="nzr111@protonmail.com",
      url = '',
      packages = ['xspectratools','xspectratools.diffraction'],
      py_modules=[
      'io',
      'xas',
      'rixs',
      'arpes',
      'uncertainty',
      'sampling',
      'diffraction'

      ],
      package_dir={"xspectratools":"xspectratools",
                    "xspectratools.diffraction":"xspectratools/diffraction"},
      install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'adaptive','scipy'
    ]
      )