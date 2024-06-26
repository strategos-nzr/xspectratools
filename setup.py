# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 15:12:59 2022

@author: Nick
"""

from setuptools import setup

setup(name='xspectratools',
      version = '0.0.9',
      description='Read and Process RIXS data',
      author = "Nick Russo",
      author_email="nzr111@protonmail.com",
      url = '',
      packages = ['xspectratools'],
      py_modules=[
      'io',
      'xas',
      'rixs'

      ],
      package_dir={"xspectratools":"xspectratools"},
      install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'scipy'
    ]
      )