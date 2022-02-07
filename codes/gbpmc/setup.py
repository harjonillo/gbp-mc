#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext = [Extension( "intersectFast", sources=["intersectFast.pyx"],
include_dirs = [np.get_include()] )]

setup(
   name = "testing",
   cmdclass={'build_ext' : build_ext},
   include_path=['/usr/local/lib/python3.7/site-packages/numpy/core/include'],
   ext_modules=ext
   )
