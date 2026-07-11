import os
from setuptools import setup, Extension
import numpy as np

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

ext = ".pyx" if USE_CYTHON else ".c"

extensions = [
    Extension(
        "perfect_cmaps._optimization",
        [os.path.join("perfect_cmaps", "_optimization" + ext)],
        include_dirs=[np.get_include()],
    )
]

if USE_CYTHON:
    extensions = cythonize(extensions, compiler_directives={"language_level": "3"})

setup(
    ext_modules=extensions,
)