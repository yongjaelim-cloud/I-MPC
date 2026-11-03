#!/usr/bin/env python3
from __future__ import print_function

import os
import shutil
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

HERE = Path(__file__).resolve().parent


def _pybind_include():
    try:
        import pybind11
        return pybind11.get_include()
    except Exception:
        candidates = [
            Path('/usr/include/pybind11'),
            Path('/usr/local/include/pybind11'),
        ]
        try:
            import site
            for root in site.getsitepackages():
                candidates.append(Path(root) / 'torch' / 'include' / 'pybind11')
        except Exception:
            pass
        for c in candidates:
            if c.is_dir():
                return str(c.parent)
        raise RuntimeError('pybind11 headers not found; install pybind11 for the active python3')


def _find_nvcc():
    env = os.environ.get('NVCC')
    if env and Path(env).is_file():
        return env
    found = shutil.which('nvcc')
    if found:
        return found
    for root in filter(None, [os.environ.get('CUDA_HOME'), os.environ.get('CUDA_PATH'), '/usr/local/cuda']):
        p = Path(root) / 'bin' / 'nvcc'
        if p.is_file():
            return str(p)
    return None


def _cuda_home(nvcc):
    if not nvcc:
        return None
    return Path(nvcc).resolve().parent.parent


def _cuda_libdir(home):
    if home is None:
        return None
    candidates = [home / 'lib64', home / 'lib', home / 'targets' / 'x86_64-linux' / 'lib']
    for p in candidates:
        if p.is_dir():
            return str(p)
    return str(home / 'lib64')


NVCC = _find_nvcc()
REQUIRE_CUDA = os.environ.get('REQUIRE_CUDA', '1').strip().lower() not in ('0', 'false', 'no', 'off')
if REQUIRE_CUDA and not NVCC:
    raise RuntimeError('nvcc not found. Native CUDA build requested; set CUDA_HOME/NVCC or REQUIRE_CUDA=0 for CPU fallback build.')
CUDA_HOME = _cuda_home(NVCC)
CUDA_LIB = _cuda_libdir(CUDA_HOME)


class NativeCudaBuildExt(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            if NVCC:
                Path(self.build_temp).mkdir(parents=True, exist_ok=True)
                obj = str(Path(self.build_temp) / 'collision_lin_cuda.o')
                cmd = [
                    NVCC, '-O3', '-std=c++14', '-c',
                    str(HERE / 'collision_lin_cuda.cu'), '-o', obj,
                    '-Xcompiler', '-fPIC',
                    '-I', str(HERE),
                ]
                # Keep IEEE exp/log behavior; intentionally no --use_fast_math.
                arch = os.environ.get('CUDA_ARCH')
                if arch:
                    cmd += ['-arch=' + arch]
                print('[build] nvcc:', ' '.join(cmd))
                subprocess.check_call(cmd)
                ext.extra_objects = list(ext.extra_objects or []) + [obj]
            else:
                ext.sources = list(ext.sources) + [str(HERE / 'collision_lin_cuda_stub.cpp')]
        super().build_extensions()


sources = [str(HERE / 'collision_lin_cpp.cpp')]
libraries = []
library_dirs = []
runtime_library_dirs = []
if NVCC:
    libraries.append('cudart')
    library_dirs.append(CUDA_LIB)
    if os.name == 'posix':
        runtime_library_dirs.append(CUDA_LIB)

ext = Extension(
    'collision_lin_cpp',
    sources=sources,
    include_dirs=[_pybind_include(), str(HERE)],
    language='c++',
    extra_compile_args=['-O3', '-DNDEBUG', '-std=c++14', '-fopenmp'],
    extra_link_args=['-fopenmp'],
    libraries=libraries,
    library_dirs=library_dirs,
    runtime_library_dirs=runtime_library_dirs,
)

setup(
    name='collision_lin_cpp',
    version='28.0',
    ext_modules=[ext],
    cmdclass={'build_ext': NativeCudaBuildExt},
)
