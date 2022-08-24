from distutils.core import setup, Extension
from glob import glob
from pathlib import Path
from sys import platform

lzham_alpha8_dir = Path("lzham/lzham_alpha8/")

ext_sources = [
    'lzham/lzham_decompress.cpp',
    *glob(str(lzham_alpha8_dir / 'lzhamdecomp/*.cpp')),
    lzham_alpha8_dir / 'lzhamcomp/lzham_lzbase.cpp',
    lzham_alpha8_dir / 'lzhamcomp/lzham_lzcomp_internal.cpp',
    lzham_alpha8_dir / 'lzhamcomp/lzham_lzcomp_state.cpp',
    lzham_alpha8_dir / 'lzhamcomp/lzham_lzcomp.cpp',
    lzham_alpha8_dir / 'lzhamcomp/lzham_match_accel.cpp',
    "lzham/lzham_codec/lzhamcomp/lzham_pthreads_threading.cpp" if platform != "win32" else "lzham/lzham_codec/lzhamcomp/lzham_win32_threading.cpp",
    lzham_alpha8_dir / 'lzhamlib/lzham_lib.cpp',
]

extra_compile_args = ["-Wno-unused-value"] if platform != "win32" else ["/Wno-unused-value"]

setup(
    name="pycnkdec",
    version="0.0.1",
    description="Python Forgelight CNK decompressor",
    ext_modules=[
        Extension('cnkdec', list(map(str, ext_sources)), list(map(str, [lzham_alpha8_dir / "include", lzham_alpha8_dir / "lzhamdecomp", lzham_alpha8_dir / "lzhamcomp"])), extra_compile_args=extra_compile_args)
    ],
    ext_package="cnkdec",
    package_dir={'': 'lzham'},
    packages=['', 'cnkdec']
)