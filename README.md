# Rust Linear Solver Toolbox

The Rust Linear Solver toolbox is an in-development project for
dense and sparse linear algebra routines in Rust.

## LICENSE

This work is dual-licensed under the Apache 2.0 and MIT license.
You can choose between one of them if you use this work.

`SPDX-License-Identifier: (Apache-2.0 OR MIT)`

Some optional dependencies of the library have different licenses that
may change the license of compiled library components. 

Note that when linking against FFTW (either by the flag `fftw_system` or
the flag `fftw_source`) the GPL license of FFTW needs to be adhered to.

## System FFTW

Enable `fftw,fftw_system` to use an installed FFTW library. Both double- and
single-precision libraries (`fftw3` and `fftw3f`) are required.

- macOS (Homebrew): `brew install fftw pkg-config`
- Debian/Ubuntu: `sudo apt install libfftw3-dev pkg-config`
- Fedora: `sudo dnf install fftw-devel pkgconf-pkg-config`

The build uses `pkg-config` to discover library paths, with a fallback to
standard native Homebrew FFTW locations on macOS. For custom installations,
set `PKG_CONFIG_PATH` to the directory containing `fftw3.pc` and `fftw3f.pc`.
Cross-compilation requires target libraries and the corresponding `pkg-config`
configuration; host Homebrew paths are not used as a fallback.
This discovery does not affect the `fftw_source` or `fftw_mkl` backends.

## Notes

This library is the result of the merger of two experimental linear algebra projects

- Householder (github.com/UCL-ARC/householder)
- sandbox (github.com/linalg-rs/sandbox)

Both projects are MIT + Apache-2.0 dual licensed. The Rust Linear Solver
toolbox is the successor of both projects.
