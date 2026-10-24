# Rust Linear Solver Toolbox

[![run-tests](https://github.com/tbetcke/rlst/actions/workflows/run-tests.yml/badge.svg)](https://github.com/tbetcke/rlst/actions/workflows/run-tests.yml)

The Rust Linear Solver toolbox is an in-development project for
dense and sparse linear algebra routines in Rust.

## LICENSE

This work is dual-licensed under the Apache 2.0 and MIT license.
You can choose between one of them if you use this work.

`SPDX-License-Identifier: (Apache-2.0 OR MIT)`

Some optional dependencies of the library have different licenses that
may change the license of compiled library components. 

> **Warning: FFTW is licensed under the GNU General Public License (version 2 or later).**
> RLST's optional `fftw` feature links FFTW and the GPL-licensed `fftw-sys` bindings. RLST
> itself is licensed MIT / Apache-2.0, but a program built with this feature links
> GPL-licensed code, and distributing such a
> program is subject to the terms of the GPL unless you hold a non-free FFTW licence from
> MIT's Technology Licensing Office. Cargo enables a feature for the whole build when any crate in the dependency graph requests it,
> so check your graph (for example with `cargo tree -e features`). The feature is off by
> default. This note is not legal advice; check the licences that apply to your use.

## System FFTW

Enable `fftw_system` (which implies `fftw`) to use an installed FFTW library, version
3.3.4 or later; 3.3.10 or later is recommended. Both double- and single-precision
libraries (`fftw3` and `fftw3f`) are required.

- macOS (Homebrew): `brew install fftw pkg-config`
- Debian/Ubuntu: `sudo apt install libfftw3-dev pkg-config`
- Fedora: `sudo dnf install fftw-devel pkgconf-pkg-config`

The build uses `pkg-config` to discover library paths, with a fallback to
standard native Homebrew FFTW locations on macOS. For custom installations,
set `PKG_CONFIG_PATH` to the directory containing `fftw3.pc` and `fftw3f.pc`.
Cross-compilation requires target libraries and the corresponding `pkg-config`
configuration; host Homebrew paths are not used as a fallback.

Enable `fftw_threads` (which implies `fftw_system`) for threaded FFTW plans. It
also links FFTW's threads libraries, `fftw3_threads` and `fftw3f_threads`, which
must be installed beside `fftw3` and `fftw3f` (the Homebrew and Debian/Ubuntu
packages above include them; a source build needs `--enable-threads`), and it
needs FFTW 3.3.6 or later. FFTW's OpenMP libraries are not used.

The guide `rlst::doc::optional_integrations` describes the FFTW interface, its versions and
its rules on threads and determinism; the module documentation of `rlst::fftw` lists its
parts.

## Notes

This library is the result of the merger of two experimental linear algebra projects

- Householder (github.com/UCL-ARC/householder)
- sandbox (github.com/linalg-rs/sandbox)

Both projects are MIT + Apache-2.0 dual licensed. The Rust Linear Solver
toolbox is the successor of both projects.
