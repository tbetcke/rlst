# rlst-suitesparse

`rlst-suitesparse` provides SuiteSparse bindings for RLST sparse matrices. It supports UMFPACK symbolic and numeric factorization of `rlst::sparse::csr_mat::CsrMatrix<f64>` and `rlst::sparse::csr_mat::CsrMatrix<rlst::c64>`, solving real and complex double-precision square systems. Both matrix types implement the `UmfpackSymbolic` trait, so symbolic analysis starts with the same `matrix.analyze()` call.

## Native dependency

A native SuiteSparse installation with UMFPACK is required. The build script searches these installation prefixes automatically:

- `/opt/homebrew/opt/suite-sparse` (Homebrew on Apple Silicon)
- `/usr/local/opt/suite-sparse` (Homebrew on Intel macOS)
- `/usr/local`
- `/usr` (including Debian/Ubuntu multiarch library directories)

Set `SUITESPARSE_PATH` to override the search. Its value must be the SuiteSparse installation prefix containing `include/` and `lib/`:

```bash
export SUITESPARSE_PATH=/custom/prefix
cargo test -p rlst-suitesparse
```

## License

This crate is dual-licensed under MIT and Apache-2.0, matching RLST. Native SuiteSparse components have their own licensing terms.
