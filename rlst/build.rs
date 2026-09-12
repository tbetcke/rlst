fn main() {
    #[cfg(feature = "fftw_system")]
    discover_fftw();
}

#[cfg(feature = "fftw_system")]
fn discover_fftw() {
    // fftw-sys links both precisions, but does not discover system search paths.
    for library in ["fftw3", "fftw3f"] {
        if let Err(error) = pkg_config::Config::new().probe(library) {
            // Never substitute host Homebrew libraries when cross-compiling.
            let native_macos = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
                && std::env::var("HOST").ok() == std::env::var("TARGET").ok();
            let prefix = match std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
                Ok("aarch64") => "/opt/homebrew/opt/fftw/lib",
                _ => "/usr/local/opt/fftw/lib",
            };
            let dylib = std::path::Path::new(prefix).join(format!("lib{library}.dylib"));
            if native_macos && dylib.is_file() {
                println!("cargo:rerun-if-changed={}", dylib.display());
                println!("cargo:rustc-link-search=native={prefix}");
                println!("cargo:rustc-link-lib={library}");
            } else {
                panic!(
                    "Could not find system {library}: {error}\n\
                     Install FFTW and pkg-config (Homebrew: brew install fftw pkg-config; \
                     Debian/Ubuntu: apt install libfftw3-dev pkg-config; \
                     Fedora: dnf install fftw-devel pkgconf-pkg-config). \
                     For a custom installation, set PKG_CONFIG_PATH to its lib/pkgconfig directory."
                );
            }
        }
    }
}
