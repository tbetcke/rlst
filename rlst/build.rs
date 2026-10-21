fn main() {
    #[cfg(feature = "fftw_system")]
    discover_fftw();
}

// The version numbers and the minimum, shared with the crate's run-time check.
#[cfg(feature = "fftw_system")]
#[path = "src/fftw/version.rs"]
mod fftw_version;

#[cfg(feature = "fftw_system")]
fn discover_fftw() {
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    // rustdoc does not link, so docs.rs builds the documentation without FFTW installed.
    if std::env::var_os("DOCS_RS").is_some() {
        return;
    }

    // fftw-sys links both precisions, but does not discover system search paths.
    for library in ["fftw3", "fftw3f"] {
        match pkg_config::Config::new().probe(library) {
            Ok(found) => check_fftw_version(library, &found.version),
            Err(error) => {
                // Never substitute host Homebrew libraries when cross-compiling.
                let native_macos = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
                    && std::env::var("HOST").ok() == std::env::var("TARGET").ok();
                let prefix = match std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
                    Ok("aarch64") => "/opt/homebrew/opt/fftw/lib",
                    _ => "/usr/local/opt/fftw/lib",
                };
                let dylib = std::path::Path::new(prefix).join(format!("lib{library}.dylib"));
                if native_macos && dylib.is_file() {
                    // No version is known here; the crate checks it at run time.
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
}

/// Fail the build if pkg-config reports a version of `library` below the minimum.
#[cfg(feature = "fftw_system")]
fn check_fftw_version(library: &str, version: &str) {
    if !fftw_version::parse_numbers(version).is_some_and(fftw_version::meets_minimum) {
        panic!(
            "pkg-config found {library} version {version:?}, but RLST needs FFTW {} or later.\n\
             Install a newer FFTW (3.3.10 or later is recommended), and set PKG_CONFIG_PATH to \
             its lib/pkgconfig directory.",
            fftw_version::MINIMUM_TEXT
        );
    }
}
