use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const ENV_PATH: &str = "SUITESPARSE_PATH";

fn main() {
    println!("cargo:rerun-if-env-changed={ENV_PATH}");

    let installation = suite_sparse_installation().unwrap_or_else(|message| panic!("{message}"));

    println!(
        "cargo:rustc-link-search=native={}",
        installation.lib_dir.display()
    );
    println!("cargo:rustc-link-lib=umfpack");
    println!(
        "cargo:warning=linking against SuiteSparse from {}",
        installation.prefix.display()
    );
}

struct SuiteSparseInstallation {
    prefix: PathBuf,
    lib_dir: PathBuf,
}

fn suite_sparse_installation() -> Result<SuiteSparseInstallation, String> {
    if let Some(path) = env::var_os(ENV_PATH) {
        let prefix = PathBuf::from(path);
        return validate_prefix(prefix, true);
    }

    let candidates = [
        "/opt/homebrew/opt/suite-sparse",
        "/usr/local/opt/suite-sparse",
        "/usr/local",
        "/usr",
    ];

    for candidate in candidates {
        if let Ok(installation) = validate_prefix(PathBuf::from(candidate), false) {
            return Ok(installation);
        }
    }

    Err(format!(
        "SuiteSparse with UMFPACK was not found. Install SuiteSparse, or set {ENV_PATH} to its installation prefix (the directory containing include/ and lib/)."
    ))
}

fn validate_prefix(prefix: PathBuf, explicit: bool) -> Result<SuiteSparseInstallation, String> {
    let header_found = [
        prefix.join("include/suitesparse/umfpack.h"),
        prefix.join("include/umfpack.h"),
    ]
    .iter()
    .any(|header| header.is_file());
    let lib_dir = library_dir(&prefix);

    if header_found && let Some(lib_dir) = lib_dir {
        return Ok(SuiteSparseInstallation { prefix, lib_dir });
    }

    if explicit {
        Err(format!(
            "{ENV_PATH}={} is not a SuiteSparse installation prefix with UMFPACK headers and libraries",
            prefix.display()
        ))
    } else {
        Err("not a SuiteSparse prefix".to_owned())
    }
}

fn library_dir(prefix: &Path) -> Option<PathBuf> {
    let lib_root = prefix.join("lib");
    if library_exists(&lib_root) {
        return Some(lib_root);
    }
    if library_exists(&prefix.join("lib64")) {
        return Some(prefix.join("lib64"));
    }

    // Debian and Ubuntu install architecture-specific libraries below
    // /usr/lib/<target-triplet>. Looking one level below lib/ also supports
    // other multiarch layouts without hard-coding a target triplet.
    fs::read_dir(lib_root).ok()?.flatten().find_map(|entry| {
        let path = entry.path();
        (path.is_dir() && library_exists(&path)).then_some(path)
    })
}

fn library_exists(lib_dir: &Path) -> bool {
    [
        "libumfpack.dylib",
        "libumfpack.so",
        "libumfpack.a",
        "umfpack.lib",
    ]
    .iter()
    .any(|library| lib_dir.join(library).is_file())
}
