//! Bakes the asset manifest handed over by `build_tool` into the binary. Without one the
//! player is "unbound" and loads whatever project directory it is given at runtime.

use std::{env, fs, path::PathBuf};

use assets::{AssetManifest, codegen};

const MANIFEST_ENV: &str = "BLACKBIRD_ASSET_MANIFEST";
const PACK_ID_ENV: &str = "BLACKBIRD_PACK_ID";

fn main() {
    println!("cargo:rerun-if-env-changed={MANIFEST_ENV}");
    println!("cargo:rerun-if-env-changed={PACK_ID_ENV}");

    let manifest = match env::var_os(MANIFEST_ENV) {
        Some(path) => {
            let path = PathBuf::from(path);
            println!("cargo:rerun-if-changed={}", path.display());
            let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
            AssetManifest::parse(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        }
        None => AssetManifest::default(),
    };
    let pack_id = env::var(PACK_ID_ENV)
        .ok()
        .map(|id| u64::from_str_radix(&id, 16).unwrap_or_else(|_| panic!("{PACK_ID_ENV} must be hex, got {id}")));

    let out = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR")).join("assets.rs");
    fs::write(&out, codegen::generate(&manifest, pack_id)).unwrap_or_else(|e| panic!("writing {}: {e}", out.display()));
}
