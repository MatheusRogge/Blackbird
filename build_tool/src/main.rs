use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use build_tool::{BuildKind, BuildOptions, format_bytes};
use engine::assets::pack::{Codec, PackReader};

const USAGE: &str = "\
usage:
  blackbird build <project-dir> [--production] [--out <dir>] [--target <triple>] [--level <1-22>]
  blackbird assets <project-dir>      list the assets a build would ship
  blackbird inspect <file.bbpak>      list the contents of a pack";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("build") => build(&args[1..]),
        Some("assets") => list_assets(&args[1..]),
        Some("inspect") => inspect(&args[1..]),
        _ => {
            eprintln!("{USAGE}");
            std::process::exit(2);
        }
    }
}

fn build(args: &[String]) -> Result<()> {
    let mut project_dir = None;
    let mut kind = BuildKind::Development;
    let mut out_dir = None;
    let mut target = None;
    let mut level = None;

    let mut args = args.iter();
    while let Some(arg) = args.next() {
        let mut value = |flag: &str| args.next().cloned().with_context(|| format!("{flag} needs a value"));
        match arg.as_str() {
            "--production" | "--release" => kind = BuildKind::Production,
            "--development" | "--dev" => kind = BuildKind::Development,
            "--out" => out_dir = Some(PathBuf::from(value("--out")?)),
            "--target" => target = Some(value("--target")?),
            "--level" => level = Some(value("--level")?.parse().context("--level must be a number")?),
            flag if flag.starts_with("--") => bail!("unknown option {flag}\n{USAGE}"),
            dir if project_dir.is_none() => project_dir = Some(PathBuf::from(dir)),
            extra => bail!("unexpected argument {extra}\n{USAGE}"),
        }
    }

    let mut options = BuildOptions::new(project_dir.context(USAGE)?, kind);
    options.out_dir = out_dir;
    options.target = target;
    if let Some(level) = level {
        options.compression_level = level;
    }

    let output = build_tool::build(&options, &mut |line| eprintln!("{line}"))?;
    println!(
        "{} build: {} ({} assets, {} on disk)",
        kind,
        output.executable.display(),
        output.assets,
        format_bytes(output.stored_bytes)
    );
    Ok(())
}

fn list_assets(args: &[String]) -> Result<()> {
    let [dir] = args else { bail!(USAGE) };
    let (project, manifest) = build_tool::collect_manifest(dir.as_ref())?;
    println!("{}: {} assets, {}", project.name, manifest.len(), format_bytes(manifest.total_size()));
    for entry in manifest.entries() {
        println!("{:>10}  {}", format_bytes(entry.size), entry.path);
    }
    Ok(())
}

fn inspect(args: &[String]) -> Result<()> {
    let [path] = args else { bail!(USAGE) };
    let pack = PackReader::open(path).with_context(|| format!("opening {path}"))?;
    println!("{path}: content id {:016x}, {} entries", pack.content_id(), pack.entries().len());
    for e in pack.entries() {
        let codec = match e.codec {
            Codec::Stored => "stored",
            Codec::Zstd => "zstd",
        };
        println!("{:>10} → {:>10}  {codec:<6}  {}", format_bytes(e.raw_len), format_bytes(e.stored_len), e.path);
    }
    Ok(())
}
