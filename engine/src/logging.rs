use std::{
    collections::VecDeque,
    fs::{self, File},
    io::{self, Write},
    path::PathBuf,
    sync::{Arc, Mutex},
};

use thiserror::Error;

pub use log::LevelFilter;

#[derive(Error, Debug)]
pub enum LoggingError {
    #[error("opening log file {path}: {source}")]
    File { path: PathBuf, source: io::Error },

    #[error(transparent)]
    Init(#[from] log::SetLoggerError),
}

pub struct LogConfig {
    pub level: LevelFilter,
    /// Every record is also appended here; parent directories are created.
    pub file: Option<PathBuf>,
    /// Number of recent lines kept in memory for in-game consoles. `0` keeps none.
    pub history: usize,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self { level: LevelFilter::Info, file: None, history: 0 }
    }
}

/// The most recent log lines, shared with the logger.
#[derive(Clone)]
pub struct LogHistory {
    lines: Arc<Mutex<VecDeque<String>>>,
    capacity: usize,
}

impl LogHistory {
    fn new(capacity: usize) -> Self {
        Self { lines: Arc::new(Mutex::new(VecDeque::with_capacity(capacity))), capacity }
    }

    fn push(&self, line: &str) {
        if self.capacity == 0 {
            return;
        }
        let mut lines = self.lines.lock().expect("log history poisoned");
        if lines.len() == self.capacity {
            lines.pop_front();
        }
        lines.push_back(line.to_string());
    }

    pub fn snapshot(&self) -> Vec<String> {
        self.lines.lock().expect("log history poisoned").iter().cloned().collect()
    }
}

struct TeeWriter {
    file: Option<File>,
    history: LogHistory,
}

impl Write for TeeWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        io::stderr().write_all(buf)?;
        if let Some(file) = &mut self.file {
            file.write_all(buf)?;
        }
        for line in String::from_utf8_lossy(buf).lines().filter(|l| !l.is_empty()) {
            self.history.push(line);
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if let Some(file) = &mut self.file {
            file.flush()?;
        }
        io::stderr().flush()
    }
}

/// Installs the global logger: stderr, plus the optional file and in-memory history.
/// `RUST_LOG` still overrides `config.level`.
pub fn init(config: LogConfig) -> Result<LogHistory, LoggingError> {
    let file = match &config.file {
        Some(path) => {
            let open = || {
                if let Some(dir) = path.parent() {
                    fs::create_dir_all(dir)?;
                }
                File::create(path)
            };
            Some(open().map_err(|source| LoggingError::File { path: path.clone(), source })?)
        }
        None => None,
    };

    let history = LogHistory::new(config.history);
    env_logger::builder()
        .filter_level(config.level)
        .format_source_path(true)
        .format_level(true)
        .format_timestamp_millis()
        .parse_default_env()
        .target(env_logger::Target::Pipe(Box::new(TeeWriter { file, history: history.clone() })))
        .try_init()?;
    Ok(history)
}
