use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};

pub struct ShutdownSignal(Receiver<()>);

impl ShutdownSignal {
    pub fn should_stop(&self) -> bool {
        self.0.try_recv().is_ok()
    }
}

pub trait Executor: Send + Sync + 'static {
    fn spawn(&self, task: Box<dyn FnOnce() + Send + 'static>);
    fn spawn_named(&self, name: String, f: Box<dyn FnOnce(ShutdownSignal) + Send + 'static>);
    fn shutdown(&self);
}

struct NamedThread {
    name: String,
    shutdown_tx: Sender<()>,
    handle: Option<JoinHandle<()>>,
}

struct DefaultExecutorState {
    task_tx: Option<Sender<Box<dyn FnOnce() + Send + 'static>>>,
    named_threads: Vec<NamedThread>,
    worker: Option<JoinHandle<()>>,
}

pub struct DefaultExecutor {
    state: Mutex<DefaultExecutorState>,
}

impl DefaultExecutor {
    pub fn new() -> Self {
        let (task_tx, task_rx) = mpsc::channel::<Box<dyn FnOnce() + Send + 'static>>();

        let worker = thread::Builder::new()
            .name("executor-worker".into())
            .spawn(move || {
                while let Ok(task) = task_rx.recv() {
                    task();
                }
            })
            .expect("failed to spawn executor worker thread");

        Self {
            state: Mutex::new(DefaultExecutorState {
                task_tx: Some(task_tx),
                named_threads: Vec::new(),
                worker: Some(worker),
            }),
        }
    }
}

impl Default for DefaultExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl Executor for DefaultExecutor {
    fn spawn(&self, task: Box<dyn FnOnce() + Send + 'static>) {
        let state = self.state.lock().unwrap();
        if let Some(tx) = &state.task_tx {
            let _ = tx.send(task);
        }
    }

    fn spawn_named(&self, name: String, f: Box<dyn FnOnce(ShutdownSignal) + Send + 'static>) {
        let (shutdown_tx, shutdown_rx) = mpsc::channel();
        let signal = ShutdownSignal(shutdown_rx);

        let handle = thread::Builder::new()
            .name(name.clone())
            .spawn(move || f(signal))
            .expect("failed to spawn named thread");

        let mut state = self.state.lock().unwrap();
        state.named_threads.push(NamedThread {
            name,
            shutdown_tx,
            handle: Some(handle),
        });
    }

    fn shutdown(&self) {
        let mut state = self.state.lock().unwrap();

        for t in &state.named_threads {
            let _ = t.shutdown_tx.send(());
        }

        for t in &mut state.named_threads {
            if let Some(h) = t.handle.take()
                && h.join().is_err()
            {
                eprintln!("[executor] named thread '{}' panicked", t.name);
            }
        }

        state.named_threads.clear();

        // Drop task_tx so the worker thread exits on next recv()
        state.task_tx = None;
        if let Some(h) = state.worker.take() {
            let _ = h.join();
        }
    }
}

impl Drop for DefaultExecutor {
    fn drop(&mut self) {
        self.shutdown();
    }
}
