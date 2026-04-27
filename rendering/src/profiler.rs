use std::{
    collections::{HashMap, VecDeque},
    fmt,
    sync::{Arc, atomic::{AtomicBool, Ordering}},
    time::{Duration, Instant},
};

const HISTORY: usize = 60;

#[derive(Debug, Clone)]
pub struct PassStats {
    pub name: &'static str,
    pub last_ms: f32,
    pub avg_ms: f32,
    pub gpu_ms: Option<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct FrameStats {
    pub frame: u64,
    pub last_ms: f32,
    pub avg_ms: f32,
    pub fps: f32,
    pub passes: Vec<PassStats>,
}

impl fmt::Display for FrameStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "frame={} fps={:.1} cpu={:.2}ms (avg={:.2}ms)", self.frame, self.fps, self.last_ms, self.avg_ms)?;
        for pass in &self.passes {
            write!(f, "\n  {:20} cpu {:6.2}ms  avg {:6.2}ms", pass.name, pass.last_ms, pass.avg_ms)?;
            if let Some(gpu) = pass.gpu_ms {
                write!(f, "  gpu {:6.2}ms", gpu)?;
            }
        }
        Ok(())
    }
}

pub struct FrameProfiler {
    frame: u64,
    frame_times: VecDeque<f32>,
    pass_times: HashMap<&'static str, VecDeque<f32>>,
    pass_order: Vec<&'static str>,
    frame_start: Instant,
    last_stats: FrameStats,
}

impl Default for FrameProfiler {
    fn default() -> Self {
        Self {
            frame: 0,
            frame_times: VecDeque::with_capacity(HISTORY),
            pass_times: HashMap::new(),
            pass_order: Vec::new(),
            frame_start: Instant::now(),
            last_stats: FrameStats::default(),
        }
    }
}

impl FrameProfiler {
    pub fn begin_frame(&mut self) {
        self.frame_start = Instant::now();
        self.pass_order.clear();
    }

    pub fn record_pass(&mut self, name: &'static str, duration: Duration) {
        let ms = duration.as_secs_f32() * 1000.0;
        let history = self.pass_times.entry(name).or_insert_with(|| VecDeque::with_capacity(HISTORY));
        if history.len() >= HISTORY {
            history.pop_front();
        }
        history.push_back(ms);
        if !self.pass_order.contains(&name) {
            self.pass_order.push(name);
        }
    }

    pub fn end_frame(&mut self) {
        let elapsed_ms = self.frame_start.elapsed().as_secs_f32() * 1000.0;
        if self.frame_times.len() >= HISTORY {
            self.frame_times.pop_front();
        }
        self.frame_times.push_back(elapsed_ms);
        self.frame += 1;

        let avg_ms = rolling_avg(&self.frame_times);
        let fps = if avg_ms > 0.0 { 1000.0 / avg_ms } else { 0.0 };

        let passes = self.pass_order.iter().filter_map(|&name| {
            let history = self.pass_times.get(name)?;
            Some(PassStats {
                name,
                last_ms: *history.back().unwrap_or(&0.0),
                avg_ms: rolling_avg(history),
                gpu_ms: None,
            })
        }).collect();

        self.last_stats = FrameStats { frame: self.frame, last_ms: elapsed_ms, avg_ms, fps, passes };
    }

    pub fn apply_gpu_times(&mut self, gpu_times_ms: &[f32]) {
        for (pass, &gpu_ms) in self.last_stats.passes.iter_mut().zip(gpu_times_ms.iter()) {
            pass.gpu_ms = Some(gpu_ms);
        }
    }

    pub fn stats(&self) -> &FrameStats {
        &self.last_stats
    }
}

pub struct GpuProfiler {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    num_passes: u32,
    timestamp_period_ns: f32,
    ready: Arc<AtomicBool>,
    pending: bool,
    pub last_gpu_times_ms: Vec<f32>,
}

impl GpuProfiler {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, num_passes: u32) -> Self {
        let buf_size = (num_passes as u64) * 2 * 8;

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gpu_profiler"),
            ty: wgpu::QueryType::Timestamp,
            count: num_passes * 2,
        });

        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_profiler_resolve"),
            size: buf_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_profiler_staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            query_set,
            resolve_buf,
            staging_buf,
            num_passes,
            timestamp_period_ns: queue.get_timestamp_period(),
            ready: Arc::new(AtomicBool::new(false)),
            pending: false,
            last_gpu_times_ms: vec![0.0; num_passes as usize],
        }
    }

    pub fn write_begin(&self, encoder: &mut wgpu::CommandEncoder, pass_idx: u32) {
        encoder.write_timestamp(&self.query_set, pass_idx * 2);
    }

    pub fn write_end(&self, encoder: &mut wgpu::CommandEncoder, pass_idx: u32) {
        encoder.write_timestamp(&self.query_set, pass_idx * 2 + 1);
    }

    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) -> bool {
        if self.pending {
            return false;
        }
        encoder.resolve_query_set(&self.query_set, 0..self.num_passes * 2, &self.resolve_buf, 0);
        encoder.copy_buffer_to_buffer(&self.resolve_buf, 0, &self.staging_buf, 0, (self.num_passes as u64) * 2 * 8);
        true
    }

    pub fn schedule_readback(&mut self, device: &wgpu::Device) {
        if self.pending {
            return;
        }
        let ready = Arc::clone(&self.ready);
        self.staging_buf.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            if result.is_ok() {
                ready.store(true, Ordering::Release);
            }
        });
        let _ = device.poll(wgpu::PollType::Poll);
        self.pending = true;
    }

    pub fn try_read_results(&mut self) -> bool {
        if !self.pending || !self.ready.load(Ordering::Acquire) {
            return false;
        }

        {
            let raw = self.staging_buf.slice(..).get_mapped_range();
            let timestamps: &[u64] = bytemuck::cast_slice(&raw);
            for i in 0..self.num_passes as usize {
                let delta = timestamps[i * 2 + 1].saturating_sub(timestamps[i * 2]);
                self.last_gpu_times_ms[i] = (delta as f64 * self.timestamp_period_ns as f64 / 1_000_000.0) as f32;
            }
        }

        self.staging_buf.unmap();
        self.ready.store(false, Ordering::Release);
        self.pending = false;
        true
    }
}

fn rolling_avg(history: &VecDeque<f32>) -> f32 {
    if history.is_empty() {
        return 0.0;
    }
    history.iter().sum::<f32>() / history.len() as f32
}
