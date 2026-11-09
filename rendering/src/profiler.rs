use std::{
    collections::{HashMap, VecDeque},
    fmt,
    sync::{Arc, atomic::{AtomicBool, Ordering}},
    time::{Duration, Instant},
};

use crate::arena::ArenaStats;

const HISTORY: usize = 60;

#[derive(Debug, Clone)]
pub struct PassStats {
    pub name: &'static str,
    pub last_ms: f32,
    pub avg_ms: f32,
    pub gpu_ms: Option<f32>,
    /// Last frame's draws, from passes that count them (`Pass::draw_counts`).
    pub draws: Option<DrawCounts>,
}

/// A pass's mesh draws in one frame: `draws` calls drawing `instances` meshes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DrawCounts {
    pub draws: u32,
    pub instances: u32,
}

#[derive(Debug, Clone, Default)]
pub struct FrameStats {
    pub frame: u64,
    /// Time between frames (60-frame average): what the frame rate is made of, GPU waits included.
    pub frame_ms: f32,
    pub fps: f32,
    /// CPU time recording the render graph (60-frame average).
    pub cpu_ms: f32,
    pub passes: Vec<PassStats>,
    pub memory: ArenaStats,
}

impl fmt::Display for FrameStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "frame={} fps={:.1} frame={:.2}ms cpu={:.2}ms", self.frame, self.fps, self.frame_ms, self.cpu_ms)?;
        let m = &self.memory;
        write!(
            f,
            "\n  geometry: {} / {} vertices, {} / {} indices",
            m.geometry_vertices.0, m.geometry_vertices.1, m.geometry_indices.0, m.geometry_indices.1
        )?;
        for pass in &self.passes {
            write!(f, "\n  {:20} cpu {:6.2}ms  avg {:6.2}ms", pass.name, pass.last_ms, pass.avg_ms)?;
            if let Some(gpu) = pass.gpu_ms {
                write!(f, "  gpu {:6.2}ms", gpu)?;
            }
            if let Some(d) = pass.draws {
                write!(f, "  {} draws, {} meshes", d.draws, d.instances)?;
            }
        }
        Ok(())
    }
}

pub struct FrameProfiler {
    frame: u64,
    /// CPU time of each frame's graph recording.
    cpu_times: VecDeque<f32>,
    /// Time from each frame's start to the next's.
    frame_intervals: VecDeque<f32>,
    last_frame_start: Option<Instant>,
    pass_times: HashMap<&'static str, VecDeque<f32>>,
    pass_order: Vec<&'static str>,
    pass_draws: HashMap<&'static str, DrawCounts>,
    /// Latest GPU time per pass, a few frames old (timestamp readback).
    pass_gpu: HashMap<&'static str, f32>,
    frame_start: Instant,
    last_stats: FrameStats,
}

impl Default for FrameProfiler {
    fn default() -> Self {
        Self {
            frame: 0,
            cpu_times: VecDeque::with_capacity(HISTORY),
            frame_intervals: VecDeque::with_capacity(HISTORY),
            last_frame_start: None,
            pass_times: HashMap::new(),
            pass_order: Vec::new(),
            pass_draws: HashMap::new(),
            pass_gpu: HashMap::new(),
            frame_start: Instant::now(),
            last_stats: FrameStats::default(),
        }
    }
}

impl FrameProfiler {
    pub fn begin_frame(&mut self) {
        let now = Instant::now();
        if let Some(last) = self.last_frame_start {
            push_capped(&mut self.frame_intervals, (now - last).as_secs_f32() * 1000.0);
        }
        self.last_frame_start = Some(now);
        self.frame_start = now;
    }

    pub fn record_pass(&mut self, name: &'static str, duration: Duration) {
        let ms = duration.as_secs_f32() * 1000.0;
        push_capped(self.pass_times.entry(name).or_insert_with(|| VecDeque::with_capacity(HISTORY)), ms);
        if !self.pass_order.contains(&name) {
            self.pass_order.push(name);
        }
    }

    pub fn record_draws(&mut self, name: &'static str, draws: Option<DrawCounts>) {
        match draws {
            Some(draws) => self.pass_draws.insert(name, draws),
            None => self.pass_draws.remove(name),
        };
    }

    pub fn end_frame(&mut self) {
        push_capped(&mut self.cpu_times, self.frame_start.elapsed().as_secs_f32() * 1000.0);
        self.frame += 1;

        let frame_ms = rolling_avg(&self.frame_intervals);
        let fps = if frame_ms > 0.0 { 1000.0 / frame_ms } else { 0.0 };

        let passes = self.pass_order.iter().filter_map(|&name| {
            let history = self.pass_times.get(name)?;
            Some(PassStats {
                name,
                last_ms: *history.back().unwrap_or(&0.0),
                avg_ms: rolling_avg(history),
                gpu_ms: self.pass_gpu.get(name).copied(),
                draws: self.pass_draws.get(name).copied(),
            })
        }).collect();

        self.last_stats = FrameStats {
            frame: self.frame,
            frame_ms,
            fps,
            cpu_ms: rolling_avg(&self.cpu_times),
            passes,
            memory: ArenaStats::default(),
        };
    }

    pub fn set_arena_stats(&mut self, memory: ArenaStats) {
        self.last_stats.memory = memory;
    }

    /// GPU times from a read-back frame, kept until newer ones arrive: the stats are rebuilt every
    /// frame, and setting them on the last stats lost them at the next `end_frame`.
    pub fn apply_gpu_times(&mut self, gpu_times_ms: impl IntoIterator<Item = (&'static str, f32)>) {
        self.pass_gpu.extend(gpu_times_ms);
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
    /// Pass names in the order of the resolved queries.
    names: Vec<&'static str>,
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
            names: Vec::new(),
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

    pub fn num_passes(&self) -> u32 {
        self.num_passes
    }

    /// Resolves this frame's queries for `names`, the passes in query order.
    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder, names: Vec<&'static str>) -> bool {
        if self.pending {
            return false;
        }
        self.names = names;
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

    /// The pass times of the last resolved frame, once read back.
    pub fn try_read_results(&mut self) -> Option<Vec<(&'static str, f32)>> {
        if !self.pending || !self.ready.load(Ordering::Acquire) {
            return None;
        }

        {
            let raw = self.staging_buf.slice(..).get_mapped_range().expect("profiler staging buffer not mapped");
            let timestamps: &[u64] = bytemuck::cast_slice(&raw);
            for i in 0..self.num_passes as usize {
                let delta = timestamps[i * 2 + 1].saturating_sub(timestamps[i * 2]);
                self.last_gpu_times_ms[i] = (delta as f64 * self.timestamp_period_ns as f64 / 1_000_000.0) as f32;
            }
        }

        self.staging_buf.unmap();
        self.ready.store(false, Ordering::Release);
        self.pending = false;
        Some(self.names.iter().copied().zip(self.last_gpu_times_ms.iter().copied()).collect())
    }
}

fn push_capped(history: &mut VecDeque<f32>, ms: f32) {
    if history.len() >= HISTORY {
        history.pop_front();
    }
    history.push_back(ms);
}

fn rolling_avg(history: &VecDeque<f32>) -> f32 {
    if history.is_empty() {
        return 0.0;
    }
    history.iter().sum::<f32>() / history.len() as f32
}
