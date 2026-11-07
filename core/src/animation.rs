//! Keyframe curves and clips. Knows nothing about entities: a `Clip` maps targets of any type to
//! curves, and its owner writes the sampled values wherever the targets point.

/// How a curve moves from a key to the next one (set on the earlier key).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Interpolation {
    /// Holds the key's value until the next key.
    Step,
    #[default]
    Linear,
    /// Hermite with Catmull-Rom tangents: passes through every key with a continuous slope.
    Smooth,
    /// Starts slowly (cubic).
    EaseIn,
    /// Ends slowly (cubic).
    EaseOut,
    EaseInOut,
    /// Hermite with the key's `out_tangent` and the next key's `in_tangent` (glTF CUBICSPLINE).
    Cubic,
}

/// Up to four components; a curve uses the first `Curve::components`.
pub type Value = [f32; 4];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Key {
    /// Seconds.
    pub time: f32,
    pub value: Value,
    pub interpolation: Interpolation,
    /// Slopes in value per second, used by `Cubic`.
    pub in_tangent: Value,
    pub out_tangent: Value,
}

impl Key {
    pub fn new(time: f32, value: Value, interpolation: Interpolation) -> Self {
        Self { time, value, interpolation, in_tangent: [0.0; 4], out_tangent: [0.0; 4] }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Curve {
    /// 1–4.
    pub components: usize,
    /// Values are unit quaternions (xyzw): linear blends by slerp, and every result is normalized.
    pub rotation: bool,
    /// Sorted by time.
    pub keys: Vec<Key>,
}

impl Curve {
    pub fn new(components: usize, mut keys: Vec<Key>) -> Self {
        keys.sort_by(|a, b| a.time.total_cmp(&b.time));
        Self { components: components.clamp(1, 4), rotation: false, keys }
    }

    pub fn rotation(keys: Vec<Key>) -> Self {
        Self { rotation: true, ..Self::new(4, keys) }
    }

    pub fn duration(&self) -> f32 {
        self.keys.last().map_or(0.0, |k| k.time)
    }

    /// The value at `time`, holding the first / last key's value outside the keys.
    pub fn sample(&self, time: f32) -> Value {
        let keys = &self.keys;
        let (Some(first), Some(last)) = (keys.first(), keys.last()) else { return [0.0; 4] };
        if time <= first.time {
            return first.value;
        }
        if time >= last.time {
            return last.value;
        }
        let next = keys.partition_point(|k| k.time <= time);
        let (k0, k1) = (&keys[next - 1], &keys[next]);
        let span = k1.time - k0.time;
        if span <= 0.0 {
            return k1.value;
        }
        let u = (time - k0.time) / span;
        let value = match k0.interpolation {
            Interpolation::Step => return k0.value,
            Interpolation::Linear if self.rotation => return slerp(k0.value, k1.value, u),
            Interpolation::Linear => lerp(k0.value, k1.value, u),
            Interpolation::EaseIn => lerp(k0.value, k1.value, u * u * u),
            Interpolation::EaseOut => lerp(k0.value, k1.value, 1.0 - (1.0 - u).powi(3)),
            Interpolation::EaseInOut => {
                let eased = if u < 0.5 { 4.0 * u * u * u } else { 1.0 - (-2.0 * u + 2.0).powi(3) / 2.0 };
                lerp(k0.value, k1.value, eased)
            }
            Interpolation::Smooth => {
                let (m0, m1) = (self.auto_tangent(next - 1), self.auto_tangent(next));
                hermite(k0.value, m0, k1.value, m1, u, span)
            }
            Interpolation::Cubic => hermite(k0.value, k0.out_tangent, k1.value, k1.in_tangent, u, span),
        };
        if self.rotation { normalize(value) } else { value }
    }

    /// Catmull-Rom slope at key `i` (one-sided at the ends), in value per second.
    fn auto_tangent(&self, i: usize) -> Value {
        let keys = &self.keys;
        let before = &keys[i.saturating_sub(1)];
        let after = &keys[(i + 1).min(keys.len() - 1)];
        let span = after.time - before.time;
        if span <= 0.0 {
            return [0.0; 4];
        }
        std::array::from_fn(|c| (after.value[c] - before.value[c]) / span)
    }
}

/// When a clip is past its end: wrap to the start, or hold the last frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Wrap {
    #[default]
    Loop,
    Once,
}

/// Curves keyed by what they animate.
#[derive(Clone, Debug, PartialEq)]
pub struct Clip<T> {
    pub name: String,
    pub tracks: Vec<(T, Curve)>,
    pub wrap: Wrap,
}

impl<T> Clip<T> {
    pub fn duration(&self) -> f32 {
        self.tracks.iter().map(|(_, c)| c.duration()).fold(0.0, f32::max)
    }

    /// The clip-local time `elapsed` seconds after its start.
    pub fn local_time(&self, elapsed: f32) -> f32 {
        let duration = self.duration();
        match self.wrap {
            Wrap::Loop if duration > 0.0 => elapsed.rem_euclid(duration),
            _ => elapsed.clamp(0.0, duration),
        }
    }

    /// Every track's value at clip-local `time`.
    pub fn sample(&self, time: f32) -> impl Iterator<Item = (&T, Value)> {
        self.tracks.iter().map(move |(target, curve)| (target, curve.sample(time)))
    }
}

/// A clip's playhead.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Playback {
    /// Seconds since the clip started, scaled by `speed`, before wrapping.
    pub elapsed: f32,
    pub speed: f32,
    pub playing: bool,
}

impl Default for Playback {
    fn default() -> Self {
        Self { elapsed: 0.0, speed: 1.0, playing: true }
    }
}

impl Playback {
    pub fn advance(&mut self, delta: f32) {
        if self.playing {
            self.elapsed += delta * self.speed;
        }
    }
}

fn lerp(a: Value, b: Value, t: f32) -> Value {
    std::array::from_fn(|c| a[c] + (b[c] - a[c]) * t)
}

/// glTF's cubic spline: tangents are slopes per second, scaled by the segment's length.
fn hermite(p0: Value, m0: Value, p1: Value, m1: Value, u: f32, span: f32) -> Value {
    let (u2, u3) = (u * u, u * u * u);
    let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
    let h10 = u3 - 2.0 * u2 + u;
    let h01 = -2.0 * u3 + 3.0 * u2;
    let h11 = u3 - u2;
    std::array::from_fn(|c| h00 * p0[c] + h10 * span * m0[c] + h01 * p1[c] + h11 * span * m1[c])
}

fn normalize(q: Value) -> Value {
    let len = q.iter().map(|c| c * c).sum::<f32>().sqrt();
    if len > 0.0 { q.map(|c| c / len) } else { [0.0, 0.0, 0.0, 1.0] }
}

fn slerp(a: Value, b: Value, t: f32) -> Value {
    let mut cos = (0..4).map(|c| a[c] * b[c]).sum::<f32>();
    // Take the short way round.
    let b = if cos < 0.0 {
        cos = -cos;
        b.map(|c| -c)
    } else {
        b
    };
    if cos > 0.9995 {
        return normalize(lerp(a, b, t));
    }
    let angle = cos.acos();
    let sin = angle.sin();
    let (wa, wb) = (((1.0 - t) * angle).sin() / sin, (t * angle).sin() / sin);
    std::array::from_fn(|c| wa * a[c] + wb * b[c])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(keys: &[(f32, f32, Interpolation)]) -> Curve {
        Curve::new(1, keys.iter().map(|&(t, v, i)| Key::new(t, [v, 0.0, 0.0, 0.0], i)).collect())
    }

    fn at(curve: &Curve, time: f32) -> f32 {
        curve.sample(time)[0]
    }

    #[test]
    fn presets_hit_their_keys_and_differ_in_between() {
        for preset in [
            Interpolation::Linear,
            Interpolation::Smooth,
            Interpolation::EaseIn,
            Interpolation::EaseOut,
            Interpolation::EaseInOut,
            Interpolation::Cubic,
        ] {
            let curve = scalar(&[(0.0, 0.0, preset), (2.0, 10.0, preset)]);
            assert_eq!(at(&curve, 0.0), 0.0, "{preset:?}");
            assert!((at(&curve, 2.0) - 10.0).abs() < 1e-5, "{preset:?}");
        }
        let mid = |p| at(&scalar(&[(0.0, 0.0, p), (2.0, 10.0, p)]), 0.5);
        assert!((mid(Interpolation::Linear) - 2.5).abs() < 1e-5);
        assert!(mid(Interpolation::EaseIn) < 2.5);
        assert!(mid(Interpolation::EaseOut) > 2.5);
        assert_eq!(mid(Interpolation::Step), 0.0);
    }

    #[test]
    fn step_holds_until_the_next_key() {
        let curve = scalar(&[(0.0, 1.0, Interpolation::Step), (1.0, 0.05, Interpolation::Step), (1.2, 1.0, Interpolation::Linear)]);
        assert_eq!(at(&curve, 0.99), 1.0);
        assert_eq!(at(&curve, 1.0), 0.05);
        assert_eq!(at(&curve, 1.19), 0.05);
    }

    #[test]
    fn smooth_has_no_kink_at_a_key() {
        let curve = scalar(&[(0.0, 0.0, Interpolation::Smooth), (1.0, 1.0, Interpolation::Smooth), (2.0, 0.0, Interpolation::Smooth)]);
        let e = 1e-3;
        let left = (at(&curve, 0.5) - at(&curve, 0.5 - e)) / e;
        assert!(left > 0.0);
        let before = (at(&curve, 1.0) - at(&curve, 1.0 - e)) / e;
        let after = (at(&curve, 1.0 + e) - at(&curve, 1.0)) / e;
        assert!((before - after).abs() < 0.01, "{before} vs {after}");
    }

    #[test]
    fn cubic_uses_key_tangents() {
        let mut keys = vec![Key::new(0.0, [0.0; 4], Interpolation::Cubic), Key::new(1.0, [0.0; 4], Interpolation::Cubic)];
        keys[0].out_tangent = [4.0, 0.0, 0.0, 0.0];
        let curve = Curve::new(1, keys);
        assert!(at(&curve, 0.25) > 0.3);
    }

    #[test]
    fn rotation_slerps_unit_quaternions() {
        let half = std::f32::consts::FRAC_1_SQRT_2;
        let curve = Curve::rotation(vec![
            Key::new(0.0, [0.0, 0.0, 0.0, 1.0], Interpolation::Linear),
            Key::new(1.0, [0.0, half, 0.0, half], Interpolation::Linear),
        ]);
        let q = curve.sample(0.5);
        let len: f32 = q.iter().map(|c| c * c).sum::<f32>().sqrt();
        assert!((len - 1.0).abs() < 1e-5);
        // Halfway through a 90° turn about Y is 45°.
        assert!((q[1] - (std::f32::consts::PI / 8.0).sin()).abs() < 1e-4);
    }

    #[test]
    fn clips_loop_or_hold() {
        let curve = scalar(&[(0.0, 0.0, Interpolation::Linear), (4.0, 4.0, Interpolation::Linear)]);
        let mut clip = Clip { name: "c".into(), tracks: vec![((), curve)], wrap: Wrap::Loop };
        assert_eq!(clip.local_time(5.0), 1.0);
        clip.wrap = Wrap::Once;
        assert_eq!(clip.local_time(5.0), 4.0);
    }
}
