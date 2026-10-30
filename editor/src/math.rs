//! Quaternions are glTF-style `[x, y, z, w]` arrays, matching `NodeEdit::rotation`.

use ultraviolet::{Mat3, Mat4, Vec3, Vec4};

pub type Quat = [f32; 4];

pub fn quat_mul(a: Quat, b: Quat) -> Quat {
    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

pub fn quat_conjugate([x, y, z, w]: Quat) -> Quat {
    [-x, -y, -z, w]
}

pub fn quat_normalize(q: Quat) -> Quat {
    let len = q.iter().map(|c| c * c).sum::<f32>().sqrt();
    if len < 1e-8 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    q.map(|c| c / len)
}

pub fn quat_from_axis_angle(axis: Vec3, angle: f32) -> Quat {
    let axis = axis.normalized();
    let (s, c) = (angle * 0.5).sin_cos();
    [axis.x * s, axis.y * s, axis.z * s, c]
}

/// Rotation part of `m`, ignoring scale (columns are normalized first).
pub fn quat_from_matrix(m: &Mat4) -> Quat {
    let cols = m.truncate().cols.map(|c| {
        let len = c.mag();
        if len > 1e-8 { c / len } else { c }
    });
    let r = Mat3::new(cols[0], cols[1], cols[2]);
    let (m00, m11, m22) = (r.cols[0].x, r.cols[1].y, r.cols[2].z);
    let trace = m00 + m11 + m22;

    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            (r.cols[1].z - r.cols[2].y) / s,
            (r.cols[2].x - r.cols[0].z) / s,
            (r.cols[0].y - r.cols[1].x) / s,
            0.25 * s,
        ]
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        [
            0.25 * s,
            (r.cols[1].x + r.cols[0].y) / s,
            (r.cols[2].x + r.cols[0].z) / s,
            (r.cols[1].z - r.cols[2].y) / s,
        ]
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        [
            (r.cols[1].x + r.cols[0].y) / s,
            0.25 * s,
            (r.cols[2].y + r.cols[1].z) / s,
            (r.cols[2].x - r.cols[0].z) / s,
        ]
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        [
            (r.cols[2].x + r.cols[0].z) / s,
            (r.cols[2].y + r.cols[1].z) / s,
            0.25 * s,
            (r.cols[0].y - r.cols[1].x) / s,
        ]
    };
    quat_normalize(q)
}

pub fn rotate(q: Quat, v: Vec3) -> Vec3 {
    let axis = Vec3::new(q[0], q[1], q[2]);
    let t = axis.cross(v) * 2.0;
    v + t * q[3] + axis.cross(t)
}

/// `T * R * S`, the glTF node transform.
pub fn trs(translation: [f32; 3], rotation: Quat, scale: [f32; 3]) -> Mat4 {
    let x = rotate(rotation, Vec3::unit_x()) * scale[0];
    let y = rotate(rotation, Vec3::unit_y()) * scale[1];
    let z = rotate(rotation, Vec3::unit_z()) * scale[2];
    Mat4::new(
        Vec4::new(x.x, x.y, x.z, 0.0),
        Vec4::new(y.x, y.y, y.z, 0.0),
        Vec4::new(z.x, z.y, z.z, 0.0),
        Vec4::new(translation[0], translation[1], translation[2], 1.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    fn same_rotation(a: Quat, b: Quat) -> bool {
        let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        close(dot.abs(), 1.0)
    }

    #[test]
    fn axis_angle_rotates_right_handed() {
        let q = quat_from_axis_angle(Vec3::unit_y(), std::f32::consts::FRAC_PI_2);
        let v = rotate(q, Vec3::unit_x());
        assert!(close(v.x, 0.0) && close(v.y, 0.0) && close(v.z, -1.0));
    }

    #[test]
    fn mul_composes_left_after_right() {
        let a = quat_from_axis_angle(Vec3::unit_x(), 0.7);
        let b = quat_from_axis_angle(Vec3::unit_z(), -1.3);
        let v = Vec3::new(0.3, -2.0, 1.5);
        let composed = rotate(quat_mul(a, b), v);
        let sequential = rotate(a, rotate(b, v));
        assert!((composed - sequential).mag() < 1e-4);
    }

    #[test]
    fn matrix_round_trip_ignores_scale() {
        let q = quat_normalize([0.2, -0.5, 0.4, 0.7]);
        let m = trs([1.0, 2.0, 3.0], q, [2.0, 0.5, 3.0]);
        assert!(same_rotation(quat_from_matrix(&m), q));
    }

    #[test]
    fn trs_matches_component_order() {
        let q = quat_from_axis_angle(Vec3::unit_z(), std::f32::consts::FRAC_PI_2);
        let m = trs([10.0, 0.0, 0.0], q, [2.0, 1.0, 1.0]);
        let p = m.transform_point3(Vec3::unit_x());
        assert!(close(p.x, 10.0) && close(p.y, 2.0) && close(p.z, 0.0));
    }
}
