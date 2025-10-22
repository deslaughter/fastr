use crate::nwtc::{matrix::Matrix3, vector::Vector3};

// https://academicflight.com/articles/kinematics/rotation-formalisms/rotation-matrix/
// Quaternions represent a rotation from body-fixed frame to inertial frame

/// Quaternion operations for 3D rotations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    #[inline]
    pub fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    #[inline]
    pub fn from_array(arr: [f64; 4]) -> Self {
        Self {
            w: arr[0],
            x: arr[1],
            y: arr[2],
            z: arr[3],
        }
    }

    #[inline]
    pub fn as_array(&self) -> [f64; 4] {
        [self.w, self.x, self.y, self.z]
    }

    #[inline]
    pub fn norm(&self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    #[inline]
    pub fn normalize(&self) -> Self {
        let norm = self.norm();
        if norm < f64::EPSILON {
            Self::identity()
        } else {
            Self {
                w: self.w / norm,
                x: self.x / norm,
                y: self.y / norm,
                z: self.z / norm,
            }
        }
    }

    pub fn inverse(&self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    pub fn compose(&self, other: &Self) -> Self {
        Self {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }

    pub fn rotate_vector(&self, v: &Vector3) -> Vector3 {
        let v_as_quat = Quaternion::new(0.0, v.x, v.y, v.z);
        let rotated = (*self * v_as_quat) * self.inverse();
        Vector3::new(rotated.x, rotated.y, rotated.z)
    }

    pub fn from_vector(rv: Vector3) -> Self {
        let theta = (rv.x * rv.x + rv.y * rv.y + rv.z * rv.z).sqrt();
        if theta < f64::EPSILON {
            Self::identity()
        } else {
            let half_theta = theta / 2.0;
            let (sin_half_theta, cos_half_theta) = half_theta.sin_cos();
            Self {
                w: cos_half_theta,
                x: rv.x * sin_half_theta / theta,
                y: rv.y * sin_half_theta / theta,
                z: rv.z * sin_half_theta / theta,
            }
        }
    }

    pub fn as_vector(self) -> Vector3 {
        let qw = self.w.clamp(-1.0, 1.0);
        let theta = 2.0 * qw.acos();
        let sin_half_theta = (1.0 - qw * qw).sqrt();

        if sin_half_theta.abs() < f64::EPSILON {
            Vector3::zero()
        } else {
            Vector3::new(
                self.x * theta / sin_half_theta,
                self.y * theta / sin_half_theta,
                self.z * theta / sin_half_theta,
            )
        }
    }

    pub fn from_matrix(m: &Matrix3) -> Self {
        let m = &m.data;
        let trace = m[0][0] + m[1][1] + m[2][2];
        if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            Self {
                w: 0.25 * s,
                x: (m[2][1] - m[1][2]) / s,
                y: (m[0][2] - m[2][0]) / s,
                z: (m[1][0] - m[0][1]) / s,
            }
        } else if (m[0][0] > m[1][1]) && (m[0][0] > m[2][2]) {
            let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
            Self {
                w: (m[2][1] - m[1][2]) / s,
                x: 0.25 * s,
                y: (m[0][1] + m[1][0]) / s,
                z: (m[0][2] + m[2][0]) / s,
            }
        } else if m[1][1] > m[2][2] {
            let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
            Self {
                w: (m[0][2] - m[2][0]) / s,
                x: (m[0][1] + m[1][0]) / s,
                y: 0.25 * s,
                z: (m[1][2] + m[2][1]) / s,
            }
        } else {
            let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
            Self {
                w: (m[1][0] - m[0][1]) / s,
                x: (m[0][2] + m[2][0]) / s,
                y: (m[1][2] + m[2][1]) / s,
                z: 0.25 * s,
            }
        }
    }

    pub fn as_matrix(&self) -> Matrix3 {
        Matrix3 {
            data: [
                [
                    1.0 - 2.0 * (self.y * self.y + self.z * self.z),
                    2.0 * (self.x * self.y - self.z * self.w),
                    2.0 * (self.x * self.z + self.y * self.w),
                ],
                [
                    2.0 * (self.x * self.y + self.z * self.w),
                    1.0 - 2.0 * (self.x * self.x + self.z * self.z),
                    2.0 * (self.y * self.z - self.x * self.w),
                ],
                [
                    2.0 * (self.x * self.z - self.y * self.w),
                    2.0 * (self.y * self.z + self.x * self.w),
                    1.0 - 2.0 * (self.x * self.x + self.y * self.y),
                ],
            ],
        }
    }
}

impl std::ops::Mul for Quaternion {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.compose(&other)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    const EPSILON: f64 = 1e-12;
    use std::f64::consts::PI;

    fn assert_quaternion_eq(act: Quaternion, exp: Quaternion) {
        assert_relative_eq!(act.w, exp.w, epsilon = EPSILON);
        assert_relative_eq!(act.x, exp.x, epsilon = EPSILON);
        assert_relative_eq!(act.y, exp.y, epsilon = EPSILON);
        assert_relative_eq!(act.z, exp.z, epsilon = EPSILON);
    }

    fn assert_vector3_eq(act: Vector3, exp: Vector3) {
        assert_relative_eq!(act.x, exp.x, epsilon = EPSILON);
        assert_relative_eq!(act.y, exp.y, epsilon = EPSILON);
        assert_relative_eq!(act.z, exp.z, epsilon = EPSILON);
    }

    #[test]
    fn test_identity() {
        let q = Quaternion::identity();
        assert_eq!(q.w, 1.0);
        assert_eq!(q.x, 0.0);
        assert_eq!(q.y, 0.0);
        assert_eq!(q.z, 0.0);
    }

    #[test]
    fn test_from_array() {
        let arr = [1.0, 2.0, 3.0, 4.0];
        let q = Quaternion::from_array(arr);
        assert_eq!(q.w, 1.0);
        assert_eq!(q.x, 2.0);
        assert_eq!(q.y, 3.0);
        assert_eq!(q.z, 4.0);
    }

    #[test]
    fn test_as_array() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let arr = q.as_array();
        assert_eq!(arr, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_norm() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        assert!((q.norm() - 5.477225575051661).abs() < EPSILON);
    }

    #[test]
    fn test_normalize() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let normalized = q.normalize();
        assert!((normalized.norm() - 1.0).abs() < EPSILON);

        // Test normalization of near-zero quaternion
        let q_small = Quaternion::new(0.0, 1e-16, 1e-16, 1e-16);
        let normalized_small = q_small.normalize();
        assert_quaternion_eq(normalized_small, Quaternion::identity());
    }

    #[test]
    fn test_inverse() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let inv = q.inverse();
        assert_eq!(inv.w, 1.0);
        assert_eq!(inv.x, -2.0);
        assert_eq!(inv.y, -3.0);
        assert_eq!(inv.z, -4.0);
    }

    #[test]
    fn test_compose() {
        let q_base = Quaternion::from_vector(Vector3::new(PI / 6., PI / 3., 0.)); // base
        let q_delta = Quaternion::from_vector(Vector3::new(PI / 4., 0., 0.)); // delta
        let result = (q_delta * q_base).as_vector();
        let expected = Vector3::new(1.2307715233860903, 1.0268512702231527, 0.4253357226664698);
        assert_vector3_eq(result, expected);
    }

    #[test]
    fn test_rotate_vector() {
        // Test 90-degree rotation around Z-axis
        let q = Quaternion::from_vector(Vector3::new(0., 0., PI / 2.));
        let v = Vector3::new(1.0, 0.0, 0.0);
        let rotated = q.rotate_vector(&v);
        assert_vector3_eq(rotated, Vector3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn test_from_rotation_vector() {
        // Test rotation around X axis by 90 degrees
        let rv = Vector3::new(PI / 2.0, 0.0, 0.0);
        let q = Quaternion::from_vector(rv);
        assert!((q.w - 0.7071067811865476).abs() < EPSILON);
        assert!((q.x - 0.7071067811865475).abs() < EPSILON);
        assert!((q.y - 0.0).abs() < EPSILON);
        assert!((q.z - 0.0).abs() < EPSILON);

        // Test zero rotation
        let zero_rv = Vector3::zero();
        let zero_q = Quaternion::from_vector(zero_rv);
        assert_quaternion_eq(zero_q, Quaternion::identity());
    }

    #[test]
    fn test_as_rotation_vector() {
        // Create a quaternion representing rotation around Y-axis
        let q = Quaternion::new(0.7071067811865476, 0.0, 0.7071067811865475, 0.0); // cos(pi/4), 0, sin(pi/4), 0
        let rv = q.as_vector();
        assert_vector3_eq(rv, Vector3::new(0.0, PI / 2.0, 0.0));

        // Test identity
        let identity_rv = Quaternion::identity().as_vector();
        assert_vector3_eq(identity_rv, Vector3::zero());
    }

    #[test]
    fn test_from_rotation_matrix() {
        // Create a rotation matrix for 90 degrees around Z axis
        let matrix = Matrix3::new([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
        let q = Quaternion::from_matrix(&matrix);
        let expected = Quaternion::new(0.7071067811865476, 0.0, 0.0, 0.7071067811865475);
        assert_quaternion_eq(q, expected);
    }

    #[test]
    fn test_as_rotation_matrix() {
        // 90 degrees around X axis
        let q = Quaternion::new(0.7071067811865476, 0.7071067811865475, 0.0, 0.0);
        let matrix = q.as_matrix();
        let expected = Matrix3::new([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]);

        for i in 0..3 {
            for j in 0..3 {
                assert!((matrix.get(i, j) - expected.get(i, j)).abs() < EPSILON);
            }
        }
    }

    #[test]
    fn test_roundtrip_rotation_vector() {
        let original_rv = Vector3::new(0.1, 0.2, 0.3);
        let q = Quaternion::from_vector(original_rv);
        let recovered_rv = q.as_vector();
        assert_vector3_eq(original_rv, recovered_rv);
    }

    #[test]
    fn test_roundtrip_rotation_matrix() {
        let original_matrix =
            Matrix3::new([[0.36, 0.48, -0.8], [-0.8, 0.6, 0.0], [0.48, 0.64, 0.6]]);
        let q = Quaternion::from_matrix(&original_matrix);
        let recovered_matrix = q.as_matrix();

        let diff = original_matrix - recovered_matrix;
        for i in 0..3 {
            for j in 0..3 {
                assert!(diff.get(i, j).abs() < EPSILON);
            }
        }
    }
}
