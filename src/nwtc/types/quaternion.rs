use crate::nwtc::{matrix::Matrix3, vector::Vector3};

// https://academicflight.com/articles/kinematics/rotation-formalisms/rotation-matrix/
// Quaternions represent a rotation from body-fixed frame to inertial frame

/// A quaternion for representing 3D rotations.
///
/// Quaternions provide a compact and numerically stable way to represent rotations
/// in 3D space. They avoid gimbal lock and are efficient for composition of rotations.
/// This implementation represents rotations from a body-fixed frame to an inertial frame.
///
/// A quaternion consists of a scalar part (w) and a vector part (x, y, z), often
/// written as q = w + xi + yj + zk, where i, j, k are the fundamental quaternion units.
///
/// For unit quaternions representing rotations:
/// - w = cos(θ/2) where θ is the rotation angle
/// - (x, y, z) = sin(θ/2) * axis where axis is the unit rotation axis
///
/// # Examples
///
/// ```
/// use fastr::nwtc::Quaternion;
/// use fastr::nwtc::Vector3;
/// use std::f64::consts::PI;
///
/// // Create identity quaternion (no rotation)
/// let identity = Quaternion::identity();
///
/// // Create quaternion from rotation vector (axis-angle representation)
/// let rotation_vector = Vector3::new(0.0, 0.0, PI/2.0); // 90° around Z-axis
/// let q = Quaternion::from_vector(rotation_vector);
///
/// // Rotate a vector
/// let vector = Vector3::new(1.0, 0.0, 0.0);
/// let rotated = q.rotate_vector(&vector);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    /// The scalar (real) component of the quaternion
    pub w: f64,
    /// The i component of the quaternion vector part
    pub x: f64,
    /// The j component of the quaternion vector part
    pub y: f64,
    /// The k component of the quaternion vector part
    pub z: f64,
}

impl Quaternion {
    /// Creates a new quaternion with the given components.
    ///
    /// # Arguments
    ///
    /// * `w` - The scalar (real) component
    /// * `x` - The i component of the vector part
    /// * `y` - The j component of the vector part
    /// * `z` - The k component of the vector part
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::nwtc::Quaternion;
    ///
    /// let q = Quaternion::new(1.0, 0.0, 0.0, 0.0); // Identity quaternion
    /// ```
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    /// Creates the identity quaternion representing no rotation.
    ///
    /// The identity quaternion has w = 1 and x = y = z = 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::nwtc::Quaternion;
    /// use fastr::nwtc::Vector3;
    /// let identity = Quaternion::identity();
    /// let vector = Vector3::new(1.0, 2.0, 3.0);
    /// let rotated = identity.rotate_vector(&vector);
    /// // rotated == vector (no rotation applied)
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Creates a quaternion from a 4-element array [w, x, y, z].
    ///
    /// # Arguments
    ///
    /// * `arr` - Array containing [w, x, y, z] components
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::nwtc::Quaternion;
    ///
    /// let q = Quaternion::from_array([1.0, 0.0, 0.0, 0.0]);
    /// assert_eq!(q.w, Quaternion::identity().w);
    /// assert_eq!(q.x, Quaternion::identity().x);
    /// assert_eq!(q.y, Quaternion::identity().y);
    /// assert_eq!(q.z, Quaternion::identity().z);
    /// ```
    #[inline]
    pub fn from_array(arr: [f64; 4]) -> Self {
        Self {
            w: arr[0],
            x: arr[1],
            y: arr[2],
            z: arr[3],
        }
    }

    /// Converts the quaternion to a 4-element array [w, x, y, z].
    ///
    /// # Returns
    ///
    /// An array containing the quaternion components in order [w, x, y, z]
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::nwtc::Quaternion;
    ///
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let arr = q.to_array();
    /// assert_eq!(arr, [1.0, 2.0, 3.0, 4.0]);
    /// ```
    #[inline]
    pub fn to_array(&self) -> [f64; 4] {
        [self.w, self.x, self.y, self.z]
    }

    /// Calculates the norm (magnitude) of the quaternion.
    ///
    /// The norm is calculated as √(w² + x² + y² + z²).
    /// For unit quaternions representing rotations, this should be 1.0.
    ///
    /// # Returns
    ///
    /// The magnitude of the quaternion
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::nwtc::Quaternion;
    ///
    /// let q = Quaternion::identity();
    /// assert_eq!(q.norm(), 1.0);
    /// ```
    #[inline]
    pub fn norm(&self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Returns a normalized (unit) quaternion.
    ///
    /// Normalizing ensures the quaternion has magnitude 1, which is required
    /// for representing pure rotations. If the quaternion has near-zero magnitude,
    /// returns the identity quaternion to avoid division by zero.
    ///
    /// # Returns
    ///
    /// A unit quaternion in the same direction, or identity if input is near-zero
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::nwtc::Quaternion;
    ///
    /// let q = Quaternion::new(2.0, 0.0, 0.0, 0.0);
    /// let normalized = q.normalize();
    /// assert!((normalized.norm() - 1.0).abs() < f64::EPSILON);
    /// ```
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

    /// Returns the inverse (conjugate) of the quaternion.
    ///
    /// For unit quaternions, the inverse equals the conjugate: q* = (w, -x, -y, -z).
    /// The inverse quaternion represents the opposite rotation.
    ///
    /// # Returns
    ///
    /// The inverse quaternion
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::nwtc::Quaternion;
    ///
    /// let q = Quaternion::new(0.7071, 0.7071, 0.0, 0.0); // 90° around X
    /// let q_inv = q.inverse();
    /// let identity = q * q_inv; // Should be close to identity
    /// ```
    pub fn inverse(&self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Composes this quaternion with another quaternion.
    ///
    /// Quaternion composition represents the application of multiple rotations.
    /// The result represents first applying this rotation, then the other rotation.
    /// This operation is equivalent to quaternion multiplication.
    ///
    /// # Arguments
    ///
    /// * `other` - The quaternion to compose with this one
    ///
    /// # Returns
    ///
    /// A new quaternion representing the composed rotation
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::nwtc::Quaternion;
    /// use fastr::nwtc::Vector3;
    /// use std::f64::consts::PI;
    ///
    /// let q1 = Quaternion::from_vector(Vector3::new(PI/2., 0.0, 0.0)); // 90° around X
    /// let q2 = Quaternion::from_vector(Vector3::new(0.0, PI/2., 0.0)); // 90° around Y
    /// let composed = q1.compose(&q2);
    /// ```
    pub fn compose(&self, other: &Self) -> Self {
        Self {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }

    /// Rotates a vector using this quaternion.
    ///
    /// This applies the rotation represented by this quaternion to the given vector.
    /// The operation is performed using the formula: v' = q * v * q^(-1),
    /// where v is treated as a pure quaternion (0, vx, vy, vz).
    ///
    /// # Arguments
    ///
    /// * `v` - The vector to rotate
    ///
    /// # Returns
    ///
    /// The rotated vector
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::nwtc::Quaternion;
    /// use fastr::nwtc::Vector3;
    /// use std::f64::consts::PI;
    ///
    /// let q = Quaternion::from_vector(Vector3::new(0.0, 0.0, PI/2.)); // 90° around Z
    /// let v = Vector3::new(1.0, 0.0, 0.0); // X-axis vector
    /// let rotated = q.rotate_vector(&v);
    /// // rotated should be approximately (0, 1, 0) - Y-axis vector
    /// ```
    pub fn rotate_vector(&self, v: &Vector3) -> Vector3 {
        let qv = Quaternion::new(0.0, v.x, v.y, v.z);
        let rotated = (*self * qv) * self.inverse();
        Vector3::new(rotated.x, rotated.y, rotated.z)
    }

    /// Creates a quaternion from a rotation vector (axis-angle representation).
    ///
    /// The rotation vector encodes both the axis of rotation (as the vector direction)
    /// and the angle of rotation (as the vector magnitude). This is also known as
    /// the axis-angle representation.
    ///
    /// # Arguments
    ///
    /// * `rv` - The rotation vector where magnitude is angle and direction is axis
    ///
    /// # Returns
    ///
    /// A unit quaternion representing the rotation
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::nwtc::{Quaternion, Vector3};
    /// use std::f64::consts::PI;
    ///
    /// let rotation_vector = Vector3::new(PI/2., 0.0, 0.0); // 90° around X-axis
    /// let q = Quaternion::from_vector(rotation_vector);
    /// ```
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

    /// Converts the quaternion to a rotation vector (axis-angle representation).
    ///
    /// The returned vector's direction represents the rotation axis and its magnitude
    /// represents the rotation angle in radians.
    ///
    /// # Returns
    ///
    /// A rotation vector representing the same rotation
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::nwtc::Quaternion;
    ///
    /// let q = Quaternion::new(0.7071, 0.7071, 0.0, 0.0); // 90° around X
    /// let rotation_vector = q.to_vector();
    /// // rotation_vector should be approximately (PI/2., 0, 0)
    /// ```
    pub fn to_vector(self) -> Vector3 {
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

    /// Creates a quaternion from a rotation matrix.
    ///
    /// Converts a 3x3 rotation matrix to its equivalent quaternion representation.
    /// This uses Shepperd's method for numerical stability, choosing the computation
    /// path based on the matrix trace and diagonal elements.
    ///
    /// # Arguments
    ///
    /// * `m` - The 3x3 rotation matrix to convert
    ///
    /// # Returns
    ///
    /// A unit quaternion representing the same rotation
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::nwtc::{Matrix3, Quaternion};
    ///
    /// // 90° rotation around Z-axis
    /// let matrix = Matrix3::new([
    ///     [0.0, -1.0, 0.0],
    ///     [1.0,  0.0, 0.0],
    ///     [0.0,  0.0, 1.0]
    /// ]);
    /// let q = Quaternion::from_matrix(&matrix);
    /// ```
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

    /// Converts the quaternion to a rotation matrix.
    ///
    /// Returns a 3x3 rotation matrix that represents the same rotation as this quaternion.
    /// The matrix can be used to rotate vectors by matrix multiplication.
    ///
    /// # Returns
    ///
    /// A 3x3 rotation matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::nwtc::Quaternion;
    /// use fastr::nwtc::Vector3;
    /// use std::f64::consts::PI;
    ///
    /// let q = Quaternion::from_vector(Vector3::new(0.0, 0.0, PI/2.)); // 90° around Z
    /// let matrix = q.to_matrix();
    /// let vector = Vector3::new(1.0, 0.0, 0.0);
    /// let rotated = matrix * vector; // Should give (0, 1, 0)
    /// ```
    pub fn to_matrix(&self) -> Matrix3 {
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

    /// Calculates the dot product of two quaternions.
    ///
    /// The dot product is useful for measuring the similarity between two rotations
    /// and is used internally by other functions like `angle_to` and `slerp`.
    ///
    /// # Arguments
    ///
    /// * `other` - The other quaternion to compute the dot product with
    ///
    /// # Returns
    ///
    /// The dot product as an f64 value between -1.0 and 1.0 for unit quaternions
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::nwtc::Quaternion;
    ///
    /// let q1 = Quaternion::identity();
    /// let q2 = Quaternion::identity();
    /// let dot = q1.dot(&q2);
    /// assert_eq!(dot, 1.0); // Same rotation
    /// ```
    pub fn dot(&self, other: &Self) -> f64 {
        self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Calculates the angle between two quaternions.
    ///
    /// Returns the angular distance between the rotations represented by this
    /// quaternion and the other quaternion. The result is always positive
    /// and represents the shortest rotation between the two orientations.
    ///
    /// # Arguments
    ///
    /// * `other` - The other quaternion to measure the angle to
    ///
    /// # Returns
    ///
    /// The angle in radians between 0.0 and π
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::nwtc::Quaternion;
    /// use fastr::nwtc::Vector3;
    /// use std::f64::consts::PI;
    ///
    /// let q1 = Quaternion::identity();
    /// let q2 = Quaternion::from_vector(Vector3::new(PI/2., 0.0, 0.0)); // 90° around X
    /// let angle = q1.angle_to(&q2);
    /// assert!((angle - PI/2.).abs() < 1e-10);
    /// ```
    pub fn angle_to(&self, other: &Self) -> f64 {
        let dot = self.dot(other).abs().clamp(0.0, 1.0);
        2.0 * dot.acos()
    }

    /// Spherical linear interpolation between two quaternions.
    ///
    /// SLERP provides smooth interpolation between two rotations along the shortest
    /// path on the 4D unit sphere. This is the preferred method for animating
    /// rotations as it maintains constant angular velocity.
    ///
    /// # Arguments
    ///
    /// * `other` - The target quaternion to interpolate towards
    /// * `t` - The interpolation parameter between 0.0 and 1.0
    ///   - t = 0.0 returns this quaternion
    ///   - t = 1.0 returns the other quaternion
    ///   - Values between 0.0 and 1.0 return interpolated rotations
    ///
    /// # Returns
    ///
    /// The interpolated quaternion
    ///
    /// # Examples
    ///
    /// ```
    /// use fastr::nwtc::Quaternion;
    /// use fastr::nwtc::Vector3;
    /// use std::f64::consts::PI;
    ///
    /// let q1 = Quaternion::identity();
    /// let q2 = Quaternion::from_vector(Vector3::new(PI/2., 0.0, 0.0)); // 90° around X
    /// let halfway = q1.slerp(&q2, 0.5); // 45° rotation around X
    /// ```
    pub fn slerp(&self, other: &Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);

        if t == 0.0 {
            return *self;
        }
        if t == 1.0 {
            return *other;
        }

        let mut dot = self.dot(other);

        // Choose the shortest path by flipping one quaternion if needed
        let other = if dot < 0.0 {
            dot = -dot;
            Quaternion::new(-other.w, -other.x, -other.y, -other.z)
        } else {
            *other
        };

        // If quaternions are very close, use linear interpolation to avoid division by zero
        if dot > 0.9995 {
            let result = Quaternion::new(
                self.w + t * (other.w - self.w),
                self.x + t * (other.x - self.x),
                self.y + t * (other.y - self.y),
                self.z + t * (other.z - self.z),
            );
            return result.normalize();
        }

        // Calculate the angle and perform spherical interpolation
        let theta_0 = dot.acos();
        let sin_theta_0 = theta_0.sin();
        let theta = theta_0 * t;
        let sin_theta = theta.sin();

        let s0 = (theta_0 - theta).cos() - dot * sin_theta / sin_theta_0;
        let s1 = sin_theta / sin_theta_0;

        Quaternion::new(
            s0 * self.w + s1 * other.w,
            s0 * self.x + s1 * other.x,
            s0 * self.y + s1 * other.y,
            s0 * self.z + s1 * other.z,
        )
    }
}

/// Implementation of quaternion multiplication for Quaternion.
///
/// Quaternion multiplication represents composition of rotations. The operation
/// is non-commutative, meaning q1 * q2 ≠ q2 * q1 in general.
///
/// This delegates to the `compose` method for the actual implementation.
///
/// # Examples
///
/// ```
/// use fastr::nwtc::{Quaternion, Vector3};
/// use std::f64::consts::PI;
///
/// let q1 = Quaternion::from_vector(Vector3::new(PI/4., 0.0, 0.0));
/// let q2 = Quaternion::from_vector(Vector3::new(0.0, PI/4., 0.0));
/// let composed = q1 * q2; // Apply q1 rotation, then q2 rotation
/// ```
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
    fn test_to_array() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let arr = q.to_array();
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
        let result = (q_delta * q_base).to_vector();
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
    fn test_to_vector() {
        // Create a quaternion representing rotation around Y-axis
        let q = Quaternion::new(0.7071067811865476, 0.0, 0.7071067811865475, 0.0); // cos(pi/4), 0, sin(pi/4), 0
        let rv = q.to_vector();
        assert_vector3_eq(rv, Vector3::new(0.0, PI / 2.0, 0.0));

        // Test identity
        let identity_rv = Quaternion::identity().to_vector();
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
    fn test_to_matrix() {
        // 90 degrees around X axis
        let q = Quaternion::new(0.7071067811865476, 0.7071067811865475, 0.0, 0.0);
        let matrix = q.to_matrix();
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
        let recovered_rv = q.to_vector();
        assert_vector3_eq(original_rv, recovered_rv);
    }

    #[test]
    fn test_roundtrip_rotation_matrix() {
        let original_matrix =
            Matrix3::new([[0.36, 0.48, -0.8], [-0.8, 0.6, 0.0], [0.48, 0.64, 0.6]]);
        let q = Quaternion::from_matrix(&original_matrix);
        let recovered_matrix = q.to_matrix();

        let diff = original_matrix - recovered_matrix;
        for i in 0..3 {
            for j in 0..3 {
                assert!(diff.get(i, j).abs() < EPSILON);
            }
        }
    }

    #[test]
    fn test_dot() {
        let q1 = Quaternion::identity();
        let q2 = Quaternion::identity();
        assert_eq!(q1.dot(&q2), 1.0);

        // Test dot product of two 90° rotations around X-axis in opposite directions
        let q3 = Quaternion::new(0.7071067811865476, 0.7071067811865475, 0.0, 0.0); // +90° around X
        let q4 = Quaternion::new(0.7071067811865476, -0.7071067811865475, 0.0, 0.0); // -90° around X
        let expected_dot =
            0.7071067811865476 * 0.7071067811865476 + 0.7071067811865475 * (-0.7071067811865475);
        assert!((q3.dot(&q4) - expected_dot).abs() < EPSILON);

        // Test orthogonal quaternions (90° around different axes)
        let qx = Quaternion::from_vector(Vector3::new(PI / 2.0, 0.0, 0.0)); // 90° around X
        let qy = Quaternion::from_vector(Vector3::new(0.0, PI / 2.0, 0.0)); // 90° around Y
        let dot_orthogonal = qx.dot(&qy);
        assert!((dot_orthogonal - 0.5).abs() < EPSILON); // cos(π/2) for half-angle = cos(π/4) = √2/2 ≈ 0.707, but for quaternions it's different
    }

    #[test]
    fn test_angle_to() {
        let q1 = Quaternion::identity();
        let q2 = Quaternion::from_vector(Vector3::new(PI / 2.0, 0.0, 0.0)); // 90° around X
        let angle = q1.angle_to(&q2);
        assert!((angle - PI / 2.0).abs() < EPSILON);

        // Test angle between identical quaternions
        let angle_same = q1.angle_to(&q1);
        assert!(angle_same.abs() < EPSILON);

        // Test angle between opposite rotations
        let q3 = Quaternion::from_vector(Vector3::new(PI, 0.0, 0.0)); // 180° around X
        let angle_opposite = q1.angle_to(&q3);
        assert!((angle_opposite - PI).abs() < EPSILON);
    }

    #[test]
    fn test_slerp() {
        let q1 = Quaternion::identity();
        let q2 = Quaternion::from_vector(Vector3::new(PI / 2.0, 0.0, 0.0)); // 90° around X

        // Test endpoints
        let start = q1.slerp(&q2, 0.0);
        assert_quaternion_eq(start, q1);

        let end = q1.slerp(&q2, 1.0);
        assert_quaternion_eq(end, q2);

        // Test midpoint
        let mid = q1.slerp(&q2, 0.5);
        let expected_angle = PI / 4.0; // 45° around X
        let expected_mid = Quaternion::from_vector(Vector3::new(expected_angle, 0.0, 0.0));
        assert_quaternion_eq(mid, expected_mid);

        // Test that interpolated quaternion is normalized
        assert!((mid.norm() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_slerp_close_quaternions() {
        // Test SLERP with very close quaternions (should use linear interpolation path)
        let q1 = Quaternion::identity();
        let small_rotation = Vector3::new(0.001, 0.0, 0.0); // Very small rotation
        let q2 = Quaternion::from_vector(small_rotation);

        let mid = q1.slerp(&q2, 0.5);
        assert!((mid.norm() - 1.0).abs() < EPSILON);

        // The result should be approximately halfway
        let mid_angle = q1.angle_to(&mid);
        let total_angle = q1.angle_to(&q2);
        assert!((mid_angle - total_angle * 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_slerp_shortest_path() {
        // Test that SLERP takes the shortest path (flips quaternion if needed)
        let q1 = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let q2 = Quaternion::new(-1.0, 0.0, 0.0, 0.0); // Represents same rotation but opposite sign

        let mid = q1.slerp(&q2, 0.5);

        // The result should be very close to the original (since they represent the same rotation)
        let angle = q1.angle_to(&mid);
        assert!(angle < 0.1); // Should be very small angle
    }
}
