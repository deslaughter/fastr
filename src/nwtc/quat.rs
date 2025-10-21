use ndarray::prelude::*;

pub struct Quaternion {
    w: f64,
    i: f64,
    j: f64,
    k: f64,
}

impl Quaternion {
    pub fn new(w: f64, i: f64, j: f64, k: f64) -> Self {
        Self { w, i, j, k }
    }

    pub fn from_axis_angle(axis: &[f64; 3], angle: f64) -> Self {
        let half_angle = angle / 2.0;
        let sin_half_angle = half_angle.sin();
        Self {
            w: half_angle.cos(),
            i: axis[0] * sin_half_angle,
            j: axis[1] * sin_half_angle,
            k: axis[2] * sin_half_angle,
        }
    }

    pub fn from_rotation_vector(rvec: &[f64; 3]) -> Self {
        let theta = (rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]).sqrt();
        if theta < 1e-12 {
            Self {
                w: 1.0,
                i: 0.0,
                j: 0.0,
                k: 0.0,
            }
        } else {
            let half_theta = theta / 2.0;
            let sin_half_theta = half_theta.sin();
            Self {
                w: half_theta.cos(),
                i: rvec[0] * sin_half_theta / theta,
                j: rvec[1] * sin_half_theta / theta,
                k: rvec[2] * sin_half_theta / theta,
            }
        }
    }

    pub fn to_rotation_vector(&self) -> [f64; 3] {
        let angle = 2.0 * self.w.acos();
        let s = (1.0 - self.w * self.w).sqrt();
        if s < 1e-12 {
            [0.0, 0.0, 0.0]
        } else {
            [angle * self.i / s, angle * self.j / s, angle * self.k / s]
        }
    }

    pub fn to_rotation_matrix(&self) -> [[f64; 3]; 3] {
        let w2 = self.w * self.w;
        let i2 = self.i * self.i;
        let j2 = self.j * self.j;
        let k2 = self.k * self.k;

        [
            [
                w2 + i2 - j2 - k2,
                2.0 * (self.i * self.j - self.w * self.k),
                2.0 * (self.i * self.k + self.w * self.j),
            ],
            [
                2.0 * (self.i * self.j + self.w * self.k),
                w2 - i2 + j2 - k2,
                2.0 * (self.j * self.k - self.w * self.i),
            ],
            [
                2.0 * (self.i * self.k - self.w * self.j),
                2.0 * (self.j * self.k + self.w * self.i),
                w2 - i2 - j2 + k2,
            ],
        ]
    }

    pub fn from_rotation_matrix(m: [[f64; 3]; 3]) -> Self {
        let trace = m[0][0] + m[1][1] + m[2][2];
        if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            Self {
                w: 0.25 * s,
                i: (m[2][1] - m[1][2]) / s,
                j: (m[0][2] - m[2][0]) / s,
                k: (m[1][0] - m[0][1]) / s,
            }
        } else if (m[0][0] > m[1][1]) && (m[0][0] > m[2][2]) {
            let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
            Self {
                w: (m[2][1] - m[1][2]) / s,
                i: 0.25 * s,
                j: (m[0][1] + m[1][0]) / s,
                k: (m[0][2] + m[2][0]) / s,
            }
        } else if m[1][1] > m[2][2] {
            let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
            Self {
                w: (m[0][2] - m[2][0]) / s,
                i: (m[0][1] + m[1][0]) / s,
                j: 0.25 * s,
                k: (m[1][2] + m[2][1]) / s,
            }
        } else {
            let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
            Self {
                w: (m[1][0] - m[0][1]) / s,
                i: (m[0][2] + m[2][0]) / s,
                j: (m[1][2] + m[2][1]) / s,
                k: 0.25 * s,
            }
        }
    }

    pub fn normalize(&mut self) {
        let norm = (self.w * self.w + self.i * self.i + self.j * self.j + self.k * self.k).sqrt();
        if norm > 0.0 {
            self.w /= norm;
            self.i /= norm;
            self.j /= norm;
            self.k /= norm;
        }
    }

    pub fn compose(&self, other: &Quaternion) -> Quaternion {
        Quaternion {
            w: self.w * other.w - self.i * other.i - self.j * other.j - self.k * other.k,
            i: self.w * other.i + self.i * other.w + self.j * other.k - self.k * other.j,
            j: self.w * other.j - self.i * other.k + self.j * other.w + self.k * other.i,
            k: self.w * other.k + self.i * other.j - self.j * other.i + self.k * other.w,
        }
    }

    pub fn to_array(&self) -> [f64; 4] {
        [self.w, self.i, self.j, self.k]
    }

    pub fn from_array(arr: [f64; 4]) -> Self {
        Self {
            w: arr[0],
            i: arr[1],
            j: arr[2],
            k: arr[3],
        }
    }

    pub fn rotate_vector(&self, vec: &[f64; 3]) -> [f64; 3] {
        let q_vec = Quaternion {
            w: 0.0,
            i: vec[0],
            j: vec[1],
            k: vec[2],
        };
        let q_conj = Quaternion {
            w: self.w,
            i: -self.i,
            j: -self.j,
            k: -self.k,
        };
        let q_res = self.compose(&q_vec).compose(&q_conj);
        q_res.ijk()
    }

    pub fn ijk(&self) -> [f64; 3] {
        [self.i, self.j, self.k]
    }

    pub fn scalar(&self) -> f64 {
        self.w
    }
}

#[inline]
pub fn quat_scalar<T: NdFloat>(q: ArrayView1<T>) -> T {
    (T::one() - q.pow2().sum()).sqrt()
}

#[inline]
pub fn quat_compose<T: NdFloat>(q1: ArrayView1<T>, q2: ArrayView1<T>, mut q: ArrayViewMut1<T>) {
    let q1_0 = quat_scalar(q1);
    let q2_0 = quat_scalar(q2);
    let q0 = q1_0 * q2_0 - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2];
    q.assign(&ArrayView1::from(&[
        q1_0 * q2[0] + q1[0] * q2_0 + q1[1] * q2[2] - q1[2] * q2[1],
        q1_0 * q2[1] - q1[0] * q2[2] + q1[1] * q2_0 + q1[2] * q2[0],
        q1_0 * q2[2] + q1[0] * q2[1] - q1[1] * q2[0] + q1[2] * q2_0,
    ]));
    quat_normalize(q0, q)
}

#[inline]
pub fn quat_normalize<T: NdFloat>(q0: T, mut q: ArrayViewMut1<T>) {
    let m = (q0 * q0 + q.pow2().sum()).sqrt();
    if m < T::epsilon() {
        q.fill(T::zero());
    } else if q[0] < T::zero() {
        q.assign(&ArrayView1::from(&[-q[0] / m, -q[1] / m, -q[2] / m]));
    } else {
        q.assign(&ArrayView1::from(&[q[0] / m, q[1] / m, q[2] / m]));
    }
}

#[inline]
pub fn quat_from_rvec<T: NdFloat>(rvec: ArrayView1<T>, mut q: ArrayViewMut1<T>) {
    let theta = rvec.pow2().sum().sqrt();
    if theta < T::epsilon() {
        q.fill(T::zero());
    } else {
        let half_theta = theta / T::from(2.).unwrap();
        let sin_half_theta = half_theta.sin();
        let q0 = half_theta.cos();
        q[0] = rvec[0] * sin_half_theta / theta;
        q[1] = rvec[1] * sin_half_theta / theta;
        q[2] = rvec[2] * sin_half_theta / theta;
        if q0 < T::zero() {
            q *= -T::one();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_array_view() {
        let a = ArrayView1::from(&[1.0, 2.0, 30.0]);
        let b = a.to_slice().unwrap();
        assert_relative_eq!(*b, [1.0, 2.0, 30.0], epsilon = 1e-12);

        let mut c = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        c[1] = [7.0, 8.0, 9.0];
        assert_relative_eq!(c[1][0], 7.0, epsilon = 1e-12);

        let mut d = ArrayViewMut1::from(&mut c[1]);
        d.assign(&ArrayView::from(&[1.0, 5.0, 6.0]));
    }

    #[test]
    fn test_quat_scalar() {
        let q = array![0.5, 0.5, 0.5];
        let scalar = quat_scalar(q.view());
        assert_relative_eq!(scalar, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_quat_normalize() {
        let q0: f64 = 0.5;
        let mut q = array![0.5, 0.5, 0.5];
        quat_normalize(q0, q.view_mut());
        let norm = (q0 * q0 + q.dot(&q)).sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quat_from_rvec() {
        let rvec = array![0.1, 0.2, 0.3];
        let mut q = Array1::<f64>::zeros(3);
        quat_from_rvec(rvec.view(), q.view_mut());

        let theta = rvec.dot(&rvec).sqrt();
        let half_theta = theta / 2.0;
        let sin_half_theta = half_theta.sin();
        let q0 = half_theta.cos();

        assert_relative_eq!(q[0], rvec[0] * sin_half_theta / theta, epsilon = 1e-10);
        assert_relative_eq!(q[1], rvec[1] * sin_half_theta / theta, epsilon = 1e-10);
        assert_relative_eq!(q[2], rvec[2] * sin_half_theta / theta, epsilon = 1e-10);
    }

    #[test]
    fn test_quat_compose() {
        let q1 = array![0.1, 0.2, 0.3];
        let q2 = array![0.4, 0.5, 0.6];
        let mut q = Array1::<f64>::zeros(3);

        quat_compose(q1.view(), q2.view(), q.view_mut());

        let q1_0 = quat_scalar(q1.view());
        let q2_0 = quat_scalar(q2.view());

        let expected_q0 = q1_0 * q2_0 - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2];
        let mut expected_q = array![
            q1_0 * q2[0] + q1[0] * q2_0 + q1[1] * q2[2] - q1[2] * q2[1],
            q1_0 * q2[1] - q1[0] * q2[2] + q1[1] * q2_0 + q1[2] * q2[0],
            q1_0 * q2[2] + q1[0] * q2[1] - q1[1] * q2[0] + q1[2] * q2_0,
        ];

        quat_normalize(expected_q0, expected_q.view_mut());

        assert_relative_eq!(q[0], expected_q[0], epsilon = 1e-10);
        assert_relative_eq!(q[1], expected_q[1], epsilon = 1e-10);
        assert_relative_eq!(q[2], expected_q[2], epsilon = 1e-10);
    }

    #[test]
    fn test_quat_compose_identity() {
        // Identity quaternion (0,0,0) with scalar part 1
        let identity = array![0.0, 0.0, 0.0];
        let q = array![0.1, 0.2, 0.3];
        let mut result = Array1::<f64>::zeros(3);

        // q * identity = q
        quat_compose(q.view(), identity.view(), result.view_mut());
        assert_relative_eq!(result[0], q[0], epsilon = 1e-10);
        assert_relative_eq!(result[1], q[1], epsilon = 1e-10);
        assert_relative_eq!(result[2], q[2], epsilon = 1e-10);

        // identity * q = q
        quat_compose(identity.view(), q.view(), result.view_mut());
        assert_relative_eq!(result[0], q[0], epsilon = 1e-10);
        assert_relative_eq!(result[1], q[1], epsilon = 1e-10);
        assert_relative_eq!(result[2], q[2], epsilon = 1e-10);
    }
}
