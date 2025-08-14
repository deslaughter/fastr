use ndarray::prelude::*;

#[inline]
pub fn quat_magnitude(q: ArrayView1<f64>) -> f64 {
    (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt()
}

#[inline]
pub fn quat_compose(q1: ArrayView1<f64>, q2: ArrayView1<f64>, mut q_out: ArrayViewMut1<f64>) {
    q_out[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3];
    q_out[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2];
    q_out[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1];
    q_out[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0];
    quat_normalize(q_out);
}

#[inline]
pub fn quat_normalize(mut q: ArrayViewMut1<f64>) {
    let m = q.pow2().sum().sqrt();
    if m < f64::EPSILON {
        q.fill(0.0);
        q[0] = 1.0;
    } else if q[0] < 0.0 {
        q /= -m;
    } else {
        q /= m;
    }
}

#[inline]
pub fn quat_from_rvec(rvec: ArrayView1<f64>) -> Array1<f64> {
    let theta = rvec.powi(2).sum().sqrt();
    let half_theta = theta / 2.0;
    let sin_half_theta = half_theta.sin();
    let mut q = Array1::zeros(4);
    q[0] = half_theta.cos();
    q[1] = rvec[0] * sin_half_theta / theta;
    q[2] = rvec[1] * sin_half_theta / theta;
    q[3] = rvec[2] * sin_half_theta / theta;
    q
}
