use faer::{ColMut, ColRef};

#[inline]
pub fn quat_compose(q1: ColRef<f64>, q2: ColRef<f64>, mut q_out: ColMut<f64>) {
    q_out[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3];
    q_out[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2];
    q_out[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1];
    q_out[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0];
    let m = q_out.norm_l2();
    q_out[0] /= m;
    q_out[1] /= m;
    q_out[2] /= m;
    q_out[3] /= m;
}

#[inline]
pub fn quat_normalize(mut q: ColMut<f64>) {
    let m = q.norm_l2();
    if m == 0.0 {
        q[0] = 1.0;
        q[1] = 0.0;
        q[2] = 0.0;
        q[3] = 0.0;
        return;
    }
    if q[0] < 0.0 {
        q[0] /= -m;
        q[1] /= -m;
        q[2] /= -m;
        q[3] /= -m;
        return;
    }
    q[0] /= m;
    q[1] /= m;
    q[2] /= m;
    q[3] /= m;
}
