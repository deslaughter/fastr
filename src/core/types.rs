use faer::{col, mat, Col, Mat};
use glam::{DMat3, DQuat, DVec3, DVec4};

pub use serde::{Deserialize, Serialize};

pub type Quaternion = DQuat;
pub type Vector3 = DVec3;
pub type Vector4 = DVec4;
pub type Matrix3 = DMat3;

pub trait Vector3Ext {
    fn skew_symmetric(&self) -> Matrix3;
    fn into_faer(&self) -> Col<f64>;
}

impl Vector3Ext for Vector3 {
    fn skew_symmetric(&self) -> Matrix3 {
        DMat3::from_cols(
            Vector3::new(0.0, self.z, -self.y),
            Vector3::new(-self.z, 0.0, self.x),
            Vector3::new(self.y, -self.x, 0.0),
        )
    }
    fn into_faer(&self) -> Col<f64> {
        col![self.x, self.y, self.z]
    }
}

pub trait Matrix3Ext {
    fn into_faer(&self) -> Mat<f64>;
}

impl Matrix3Ext for Matrix3 {
    fn into_faer(&self) -> Mat<f64> {
        mat![
            [self.x_axis.x, self.y_axis.x, self.z_axis.x],
            [self.x_axis.y, self.y_axis.y, self.z_axis.y],
            [self.x_axis.z, self.y_axis.z, self.z_axis.z],
        ]
    }
}
