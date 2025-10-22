use crate::nwtc::{Quaternion, Vector3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Node {
    pub id: usize,
    pub x0: Vector3,
    pub ux: Vector3,
    pub vx: Vector3,
    pub ax: Vector3,
    pub r0: Quaternion,
    pub ur: Quaternion,
    pub vr: Vector3,
    pub ar: Vector3,
    pub f: Vector3,
    pub m: Vector3,
}

impl Node {
    pub fn new(id: usize, x0: Vector3, r0: Quaternion) -> Self {
        Self {
            id,
            x0,
            ux: Vector3::zero(),
            vx: Vector3::zero(),
            ax: Vector3::zero(),
            r0,
            ur: Quaternion::identity(),
            vr: Vector3::zero(),
            ar: Vector3::zero(),
            f: Vector3::zero(),
            m: Vector3::zero(),
        }
    }

    pub fn translate(&mut self, dux: Vector3) -> &mut Node {
        self.ux += dux;
        self
    }

    pub fn rotate(&mut self, dq: Quaternion) {
        self.ur = dq * self.ur;
    }

    pub fn x(&self) -> Vector3 {
        self.x0 + self.ux
    }

    pub fn r(&self) -> Quaternion {
        self.ur * self.r0
    }

    pub fn set_x(&mut self, x: Vector3) -> &mut Node {
        self.ux = x - self.x0;
        self
    }

    pub fn set_r(&mut self, q: Quaternion) -> &mut Node {
        self.ur = self.r0.inverse() * q;
        self
    }

    pub fn copy_motion_from(&mut self, other: &Node) -> &mut Self {
        self.x0 = other.x0;
        self.ux = other.ux;
        self.vx = other.vx;
        self.ax = other.ax;
        self.r0 = other.r0;
        self.ur = other.ur;
        self.vr = other.vr;
        self.ar = other.ar;
        self
    }

    pub fn reset_loads(&mut self) -> &mut Self {
        self.f = Vector3::zero();
        self.m = Vector3::zero();
        self
    }
}
