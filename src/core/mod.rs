pub mod mesh;
pub mod types;

use bitflags::bitflags;
use faer::Mat;
pub use mesh::Mesh;
pub use types::*;

use crate::modules::ModuleId;

bitflags! {
    /// Represents which fields are active in a mesh.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
    pub struct Jacobian: u32 {
        const dXdx = 1 << 0;
        const dXdu = 1 << 1;
        const dYdx = 1 << 2;
        const dYdu = 1 << 3;
    }
}

#[repr(usize)]
pub enum InputIndex {
    Previous,
    Current,
    Next,
}

#[repr(usize)]
pub enum StateIndex {
    Previous,
    Current,
}

pub struct ModuleLinearization {
    pub a: Mat<f64>, // dXdx
    pub b: Mat<f64>, // dXdu
    pub c: Mat<f64>, // dYdx
    pub d: Mat<f64>, // dYdu
}

pub trait Module {
    fn id() -> ModuleId;
    fn name() -> &'static str;
    fn version() -> &'static str;
    fn n_instances(&self) -> usize;
    fn calc_output(&self) -> anyhow::Result<()>;
    fn update_states(&self) -> anyhow::Result<()>;
    fn get_operating_point(&self) -> anyhow::Result<()>;
    fn set_operating_point(&self) -> anyhow::Result<()>;
    fn calc_jacobian(
        &self,
        input: InputIndex,
        state: StateIndex,
        jacobians: Jacobian,
    ) -> anyhow::Result<()>;
}
