pub mod mesh;
pub mod types;

pub use mesh::Mesh;
pub use types::*;

use crate::modules::ModuleId;

pub trait Module {
    fn id() -> ModuleId;
    fn name() -> &'static str;
    fn version() -> &'static str;
    fn n_instances(&self) -> usize;
    fn calc_output(&self) -> anyhow::Result<()>;
    fn update_states(&self) -> anyhow::Result<()>;
    fn jacobian_p_input(&self) -> anyhow::Result<()>;
    fn jacobian_p_continuous_state(&self) -> anyhow::Result<()>;
    fn get_operating_point(&self) -> anyhow::Result<()>;
    fn set_operating_point(&self) -> anyhow::Result<()>;
}
