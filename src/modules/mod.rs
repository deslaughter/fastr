use std::collections::HashMap;

use crate::core::types::*;
use faer::prelude::*;

use crate::core::{mesh::Fields, Mesh};

pub mod elastodyn;
pub use elastodyn::ElastoDyn;

pub mod input;
pub use input::Input;

pub mod output;
pub use output::Output;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModuleId {
    None,
    Glue,
    ElastoDyn,
    BeamDyn,
    AeroDyn,
    ServoDyn,
    InflowWind,
    HydroDyn,
    SubDyn,
    MoorDyn,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StateType {
    TranslationalDisplacement,
    TranslationalVelocity,
    AngularDisplacement,
    AngularVelocity,
}

pub struct ModuleData {
    // Module identifier
    pub module_id: ModuleId,

    // Instance identifier
    pub instance: usize,

    // State vector
    pub x: Col<f64>,

    // Input vector
    pub u: Col<f64>,

    // Output vector
    pub y: Col<f64>,

    // Change in state with respect to state perturbations
    pub dxdx: Mat<f64>,

    // Change in output with respect to state perturbations
    pub dydx: Mat<f64>,

    // Change in state with respect to input perturbations
    pub dxdu: Mat<f64>,

    // Change in output with respect to input perturbations
    pub dydu: Mat<f64>,

    pub dudu: Mat<f64>,
    pub dudy: Mat<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModuleVars {
    nx: usize,
    pub x: Vec<StateVar>,
    nu: usize,
    pub u: Vec<InputOutputVar>,
    ny: usize,
    pub y: Vec<InputOutputVar>,
}

impl ModuleVars {
    pub fn new() -> Self {
        ModuleVars {
            nx: 0,
            x: Vec::new(),
            nu: 0,
            u: Vec::new(),
            ny: 0,
            y: Vec::new(),
        }
    }

    pub fn add_state_var(&mut self, name: &str, state_type: StateType) {
        let v = StateVar::new(name, state_type, self.nx);
        self.nx += 1;
        self.x.push(v);
    }

    pub fn add_input_var(&mut self, name: &str, mesh_fields: Fields, mesh: &Mesh) {
        let v = InputOutputVar::new(name, mesh_fields, mesh, self.nu);
        self.nu += v.n;
        self.u.push(v);
    }

    pub fn add_output_var(&mut self, name: &str, mesh_fields: Fields, mesh: &Mesh) {
        let v = InputOutputVar::new(name, mesh_fields, mesh, self.ny);
        self.ny += v.n;
        self.y.push(v);
    }

    pub fn n_states(&self) -> usize {
        self.nx
    }

    pub fn n_inputs(&self) -> usize {
        self.nu
    }

    pub fn n_outputs(&self) -> usize {
        self.ny
    }

    pub fn set_module_info(&mut self, module_id: ModuleId, instance: usize) {
        for var in &mut self.x {
            var.module_id = module_id.clone();
            var.instance = instance;
        }
        for var in &mut self.u {
            var.module_id = module_id.clone();
            var.instance = instance;
        }
        for var in &mut self.y {
            var.module_id = module_id.clone();
            var.instance = instance;
        }
    }

    pub fn set_state_rows(&mut self, first_row: usize) {
        let mut row = first_row;

        let mut var_row_hash: HashMap<String, usize> = HashMap::new();

        for var in &mut self.x {
            match var_row_hash.get(&var.name) {
                Some(r) => var.row = *r,
                None => {
                    var.row = row;
                    var_row_hash.insert(var.name.clone(), var.row);
                    row += 1;
                }
            };
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateVar {
    pub name: String,
    pub module_id: ModuleId,
    pub instance: usize,
    pub state_type: StateType,
    pub i: usize,   // Local state value index
    pub row: usize, // Global state row index
    pub col: usize, // Global state column index
}

impl StateVar {
    pub fn new(name: &str, state_type: StateType, i: usize) -> Self {
        let col = match state_type {
            StateType::TranslationalDisplacement => 0,
            StateType::TranslationalVelocity => 1,
            StateType::AngularDisplacement => 0,
            StateType::AngularVelocity => 1,
        };

        Self {
            name: name.to_string(),
            module_id: ModuleId::None,
            instance: 0,
            state_type,
            i,
            row: 0,
            col,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputOutputVar {
    pub name: String,
    pub module_id: ModuleId,
    pub instance: usize,
    pub mesh_id: usize,
    pub mesh_fields: Fields,
    pub i: usize,
    pub n: usize,
}

impl InputOutputVar {
    pub fn new(name: &str, mesh_fields: Fields, mesh: &Mesh, i: usize) -> Self {
        if !mesh.fields.contains(mesh_fields) {
            panic!(
                "Mesh ID {} does not contain fields {:?} required for variable '{}'.",
                mesh.id, mesh_fields, name
            );
        }
        Self {
            name: name.to_string(),
            module_id: ModuleId::None,
            instance: 0,
            mesh_id: mesh.id,
            mesh_fields,
            i,
            n: mesh.n_values(mesh_fields),
        }
    }
}

#[derive(Clone, Debug)]
pub struct InitInputBase {
    pub time_step: f64,
    pub interpolation_order: usize,
}
