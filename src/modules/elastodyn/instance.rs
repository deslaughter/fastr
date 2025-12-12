use std::f64::consts::FRAC_PI_2;
use std::f64::consts::PI;

use glam::DQuat;
use glam::DVec3;
use itertools::Itertools;

use crate::core;
use crate::core::mesh::Fields;
use crate::modules;
use crate::modules::ModuleVars;
use crate::modules::StateType;

use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Instance {
    pub parameters: Parameters,
    pub input_times: Vec<f64>,
    pub inputs: Vec<Input>,
    pub states: Vec<State>,
    pub u: Input,
    pub y: Output,
}

pub struct InitInput {
    pub base: modules::InitInputBase,
    pub n_blades: usize,
    pub hub_diameter: f64,
    pub rotor_speed: f64,
    pub azimuth: f64,
}

pub struct InitOutput {
    _module_vars: ModuleVars,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Parameters {
    n_blades: usize,
    u_mesh_ids: InputMeshIds,
    y_mesh_ids: OutputMeshIds,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputMeshIds {
    pub hub_load: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutputMeshIds {
    pub hub_motion: usize,
    pub blade_root_motion: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    // Hub translational displacement
    pub hub_ut: DVec3,

    // Hub rotational displacement
    pub hub_ur: DQuat,

    // Hub translational velocity
    pub hub_vt: DVec3,

    // Hub rotational velocity
    pub hub_vr: DVec3,
}

impl Instance {
    pub fn new(init_input: &InitInput) -> (Self, anyhow::Result<InitOutput>) {
        //----------------------------------------------------------------------
        // Meshes
        //----------------------------------------------------------------------

        // Create hub motion meshe
        let mut mb = core::mesh::Builder::new();
        mb.add_node().set_position(0., 0., 0.).build();
        let mut hub_motion_mesh = mb.build();

        // Create hub load mesh as sibling of hub motion mesh
        let hub_load_mesh = hub_motion_mesh.create_sibling();

        // Create blade root meshes
        let blade_root_motion_meshes = (0..init_input.n_blades)
            .map(|i| {
                let angle = 2.0 * PI / (init_input.n_blades as f64) * (i as f64);
                let mut mb = core::mesh::Builder::new();
                mb.add_node()
                    .set_position(init_input.hub_diameter / 2., 0., 0.)
                    .set_orientation(DQuat::IDENTITY)
                    .rotate(DQuat::from_rotation_y(-FRAC_PI_2))
                    .rotate(DQuat::from_rotation_x(angle))
                    .build();
                mb.build()
            })
            .collect_vec();

        //----------------------------------------------------------------------
        // Module variables
        //----------------------------------------------------------------------

        let mut mv = ModuleVars::new();

        // Add state variables
        mv.add_state_var("surge", StateType::TranslationalDisplacement);
        mv.add_state_var("sway", StateType::TranslationalDisplacement);
        // mv.add_state_var("heave", StateType::TranslationalDisplacement);
        // mv.add_state_var("roll", StateType::AngularDisplacement);
        // mv.add_state_var("pitch", StateType::AngularDisplacement);
        // mv.add_state_var("yaw", StateType::AngularDisplacement);

        mv.add_state_var("surge", StateType::TranslationalVelocity);
        mv.add_state_var("sway", StateType::TranslationalVelocity);
        // mv.add_state_var("heave", StateType::TranslationalVelocity);
        // mv.add_state_var("roll", StateType::AngularVelocity);
        // mv.add_state_var("pitch", StateType::AngularVelocity);
        // mv.add_state_var("yaw", StateType::AngularVelocity);

        // Add input variables
        mv.add_input_var("Hub Loads", Fields::Loads, &hub_load_mesh);

        // Add output variables
        mv.add_output_var("Hub Motion", Fields::Motion, &hub_motion_mesh);
        blade_root_motion_meshes.iter().for_each(|mesh| {
            mv.add_output_var(&format!("B{} Root Motion", mesh.id), Fields::Motion, mesh);
        });

        //----------------------------------------------------------------------
        // Input, Output, and State
        //----------------------------------------------------------------------

        // Create input
        let mut u = modules::Input::new();

        // Create output
        let mut y = modules::Output::new();

        //----------------------------------------------------------------------
        // Instance
        //----------------------------------------------------------------------

        // Initialize the ElastoDyn instance with given parameters
        let ins = Instance {
            parameters: Parameters {
                n_blades: init_input.n_blades,
                u_mesh_ids: InputMeshIds {
                    hub_load: u.add_mesh(hub_load_mesh),
                },
                y_mesh_ids: OutputMeshIds {
                    hub_motion: y.add_mesh(hub_motion_mesh),
                    blade_root_motion: blade_root_motion_meshes
                        .into_iter()
                        .map(|mesh| y.add_mesh(mesh))
                        .collect_vec(),
                },
            },
            input_times: vec![0.0; init_input.base.interpolation_order + 1],
            inputs: vec![u.clone(); init_input.base.interpolation_order + 1],
            states: Vec::new(),
            u,
            y,
        };

        // Create initialization output
        let init_out = InitOutput { _module_vars: mv };

        (ins, Ok(init_out))
    }

    pub fn _output(&self) -> anyhow::Result<()> {
        // Output calculation logic for ElastoDyn
        Ok(())
    }

    pub fn _continuous_state_derivatives(&self) -> anyhow::Result<()> {
        // Continuous state derivative calculation logic for ElastoDyn
        Ok(())
    }

    pub fn _jacobian_p_state(&self) -> anyhow::Result<()> {
        // Jacobian with respect to state logic for ElastoDyn
        Ok(())
    }

    pub fn _jacobian_p_input(&self) -> anyhow::Result<()> {
        // Jacobian with respect to input logic for ElastoDyn
        Ok(())
    }
}
