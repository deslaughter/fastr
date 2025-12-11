pub mod solver;

use crate::modules;

pub struct Turbine {
    pub elastodyn: modules::ElastoDyn,
}

pub struct TurbineInitInput {
    pub elastodyn: Vec<modules::elastodyn::InitInput>,
}

impl Turbine {
    pub fn new(init_input: &TurbineInitInput) -> Self {

        //
        let mut elastodyn = modules::ElastoDyn::new();
        for ed_init in &init_input.elastodyn {
            elastodyn.add_instance(ed_init).unwrap();
        }

        Turbine { elastodyn }
    }
}
