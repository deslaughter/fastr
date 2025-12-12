use crate::{core::types::*, modules};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct System {
    pub elastodyn: modules::ElastoDyn,
}

pub struct SystemInitInput {
    pub elastodyn: Vec<modules::elastodyn::InitInput>,
}

impl System {
    pub fn new(init_input: &SystemInitInput) -> Self {
        //
        let mut elastodyn = modules::ElastoDyn::new();
        let _ed_init_output = init_input
            .elastodyn
            .iter()
            .map(|init_input| elastodyn.add_instance(init_input).unwrap());

        System { elastodyn }
    }
}
