pub mod nwtc;

pub struct Var {
    pub name: String,
    pub i_mod: [usize; 2],
    pub i_val: [usize; 2],
}

pub struct VarSet {
    pub n_values: usize,
    pub vars: Vec<Var>,
    pub values: Vec<f64>,
}

pub struct ModVars {
    pub x: VarSet,
    pub u: VarSet,
    pub y: VarSet,
}

pub struct OperatingPoint {
    pub x: Vec<f64>,
    pub dx: Vec<f64>,
    pub u: Vec<f64>,
    pub y: Vec<f64>,
}

pub trait Module {
    fn calc_output(&self);
    fn calc_continuous_state_deriv(&self);
    fn update_states(&self);
    fn jacobian_p_input(&self);
    fn jacobian_p_continuous_states(&self);
    fn copy_states(&self, src: usize, dst: usize);
    fn copy_input(&self, src: usize, dst: usize);
    fn extrapolate_interpolate(&self, t: f64);
    fn get_operating_point(&self, vars: &ModVars) -> OperatingPoint;
    fn set_operating_point(&self, vars: &ModVars) -> OperatingPoint;
}
