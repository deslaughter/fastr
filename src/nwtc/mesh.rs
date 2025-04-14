use faer::{unzip, zip, Mat, MatRef};
use itertools::izip;
use std::{
    cmp::max,
    collections::HashSet,
    sync::atomic::{AtomicUsize, Ordering},
};
use thiserror::Error;

use super::quaternion::{quat_compose, quat_normalize};

#[derive(Debug, Clone)]
pub struct Mesh {
    /// Unique ID of the mesh
    id: usize,

    /// Name of the mesh
    name: String,

    /// Usage of the mesh (Input, Output, State)
    usage: Usage,

    /// Sibling meshes identifiers
    sibling_ids: HashSet<usize>,

    /// Element connectivity
    pub elements: Elements,

    /// Number of nodes in the mesh
    n_nodes: usize,

    /// Initial translational position of the nodes
    pub x0_trans: Mat<f64>,

    /// Initial orientation of the nodes
    x0_rot: Mat<f64>,

    /// Current translational position of the nodes
    x_trans: Mat<f64>,

    /// Current orientation of the nodes
    x_rot: Mat<f64>,

    /// Translational displacement of the nodes
    dx_trans: Option<Mat<f64>>,

    /// Rotational displacement of the nodes
    dx_rot: Option<Mat<f64>>,

    /// Translational velocity of the nodes
    v_trans: Option<Mat<f64>>,

    /// Rotational velocity of the nodes
    v_rot: Option<Mat<f64>>,

    /// Translational acceleration of the nodes
    a_trans: Option<Mat<f64>>,

    /// Rotational acceleration of the nodes
    a_rot: Option<Mat<f64>>,

    /// Translational force on the nodes
    force: Option<Mat<f64>>,

    /// Rotational moment on the nodes
    moment: Option<Mat<f64>>,

    /// Scalars associated with the nodes
    scalars: Option<Mat<f64>>,
}

impl Mesh {
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    pub fn x_trans(&mut self) -> MatRef<f64> {
        if let Some(dx_trans) = &self.dx_trans {
            zip!(&mut self.x_trans, &self.x0_trans, dx_trans)
                .for_each(|unzip!(x, x0, dx)| *x = *x0 + *dx);
        }
        self.x_trans.as_ref()
    }

    pub fn x_rot(&mut self) -> MatRef<f64> {
        if let Some(dx_rot) = &self.dx_rot {
            izip!(
                self.x_rot.col_iter_mut(),
                self.x0_rot.col_iter(),
                dx_rot.col_iter()
            )
            .for_each(|(x, x0, dx)| {
                quat_compose(dx, x0, x);
            });
        }
        self.x_rot.as_ref()
    }

    pub fn create_sibling(&mut self) -> Mesh {
        let mut sibling = self.clone();
        sibling.id = get_id();
        self.sibling_ids.insert(sibling.id);
        sibling.sibling_ids.insert(self.id);
        sibling
    }
}

#[derive(Debug, Clone)]
pub enum Usage {
    Input,
    Output,
    State,
}

#[derive(Debug, Clone)]
pub struct Elements {
    pub n_elements: usize,
    pub points: Vec<usize>,
    pub lines: Vec<[usize; 2]>,
}

impl Elements {
    pub fn n_elements(&self) -> usize {
        self.points.len() + self.lines.len()
    }

    pub fn n_nodes(&self) -> usize {
        let mut max_node_id: usize = 0;
        for point in &self.points {
            max_node_id = max(max_node_id, *point);
        }
        for line in &self.lines {
            max_node_id = max(max_node_id, line[0]);
            max_node_id = max(max_node_id, line[1]);
        }
        max_node_id + 1
    }

    pub fn unused_nodes(&self) -> Vec<usize> {
        // Get number of nodes used in the mesh
        let n_nodes = self.n_nodes();

        // Create a vector to track used nodes
        let mut used_nodes = vec![false; n_nodes];

        // Mark nodes used in points and lines
        for point in &self.points {
            used_nodes[*point] = true;
        }
        for line in &self.lines {
            used_nodes[line[0]] = true;
            used_nodes[line[1]] = true;
        }

        // Collect unused nodes
        (0..n_nodes).filter(|&i| !used_nodes[i]).collect()
    }
}

pub struct MeshBuilder {
    elements: Elements,
    node_position: Vec<[f64; 3]>,
    node_orientation: Vec<[f64; 4]>,
    has_translational_displacement: bool,
    has_rotational_displacement: bool,
    has_translational_velocity: bool,
    has_rotational_velocity: bool,
    has_translational_acceleration: bool,
    has_rotational_acceleration: bool,
    has_moment: bool,
    has_force: bool,
    n_scalars: usize,
}
impl MeshBuilder {
    pub fn new() -> Self {
        Self {
            elements: Elements {
                n_elements: 0,
                points: Vec::new(),
                lines: Vec::new(),
            },
            node_position: Vec::new(),
            node_orientation: Vec::new(),
            has_translational_displacement: false,
            has_rotational_displacement: false,
            has_translational_velocity: false,
            has_rotational_velocity: false,
            has_translational_acceleration: false,
            has_rotational_acceleration: false,
            has_force: false,
            has_moment: false,
            n_scalars: 0,
        }
    }

    pub fn with_node_positions(mut self, position: &[[f64; 3]]) -> Self {
        self.node_position.extend_from_slice(position);
        self
    }

    pub fn with_node_orientations(mut self, orientation: &[[f64; 4]]) -> Self {
        self.node_orientation.extend_from_slice(orientation);
        self
    }

    pub fn with_translational_displacement(mut self) -> Self {
        self.has_translational_displacement = true;
        self
    }

    pub fn with_rotational_displacement(mut self) -> Self {
        self.has_rotational_displacement = true;
        self
    }

    pub fn with_translational_velocity(mut self) -> Self {
        self.has_translational_velocity = true;
        self
    }

    pub fn with_rotational_velocity(mut self) -> Self {
        self.has_rotational_velocity = true;
        self
    }

    pub fn with_translational_acceleration(mut self) -> Self {
        self.has_translational_acceleration = true;
        self
    }

    pub fn with_rotational_acceleration(mut self) -> Self {
        self.has_rotational_acceleration = true;
        self
    }

    pub fn with_force(mut self) -> Self {
        self.has_force = true;
        self
    }

    pub fn with_moment(mut self) -> Self {
        self.has_moment = true;
        self
    }

    pub fn with_scalars(mut self, n_scalars_per_node: usize) -> Self {
        self.n_scalars = n_scalars_per_node;
        self
    }

    pub fn with_point(mut self, point: usize) -> Self {
        self.elements.points.push(point);
        self.elements.n_elements += 1;
        self
    }

    pub fn with_points(mut self, points: &[usize]) -> Self {
        self.elements.points.extend(points);
        self.elements.n_elements += points.len();
        self
    }

    pub fn with_line(mut self, line: [usize; 2]) -> Self {
        self.elements.lines.push(line);
        self.elements.n_elements += 1;
        self
    }

    pub fn with_lines(mut self, lines: &[[usize; 2]]) -> Self {
        self.elements.lines.extend(lines);
        self.elements.n_elements += lines.len();
        self
    }

    pub fn build(&self, name: &str, usage: Usage) -> Result<Mesh, MeshBuilderError> {
        // Check if elements are defined
        if self.elements.n_elements() == 0 {
            return Err(MeshBuilderError::NoElementsDefined);
        }

        // Get number of nodes expected by elements
        let n_nodes = self.elements.n_nodes();
        if n_nodes == 0 {
            return Err(MeshBuilderError::NoNodesDefined);
        }

        // Insufficient nodes in mesh
        if n_nodes < self.node_position.len() {
            return Err(MeshBuilderError::InsufficientNodes(
                n_nodes,
                self.node_position.len(),
            ));
        }

        // To many nodes in mesh
        if n_nodes > self.node_position.len() {
            return Err(MeshBuilderError::ExtraNodes(
                n_nodes,
                self.node_position.len(),
            ));
        }

        // Get unused nodes
        let unused_nodes = self.elements.unused_nodes();
        if unused_nodes.len() > 0 {
            return Err(MeshBuilderError::UnusedNode(unused_nodes[0]));
        }

        // If node orientation is defined, but does not match the number of nodes
        if self.node_orientation.len() > 0 && self.node_orientation.len() != n_nodes {
            return Err(MeshBuilderError::InsufficientNodes(
                n_nodes,
                self.node_orientation.len(),
            ));
        }

        let x_trans = Mat::from_fn(3, n_nodes, |i, j| self.node_position[j][i]);
        let x_rot = if self.node_orientation.len() == 0 {
            Mat::from_fn(4, n_nodes, |i, _| if i == 0 { 1.0 } else { 0.0 })
        } else {
            let mut tmp = Mat::from_fn(4, n_nodes, |i, j| self.node_orientation[j][i]);
            tmp.col_iter_mut().for_each(quat_normalize);
            tmp
        };

        let is_motion_mesh = self.has_translational_displacement
            | self.has_rotational_displacement
            | self.has_translational_velocity
            | self.has_rotational_velocity
            | self.has_translational_acceleration
            | self.has_rotational_acceleration;

        let is_load_mesh = self.has_force | self.has_moment;

        let is_input_mesh = matches!(usage, Usage::Input);

        Ok(Mesh {
            id: get_id(),
            name: name.to_string(),
            sibling_ids: HashSet::new(),
            usage,
            n_nodes,
            elements: self.elements.clone(),
            x0_trans: x_trans.clone(),
            x0_rot: x_rot.clone(),
            x_trans,
            x_rot,
            dx_trans: if self.has_translational_displacement || (is_motion_mesh && is_input_mesh) {
                Some(Mat::zeros(3, n_nodes))
            } else {
                None
            },
            dx_rot: if self.has_rotational_displacement {
                Some(Mat::from_fn(
                    4,
                    n_nodes,
                    |i, _| if i == 0 { 1.0 } else { 0.0 },
                ))
            } else {
                None
            },
            v_trans: if self.has_translational_velocity || (is_motion_mesh && is_input_mesh) {
                Some(Mat::zeros(3, n_nodes))
            } else {
                None
            },
            v_rot: if self.has_rotational_velocity {
                Some(Mat::zeros(3, n_nodes))
            } else {
                None
            },
            a_trans: if self.has_translational_acceleration || (is_motion_mesh && is_input_mesh) {
                Some(Mat::zeros(3, n_nodes))
            } else {
                None
            },
            a_rot: if self.has_rotational_acceleration {
                Some(Mat::zeros(3, n_nodes))
            } else {
                None
            },
            force: if self.has_force || (is_load_mesh && is_input_mesh) {
                Some(Mat::zeros(3, n_nodes))
            } else {
                None
            },
            moment: if self.has_moment || (is_load_mesh && is_input_mesh) {
                Some(Mat::zeros(3, n_nodes))
            } else {
                None
            },
            scalars: if self.n_scalars > 0 {
                Some(Mat::zeros(self.n_scalars, n_nodes))
            } else {
                None
            },
        })
    }
}

fn get_id() -> usize {
    static COUNTER: AtomicUsize = AtomicUsize::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[derive(Error, Debug)]
pub enum MeshBuilderError {
    #[error("No nodes defined")]
    NoNodesDefined,
    #[error("No elements defined")]
    NoElementsDefined,
    #[error("Node {0} is not used in any element")]
    UnusedNode(usize),
    #[error("Elements require {0} nodes, but only {1} are defined")]
    InsufficientNodes(usize, usize),
    #[error("Elements use {0} nodes, {1} are defined")]
    ExtraNodes(usize, usize),
}

mod test {
    use faer::mat;

    use super::*;

    #[test]
    fn test_motion_mesh_builder() {
        let mut mesh1 = MeshBuilder::new()
            .with_node_positions(&[[0., 0., 0.], [1., 0., 0.]])
            .with_translational_displacement()
            .with_line([0, 1])
            .build("t1", Usage::Input)
            .unwrap();

        assert_eq!(mesh1.id, 1);
        assert_eq!(mesh1.name, "t1");
        assert_eq!(mesh1.n_nodes(), 2);

        assert_eq!(mesh1.elements.n_elements(), 1);
        assert_eq!(mesh1.elements.n_nodes(), 2);
        assert_eq!(mesh1.elements.points.len(), 0);
        assert_eq!(mesh1.elements.lines.len(), 1);
        assert_eq!(mesh1.elements.lines[0], [0, 1]);

        assert_eq!(mesh1.x_trans.nrows(), 3);
        assert_eq!(mesh1.x_trans, mat![[0., 0., 0.], [1., 0., 0.]].transpose());
        assert_eq!(mesh1.x_rot.nrows(), 4);
        assert_eq!(
            mesh1.x_rot,
            mat![[1., 0., 0., 0.], [1., 0., 0., 0.]].transpose()
        );
        assert_eq!(mesh1.dx_trans, Some(Mat::zeros(3, 2)));
        assert_eq!(mesh1.dx_rot, None);
        assert_eq!(mesh1.v_trans, Some(Mat::zeros(3, 2)));
        assert_eq!(mesh1.v_rot, None);
        assert_eq!(mesh1.a_trans, Some(Mat::zeros(3, 2)));
        assert_eq!(mesh1.a_rot, None);
        assert_eq!(mesh1.force, None);
        assert_eq!(mesh1.moment, None);
        assert_eq!(mesh1.scalars, None);

        assert_eq!(mesh1.sibling_ids.len(), 0);
        assert_eq!(matches!(mesh1.usage, Usage::Input), true);

        // Create a sibling mesh and check the sibling relationship
        let mesh2 = mesh1.create_sibling();
        assert_eq!(mesh1.sibling_ids.len(), 1);
        assert_eq!(mesh2.sibling_ids.len(), 1);
        assert_eq!(mesh1.sibling_ids.contains(&mesh2.id), true);
        assert_eq!(mesh2.sibling_ids.contains(&mesh1.id), true);
        assert_eq!(mesh2.sibling_ids.contains(&mesh2.id), false);
        assert_eq!(mesh1.sibling_ids.contains(&mesh1.id), false);
    }
}
