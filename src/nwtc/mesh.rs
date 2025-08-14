use itertools::Itertools;
use kiddo::float::kdtree::KdTree;
use kiddo::SquaredEuclidean;
use ndarray::prelude::*;
use std::{
    collections::HashSet,
    sync::atomic::{AtomicUsize, Ordering},
};
use thiserror::Error;

use crate::nwtc::quaternion::quat_compose;

use super::quaternion::quat_normalize;

fn get_id() -> usize {
    static COUNTER: AtomicUsize = AtomicUsize::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, Clone)]
struct Metadata {
    /// Unique ID of the mesh
    id: usize,

    /// Name of the mesh
    name: String,

    /// Usage of the mesh (Input, Output, State)
    usage: Usage,

    /// Sibling meshes identifiers
    sibling_ids: HashSet<usize>,
}

impl Metadata {
    fn new(name: &str, usage: &Usage) -> Self {
        Self {
            id: get_id(),
            name: name.to_string(),
            usage: usage.clone(),
            sibling_ids: HashSet::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PointElement {
    pub id: usize,
    pub node_id: usize,
}

#[derive(Debug, Clone)]
pub struct Line2Element {
    pub id: usize,
    pub node_1_id: usize,
    pub node_2_id: usize,
}

/// Element types
#[derive(Debug, Clone)]
pub enum Elements {
    None,
    Point(Vec<PointElement>),
    Line2(Vec<Line2Element>),
}

impl Elements {
    pub fn n_elements(&self) -> usize {
        match self {
            Elements::None => 0,
            Elements::Point(ref points) => points.len(),
            Elements::Line2(ref lines) => lines.len(),
        }
    }

    pub fn n_point_elements(&self) -> usize {
        match self {
            Elements::Point(ref points) => points.len(),
            _ => 0,
        }
    }

    pub fn n_line2_elements(&self) -> usize {
        match self {
            Elements::Line2(ref lines) => lines.len(),
            _ => 0,
        }
    }
}

/// Defines the usage of the mesh
#[derive(Debug, Clone)]
pub enum Usage {
    Input,
    Output,
    State,
}

//------------------------------------------------------------------------------
// Mesh
//------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Mesh {
    /// Metadata
    pub meta: Metadata,

    /// Element connectivity
    pub elements: Elements,

    /// Initial translational position of the nodes
    pub xref_trans: Array2<f64>,

    /// Initial orientation of the nodes
    xref_rot: Array2<f64>,

    /// Current translational position of the nodes
    x_trans: Array2<f64>,

    /// Current orientation of the nodes
    x_rot: Array2<f64>,

    /// Translational displacement of the nodes
    dx_trans: Array2<f64>,

    /// Rotational displacement of the nodes
    dx_rot: Array2<f64>,

    /// Translational velocity of the nodes
    v_trans: Array2<f64>,

    /// Rotational velocity of the nodes
    v_rot: Array2<f64>,

    /// Translational acceleration of the nodes
    a_trans: Array2<f64>,

    /// Rotational acceleration of the nodes
    a_rot: Array2<f64>,

    /// Force applied to the nodes
    force: Array2<f64>,

    /// Moment applied to the nodes
    moment: Array2<f64>,
}

impl Mesh {
    pub fn n_nodes(&self) -> usize {
        self.xref_trans.ncols()
    }

    /// Updates the current translational position of the nodes and returns a view.
    pub fn x_trans(&mut self) -> ArrayView2<f64> {
        if !self.dx_trans.is_empty() {
            ndarray::Zip::from(&mut self.x_trans)
                .and(self.xref_trans.view())
                .and(self.dx_trans.view())
                .for_each(|x, xref, dx| *x = *xref + *dx);
        }
        self.x_trans.view()
    }

    /// Updates the current rotational position of the nodes and returns a view.
    pub fn x_rot(&mut self) -> ArrayView2<f64> {
        if !self.dx_rot.is_empty() {
            ndarray::Zip::from(self.x_rot.columns_mut())
                .and(self.xref_rot.columns())
                .and(self.dx_rot.columns())
                .for_each(|x, xref, dx| quat_compose(dx, xref, x));
        }
        self.x_rot.view()
    }

    pub fn build_mapping(&self, destination: &Mesh) -> Result<MeshMapping, MappingError> {
        match (&self.elements, &destination.elements) {
            (Elements::None, _) => Err(MappingError::NoElementsInSourceMesh),
            (_, Elements::None) => Err(MappingError::NoElementsInDestinationMesh),
            (Elements::Point(source_elems), Elements::Point(destination_elems)) => {
                Ok(map_point_to_point(
                    source_elems,
                    self.x_trans.view(), // Source mesh current position
                    destination_elems,
                    destination.x_trans.view(), // Destination mesh current position
                ))
            }
            (Elements::Point(source_elems), Elements::Line2(destination_elems)) => {
                Err(MappingError::PointToLine2NotImplemented)
            }
            (Elements::Line2(source_elems), Elements::Point(destination_elems)) => {
                Err(MappingError::Line2ToPointNotImplemented)
            }
            (Elements::Line2(source_elems), Elements::Line2(destination_elems)) => {
                Err(MappingError::Line2ToLine2NotImplemented)
            }
        }
    }
}

fn map_point_to_point(
    source_elems: &[PointElement],
    source_x: ArrayView2<f64>,
    destination_elems: &[PointElement],
    destination_x: ArrayView2<f64>,
) -> MeshMapping {
    let mut kdtree: kiddo::float::kdtree::KdTree<f64, usize, 3, 32, u32> =
        KdTree::with_capacity(destination_x.ncols());

    // Add applicable point element node positions to the kdtree
    destination_elems.iter().for_each(|pe| {
        let c = destination_x.column(pe.node_id);
        kdtree.add(&[c[0], c[1], c[2]], pe.id);
    });

    MeshMapping {
        element_mapping: source_x
            .columns()
            .into_iter()
            .map(|source_node_x| {
                // Get nearest element node in destination mesh
                let nearest = kdtree.nearest_one::<SquaredEuclidean>(&[
                    source_node_x[0],
                    source_node_x[1],
                    source_node_x[2],
                ]);

                // Get get corresponding destination node position
                let destination_node_x = destination_x.column(nearest.item);

                // Create element mapping
                ElementMapping {
                    destination_elem_id: nearest.item,
                    distance: nearest.distance.sqrt(),
                    couple_arm: &destination_node_x - &source_node_x,
                    shape_function: Array1::default(0), // No shape function in point mapping
                }
            })
            .collect(),
    }
}

pub struct MeshMapping {
    /// Mapping data structure for loads on the mesh [-]
    pub element_mapping: Vec<ElementMapping>,
}

#[derive(Error, Debug)]
pub enum MappingError {
    #[error("No elements in source mesh")]
    NoElementsInSourceMesh,
    #[error("No elements in destination mesh")]
    NoElementsInDestinationMesh,
    #[error("Line2 to Point mapping not implemented")]
    Line2ToPointNotImplemented,
    #[error("Point to Line2 mapping not implemented")]
    PointToLine2NotImplemented,
    #[error("Line2 to Line2 mapping not implemented")]
    Line2ToLine2NotImplemented,
}

/// Mesh Builder
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
    has_force: bool,
    has_moment: bool,
}
impl MeshBuilder {
    pub fn new() -> Self {
        Self {
            elements: Elements::None,
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

    pub fn with_point_elements(mut self, node_ids: &[usize]) -> Self {
        if !matches!(self.elements, Elements::None) {
            panic!("Attempted to add point elements to a mesh that already has elements");
        }
        self.elements = Elements::Point(
            node_ids
                .iter()
                .enumerate()
                .map(|(i, &node_id)| PointElement { id: i, node_id })
                .collect(),
        );
        self
    }

    pub fn with_line2_elements(mut self, node_ids: &[(usize, usize)]) -> Self {
        if !matches!(self.elements, Elements::None) {
            panic!("Attempted to add line elements to a mesh that already has elements");
        }
        self.elements = Elements::Line2(
            node_ids
                .iter()
                .enumerate()
                .map(|(i, &(node_1_id, node_2_id))| Line2Element {
                    id: i,
                    node_1_id,
                    node_2_id,
                })
                .collect(),
        );
        self
    }

    pub fn build(&self, name: &str, usage: Usage) -> Result<Mesh, MeshBuilderError> {
        // Check if elements are defined
        if self.elements.n_elements() == 0 {
            return Err(MeshBuilderError::NoElementsDefined);
        }

        // Get list of node_ids used by elements
        let elem_node_ids: Vec<usize> = match self.elements {
            Elements::None => vec![],
            Elements::Point(ref points) => points.iter().map(|e| e.node_id).unique().collect(),
            Elements::Line2(ref lines) => lines
                .iter()
                .flat_map(|e| vec![e.node_1_id, e.node_2_id])
                .unique()
                .collect(),
        };

        // Get number of nodes expected by elements
        let n_nodes = 1 + elem_node_ids.iter().max().unwrap();

        // If no nodes
        if n_nodes == 0 {
            return Err(MeshBuilderError::NoNodesDefined);
        }

        // Insufficient nodes in mesh
        if n_nodes > self.node_position.len() {
            return Err(MeshBuilderError::InsufficientNodes(
                n_nodes,
                self.node_position.len(),
            ));
        }

        // To many nodes in mesh
        if n_nodes < self.node_position.len() {
            return Err(MeshBuilderError::ExtraNodes(
                n_nodes,
                self.node_position.len(),
            ));
        }

        // Check if all nodes are used, return error if not
        let mut node_used = vec![false; n_nodes];
        elem_node_ids.iter().for_each(|&n| {
            node_used[n] = true;
        });
        for (node_id, &used) in node_used.iter().enumerate() {
            if !used {
                return Err(MeshBuilderError::UnusedNode(node_id));
            }
        }

        // If node orientation is defined, but does not match the number of nodes
        if self.node_orientation.len() > 0 && self.node_orientation.len() != n_nodes {
            return Err(MeshBuilderError::InsufficientNodes(
                n_nodes,
                self.node_orientation.len(),
            ));
        }

        let x_trans: Array2<f64> =
            Array2::<f64>::from_shape_fn((n_nodes, 3), |(i, j)| self.node_position[i][j]);
        let x_rot = if self.node_orientation.len() == 0 {
            Array2::from_shape_fn((n_nodes, 4), |(i, _)| if i == 0 { 1.0 } else { 0.0 })
        } else {
            let mut tmp = Array2::from_shape_fn((n_nodes, 4), |(i, j)| self.node_orientation[i][j]);
            tmp.columns_mut()
                .into_iter()
                .for_each(|col| quat_normalize(col));
            tmp
        };

        let is_input_mesh = matches!(usage, Usage::Input);
        let has_translational_displacement = self.has_translational_displacement || is_input_mesh;
        let has_translational_velocity = self.has_translational_velocity || is_input_mesh;
        let has_translational_acceleration = self.has_translational_acceleration || is_input_mesh;

        Ok(Mesh {
            meta: Metadata {
                id: get_id(),
                name: name.to_string(),
                sibling_ids: HashSet::new(),
                usage,
            },
            elements: self.elements.clone(),
            xref_trans: x_trans.clone(),
            xref_rot: x_rot.clone(),
            x_trans,
            x_rot,
            dx_trans: if has_translational_displacement {
                Array2::default((n_nodes, 3))
            } else {
                Array2::default((0, 0))
            },
            dx_rot: if self.has_rotational_displacement {
                Array2::default((n_nodes, 4))
            } else {
                Array2::default((0, 0))
            },
            v_trans: if has_translational_velocity {
                Array2::default((n_nodes, 3))
            } else {
                Array2::default((0, 0))
            },
            v_rot: if self.has_rotational_velocity {
                Array2::default((n_nodes, 3))
            } else {
                Array2::default((0, 0))
            },
            a_trans: if has_translational_acceleration {
                Array2::default((n_nodes, 3))
            } else {
                Array2::default((0, 0))
            },
            a_rot: if self.has_rotational_acceleration {
                Array2::default((n_nodes, 3))
            } else {
                Array2::default((0, 0))
            },
            force: if self.has_force {
                Array2::default((n_nodes, 3))
            } else {
                Array2::default((0, 0))
            },
            moment: if self.has_moment {
                Array2::default((n_nodes, 3))
            } else {
                Array2::default((0, 0))
            },
        })
    }
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

mod test_mesh {
    use std::array;

    use crate::nwtc::quaternion::quat_from_rvec;

    use super::*;

    #[test]
    fn test_create_output_mesh() {
        // Create output motion mesh
        let n_nodes = 1;
        let mut m1 = MeshBuilder::new()
            .with_node_positions(&(0..n_nodes).map(|i| [0., 0., i as f64]).collect_vec())
            .with_node_orientations(
                &(0..n_nodes)
                    .map(|_| {
                        let angle = 20. * std::f64::consts::PI / 180.;
                        let q = quat_from_rvec(array![0., 0., angle].view());
                        [q[0], q[1], q[2], q[3]]
                    })
                    .collect_vec(),
            )
            .with_point_elements(&(0..n_nodes).collect_vec())
            .with_translational_displacement()
            .with_translational_velocity()
            .with_translational_acceleration()
            .with_rotational_velocity()
            .with_rotational_acceleration()
            .build("m1", Usage::Output)
            .unwrap();

        // Initialize mesh properties
        (0..n_nodes).for_each(|i| {
            // Translational displacement
            m1.dx_trans.row_mut(i).assign(&array![2., 0., 0.]);

            // Translational velocity
            m1.v_trans.row_mut(i).assign(&array![0.5, 0., 0.]);

            // Rotational velocity
            m1.v_rot.row_mut(i).assign(&array![2., 0., 0.]);

            // Translational acceleration
            m1.a_trans.row_mut(i).assign(&array![0., 0., 0.]);

            // Rotational acceleration
            m1.a_rot.row_mut(i).assign(&array![2., 0., 0.]);
        });

        // Create output loads mesh
        let n_nodes = 5;
        let dz = 1.0 / (n_nodes - 1) as f64;
        let mut m2 = MeshBuilder::new()
            .with_node_positions(&(0..n_nodes).map(|i| [0., 0., dz * i as f64]).collect_vec())
            .with_node_orientations(
                &(0..n_nodes)
                    .map(|i| {
                        let angle = (-25. + (i as f64 + 1.).powi(2)) * std::f64::consts::PI / 180.;
                        let q = quat_from_rvec(array![0., 0., angle].view());
                        [q[0], q[1], q[2], q[3]]
                    })
                    .collect_vec(),
            )
            .with_point_elements(&(0..n_nodes).collect_vec())
            .with_force()
            .with_moment()
            .build("m2", Usage::Output)
            .unwrap();

        // Set loads
        (0..n_nodes).for_each(|i| {
            m2.force.row_mut(i).assign(&array![1., 0., 0.]);
            m2.moment.row_mut(i).assign(&array![0., 0., 0.]);
        });
    }

    #[test]
    fn test_mesh_builder() {
        let mesh1 = MeshBuilder::new()
            .with_node_positions(&[[0., 0., 0.], [1., 0., 0.]])
            .with_translational_displacement()
            .with_line2_elements(&[(0, 1)])
            .build("t1", Usage::Input)
            .unwrap();

        assert_eq!(mesh1.meta.id, 1);
        assert_eq!(mesh1.meta.name, "t1");
        assert_eq!(mesh1.n_nodes(), 2);

        assert_eq!(mesh1.elements.n_elements(), 1);
        assert_eq!(mesh1.elements.n_point_elements(), 0);
        assert_eq!(mesh1.elements.n_line2_elements(), 1);

        assert_eq!(mesh1.x_trans.nrows(), 3);
        assert_eq!(mesh1.x_trans, array![[0., 0., 0.], [1., 0., 0.]].t());
        assert_eq!(mesh1.x_rot.nrows(), 4);
        assert_eq!(mesh1.x_rot, array![[1., 0., 0., 0.], [1., 0., 0., 0.]].t());
        assert_eq!(mesh1.dx_trans.dim(), (3, 2));
        assert_eq!(mesh1.dx_rot.is_empty(), true);
        assert_eq!(mesh1.v_trans.dim(), (3, 2));
        assert_eq!(mesh1.v_rot.is_empty(), true);
        assert_eq!(mesh1.a_trans.dim(), (3, 2));
        assert_eq!(mesh1.a_rot.is_empty(), true);
        // assert_eq!(mesh1.force.is_empty(), true);
        // assert_eq!(mesh1.moment.is_empty(), true);

        // assert_eq!(mesh1.sibling_ids.len(), 0);
        // assert_eq!(matches!(mesh1.usage, Usage::Input), true);

        // // Create a sibling mesh and check the sibling relationship
        // let mesh2 = mesh1.create_sibling();
        // assert_eq!(mesh1.sibling_ids.len(), 1);
        // assert_eq!(mesh2.sibling_ids.len(), 1);
        // assert_eq!(mesh1.sibling_ids.contains(&mesh2.id), true);
        // assert_eq!(mesh2.sibling_ids.contains(&mesh1.id), true);
        // assert_eq!(mesh2.sibling_ids.contains(&mesh2.id), false);
        // assert_eq!(mesh1.sibling_ids.contains(&mesh1.id), false);
    }
}

//------------------------------------------------------------------------------
// Mesh Mapping
//------------------------------------------------------------------------------

pub struct ElementMapping {
    pub destination_elem_id: usize,
    pub distance: f64,
    pub couple_arm: Array1<f64>,
    pub shape_function: Array1<f64>,
}

// mod test_mapping {
//     use faer::col;

//     use super::*;

//     #[test]
//     fn test_motion_mesh_builder() {
//         let source_mesh = MeshBuilder::new()
//             .with_translational_displacement()
//             .with_node_positions(&[[0., 0., 0.]])
//             .with_point(0)
//             .build("t1", Usage::Input)
//             .unwrap();

//         let destination_mesh = MeshBuilder::new()
//             .with_translational_displacement()
//             .with_node_positions(&[[1., 0., 0.], [2., 0., 0.]])
//             .with_point(0)
//             .with_point(1)
//             .build("t2", Usage::Input)
//             .unwrap();

//         let mapping = source_mesh.build_mapping(&destination_mesh);

//         assert_eq!(mapping.element_mapping.len(), 1);
//         assert_eq!(mapping.element_mapping[0].destination_elem_id, 0);
//         assert_eq!(mapping.element_mapping[0].distance, 1.0);
//         assert_eq!(mapping.element_mapping[0].couple_arm, col![1., 0., 0.]);
//     }
// }
