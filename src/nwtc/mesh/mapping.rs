use itertools::Itertools;
use kiddo::{float::kdtree::KdTree, SquaredEuclidean};

use crate::nwtc::mesh::Element;
use crate::nwtc::mesh::Mesh;
use crate::nwtc::Vector3;

/// Represents a mapping between elements in different meshes.
///
/// This structure stores the relationship between a source element and a destination
/// element, including spatial information and interpolation data needed for
/// transferring motion and loads between meshes.
///
/// # Fields
///
/// * `destination_elem_id` - The ID of the destination element in the target mesh
/// * `distance` - The squared Euclidean distance between source and destination elements
/// * `couple_arm` - The vector from destination to source element position
/// * `shape_function` - Interpolation weights for multi-node elements (currently unused for point elements)
pub struct ElementMapping {
    /// The ID of the destination element in the target mesh
    pub destination_elem_id: usize,
    /// The squared Euclidean distance between source and destination elements
    pub distance: f64,
    /// The vector from destination to source element position
    pub couple_arm: Vector3,
    /// Interpolation weights for multi-node elements (currently unused for point elements)
    pub shape_function: Vec<f64>,
}

/// A comprehensive mapping between two meshes for motion and load transfer.
///
/// This structure establishes relationships between elements of different types
/// in source and destination meshes. It supports various mapping strategies
/// including point-to-point, line-to-point, point-to-line, and line-to-line
/// mappings for transferring motion, velocities, accelerations, and loads.
///
/// The mapping is built using spatial proximity algorithms (KD-tree) to find
/// the nearest elements between meshes and establish coupling relationships.
///
pub struct Mapping {
    /// The ID of the source mesh
    pub source_id: usize,
    /// The ID of the destination mesh
    pub destination_id: usize,
    /// Mapping from source nodes to destination point elements
    point_to_point: Vec<ElementMapping>,
    // Future mapping types (currently commented out):
    // /// Mapping from source line elements to destination point elements
    // line2_to_point: Vec<ElementMapping>,
    // /// Mapping from source nodes to destination line elements
    // point_to_line2: Vec<ElementMapping>,
    // /// Mapping from source line elements to destination line elements
    // line2_to_line2: Vec<ElementMapping>,
}

impl Mapping {
    /// Creates a new mesh mapping between source and destination meshes.
    ///
    /// This constructor builds spatial relationships between the meshes using
    /// a KD-tree for efficient nearest neighbor searches. Currently supports
    /// point-to-point mapping where each source node is mapped to the nearest
    /// destination point element.
    ///
    /// # Arguments
    ///
    /// * `source` - The source mesh containing elements that will drive the motion
    /// * `destination` - The destination mesh that will receive the transferred motion
    ///
    /// # Returns
    ///
    /// A new `MeshMapping` instance with established element relationships
    ///
    /// # Algorithm
    ///
    /// 1. Builds a KD-tree from all point elements in the destination mesh
    /// 2. For each source node, finds the nearest destination point element
    /// 3. Stores the mapping relationship with distance and coupling arm information
    ///
    pub fn new(source: &Mesh, destination: &Mesh) -> Self {
        // Build KD-tree for efficient spatial searching of destination elements
        let mut kdtree: kiddo::float::kdtree::KdTree<f64, usize, 3, 32, u32> =
            KdTree::with_capacity(destination.nodes.len());

        // Add applicable elements to the kd-tree
        // Filter for point elements only and ensure uniqueness by node ID
        destination
            .elements
            .iter()
            .enumerate()
            .filter_map(|(elem_id, e)| match e {
                Element::Point(node_id) => Some((elem_id, node_id)),
                _ => None,
            })
            .unique_by(|(_, &node_id)| node_id)
            .for_each(|(elem_id, &node_id)| {
                kdtree.add(&destination.nodes[node_id].x0.to_array(), elem_id);
            });

        Self {
            source_id: source.id,
            destination_id: destination.id,
            // Create point-to-point mappings for all source nodes
            point_to_point: source
                .nodes
                .iter()
                .map(|node_source| {
                    // Find the nearest destination point element
                    let nearest_node =
                        kdtree.nearest_one::<SquaredEuclidean>(&node_source.x0.to_array());
                    let destination_elem_id = nearest_node.item;
                    let destination_elem = &destination.elements[destination_elem_id];
                    let destination_node_id = match destination_elem {
                        Element::Point(nid) => *nid,
                        _ => panic!("Mapped element is not a point element"),
                    };
                    ElementMapping {
                        destination_elem_id,
                        distance: nearest_node.distance,
                        // Calculate coupling arm from destination to source
                        couple_arm: node_source.x0 - destination.nodes[destination_node_id].x0,
                        shape_function: Vec::new(), // Not used for point-to-point mapping
                    }
                })
                .collect(),
            // Future mapping types (currently not implemented):
            // line2_to_point: Vec::new(),
            // point_to_line2: Vec::new(),
            // line2_to_line2: Vec::new(),
        }
    }

    /// Maps motion, velocity, and acceleration from source mesh to destination mesh.
    ///
    /// This function transfers kinematic quantities (displacements, rotations, velocities,
    /// and accelerations) from the source mesh to the destination mesh using the
    /// established mapping relationships. It accounts for rigid body motion including
    /// the coupling between translational and rotational motion at offset points.
    ///
    /// # Arguments
    ///
    /// * `source` - The source mesh containing the motion to be transferred
    /// * `destination` - The destination mesh that will receive the transferred motion (mutable)
    ///
    /// # Physics
    ///
    /// The motion transfer accounts for rigid body kinematics:
    ///
    /// **Displacement:**
    /// - `u_d = u_s + p0 - R_s * p0` where p0 is the initial coupling arm
    ///
    /// **Velocity:**
    /// - `v_d = v_s + p × ω_s` where p is the current coupling arm and ω_s is angular velocity
    ///
    /// **Acceleration:**
    /// - `a_d = a_s + p × α_s + ω_s × (p × ω_s)` including centripetal acceleration
    ///
    pub fn map_motion(&self, source: &Mesh, destination: &mut Mesh) {
        // Map point-to-point motions
        self.point_to_point
            .iter()
            .enumerate()
            .for_each(|(source_node_id, mapping)| {
                let node_s = &source.nodes[source_node_id];
                match &destination.elements[mapping.destination_elem_id] {
                    Element::Point(destination_node_id) => {
                        let node_d = &mut destination.nodes[*destination_node_id];

                        // Initial position vector from source to destination
                        let p0 = node_s.x0 - node_d.x0;

                        // Current position vector from source to destination (rotated)
                        let p = node_s.ur.rotate_vector(&p0);

                        // Map translational displacement
                        // u_d = u_s + (p0 - R*p0) accounts for rotation of the coupling arm
                        node_d.ut = node_s.ut + p0 - node_s.ur.rotate_vector(&p0);

                        // Map rotational displacement (rigid body assumption)
                        node_d.ur = node_s.ur;

                        // Map translational velocity
                        // v_d = v_s + p × ω_s (velocity due to rotation at offset point)
                        node_d.vt = node_s.vt + p.cross(&node_s.vr);

                        // Map rotational velocity (rigid body assumption)
                        node_d.vr = node_s.vr;

                        // Map translational acceleration
                        // a_d = a_s + p × α_s + ω_s × (p × ω_s)
                        // Includes both angular acceleration and centripetal acceleration terms
                        node_d.ax =
                            node_s.ax + p.cross(&node_s.ar) + node_s.vr.cross(&p.cross(&node_s.vr));

                        // Map rotational acceleration (rigid body assumption)
                        node_d.ar = node_s.ar;
                    }
                    _ => panic!("Mapped element is not a point element"),
                }
            });
    }

    pub fn map_loads(&self, source: &Mesh, destination: &mut Mesh) {
        // Map point-to-point loads
        self.point_to_point
            .iter()
            .enumerate()
            .for_each(|(source_node_id, mapping)| {
                let node_s = &source.nodes[source_node_id];
                match &destination.elements[mapping.destination_elem_id] {
                    Element::Point(destination_node_id) => {
                        let node_d = &mut destination.nodes[*destination_node_id];

                        // Map forces directly
                        node_d.f += node_s.f;

                        // Calculate moment arm in current configuration
                        let moment_arm = node_s.ur.rotate_vector(&mapping.couple_arm);

                        // Map moments including couple from force at offset point
                        node_d.m += node_s.m + moment_arm.cross(&node_s.f);
                    }
                    _ => panic!("Mapped element is not a point element"),
                }
            });
    }
}
